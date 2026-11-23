"""Merge trained LoRA adapters into a complete Inkling-2 checkpoint that SGLang can reload.

Only the byte ranges of the merged BF16 tensors are rewritten (same shapes and dtypes, so the
safetensors header and every offset stay valid, and the NVFP4 experts stay byte-identical):
  attention wq_du / wk_dv / wv_dv / wo_ud (24 layers), router mlp.gate.weight (24 layers),
  visual.proj_in / visual.proj_out  (122 tensors).

Default: copy the whole checkpoint dir to --dst every sync, then patch.
--in-place: copy the base checkpoint to --dst only when it is missing or its header differs from
--src (put --dst on /dev/shm for zero SSD wear), then os.pwrite just the merged tensors. Every
merge is computed from --src's base weights, so repeated syncs never compound; tensors patched by
an earlier sync and absent from this LoRA are restored from --src (tracked in a sidecar file).

python3 -m inkling2_train_opt.merge --lora /work/lora.pt --src /models/... --dst /dev/shm/ckpt --in-place
"""

import argparse
import json
import os
import re
import shutil
import struct
import time

ATTN = {"wq": "wq_du", "wk": "wk_dv", "wv": "wv_dv", "wo": "wo_ud"}
ST_NAME = "model-00001-of-00001.safetensors"
PATCHED_SIDECAR = ".lapis_patched_tensors.json"


def target_name(module):
    m = re.fullmatch(r"layers\.(\d+)\.attn\.(wq|wk|wv|wo)", module)
    if m:
        return f"model.llm.layers.{m.group(1)}.attn.{ATTN[m.group(2)]}.weight"
    m = re.fullmatch(r"layers\.(\d+)\.router\.lin", module)
    if m:
        return f"model.llm.layers.{m.group(1)}.mlp.gate.weight"
    m = re.fullmatch(r"visual\.(proj_in|proj_out)", module)
    if m:
        return f"model.visual.{m.group(1)}.weight"
    raise KeyError(module)


# --------------------------------------------------------------------------- safetensors bytes (stdlib only)


def read_header(path):
    """-> (header dict, absolute offset of the data section, raw header bytes incl. the 8-byte length)."""
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        hlen = struct.unpack("<Q", raw_len)[0]
        raw = fh.read(hlen)
    if len(raw) != hlen:
        raise ValueError(f"{path}: truncated header")
    return json.loads(raw), 8 + hlen, raw_len + raw


def tensor_span(header, base, name):
    """Absolute [start, end) byte range of tensor ``name``."""
    start, end = header[name]["data_offsets"]
    return base + start, base + end


def pwrite_tensors(path, patches, dtype="BF16"):
    """Overwrite tensors in place with ``os.pwrite``. ``patches``: {tensor name: bytes}. Each must
    already exist with ``dtype`` and exactly its byte length, so the header is never touched.
    Returns the number of bytes written."""
    header, base, _ = read_header(path)
    spans = {}
    for name, data in patches.items():
        info = header.get(name)
        if info is None:
            raise KeyError(f"{name} not in {path}")
        if info["dtype"] != dtype:
            raise ValueError(f"{name}: dtype {info['dtype']} != {dtype}")
        start, end = tensor_span(header, base, name)
        if len(data) != end - start:
            raise ValueError(f"{name}: {len(data)} bytes != {end - start}")
        spans[name] = start
    size = os.path.getsize(path)
    written = 0
    fd = os.open(path, os.O_RDWR)
    try:
        for name, data in patches.items():
            mv, off = memoryview(data), spans[name]
            while mv:
                n = os.pwrite(fd, mv, off)
                mv, off, written = mv[n:], off + n, written + n
    finally:
        os.close(fd)
    assert os.path.getsize(path) == size, "pwrite changed the file size"
    return written


def read_tensor_bytes(path, names):
    header, base, _ = read_header(path)
    out = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        for name in names:
            start, end = tensor_span(header, base, name)
            out[name] = os.pread(fd, end - start, start)
    finally:
        os.close(fd)
    return out


def ensure_base_copy(src_dir, dst_dir):
    """Make --dst a full copy of --src once. Re-copies the weights only when they are missing, a
    different size, or their header bytes differ from --src's. Returns True when it copied."""
    os.makedirs(dst_dir, exist_ok=True)
    copied = False
    for name in os.listdir(src_dir):
        s, d = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
            continue
        if name == ST_NAME:
            fresh = (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s)
                     and read_header(d)[2] == read_header(s)[2])
            if fresh:
                continue
            tmp = d + ".partial"
            shutil.copyfile(s, tmp)  # never hardlink: the weights file is patched in place
            os.replace(tmp, d)
            sidecar = os.path.join(dst_dir, PATCHED_SIDECAR)
            if os.path.exists(sidecar):
                os.remove(sidecar)
            copied = True
        elif not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s):
            shutil.copyfile(s, d)
    return copied


def load_patched(dst_dir):
    p = os.path.join(dst_dir, PATCHED_SIDECAR)
    return set(json.load(open(p))) if os.path.exists(p) else set()


def save_patched(dst_dir, names):
    p = os.path.join(dst_dir, PATCHED_SIDECAR)
    with open(p + ".tmp", "w") as fh:
        json.dump(sorted(names), fh)
    os.replace(p + ".tmp", p)


# --------------------------------------------------------------------------- merge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--src", default="/models/thinkingmachines--Inkling-2-Mini-heretic")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--in-place", action="store_true",
                    help="copy the base once, then pwrite only the merged tensors (use a /dev/shm --dst)")
    args = ap.parse_args()

    import torch  # heavy; kept out of module import so the byte helpers test offline
    from safetensors import safe_open

    t0 = time.time()
    blob = torch.load(args.lora, map_location="cpu")
    rank, state = blob["rank"], blob["state"]
    scale = blob.get("alpha", rank) / rank
    modules = sorted({k.rsplit(".", 1)[0] for k in state})
    if args.in_place:
        copied_base = ensure_base_copy(args.src, args.dst)
    else:
        os.makedirs(args.dst, exist_ok=True)
        for name in os.listdir(args.src):
            s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copyfile(s, d)  # never hardlink: the weights file is patched below
        copied_base = True
    t_copy = time.time()
    path = os.path.join(args.dst, ST_NAME)
    header, _, _ = read_header(path)
    src_path = os.path.join(args.src, ST_NAME)
    src = safe_open(src_path, "pt", device="cpu")
    patches, changed = {}, []
    for mod in modules:
        tname = target_name(mod)
        a, b = state[mod + ".lora_A"].float(), state[mod + ".lora_B"].float()
        w = src.get_tensor(tname)
        merged = (w.float() + scale * (b @ a)).to(w.dtype).contiguous()
        info = header[tname]
        assert list(merged.shape) == info["shape"] and info["dtype"] == "BF16", (tname, info)
        patches[tname] = merged.view(torch.uint8).numpy().tobytes()
        changed.append({"tensor": tname, "delta_rel_fro": ((merged.float() - w.float()).norm() / w.float().norm()).item()})
    t_merge = time.time()
    restored = []
    if args.in_place:
        stale = sorted(load_patched(args.dst) - set(patches))
        if stale:
            pwrite_tensors(path, read_tensor_bytes(src_path, stale))
            restored = stale
    nbytes = pwrite_tensors(path, patches)
    if args.in_place:
        save_patched(args.dst, patches)
    t_write = time.time()
    out = {"dst": args.dst, "in_place": args.in_place, "copied_base": copied_base,
           "merged_tensors": len(changed), "restored_tensors": len(restored), "bytes_written": nbytes,
           "seconds": t_write - t0, "copy_s": t_copy - t0, "merge_s": t_merge - t_copy, "pwrite_s": t_write - t_merge,
           "max_delta_rel_fro": max(c["delta_rel_fro"] for c in changed), "tensors": changed}
    json.dump(out, open(os.path.join(os.path.dirname(args.lora), "merge_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "tensors"}))


if __name__ == "__main__":
    main()
