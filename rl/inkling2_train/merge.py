"""Merge trained LoRA adapters into a complete Inkling-2 checkpoint that SGLang can reload.

Copies the checkpoint dir, then overwrites in place only the byte ranges of the merged BF16
tensors (same shapes and dtypes, so every safetensors offset stays valid):
  attention wq_du / wk_dv / wv_dv / wo_ud (24 layers), router mlp.gate.weight (24 layers),
  visual.proj_in / visual.proj_out.

python3 -m inkling2_train.merge --lora /work/lora.pt --src /models/... --dst /merged/ckpt
"""

import argparse
import json
import os
import re
import shutil
import struct
import time

import torch
from safetensors import safe_open

ATTN = {"wq": "wq_du", "wk": "wk_dv", "wv": "wv_dv", "wo": "wo_ud"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--src", default="/models/thinkingmachines--Inkling-2-Mini-heretic")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    t0 = time.time()
    blob = torch.load(args.lora, map_location="cpu")
    rank, state = blob["rank"], blob["state"]
    scale = blob.get("alpha", rank) / rank
    modules = sorted({k.rsplit(".", 1)[0] for k in state})
    os.makedirs(args.dst, exist_ok=True)
    st_name = "model-00001-of-00001.safetensors"
    for name in os.listdir(args.src):
        s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copyfile(s, d)  # never hardlink: the weights file is patched below
    path = os.path.join(args.dst, st_name)
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hlen))
    base = 8 + hlen
    src = safe_open(os.path.join(args.src, st_name), "pt", device="cpu")
    changed = []
    with open(path, "r+b") as fh:
        for mod in modules:
            tname = target_name(mod)
            a, b = state[mod + ".lora_A"].float(), state[mod + ".lora_B"].float()
            w = src.get_tensor(tname)
            merged = (w.float() + scale * (b @ a)).to(w.dtype).contiguous()
            info = header[tname]
            assert list(merged.shape) == info["shape"] and info["dtype"] == "BF16", (tname, info)
            start, end = info["data_offsets"]
            data = merged.view(torch.uint8).numpy().tobytes()
            assert len(data) == end - start
            fh.seek(base + start)
            fh.write(data)
            changed.append({"tensor": tname, "delta_rel_fro": ((merged.float() - w.float()).norm() / w.float().norm()).item()})
    out = {"dst": args.dst, "merged_tensors": len(changed), "seconds": time.time() - t0,
           "max_delta_rel_fro": max(c["delta_rel_fro"] for c in changed), "tensors": changed}
    json.dump(out, open(os.path.join(os.path.dirname(args.lora), "merge_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "tensors"}))


if __name__ == "__main__":
    main()
