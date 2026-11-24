"""Dense, high-resolution DESKTOP grounding data for Studio's L3_VLM grounder (closing the ScreenSpot-Pro gap).

Why: Stage 1-5 grounding was mostly 1024x768 / 1280x720 web pages with large targets, while ScreenSpot-Pro is
2.5K-4K pro-app screenshots (IDE, CAD, creative, office, macOS) whose targets are 12-20 px toolbar icons. Studio
downsizes every screenshot to <= 1536 px wide (stage1_common.MAX_W), so on a 2940 px Retina capture a 16 px icon
shrinks to ~8 px. This pipeline adds (a) dense high-res desktop screenshots with small targets and (b) native-pixel
1280x800 crops around the target (no resize at all in Studio's pipeline), so the ViT sees sharp tiny icons.

Sources (both disjoint from the held-out evals, asserted per image by sha256, basename and a 256-bit dHash):
  ServiceNow/GroundCUA (MIT)        87 desktop apps (IDEs, CAD, GIMP/Krita/Blender, LibreOffice/OnlyOffice, ...),
                                    human-annotated boxes + LLM-written instructions from instruction_tuning.tar.gz.
                                    Only images >= --min-width px wide are taken; PNGs fetched one by one.
  OS-Copilot/OS-Atlas-data macOS    AX-tree boxes on real macOS screenshots; the 5.5 GB zip is never downloaded:
                                    its central directory and only the chosen members are read with HTTP ranges.

Every row (one JSONL line) uses fetch_grounding.py's L3_VLM grounder schema plus training fields:
  {"id", "image_path" (absolute), "width", "height", "instruction", "bbox_1000": [x0,y0,x1,y1], "center_1000": [x,y],
   "source", "split" ("train"/"val", 90/10 per IMAGE; crops inherit their parent's split), "platform", "application",
   "ui_type", "bbox_px", "area_frac", "variant" ("full"/"crop"), "parent_id", "crop_xywh", "image_sha256",
   "instruction_kind", "prompt" (sf.grounder_prompt), "target" ('{"point":[x,y]}', compact, as build_mix writes)}

Outputs under <root> (default ~/cua_datasets/dense_grounding, everything counted against --cap-gb, default 4.0):
  raw/...                    downloaded images + instruction tarball
  crops/<sha1>.jpg           1280x800 JPEG q=80 crops (Studio's own JPEG quality; studio_jpeg passes them untouched)
  grounding/dense_{train,val,all}.jsonl, grounding/manifest.json, eval_index.json (cached eval-image hashes)

  uv run python data_prep/fetch_dense_grounding.py               # full fetch + crops + leakage gate
  uv run python data_prep/fetch_dense_grounding.py --verify-only # re-run the gates on the written jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
sys.path.insert(0, os.path.join(RL, "overnight_stage1"))
import studio_format as sf  # noqa: E402

HF = "https://huggingface.co"
SPLIT_SALT = "lapis-dense-grounding-v1"
CROP_W, CROP_H = 1280, 800
CROP_JPEG_Q = 80  # == stage1_common.JPEG_Q: the exact bytes Studio would send for a <= 1536 px JPEG
HELDOUT_SOURCES = ("screenspot_pro", "osworld_g")
SMALL_AREA = 0.01  # "small box": < 1% of the screen
DENSE_AREA = 0.02  # selection preference: < 2% of the screen
GCUA_REPO = "ServiceNow/GroundCUA"
ATLAS_REPO = "OS-Copilot/OS-Atlas-data"
# pro-desktop apps closest to ScreenSpot-Pro's categories (dev, creative, CAD, office, scientific): 2 picks per round
PRO_APPS = {
    "Arduino IDE", "Atom", "Blender", "Brackets", "Code::Blocks", "Darktable", "Eclipse", "FontForge", "FreeCAD", "GIMP",
    "GNU Octave", "Geany", "Gnumeric", "GrassGIS", "Inkscape", "IntelliJ IDEA", "KDevelop", "Komodo Edit", "Krita",
    "LibreOffice Calc", "LibreOffice Draw", "LibreOffice Impress", "LibreOffice Writer", "Lightworks", "MuseScore",
    "Natron", "NetBeans", "OBS Studio", "OnlyOffice Document Editor", "OnlyOffice Presentation", "OnlyOffice Spreadsheet",
    "OpenShot", "OpenToonz", "PyCharm", "QGIS", "Qt Creator", "RStudio", "Scribus", "Shotcut", "Spyder", "VSCode",
    "Veusz", "draw.io", "Audacity",
}


# --------------------------------------------------------------------------- small utils


def _headers(extra: dict | None = None) -> dict:
    h = {"User-Agent": "dense-grounding-fetch/1.0"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        h["Authorization"] = "Bearer " + tok
    h.update(extra or {})
    return h


def _open(url: str, headers: dict | None = None, timeout: int = 120, retries: int = 6):
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=_headers(headers)), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError("unreachable")


def resolve_url(repo: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def du_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class Budget:
    """Hard cap on everything stored under root. check() before every batch that writes."""

    def __init__(self, root: str, cap_bytes: int):
        self.root, self.cap = root, cap_bytes

    def used(self) -> int:
        return du_bytes(self.root)

    def left(self) -> int:
        return self.cap - self.used()

    def check(self, incoming: int, what: str) -> None:
        used = self.used()
        if used + incoming > self.cap:
            raise SystemExit(f"cap: {what} needs {incoming / 1e9:.3f} GB, {used / 1e9:.3f} GB used of {self.cap / 1e9:.2f} GB")


def download(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with _open(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def png_dims(head: bytes) -> tuple[int, int] | None:
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def dhash256(path: str) -> int:
    """256-bit difference hash on a 17x16 grayscale thumbnail: survives resizing / re-encoding of the same screenshot."""
    from PIL import Image

    with Image.open(path) as im:
        im.draft("L", (68, 64))
        g = im.convert("L").resize((17, 16), Image.Resampling.BILINEAR)
    px = g.tobytes()
    bits = 0
    for y in range(16):
        for x in range(16):
            bits = (bits << 1) | (px[y * 17 + x] > px[y * 17 + x + 1])
    return bits


def split_for(source: str, image_key: str, val_pct: int = 10) -> str:
    h = int(hashlib.sha1(f"{SPLIT_SALT}|{source}|{image_key}".encode()).hexdigest(), 16) % 100
    return "val" if h < val_pct else "train"


def hrank(*parts) -> int:
    """Deterministic pseudo-random order key."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def to_1000(v: float, d: int) -> int:
    return max(0, min(1000, round(v / d * 1000)))


def box_1000(px_box, w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = px_box
    return [to_1000(min(x0, x1), w), to_1000(min(y0, y1), h), to_1000(max(x0, x1), w), to_1000(max(y0, y1), h)]


def center_of(b: list[int]) -> list[int]:
    return [round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)]


def strictly_inside(c, b) -> bool:
    return b[0] < c[0] < b[2] and b[1] < c[1] < b[3]


def clean_instruction(s: str) -> str:
    return " ".join(s.replace("`", "").split())


def make_row(*, sid, image_path, w, h, px_box, instruction, kind, source, split, sha, **extra) -> dict | None:
    """One L3_VLM grounder row, or None when the box is degenerate at 0..1000 resolution (center not strictly inside).
    The pixel box is rounded to 0.01 px first so the stored bbox_px reproduces bbox_1000 exactly."""
    px_box = [round(v, 2) for v in px_box]
    b = box_1000(px_box, w, h)
    c = center_of(b)
    if not strictly_inside(c, b) or not instruction:
        return None
    x0, y0, x1, y1 = px_box
    row = {"id": sid, "image_path": image_path, "width": int(w), "height": int(h), "instruction": instruction,
           "bbox_1000": b, "center_1000": c, "source": source, "split": split,
           "bbox_px": [round(v, 2) for v in (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))],
           "area_frac": round(abs(x1 - x0) * abs(y1 - y0) / (w * h), 8), "image_sha256": sha, "instruction_kind": kind}
    row.update({k: v for k, v in extra.items() if v is not None})
    return add_prompt(row)


def add_prompt(row: dict) -> dict:
    """Studio's exact L3_VLM grounder turn: buildInklingGroundRequest prompt -> compact {"point":[x,y]} reply."""
    row["prompt"] = sf.grounder_prompt(row["instruction"])
    row["target"] = json.dumps({"point": [int(row["center_1000"][0]), int(row["center_1000"][1])]}, separators=(",", ":"))
    return row


# --------------------------------------------------------------------------- held-out eval index + leakage gate


def _eval_entry(path: str) -> dict:
    return {"path": path, "basename": os.path.basename(path), "sha256": sha256_file(path), "dhash": str(dhash256(path)),
            "size": os.path.getsize(path), "mtime": int(os.path.getmtime(path))}


def eval_jsonl_paths(grounding_dir: str) -> list[str]:
    out = []
    for name in HELDOUT_SOURCES:
        out += sorted(glob.glob(os.path.join(grounding_dir, f"{name}*.jsonl")))
    return out


def build_eval_index(grounding_dir: str, cache: str, workers: int = 8) -> dict:
    """sha256 / basename / dHash of every ScreenSpot-Pro and OSWorld-G image (cached; recomputed on size/mtime change)."""
    paths: set[str] = set()
    jsonls = eval_jsonl_paths(grounding_dir)
    if not jsonls:
        raise SystemExit(f"no held-out eval jsonl under {grounding_dir}: refusing to run without a leakage gate")
    for p in jsonls:
        for line in open(p):
            paths.add(json.loads(line)["image_path"])
    old = {}
    if os.path.exists(cache):
        old = {e["path"]: e for e in json.load(open(cache))["images"]}
    todo, keep = [], []
    for p in sorted(paths):
        e = old.get(p)
        if e and os.path.exists(p) and e["size"] == os.path.getsize(p) and e["mtime"] == int(os.path.getmtime(p)):
            keep.append(e)
        elif os.path.exists(p):
            todo.append(p)
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            keep += list(ex.map(_eval_entry, todo, chunksize=8))
    idx = {"jsonls": jsonls, "n_images": len(keep), "missing": len(paths) - len(keep), "images": keep}
    tmp = cache + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f)
    os.replace(tmp, cache)
    return idx


class LeakGate:
    def __init__(self, idx: dict, dhash_max: int = 8):
        self.sha = {e["sha256"] for e in idx["images"]}
        self.names = {e["basename"] for e in idx["images"]}
        self.dh = [int(e["dhash"]) for e in idx["images"]]
        self.dhash_max = dhash_max
        self.dirs = sorted({os.path.dirname(e["path"]) for e in idx["images"]})

    def why(self, path: str, sha: str, dh: int | None, source: str) -> str | None:
        """Reason this image must be excluded, or None if it is clean."""
        if source in HELDOUT_SOURCES or any(s in source for s in HELDOUT_SOURCES):
            return "source"
        rp = os.path.realpath(path)
        if any(rp.startswith(os.path.realpath(d) + os.sep) for d in self.dirs):
            return "path"
        if sha in self.sha:
            return "sha256"
        if os.path.basename(path) in self.names:
            return "basename"
        if dh is not None and any((dh ^ e).bit_count() <= self.dhash_max for e in self.dh):
            return "dhash"
        return None


def _hash_pair(path: str) -> tuple[str, str, str | None]:
    """dhash256 fully decodes the image, so a truncated / corrupt file comes back with dhash None."""
    try:
        dh = str(dhash256(path))
    except (OSError, SyntaxError, ValueError):
        dh = None
    return path, sha256_file(path), dh


def hash_images(paths: list[str], cache_path: str, workers: int = 8) -> dict[str, tuple[str, int] | None]:
    """path -> (sha256, dhash), or None for an image that does not decode. Cached by path+size."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [p for p in paths if p not in cache or cache[p][2] != os.path.getsize(p)]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for p, s, d in ex.map(_hash_pair, todo, chunksize=16):
                cache[p] = [s, d, os.path.getsize(p)]
        with open(cache_path + ".tmp", "w") as f:
            json.dump(cache, f)
        os.replace(cache_path + ".tmp", cache_path)
    return {p: (cache[p][0], int(cache[p][1])) if cache[p][1] is not None else None for p in paths}


def assert_no_eval_overlap(rows: list[dict], gate: LeakGate, hashes: dict[str, tuple[str, int]]) -> dict:
    """Hard gate: no row (full or crop, via its source image) may share source/path/sha256/basename/dHash with an eval
    image. Raises AssertionError on any hit; returns the counters it checked."""
    checked, bad = set(), []
    for r in rows:
        src_img = r.get("source_image_path") or r["image_path"]
        if src_img in checked:
            continue
        checked.add(src_img)
        if hashes[src_img] is None:
            bad.append((src_img, "undecodable"))
            continue
        sha, dh = hashes[src_img]
        why = gate.why(src_img, sha, dh, r["source"])
        if why:
            bad.append((src_img, why))
        if r.get("image_sha256") and r["image_sha256"] != sha:
            bad.append((src_img, "sha_mismatch"))
    assert not bad, f"eval leakage: {bad[:5]} ({len(bad)} total)"
    return {"source_images_checked": len(checked), "eval_images": len(gate.sha), "hits": 0}


# --------------------------------------------------------------------------- GroundCUA


def hf_tree_dir(repo: str, path: str) -> list[dict]:
    url: str | None = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"
    out: list[dict] = []
    while url:
        with _open(url, timeout=120) as r:
            out += [x for x in json.loads(r.read()) if x.get("type") == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = m.group(1) if m else None
    return out


def load_groundcua_instructions(tar_path: str) -> dict[str, dict]:
    """image id -> {"platform", "elements": [{"bbox", "text", "category", "instr": {kind: text}}]} (bbox-bearing files)."""
    imgs: dict[str, dict] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".json"):
                continue
            for e in json.load(tf.extractfile(m)):
                if "bbox" not in e or len(e["bbox"]) != 4:
                    continue  # spatial / *_extra rows carry only a point: no box, no strictly-inside check
                img = imgs.setdefault(e["id"], {"platform": e["platform"], "elements": {}})
                key = tuple(round(v, 1) for v in e["bbox"])
                el = img["elements"].setdefault(key, {"bbox": e["bbox"], "text": e.get("text") or "",
                                                     "category": e.get("category"), "instr": {}})
                for k, v in (e.get("all_instructions") or {}).items():
                    if isinstance(v, str) and v.strip():
                        el["instr"].setdefault(k, v)
    for v in imgs.values():
        v["elements"] = list(v["elements"].values())
    return imgs


ACTION_KINDS = ("functional_instruction", "general_instruction", "logo_and_icon_instruction", "text_instruction")


def pick_instruction(el: dict, key: str, for_crop: bool = False) -> tuple[str, str] | tuple[None, None]:
    """Deterministic instruction for one element. Mix: action-style (like ScreenSpot-Pro's commands), the visible label
    (Studio's simplified target), or the LLM description. Descriptions mention whole-window positions ("top-right of the
    window"), so crops never use them. Templated simple_instruction ("How about you hover...") is never used."""
    ins = el["instr"]
    text = clean_instruction(el.get("text") or "")
    desc = ins.get("description", "")
    # a label is trusted only when the description repeats it (drops OCR noise like "genapuacxa")
    label_ok = 2 <= len(text) <= 60 and text.lower() in desc.lower()
    opts = []
    actions = [k for k in ACTION_KINDS if k in ins]
    if actions:
        opts += [("action", actions[hrank(key, "a") % len(actions)])] * 4
    if label_ok:
        opts += [("label", None)] * 3
    if desc and not for_crop:
        opts += [("description", "description")] * 2
    if not opts:
        return None, None
    kind, k = opts[hrank(key, "kind") % len(opts)]
    if kind == "label":
        return text, "label"
    return clean_instruction(ins[k]), k


def select_elements(elements: list[dict], w: int, h: int, key: str, k: int) -> list[dict]:
    """Up to k elements per screenshot, dense-first: < 2% of the screen preferred, then < 10%; inside the frame, >= 4 px."""
    good = []
    for el in elements:
        x0, y0, x1, y1 = el["bbox"]
        bw, bh = x1 - x0, y1 - y0
        if bw < 4 or bh < 4 or x0 < 0 or y0 < 0 or x1 > w + 1 or y1 > h + 1:
            continue
        a = bw * bh / (w * h)
        if a > 0.10:
            continue
        good.append((0 if a < DENSE_AREA else 1, hrank(key, el["bbox"]), el))
    good.sort(key=lambda t: t[:2])
    return [g[2] for g in good[:k]]


def probe_png_dims(url: str) -> tuple[int, int] | None:
    with _open(url, headers={"Range": "bytes=0-32"}, timeout=60) as r:
        return png_dims(r.read())


def groundcua(root: str, budget: Budget, gate: LeakGate, gb: float, min_w: int, per_platform_probe: int,
              per_image: int, workers: int) -> list[dict]:
    raw = os.path.join(root, "raw", "groundcua")
    tar_path = download(resolve_url(GCUA_REPO, "instruction_tuning.tar.gz"), os.path.join(raw, "instruction_tuning.tar.gz"))
    t0 = time.time()
    instr = load_groundcua_instructions(tar_path)
    print(f"  groundcua instructions: {len(instr)} images, {sum(len(v['elements']) for v in instr.values())} boxed "
          f"elements ({time.time() - t0:.1f}s)")

    tree_cache = os.path.join(raw, "tree.json")
    if os.path.exists(tree_cache):
        tree = json.load(open(tree_cache))
    else:
        platforms = sorted({v["platform"] for v in instr.values()})
        with ThreadPoolExecutor(8) as ex:
            lists = list(ex.map(lambda p: hf_tree_dir(GCUA_REPO, f"images/{p}"), platforms))
        tree = {f["path"]: f["size"] for lst in lists for f in lst}
        json.dump(tree, open(tree_cache, "w"))
    by_plat: dict[str, list[str]] = {}
    for iid, v in instr.items():
        path = f"images/{v['platform']}/{iid}.png"
        if path in tree:
            by_plat.setdefault(v["platform"], []).append(iid)
    for p in by_plat:
        by_plat[p].sort(key=lambda i: hrank("gcua", i))

    dims_cache = os.path.join(raw, "dims.json")
    dims = json.load(open(dims_cache)) if os.path.exists(dims_cache) else {}
    todo = [i for p in by_plat for i in by_plat[p][:per_platform_probe] if i not in dims]
    if todo:
        t0 = time.time()

        def probe(iid):
            try:
                return iid, probe_png_dims(resolve_url(GCUA_REPO, f"images/{instr[iid]['platform']}/{iid}.png"))
            except Exception:
                return iid, None

        with ThreadPoolExecutor(workers) as ex:
            for iid, d in ex.map(probe, todo):
                dims[iid] = d
        json.dump(dims, open(dims_cache, "w"))
        print(f"  probed {len(todo)} PNG headers in {time.time() - t0:.0f}s")

    # high-res candidates with >= 1 usable dense element, round-robin over apps (pro apps 2 per round) until budget
    cands: dict[str, list[str]] = {}
    for p, ids in by_plat.items():
        for iid in ids[:per_platform_probe]:
            d = dims.get(iid)
            if d and d[0] >= min_w and select_elements(instr[iid]["elements"], d[0], d[1], iid, 1):
                cands.setdefault(p, []).append(iid)
    cap = int(gb * 1e9)
    chosen, spent, rnd = [], 0, 0
    while spent < cap and any(cands.values()):
        for p in sorted(cands):
            for _ in range(2 if p in PRO_APPS else 1):
                if cands[p] and spent < cap:
                    iid = cands[p].pop(0)
                    sz = tree[f"images/{p}/{iid}.png"]
                    if spent + sz <= cap:
                        chosen.append(iid)
                        spent += sz
        rnd += 1
    budget.check(sum(tree[f"images/{instr[i]['platform']}/{i}.png"] for i in chosen
                     if not os.path.exists(os.path.join(raw, "images", instr[i]["platform"], i + ".png"))), "GroundCUA images")
    t0 = time.time()

    def fetch(iid):
        p = instr[iid]["platform"]
        return iid, download(resolve_url(GCUA_REPO, f"images/{p}/{iid}.png"), os.path.join(raw, "images", p, iid + ".png"))

    with ThreadPoolExecutor(workers) as ex:
        paths = dict(ex.map(fetch, chosen))
    print(f"  downloaded {len(chosen)} GroundCUA images ({spent / 1e9:.2f} GB, {rnd} rounds) in {time.time() - t0:.0f}s")

    hashes = hash_images(sorted(paths.values()), os.path.join(root, "image_hashes.json"))
    rows, dropped = [], {}
    for iid in chosen:
        path, p = paths[iid], instr[iid]["platform"]
        w, h = dims[iid]
        if hashes[path] is None:  # truncated / corrupt PNG: never train on it
            dropped["undecodable"] = dropped.get("undecodable", 0) + 1
            continue
        sha, dh = hashes[path]
        why = gate.why(path, sha, dh, "groundcua")
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        split = split_for("groundcua", iid)
        for j, el in enumerate(select_elements(instr[iid]["elements"], w, h, iid, per_image)):
            ins, kind = pick_instruction(el, f"{iid}|{j}")
            r = make_row(sid=f"gcua_{iid[:16]}_{j}", image_path=path, w=w, h=h, px_box=el["bbox"], instruction=ins,
                         kind=kind, source="groundcua", split=split, sha=sha, platform="desktop", application=p,
                         ui_type=el.get("category"), variant="full", _el=el)
            if r:
                rows.append(r)
    print(f"  groundcua rows {len(rows)} from {len(chosen)} images; eval-overlap drops {dropped or 0}")
    return rows


# --------------------------------------------------------------------------- OS-Atlas macOS (remote zip)


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a small block cache (enough for zipfile's central
    directory parse). Member payloads are fetched with one explicit range each (see RemoteZip.read_member)."""

    BLOCK = 1 << 18

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        with _open(url, headers={"Range": "bytes=0-0"}) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def range(self, a: int, b: int) -> bytes:
        """bytes [a, b) straight from the server."""
        with _open(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=300) as r:
            return r.read()

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            a = i * self.BLOCK
            self.cache[i] = self.range(a, min(self.size, a + self.BLOCK))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[o:o + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


class RemoteZip:
    def __init__(self, url: str):
        self.f = HTTPRangeFile(url)
        self.zf = zipfile.ZipFile(self.f)
        self.info = {i.filename: i for i in self.zf.infolist()}

    def _data_start(self, zi: zipfile.ZipInfo) -> int:
        hdr = self.f.range(zi.header_offset, zi.header_offset + 30)
        assert hdr[:4] == b"PK\x03\x04", zi.filename
        n, m = struct.unpack("<HH", hdr[26:30])
        return zi.header_offset + 30 + n + m

    def head(self, name: str, nbytes: int = 4096) -> bytes:
        """First bytes of the decompressed member, fetching only ~nbytes of compressed data."""
        zi = self.info[name]
        a = self._data_start(zi)
        raw = self.f.range(a, a + min(zi.compress_size, nbytes))
        if zi.compress_type == zipfile.ZIP_STORED:
            return raw
        if zi.compress_type == zipfile.ZIP_DEFLATED:
            return zlib.decompressobj(-15).decompress(raw)
        raise ValueError(f"unsupported compression {zi.compress_type}")

    def read_member(self, name: str) -> bytes:
        zi = self.info[name]
        a = self._data_start(zi)
        raw = self.f.range(a, a + zi.compress_size)
        data = raw if zi.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
        if zlib.crc32(data) & 0xFFFFFFFF != zi.CRC:
            raise IOError(f"crc mismatch {name}")
        return data


ATLAS_KEEP_TYPES = {"AXButton", "AXImage", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXTextField",
                    "AXMenuBarItem", "AXMenuItem", "AXTab", "AXSlider", "AXComboBox", "AXDisclosureTriangle",
                    "AXStaticText", "AXLink", "AXCell", "AXSearchField", "AXToolbar", "AXIncrementor", "AXColorWell"}


def atlas_macos(root: str, budget: Budget, gate: LeakGate, gb: float, min_w: int, probe_n: int, per_image: int,
                workers: int) -> list[dict]:
    raw = os.path.join(root, "raw", "os_atlas_macos")
    ann_path = download(resolve_url(ATLAS_REPO, "desktop_domain/macos_splited.json"), os.path.join(raw, "macos_splited.json"))
    ann = {a["img_filename"]: a["elements"] for a in json.load(open(ann_path))}
    rz = RemoteZip(resolve_url(ATLAS_REPO, "desktop_domain/macos_images.zip"))
    members = {os.path.basename(n): n for n in rz.info if n.lower().endswith(".png")}
    names = sorted((b for b in ann if b in members), key=lambda b: hrank("atlas", b))[:probe_n]
    dims_cache = os.path.join(raw, "dims.json")
    dims = json.load(open(dims_cache)) if os.path.exists(dims_cache) else {}
    todo = [b for b in names if b not in dims]
    if todo:
        def probe(b):
            try:
                return b, png_dims(rz.head(members[b]))
            except Exception:
                return b, None

        with ThreadPoolExecutor(workers) as ex:
            for b, d in ex.map(probe, todo):
                dims[b] = d
        json.dump(dims, open(dims_cache, "w"))

    def usable(b) -> list[dict]:
        w, h = dims[b]
        els = []
        for e in ann[b]:
            x0, y0, x1, y1 = e["bbox"]
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1) or e.get("data_type") not in ATLAS_KEEP_TYPES:
                continue
            text = clean_instruction(e.get("instruction") or "")
            if not (2 <= len(text) <= 80):
                continue
            els.append({"bbox": [x0 * w, y0 * h, x1 * w, y1 * h], "text": text, "category": e.get("data_type"), "instr": {}})
        return els

    cap, spent, chosen = int(gb * 1e9), 0, []
    for b in names:
        d = dims.get(b)
        if not d or d[0] < min_w or not select_elements(usable(b), d[0], d[1], b, 1):
            continue
        sz = rz.info[members[b]].file_size
        if spent + sz > cap:
            continue
        chosen.append(b)
        spent += sz
    budget.check(sum(rz.info[members[b]].file_size for b in chosen if not os.path.exists(os.path.join(raw, "images", b))),
                 "OS-Atlas macOS images")
    os.makedirs(os.path.join(raw, "images"), exist_ok=True)

    def fetch(b):
        dest = os.path.join(raw, "images", b)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            data = rz.read_member(members[b])
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
        return b, dest

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        paths = dict(ex.map(fetch, chosen))
    print(f"  OS-Atlas macOS: probed {len(names)}, {sum(1 for b in names if dims.get(b) and dims[b][0] >= min_w)} >= "
          f"{min_w}px, downloaded {len(chosen)} ({spent / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
    hashes = hash_images(sorted(paths.values()), os.path.join(root, "image_hashes.json"))
    rows, dropped = [], {}
    for b in chosen:
        path = paths[b]
        w, h = dims[b]
        if hashes[path] is None:  # some members are truncated inside the source zip itself (CRC matches)
            dropped["undecodable"] = dropped.get("undecodable", 0) + 1
            continue
        sha, dh = hashes[path]
        why = gate.why(path, sha, dh, "os_atlas_macos")
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        split = split_for("os_atlas_macos", b)
        for j, el in enumerate(select_elements(usable(b), w, h, b, per_image)):
            r = make_row(sid=f"atlasmac_{b.rsplit('.', 1)[0]}_{j}", image_path=path, w=w, h=h, px_box=el["bbox"],
                         instruction=el["text"], kind="label", source="os_atlas_macos", split=split, sha=sha,
                         platform="macos", application=None, ui_type=el["category"], variant="full", _el=el)
            if r:
                rows.append(r)
    print(f"  os_atlas_macos rows {len(rows)}; eval-overlap drops {dropped or 0}")
    return rows


# --------------------------------------------------------------------------- multi-crop augmentation


def crop_window(img_w: int, img_h: int, box, rng: random.Random, crop_w: int = CROP_W, crop_h: int = CROP_H,
                margin: int = 24) -> tuple[int, int] | None:
    """Random top-left (x, y) of a crop_w x crop_h window inside the image that fully contains box with >= margin px
    of context on every side (margin shrinks at the image border). None if impossible."""
    if img_w < crop_w or img_h < crop_h:
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 + 2 > crop_w or y1 - y0 + 2 > crop_h:
        return None

    def axis(lo_box, hi_box, size, crop):
        lo = max(0, int(hi_box + min(margin, size - hi_box)) - crop + 1)  # window right edge must pass hi_box
        hi = min(size - crop, int(lo_box - min(margin, lo_box)))  # window left edge must precede lo_box
        if lo > hi:
            lo, hi = max(0, int(hi_box) - crop + 1), min(size - crop, int(lo_box))
            if lo > hi:
                return None
        return rng.randint(lo, hi)

    x, y = axis(x0, x1, img_w, crop_w), axis(y0, y1, img_h, crop_h)
    if x is None or y is None:
        return None
    return x, y


def generate_highres_crops(rows: list[dict], out_dir: str, n_crops: int = 1, seed: int = 0, min_src_w: int = 1600,
                           crop_w: int = CROP_W, crop_h: int = CROP_H, jpeg_q: int = CROP_JPEG_Q,
                           extra_targets: bool = True, anchor_ids: set[str] | None = None) -> list[dict]:
    """Native-pixel crop_w x crop_h crops around each row's target, for source screenshots wider than min_src_w.

    rows: full-frame rows of ONE OR MORE images (pixel boxes in "bbox_px"). For each anchor row (every row, or only
    ids in anchor_ids) with width > min_src_w, n_crops deterministic windows (seeded by seed + row id + k) are cut so
    the target lies fully inside; every OTHER row of the same image whose box also lies fully inside the window becomes
    an extra crop row when extra_targets is True. No resampling: Studio sends a <= 1536 px JPEG untouched, so the
    model sees the icon at its true pixel size. bbox_1000 / center_1000 are recomputed in the crop frame from bbox_px;
    rows whose center is not strictly inside the recomputed box are skipped, and whole-window "description"
    instructions are swapped for a crop-safe one (or the row is skipped). Crops inherit the parent split.
    Returns new rows (variant "crop"), ids unique."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    by_img: dict[str, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image_path"], []).append(r)
    out: list[dict] = []
    emitted: set[str] = set()
    for img_path in sorted(by_img):
        group = sorted(by_img[img_path], key=lambda r: r["id"])
        if group[0]["width"] <= min_src_w:
            continue
        im = None
        for r in group:
            if anchor_ids is not None and r["id"] not in anchor_ids:
                continue
            for k in range(n_crops):
                rng = random.Random(hrank(seed, r["id"], k))
                pos = crop_window(r["width"], r["height"], r["bbox_px"], rng, crop_w, crop_h)
                if pos is None:
                    continue
                cx, cy = pos
                cid = hashlib.sha1(f"{img_path}|{cx}|{cy}|{crop_w}x{crop_h}".encode()).hexdigest()[:20]
                dest = os.path.join(out_dir, cid + ".jpg")
                if not os.path.exists(dest):
                    if im is None:
                        im = Image.open(img_path).convert("RGB")
                        assert im.size == (r["width"], r["height"]), (img_path, im.size, r["width"], r["height"])
                    im.crop((cx, cy, cx + crop_w, cy + crop_h)).save(dest + ".tmp", "JPEG", quality=jpeg_q)
                    os.replace(dest + ".tmp", dest)
                members = [r] + ([o for o in group if o is not r] if extra_targets else [])
                for o in members:
                    bx0, by0, bx1, by1 = o["bbox_px"]
                    if o is not r and not (bx0 >= cx and by0 >= cy and bx1 <= cx + crop_w and by1 <= cy + crop_h):
                        continue
                    local = [round(v, 2) for v in (bx0 - cx, by0 - cy, bx1 - cx, by1 - cy)]
                    b = box_1000(local, crop_w, crop_h)
                    c = center_of(b)
                    if not strictly_inside(c, b):
                        continue
                    nr = {k2: v for k2, v in o.items() if k2 not in ("prompt", "target", "_el")}
                    ins = o["instruction"]
                    if o.get("instruction_kind") == "description":
                        alt, kind = pick_instruction(o["_el"], o["id"] + "|crop", for_crop=True) if o.get("_el") else (None, None)
                        if not alt:
                            continue
                        ins, nr["instruction_kind"] = alt, kind
                    nid = f"{o['id']}_crop{cid[:8]}"
                    if nid in emitted:
                        continue
                    emitted.add(nid)
                    nr.update({"id": nid, "image_path": dest, "width": crop_w, "height": crop_h,
                               "instruction": ins, "bbox_1000": b, "center_1000": c,
                               "bbox_px": [round(v, 2) for v in local],
                               "area_frac": round((bx1 - bx0) * (by1 - by0) / (crop_w * crop_h), 8), "variant": "crop",
                               "parent_id": o["id"], "crop_xywh": [cx, cy, crop_w, crop_h],
                               "source_image_path": img_path, "image_sha256": o.get("image_sha256")})
                    out.append(add_prompt(nr))
        if im is not None:
            im.close()
    return out


# --------------------------------------------------------------------------- verification


def verify_rows(rows: list[dict]) -> dict:
    """Schema + geometry checks every written row must pass."""
    need = ("id", "image_path", "width", "height", "instruction", "bbox_1000", "center_1000", "source", "split",
            "prompt", "target")
    ids = set()
    split_of_img: dict[str, str] = {}
    for r in rows:
        for k in need:
            assert k in r, (r.get("id"), k)
        assert r["id"] not in ids, r["id"]
        ids.add(r["id"])
        b, c = r["bbox_1000"], r["center_1000"]
        assert all(0 <= v <= 1000 for v in b + c), r["id"]
        assert strictly_inside(c, b), (r["id"], b, c)
        assert r["split"] in ("train", "val"), r["id"]
        assert r["source"] not in HELDOUT_SOURCES, r["id"]
        assert r["prompt"] == sf.grounder_prompt(r["instruction"]), r["id"]
        assert json.loads(r["target"]) == {"point": c}, r["id"]
        src_img = r.get("source_image_path") or r["image_path"]
        assert split_of_img.setdefault(src_img, r["split"]) == r["split"], f"image in two splits: {src_img}"
        if r.get("variant") == "crop":
            assert (r["width"], r["height"]) == (CROP_W, CROP_H), r["id"]
    return {"rows": len(rows), "source_images": len(split_of_img)}


def stats(rows: list[dict]) -> dict:
    def pct(xs, f):
        return round(100 * sum(1 for x in xs if f(x)) / max(1, len(xs)), 1)

    out = {}
    for name, sel in (("all", rows), ("full", [r for r in rows if r.get("variant") == "full"]),
                      ("crop", [r for r in rows if r.get("variant") == "crop"])):
        out[name] = {"rows": len(sel), "train": sum(r["split"] == "train" for r in sel),
                     "val": sum(r["split"] == "val" for r in sel), "images": len({r["image_path"] for r in sel}),
                     "small_lt1pct": pct(sel, lambda r: r["area_frac"] < SMALL_AREA),
                     "lt2pct": pct(sel, lambda r: r["area_frac"] < DENSE_AREA)}
    full = [r for r in rows if r.get("variant") == "full"]
    out["by_source"] = {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})}
    out["applications"] = len({r.get("application") for r in full if r.get("application")})
    out["full_width_ge_1920_pct"] = pct(full, lambda r: r["width"] >= 1920)
    out["median_full_box_px"] = sorted(min(r["bbox_px"][2] - r["bbox_px"][0], r["bbox_px"][3] - r["bbox_px"][1])
                                       for r in full)[len(full) // 2] if full else None
    return out


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


# --------------------------------------------------------------------------- main


def run_gates(root: str, rows: list[dict], gate: LeakGate) -> dict:
    src_imgs = sorted({r.get("source_image_path") or r["image_path"] for r in rows})
    hashes = hash_images(src_imgs, os.path.join(root, "image_hashes.json"))
    return {"schema": verify_rows(rows), "leakage": assert_no_eval_overlap(rows, gate, hashes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/cua_datasets/dense_grounding"))
    ap.add_argument("--eval-grounding-dir", default=os.path.expanduser("~/cua_datasets/grounding"))
    ap.add_argument("--cap-gb", type=float, default=4.0, help="hard cap on everything under --root")
    ap.add_argument("--groundcua-gb", type=float, default=2.45)
    ap.add_argument("--atlas-gb", type=float, default=0.45)
    ap.add_argument("--crops-gb", type=float, default=0.55)
    ap.add_argument("--min-width", type=int, default=1600)
    ap.add_argument("--per-image", type=int, default=5)
    ap.add_argument("--probe-per-app", type=int, default=160)
    ap.add_argument("--atlas-probe", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    gdir = os.path.join(root, "grounding")
    os.makedirs(gdir, exist_ok=True)
    budget = Budget(root, int(a.cap_gb * 1e9))
    t0 = time.time()
    idx = build_eval_index(os.path.abspath(os.path.expanduser(a.eval_grounding_dir)), os.path.join(root, "eval_index.json"))
    gate = LeakGate(idx)
    print(f"eval index: {idx['n_images']} held-out images from {[os.path.basename(p) for p in idx['jsonls']]} "
          f"({time.time() - t0:.0f}s)")
    if a.verify_only:
        rows = read_jsonl(os.path.join(gdir, "dense_all.jsonl"))
        print(json.dumps(run_gates(root, rows, gate)), json.dumps(stats(rows)))
        return 0

    full = groundcua(root, budget, gate, a.groundcua_gb, a.min_width, a.probe_per_app, a.per_image, a.workers)
    full += atlas_macos(root, budget, gate, a.atlas_gb, a.min_width, a.atlas_probe, a.per_image, a.workers)

    # crops: smallest-target anchors first, <= 2 windows per screenshot, other targets in a window ride along,
    # batch by batch until --crops-gb (checked between batches; one batch is ~0.06 GB)
    t0 = time.time()
    crop_dir = os.path.join(root, "crops")
    by_img: dict[str, list[dict]] = {}
    for r in full:
        by_img.setdefault(r["image_path"], []).append(r)
    anchors, per_img = [], {}
    for r in sorted((r for r in full if r["width"] > a.min_width),
                    key=lambda r: (r["area_frac"] >= SMALL_AREA, hrank("anchor", r["id"]))):
        if per_img.get(r["image_path"], 0) < 2:
            per_img[r["image_path"]] = per_img.get(r["image_path"], 0) + 1
            anchors.append(r)
    crops: dict[str, dict] = {}
    cap = int(a.crops_gb * 1e9)
    with ProcessPoolExecutor(8) as ex:
        for i in range(0, len(anchors), 400):
            if du_bytes(crop_dir) >= cap or budget.left() < 100e6:
                break
            batch = anchors[i:i + 400]
            ids_by_img: dict[str, set[str]] = {}
            for r in batch:
                ids_by_img.setdefault(r["image_path"], set()).add(r["id"])
            jobs = [ex.submit(generate_highres_crops, by_img[p], crop_dir, 1, 0, a.min_width, anchor_ids=ids)
                    for p, ids in sorted(ids_by_img.items())]
            for j in jobs:
                for r in j.result():
                    crops.setdefault(r["id"], r)
    crops = list(crops.values())
    print(f"  crops: {len(crops)} rows on {len({r['image_path'] for r in crops})} windows "
          f"({du_bytes(crop_dir) / 1e9:.2f} GB) in {time.time() - t0:.0f}s")

    rows = full + crops
    gates = run_gates(root, rows, gate)
    write_jsonl(os.path.join(gdir, "dense_all.jsonl"), rows)
    for s in ("train", "val"):
        write_jsonl(os.path.join(gdir, f"dense_{s}.jsonl"), [r for r in rows if r["split"] == s])
    total = du_bytes(root)
    assert total <= budget.cap, f"disk {total / 1e9:.3f} GB over cap"
    manifest = {"root": root, "cap_gb": a.cap_gb, "disk_gb": round(total / 1e9, 3), "args": vars(a), "gates": gates,
                "stats": stats(rows), "eval_jsonls": idx["jsonls"], "eval_images_indexed": idx["n_images"],
                "eval_images_missing": idx["missing"]}
    with open(os.path.join(gdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["stats"], indent=1))
    print(f"gates {json.dumps(gates)}  disk {total / 1e9:.3f} GB / {a.cap_gb} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
