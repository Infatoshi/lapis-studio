"""Fetch public GUI-grounding benchmarks + MiniWoB++ and normalize them to Studio's L3_VLM grounder schema.

Every sample becomes one JSONL row:
  {"id", "image_path" (absolute), "width", "height", "instruction", "bbox_1000": [x0, y0, x1, y1],
   "center_1000": [x, y], "source", "split", "platform", "ui_type"}
with integer 0..1000 coordinates (x=0 left, y=0 top), the same frame Studio's grounder replies in
({"point":[x,y]}, src/mac/cua_executor.cpp buildInklingGroundRequest).

Splits are assigned per IMAGE (sha1 of source + image name), never per sample, so no screenshot is in two splits:
  ScreenSpot-Pro, OSWorld-G   -> heldout_eval only (headline benchmarks; never trained on)
  ScreenSpot-v2, UI-Vision    -> 70% train / 10% val / 20% heldout_eval by image hash
Only the heldout_eval slice of v2 / UI-Vision is comparable to anything; their full-benchmark public numbers are not.

Downloads go to ~/cua_datasets (outside git) and stop before the total would pass --cap-gb (default 5).

  uv run --with pyarrow python data_prep/fetch_grounding.py --root ~/cua_datasets
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

HF = "https://huggingface.co"
UA = {"User-Agent": "cua-grounding-fetch/1.0"}
SPLIT_SALT = "lapis-grounding-v1"


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
    def __init__(self, root: str, cap_bytes: int):
        self.root, self.cap = root, cap_bytes

    def check(self, incoming: int, what: str) -> None:
        used = du_bytes(self.root)
        if used + incoming > self.cap:
            raise SystemExit(f"cap: {what} needs {incoming / 1e9:.2f} GB, {used / 1e9:.2f} GB used of {self.cap / 1e9:.2f} GB")


def hf_tree(repo: str) -> list[dict]:
    """All files in a dataset repo. The tree API pages at ~1000 entries; follow Link: <...>; rel="next"."""
    url: str | None = f"{HF}/api/datasets/{repo}/tree/main?recursive=true"
    out: list[dict] = []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            out += [x for x in json.loads(r.read()) if x.get("type") == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = m.group(1) if m else None
    return out


def hf_get(repo: str, path: str, dest: str, retries: int = 3) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
    for attempt in range(retries):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return dest


def fetch_files(repo: str, files: list[dict], raw: str, budget: Budget, workers: int = 16) -> None:
    todo = [f for f in files if not os.path.exists(os.path.join(raw, f["path"]))]
    budget.check(sum(f.get("size", 0) for f in todo), repo)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda f: hf_get(repo, f["path"], os.path.join(raw, f["path"])), todo))


def split_for(source: str, image_key: str, heldout_only: bool) -> str:
    if heldout_only:
        return "heldout_eval"
    h = int(hashlib.sha1(f"{SPLIT_SALT}|{source}|{image_key}".encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 80 else "heldout_eval")


def norm_row(sid, image_path, w, h, instruction, x0, y0, x1, y1, source, heldout_only, image_key, **extra):
    def c(v, d):
        return max(0, min(1000, round(v / d * 1000)))

    b = [c(x0, w), c(y0, h), c(x1, w), c(y1, h)]
    if b[2] < b[0]:
        b[0], b[2] = b[2], b[0]
    if b[3] < b[1]:
        b[1], b[3] = b[3], b[1]
    row = {"id": sid, "image_path": image_path, "width": int(w), "height": int(h), "instruction": " ".join(instruction.split()),
           "bbox_1000": b, "center_1000": [round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)], "source": source,
           "split": split_for(source, image_key, heldout_only)}
    row.update({k: v for k, v in extra.items() if v is not None})
    return row


def image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# --------------------------------------------------------------------------- datasets


def screenspot_pro(root: str, budget: Budget) -> list[dict]:
    repo, raw = "likaixin/ScreenSpot-Pro", os.path.join(root, "screenspot_pro")
    files = [f for f in hf_tree(repo) if f["path"].startswith(("annotations/", "images/")) and not f["path"].endswith(".DS_Store")]
    fetch_files(repo, files, raw, budget)
    rows = []
    for ann in sorted(os.listdir(os.path.join(raw, "annotations"))):
        for a in json.load(open(os.path.join(raw, "annotations", ann))):
            img = os.path.join(raw, "images", a["img_filename"])
            if not os.path.exists(img):
                continue
            w, h = a.get("img_size") or image_size(img)
            x0, y0, x1, y1 = a["bbox"]
            rows.append(norm_row(f"sspro_{a['id']}", img, w, h, a["instruction"], x0, y0, x1, y1, "screenspot_pro", True,
                                 a["img_filename"], platform=a.get("platform"), ui_type=a.get("ui_type"),
                                 application=a.get("application")))
    return rows


def _safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    base = os.path.realpath(dest)
    for m in zf.infolist():
        target = os.path.realpath(os.path.join(dest, m.filename))
        if not target.startswith(base + os.sep) and target != base:
            raise SystemExit(f"unsafe path in zip: {m.filename}")
    zf.extractall(dest)


def screenspot_v2(root: str, budget: Budget) -> list[dict]:
    repo, raw = "OS-Copilot/ScreenSpot-v2", os.path.join(root, "screenspot_v2")
    files = {f["path"]: f for f in hf_tree(repo)}
    # desktop + web only: Studio drives a Mac desktop, and dropping mobile (0.33 GB) keeps every source complete under 5 GB
    anns = [p for p in files if p.endswith("_v2.json") and "mobile" not in p]
    fetch_files(repo, [files[p] for p in anns], raw, budget)
    img_dir = os.path.join(raw, "images")
    if not os.path.isdir(img_dir) or not os.listdir(img_dir):
        z = os.path.join(raw, "screenspotv2_image.zip")
        # zip + extracted copy coexist until the zip is deleted
        budget.check(2 * files["screenspotv2_image.zip"]["size"], "ScreenSpot-v2 zip + extract")
        hf_get(repo, "screenspotv2_image.zip", z)
        tmp = os.path.join(raw, "_extract")
        with zipfile.ZipFile(z) as zf:
            _safe_extract(zf, tmp)
        os.makedirs(img_dir, exist_ok=True)
        for dp, _, fs in os.walk(tmp):
            for f in fs:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    os.replace(os.path.join(dp, f), os.path.join(img_dir, f))
        shutil.rmtree(tmp)
        os.remove(z)
    for f in os.listdir(img_dir):
        if f.startswith("mobile_"):
            os.remove(os.path.join(img_dir, f))
    stale = os.path.join(raw, "screenspot_mobile_v2.json")
    if os.path.exists(stale):
        os.remove(stale)
    rows = []
    for ann in sorted(anns):
        plat = os.path.basename(ann).split("_")[1]  # mobile / web / desktop
        for i, a in enumerate(json.load(open(os.path.join(raw, ann)))):
            img = os.path.join(img_dir, a["img_filename"])
            if not os.path.exists(img):
                continue
            w, h = image_size(img)
            x, y, bw, bh = a["bbox"]  # xywh pixels
            rows.append(norm_row(f"ssv2_{plat}_{i}", img, w, h, a["instruction"], x, y, x + bw, y + bh, "screenspot_v2", False,
                                 a["img_filename"], platform=plat, ui_type=a.get("data_type"), application=a.get("data_source")))
    return rows


def osworld_g(root: str, budget: Budget) -> list[dict]:
    import pyarrow.parquet as pq

    repo, raw = "MMInstruction/OSWorld-G", os.path.join(root, "osworld_g")
    files = [f for f in hf_tree(repo) if f["path"].endswith(".parquet")]
    fetch_files(repo, files, raw, budget)
    img_dir = os.path.join(raw, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows, skipped = [], {"refusal": 0, "other": 0}
    for f in files:
        t = pq.read_table(os.path.join(raw, f["path"])).to_pylist()
        for r in t:
            name = os.path.basename(r.get("image_path") or f"{r['id']}.png")
            img = os.path.join(img_dir, name)
            if not os.path.exists(img):
                with open(img, "wb") as out:
                    out.write(r["image"]["bytes"])
            w, h = r.get("image_size") or image_size(img)
            box, kind = r.get("box_coordinates") or [], r.get("box_type")
            if kind == "bbox" and len(box) == 4:
                x, y, bw, bh = box  # xywh pixels
                x0, y0, x1, y1 = x, y, x + bw, y + bh
            elif kind == "polygon" and len(box) >= 4:  # 2-point polygons are [x0, y0, x1, y1] corner pairs
                xs, ys = box[0::2], box[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                skipped["refusal" if kind == "refusal" else "other"] += 1
                continue
            rows.append(norm_row(f"osworldg_{r['id']}", img, w, h, r["instruction"], x0, y0, x1, y1, "osworld_g", True, name,
                                 ui_type=",".join(r.get("GUI_types") or []) or None, box_type=kind))
    print(f"  osworld_g skipped {skipped} (refusal = no target on screen; Studio's not_found case)")
    return rows


def ui_vision(root: str, budget: Budget) -> list[dict]:
    repo, raw = "ServiceNow/ui-vision", os.path.join(root, "ui_vision")
    tree = hf_tree(repo)
    anns = [f for f in tree if f["path"].startswith("annotations/element_grounding/") and f["path"].endswith(".json")]
    fetch_files(repo, anns, raw, budget)
    data = []
    for a in anns:
        sub = os.path.basename(a["path"]).replace("element_grounding_", "").replace(".json", "")
        data += [(sub, i, x) for i, x in enumerate(json.load(open(os.path.join(raw, a["path"]))))]
    need = {x["image_path"] for _, _, x in data}
    imgs = [f for f in tree if f["path"].startswith("images/") and f["path"][len("images/"):] in need]
    fetch_files(repo, imgs, raw, budget)
    rows = []
    for sub, i, x in data:
        img = os.path.join(raw, "images", x["image_path"])
        if not os.path.exists(img):
            continue
        w, h = x.get("image_size") or image_size(img)
        x0, y0, x1, y1 = x["bbox"]
        rows.append(norm_row(f"uivision_{sub}_{i}", img, w, h, x["prompt_to_evaluate"], x0, y0, x1, y1, "ui_vision", False,
                             x["image_path"], platform=x.get("platform"), ui_type=x.get("element_type"),
                             category=x.get("category"), subset=sub))
    return rows


# --------------------------------------------------------------------------- MiniWoB++

MINIWOB_TASKS = ["click-button", "click-checkboxes", "enter-text", "click-tab", "choose-list"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def miniwob(root: str, budget: Budget, port: int = 8799) -> dict:
    dest = os.path.join(root, "miniwob")
    if not os.path.isdir(os.path.join(dest, ".git")):
        budget.check(60 * 1024 * 1024, "miniwob clone (est.)")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/Farama-Foundation/miniwob-plusplus.git", dest], check=True, capture_output=True)
        subprocess.run(["git", "-C", dest, "sparse-checkout", "set", "miniwob/html"], check=True, capture_output=True)
    html = os.path.join(dest, "miniwob", "html")
    n_tasks = len([f for f in os.listdir(os.path.join(html, "miniwob")) if f.endswith(".html")])
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", html],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {"path": dest, "html_tasks": n_tasks, "bytes": du_bytes(dest), "tasks": {}}
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        for t in MINIWOB_TASKS:
            url = f"http://127.0.0.1:{port}/miniwob/{t}.html"
            body = urllib.request.urlopen(url, timeout=5).read().decode()
            deps = [d for d in _script_srcs(body)]
            dep_ok = all(urllib.request.urlopen(urllib.parse.urljoin(url, d), timeout=5).status == 200 for d in deps)
            res = {"http": 200, "deps": len(deps), "deps_ok": dep_ok}
            if os.path.exists(CHROME):
                # headless (no window): a same-origin wrapper starts the episode (skipping the START cover) and
                # copies the task's generated #query instruction out, proving genProblem() runs
                wrap = os.path.join(html, "_lapis_check.html")
                with open(wrap, "w") as f:
                    f.write(WRAPPER.replace("TASK", t))
                try:
                    dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000", "--dump-dom",
                                          f"http://127.0.0.1:{port}/_lapis_check.html"], capture_output=True, text=True,
                                         timeout=60).stdout
                finally:
                    os.remove(wrap)
                m = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
                res["query"] = html_lib.unescape(m.group(1)).strip()[:120] if m else ""
                res["js_ok"] = bool(res["query"]) and not res["query"].startswith("ERR")
            out["tasks"][t] = res
    finally:
        srv.terminate()
        srv.wait()
    out["ok"] = all(r["deps_ok"] and r.get("js_ok", True) for r in out["tasks"].values())
    return out


def _script_srcs(html: str) -> list[str]:
    return re.findall(r'<script[^>]+src="([^"]+)"', html) + re.findall(r'<link[^>]+href="([^"]+\.css)"', html)


WRAPPER = """<!doctype html><html><body><iframe id="f" src="miniwob/TASK.html" width="400" height="400"></iframe>
<pre id="out"></pre><script>
document.getElementById('f').onload = function () {
  var w = this.contentWindow, out = document.getElementById('out');
  try { w.core.startEpisodeReal(); out.textContent = w.document.getElementById('query').textContent; }
  catch (e) { out.textContent = 'ERR ' + e; }
};</script></body></html>"""


# --------------------------------------------------------------------------- main

DATASETS = {"screenspot_v2": screenspot_v2, "screenspot_pro": screenspot_pro, "ui_vision": ui_vision, "osworld_g": osworld_g}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/cua_datasets"))
    ap.add_argument("--cap-gb", type=float, default=5.0)
    ap.add_argument("--only", nargs="*", choices=list(DATASETS) + ["miniwob"])
    args = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    os.makedirs(root, exist_ok=True)
    budget = Budget(root, int(args.cap_gb * 1e9))
    out_dir = os.path.join(root, "grounding")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"root": root, "cap_gb": args.cap_gb, "sources": {}}
    names = args.only or list(DATASETS) + ["miniwob"]
    all_rows: list[dict] = []
    for name in names:
        if name == "miniwob":
            continue
        t0 = time.time()
        rows = DATASETS[name](root, budget)
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "heldout_eval")}
        images = {s: len({r["image_path"] for r in rows if r["split"] == s}) for s in counts}
        manifest["sources"][name] = {"rows": len(rows), "by_split": counts, "images_by_split": images,
                                     "bytes": du_bytes(os.path.join(root, name)), "s": round(time.time() - t0, 1)}
        print(f"{name}: {len(rows)} rows {counts} images {images} {manifest['sources'][name]['bytes'] / 1e9:.2f} GB")
        all_rows += rows
    if not args.only or "miniwob" in names:
        manifest["miniwob"] = miniwob(root, budget)
        print("miniwob:", json.dumps(manifest["miniwob"]))
    if not args.only:
        for s in ("train", "val", "heldout_eval"):
            with open(os.path.join(out_dir, f"grounding_{s}.jsonl"), "w") as f:
                for r in all_rows:
                    if r["split"] == s:
                        f.write(json.dumps(r) + "\n")
        # leak check: no image in two splits
        seen: dict[str, str] = {}
        leaks = sum(1 for r in all_rows if seen.setdefault(r["image_path"], r["split"]) != r["split"])
        manifest["image_split_leaks"] = leaks
        manifest["totals"] = {s: sum(1 for r in all_rows if r["split"] == s) for s in ("train", "val", "heldout_eval")}
    manifest["total_bytes"] = du_bytes(root)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"totals {manifest.get('totals')} leaks={manifest.get('image_split_leaks')} disk {manifest['total_bytes'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
