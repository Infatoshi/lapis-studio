"""Tests for data_prep/fetch_dense_grounding.py.

Offline unit tests use synthetic screenshots. The integration tests at the bottom read the real dataset under
~/cua_datasets/dense_grounding (skipped when it has not been fetched): 5 random full rows + 5 random crop rows are
checked against their pixels, and the whole dataset is checked for zero overlap with ScreenSpot-Pro / OSWorld-G.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import random
import zipfile

import pytest
from PIL import Image, ImageChops, ImageStat

from data_prep import fetch_dense_grounding as dg

HOME = os.path.expanduser("~")
DENSE = os.path.join(HOME, "cua_datasets", "dense_grounding")
EVAL_DIR = os.path.join(HOME, "cua_datasets", "grounding")
ALL = os.path.join(DENSE, "grounding", "dense_all.jsonl")
ICON = (255, 0, 200)


def _screen(path, w=2940, h=1912, icons=()):
    """Dark synthetic screenshot (seeded random panels, so distinct files get distinct dHashes) with magenta 14x14
    'icons' at the given pixel top-lefts."""
    im = Image.new("RGB", (w, h), (30, 30, 34))
    rng = random.Random(str(path))
    for _ in range(40):
        x, y = rng.randrange(w), rng.randrange(h)
        g = rng.randrange(20, 120)
        im.paste((g, g, g + 10), (x, y, min(w, x + rng.randrange(40, w // 2)), min(h, y + rng.randrange(20, h // 2))))
    for x, y in icons:
        im.paste(ICON, (x, y, x + 14, y + 14))
    im.save(path)
    return path


def _row(path, w, h, box, rid="r0", split="train", kind="action", el=None):
    r = dg.make_row(sid=rid, image_path=str(path), w=w, h=h, px_box=box, instruction="open the export dialog", kind=kind,
                    source="groundcua", split=split, sha="0" * 64, platform="desktop", application="Blender",
                    ui_type="Button", variant="full", _el=el)
    assert r is not None
    return r


# --------------------------------------------------------------------------- schema / geometry


def test_make_row_schema_and_prompt():
    r = _row("/x.png", 2940, 1912, [100, 200, 116, 216])
    assert r["bbox_1000"] == [34, 105, 39, 113]
    assert r["center_1000"] == [36, 109]
    assert dg.strictly_inside(r["center_1000"], r["bbox_1000"])
    assert r["prompt"] == dg.sf.grounder_prompt("open the export dialog")
    assert r["prompt"].startswith('Locate the exact center of UI element: "open the export dialog".')
    assert r["target"] == '{"point":[36,109]}'
    assert r["area_frac"] == pytest.approx(16 * 16 / (2940 * 1912), rel=1e-3)
    assert r["area_frac"] < dg.SMALL_AREA


def test_make_row_rejects_degenerate_box():
    # a 2 px box on a 3840 px frame collapses to zero width at 0..1000: center cannot be strictly inside
    assert dg.make_row(sid="d", image_path="/x.png", w=3840, h=2160, px_box=[1000, 1000, 1002, 1002], instruction="x",
                       kind="label", source="groundcua", split="train", sha="0") is None
    assert dg.make_row(sid="d", image_path="/x.png", w=1920, h=1080, px_box=[10, 10, 40, 40], instruction="",
                       kind="label", source="groundcua", split="train", sha="0") is None


def test_split_is_per_image_deterministic_and_about_90_10():
    assert dg.split_for("groundcua", "abc") == dg.split_for("groundcua", "abc")
    vals = [dg.split_for("groundcua", f"img{i}") for i in range(4000)]
    frac = vals.count("val") / len(vals)
    assert 0.08 < frac < 0.12
    assert set(vals) == {"train", "val"}


def test_select_elements_prefers_dense_and_filters():
    els = [{"bbox": [0, 0, 1000, 800]},  # 14% of a 2940x1912 screen: dropped (> 10%)
           {"bbox": [10, 10, 400, 300]},  # 2% <= a < 10%: kept, after small ones
           {"bbox": [50, 50, 66, 66]}, {"bbox": [70, 50, 86, 66]},  # small icons first
           {"bbox": [5, 5, 7, 7]},  # < 4 px: dropped
           {"bbox": [2930, 10, 2990, 30]}]  # off-frame: dropped
    sel = dg.select_elements(els, 2940, 1912, "k", 5)
    assert [e["bbox"] for e in sel[:2]] == sorted([[50, 50, 66, 66], [70, 50, 86, 66]], key=lambda b: dg.hrank("k", b))
    assert sel[2]["bbox"] == [10, 10, 400, 300]
    assert len(sel) == 3


def test_pick_instruction_mix_and_crop_safety():
    el = {"text": "Export", "instr": {"description": "The Export icon sits in the top-right toolbar of the window.",
                                      "functional_instruction": "Save the drawing as a PNG file.",
                                      "simple_instruction": "How about you hover over `Export`?"}}
    kinds = {dg.pick_instruction(el, f"k{i}")[1] for i in range(300)}
    assert kinds == {"functional_instruction", "label", "description"}  # templated simple_instruction never used
    crop_kinds = {dg.pick_instruction(el, f"k{i}", for_crop=True)[1] for i in range(300)}
    assert "description" not in crop_kinds
    noisy = {"text": "genapuacxa", "instr": {"description": "The language entry in the list."}}
    assert dg.pick_instruction(noisy, "k", for_crop=True) == (None, None)  # OCR-noise label untrusted, no description
    assert dg.pick_instruction(noisy, "k")[1] == "description"


def test_png_dims():
    buf = io.BytesIO()
    Image.new("RGB", (2940, 1912)).save(buf, "PNG")
    assert dg.png_dims(buf.getvalue()[:33]) == (2940, 1912)
    assert dg.png_dims(b"GIF89a" + b"\0" * 30) is None


# --------------------------------------------------------------------------- crops


def test_crop_window_contains_box_and_is_deterministic():
    for i in range(500):
        rng = random.Random(i)
        w, h = rng.choice([(2940, 1912), (3840, 2160), (1920, 1080), (1681, 1050)])
        x0, y0 = rng.randrange(0, w - 20), rng.randrange(0, h - 20)
        box = [x0, y0, x0 + rng.randint(8, 20), y0 + rng.randint(8, 20)]
        pos = dg.crop_window(w, h, box, random.Random(i))
        assert pos == dg.crop_window(w, h, box, random.Random(i))
        cx, cy = pos
        assert 0 <= cx <= w - 1280 and 0 <= cy <= h - 800
        assert cx <= box[0] and cy <= box[1] and box[2] <= cx + 1280 and box[3] <= cy + 800
    assert dg.crop_window(1024, 768, [10, 10, 20, 20], random.Random(0)) is None  # smaller than the window
    assert dg.crop_window(2940, 1912, [0, 0, 1500, 30], random.Random(0)) is None  # target wider than the window


def test_generate_highres_crops_pixel_exact(tmp_path):
    icons = [(2900, 20), (2860, 22), (60, 1880), (1400, 900)]
    img = _screen(tmp_path / "s.png", icons=icons)
    rows = [_row(img, 2940, 1912, [x, y, x + 14, y + 14], rid=f"r{i}") for i, (x, y) in enumerate(icons)]
    out = dg.generate_highres_crops(rows, str(tmp_path / "crops"), n_crops=2, seed=7)
    again = dg.generate_highres_crops(rows, str(tmp_path / "crops"), n_crops=2, seed=7)
    assert [r["id"] for r in out] == [r["id"] for r in again]  # deterministic, same files reused
    assert [r["crop_xywh"] for r in out] == [r["crop_xywh"] for r in again]
    assert len({r["id"] for r in out}) == len(out)
    anchors = {r["parent_id"] for r in out}
    assert anchors == {r["id"] for r in rows}  # every target got a crop
    for r in out:
        assert r["variant"] == "crop" and (r["width"], r["height"]) == (1280, 800)
        assert r["split"] == "train" and r["source_image_path"] == str(img)
        assert dg.strictly_inside(r["center_1000"], r["bbox_1000"])
        assert json.loads(r["target"]) == {"point": r["center_1000"]}
        with Image.open(r["image_path"]) as c:
            assert c.size == (1280, 800) and c.format == "JPEG"
            px = c.convert("RGB").getpixel((round(r["center_1000"][0] / 1000 * 1280),
                                            round(r["center_1000"][1] / 1000 * 800)))
        assert abs(px[0] - ICON[0]) < 40 and px[1] < 60 and abs(px[2] - ICON[2]) < 40, (r["id"], px)
        # no resampling: the 14 px icon stays ~14 px in the crop frame (1000 units == 1280 px)
        assert abs((r["bbox_1000"][2] - r["bbox_1000"][0]) * 1.28 - 14) <= 1.5
    # the two neighbouring top-right icons (40 px apart) share windows: each anchor's window also carries the other
    windows = {}
    for r in out:
        windows.setdefault(r["image_path"], set()).add(r["parent_id"])
    assert any(ps >= {"r0", "r1"} for ps in windows.values())


def test_generate_highres_crops_skips_lowres_and_anchor_filter(tmp_path):
    low = _screen(tmp_path / "low.png", w=1366, h=768, icons=[(100, 100)])
    assert dg.generate_highres_crops([_row(low, 1366, 768, [100, 100, 114, 114])], str(tmp_path / "c")) == []
    img = _screen(tmp_path / "s.png", icons=[(100, 100), (2000, 1500)])
    rows = [_row(img, 2940, 1912, [100, 100, 114, 114], rid="a"), _row(img, 2940, 1912, [2000, 1500, 2014, 1514], rid="b")]
    out = dg.generate_highres_crops(rows, str(tmp_path / "c"), anchor_ids={"b"})
    assert {r["parent_id"] for r in out} == {"b"}  # "a" is 1900 px away: not in b's window, never an anchor


def test_crop_drops_description_without_safe_alternative(tmp_path):
    img = _screen(tmp_path / "s.png", icons=[(500, 500)])
    el = {"text": "zzqx", "instr": {"description": "The thing at the top-right of the window."}}
    r = _row(img, 2940, 1912, [500, 500, 514, 514], kind="description", el=el)
    r["instruction"] = el["instr"]["description"]
    assert dg.generate_highres_crops([r], str(tmp_path / "c")) == []
    el2 = dict(el, instr=dict(el["instr"], functional_instruction="Open the layer settings."))
    r2 = _row(img, 2940, 1912, [500, 500, 514, 514], rid="r2", kind="description", el=el2)
    out = dg.generate_highres_crops([r2], str(tmp_path / "c"))
    assert out and all(o["instruction_kind"] == "functional_instruction" for o in out)
    assert out[0]["prompt"] == dg.sf.grounder_prompt("Open the layer settings.")


# --------------------------------------------------------------------------- leakage gate


def _gate(tmp_path):
    ev_dir = tmp_path / "evalimgs"
    ev_dir.mkdir()
    gdir = tmp_path / "grounding"
    gdir.mkdir()
    a = _screen(ev_dir / "sspro_a.png", w=1920, h=1080, icons=[(300, 300), (900, 200)])
    b = _screen(ev_dir / "osw_b.png", w=1920, h=1080, icons=[(10, 10)])
    for name, p in (("screenspot_pro.jsonl", a), ("osworld_g.jsonl", b)):
        (gdir / name).write_text(json.dumps({"id": name, "image_path": str(p)}) + "\n")
    idx = dg.build_eval_index(str(gdir), str(tmp_path / "eval_index.json"), workers=1)
    return dg.LeakGate(idx), a


def test_leak_gate_catches_every_channel(tmp_path):
    gate, a = _gate(tmp_path)
    clean = _screen(tmp_path / "clean.png", w=1920, h=1080, icons=[(1500, 900), (1600, 100), (200, 950)])
    sha, dh = dg.sha256_file(str(clean)), dg.dhash256(str(clean))
    assert gate.why(str(clean), sha, dh, "groundcua") is None
    assert gate.why(str(clean), sha, dh, "screenspot_pro") == "source"
    assert gate.why(str(a), sha, dh, "groundcua") == "path"
    assert gate.why(str(clean), dg.sha256_file(str(a)), dh, "groundcua") == "sha256"
    same_name = tmp_path / "x"
    same_name.mkdir()
    assert gate.why(str(same_name / "sspro_a.png"), sha, dh, "groundcua") == "basename"
    # re-encoded + resized copy of an eval screenshot: different bytes and name, caught by dHash
    with Image.open(a) as im:
        im.resize((2880, 1620)).save(tmp_path / "resized.jpg", quality=85)
    rs = str(tmp_path / "resized.jpg")
    assert gate.why(rs, dg.sha256_file(rs), dg.dhash256(rs), "groundcua") == "dhash"


def test_assert_no_eval_overlap_raises_on_crop_of_eval_image(tmp_path):
    gate, a = _gate(tmp_path)
    clean = _screen(tmp_path / "clean.png", w=2940, h=1912, icons=[(2500, 100)])
    rows = [_row(clean, 2940, 1912, [2500, 100, 2514, 114])]
    rows[0]["image_sha256"] = dg.sha256_file(str(clean))
    hashes = {str(clean): (rows[0]["image_sha256"], dg.dhash256(str(clean)))}
    assert dg.assert_no_eval_overlap(rows, gate, hashes)["hits"] == 0
    bad = dict(rows[0], id="crop", image_path="/c/x.jpg", source_image_path=str(a), image_sha256=None)
    hashes[str(a)] = (dg.sha256_file(str(a)), dg.dhash256(str(a)))
    with pytest.raises(AssertionError, match="eval leakage"):
        dg.assert_no_eval_overlap(rows + [bad], gate, hashes)


def test_truncated_image_is_flagged_and_blocked(tmp_path):
    gate, _ = _gate(tmp_path)
    good = _screen(tmp_path / "g.png", w=1920, h=1080)
    bad = tmp_path / "t.png"
    bad.write_bytes(open(good, "rb").read()[:5000])  # truncated mid-IDAT, like one OS-Atlas zip member
    h = dg.hash_images([str(good), str(bad)], str(tmp_path / "h.json"), workers=1)
    assert h[str(good)] is not None and h[str(bad)] is None
    row = _row(bad, 1920, 1080, [100, 100, 120, 120])
    with pytest.raises(AssertionError, match="undecodable"):
        dg.assert_no_eval_overlap([row], gate, h)


def test_build_eval_index_refuses_without_eval_jsonls(tmp_path):
    with pytest.raises(SystemExit):
        dg.build_eval_index(str(tmp_path), str(tmp_path / "i.json"))


def test_verify_rows_catches_split_leak_and_bad_target(tmp_path):
    r1 = _row("/a.png", 2940, 1912, [100, 100, 116, 116], rid="a", split="train")
    r2 = _row("/a.png", 2940, 1912, [300, 100, 316, 116], rid="b", split="val")
    with pytest.raises(AssertionError, match="two splits"):
        dg.verify_rows([r1, r2])
    r3 = dict(r1, target='{"point":[1,1]}')
    with pytest.raises(AssertionError):
        dg.verify_rows([r3])
    assert dg.verify_rows([r1])["rows"] == 1


# --------------------------------------------------------------------------- remote zip (no network)


def test_remote_zip_reads_member_and_head(tmp_path, monkeypatch):
    buf = io.BytesIO()
    png = io.BytesIO()
    Image.new("RGB", (2880, 1800), (9, 9, 9)).save(png, "PNG")
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("macos/a.png", png.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("macos/b.png", png.getvalue(), compress_type=zipfile.ZIP_STORED)
    blob = buf.getvalue()

    def fake_init(self, url):
        io.RawIOBase.__init__(self)
        self.url, self.size, self.pos, self.cache = url, len(blob), 0, {}

    monkeypatch.setattr(dg.HTTPRangeFile, "__init__", fake_init)
    monkeypatch.setattr(dg.HTTPRangeFile, "range", lambda self, a, b: blob[a:b])
    monkeypatch.setattr(dg.HTTPRangeFile, "BLOCK", 64)
    rz = dg.RemoteZip("http://example.invalid/x.zip")
    for name in ("macos/a.png", "macos/b.png"):
        assert dg.png_dims(rz.head(name, 256)) == (2880, 1800)
        assert rz.read_member(name) == png.getvalue()


# --------------------------------------------------------------------------- integration: the real dataset


real = pytest.mark.skipif(not os.path.exists(ALL), reason="dense grounding dataset not fetched")


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line) for line in open(ALL)]


@real
def test_real_schema_geometry_and_splits(rows):
    st = dg.verify_rows(rows)
    assert st["rows"] == len(rows) > 1000
    train = [r for r in rows if r["split"] == "train"]
    frac = 1 - len(train) / len(rows)
    assert 0.05 < frac < 0.15, frac
    assert {r["source"] for r in rows} <= {"groundcua", "os_atlas_macos"}


@real
def test_real_five_full_samples_match_pixels(rows):
    full = [r for r in rows if r["variant"] == "full"]
    for r in random.Random(1).sample(full, 5):
        with Image.open(r["image_path"]) as im:
            assert im.size == (r["width"], r["height"])
        assert r["width"] >= 1600
        assert dg.sha256_file(r["image_path"]) == r["image_sha256"]
        assert r["bbox_1000"] == dg.box_1000(r["bbox_px"], r["width"], r["height"])
        assert dg.strictly_inside(r["center_1000"], r["bbox_1000"])
        assert json.loads(r["target"]) == {"point": r["center_1000"]}


@real
def test_real_five_crop_samples_match_source_region(rows):
    crops = [r for r in rows if r["variant"] == "crop"]
    assert crops
    for r in random.Random(2).sample(crops, 5):
        cx, cy, cw, ch = r["crop_xywh"]
        with Image.open(r["image_path"]) as c, Image.open(r["source_image_path"]) as src:
            assert c.size == (cw, ch) == (1280, 800)
            ref = src.convert("RGB").crop((cx, cy, cx + cw, cy + ch))
            diff = ImageStat.Stat(ImageChops.difference(c.convert("RGB"), ref)).mean
        assert max(diff) < 6, (r["id"], diff)  # same pixels up to JPEG q80 noise: no resize, no offset
        assert dg.strictly_inside(r["center_1000"], r["bbox_1000"])
        assert r["bbox_1000"] == dg.box_1000(r["bbox_px"], cw, ch)
        parent = next(p for p in rows if p["id"] == r["parent_id"])
        px = parent["bbox_px"]
        assert r["bbox_px"] == [round(v, 2) for v in (px[0] - cx, px[1] - cy, px[2] - cx, px[3] - cy)]
        assert r["split"] == parent["split"]


@real
def test_real_zero_overlap_with_heldout_evals(rows):
    eval_rows = []
    for name in ("screenspot_pro", "osworld_g"):
        for p in glob.glob(os.path.join(EVAL_DIR, f"{name}*.jsonl")):
            eval_rows += [json.loads(line) for line in open(p)]
    assert len(eval_rows) > 1000
    ev_paths = {os.path.realpath(r["image_path"]) for r in eval_rows}
    ev_names = {os.path.basename(p) for p in ev_paths}
    ev_ids = {r["id"] for r in eval_rows}
    srcs = {r.get("source_image_path") or r["image_path"] for r in rows}
    assert not {os.path.realpath(p) for p in srcs} & ev_paths
    assert not {os.path.basename(p) for p in srcs} & ev_names
    assert not {r["id"] for r in rows} & ev_ids
    assert not {r["source"] for r in rows} & {"screenspot_pro", "osworld_g"}
    # content: sha256 of every source image vs every eval image (cached index, recomputed if an eval file changed)
    idx = json.load(open(os.path.join(DENSE, "eval_index.json")))
    assert idx["n_images"] == len(ev_paths) and idx["missing"] == 0
    ev_sha = {e["sha256"] for e in idx["images"]}
    sample = random.Random(3).sample(sorted(srcs), 25)
    assert not {hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sample} & ev_sha
    hashes = json.load(open(os.path.join(DENSE, "image_hashes.json")))
    assert not {hashes[p][0] for p in srcs} & ev_sha
    dh = [int(e["dhash"]) for e in idx["images"]]
    near = [p for p in srcs if any((int(hashes[p][1]) ^ e).bit_count() <= 8 for e in dh)]
    assert not near, near[:3]


@real
def test_real_disk_under_cap_and_manifest(rows):
    total = dg.du_bytes(DENSE)
    assert total <= 4.0e9, total
    man = json.load(open(os.path.join(DENSE, "grounding", "manifest.json")))
    assert man["gates"]["leakage"]["hits"] == 0
    assert man["stats"]["all"]["rows"] == len(rows)
