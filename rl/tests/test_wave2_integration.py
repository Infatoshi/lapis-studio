"""Wave-2 integration: build_mix sources / weights / leakage gates, sync_datasets gates, collect_wave2 planning +
offline expert twin + record format, eval_wave2 held-out plan. Offline only (no Daytona, no server, no GPU)."""

from __future__ import annotations

import io
import json
import os
import pickle
import random
import sys
from pathlib import Path

import pytest

RL = Path(__file__).resolve().parents[1]
OS1 = RL / "overnight_stage1"
for p in (str(RL), str(OS1)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_mix as BM  # noqa: E402
import collect_wave2 as C  # noqa: E402
import eval_wave2 as EV  # noqa: E402
import studio_format as SF  # noqa: E402
import sync_datasets as SD  # noqa: E402
from studio_contract import parse_cua_action, parse_policy_output, parse_studio_actions  # noqa: E402
from synth import cal_board as CB  # noqa: E402
from synth import code_review as CR  # noqa: E402

M = C.M
W = EV.W
CU = Path.home() / "cua_datasets"
DENSE = CU / "dense_grounding" / "grounding"


# --------------------------------------------------------------------------- deps


def test_jmespath_in_venv():
    import jmespath

    assert W._jmespath() is jmespath
    assert jmespath.search("a.b", {"a": {"b": 3}}) == 3
    text = (RL / "pyproject.toml").read_text()
    assert '"jmespath' in text


# --------------------------------------------------------------------------- build_mix: sources + weights


def test_sources_cover_wave2():
    for s in ("code_review", "cal_board", "miniwob_46", "webclones_train", "dense_grounding", "public_grounding",
              "mac_grounding", "general_replay", "feed_arena", "tab_forge", "inbox_desk"):
        assert s in BM.SOURCES
    assert len(set(BM.SOURCES)) == len(BM.SOURCES)
    assert set(BM.PRESETS["wave2"]) == set(BM.SOURCES)


def test_wave2_preset_group_shares():
    w = BM.resolve_weights("wave2", None, BM.SOURCES)
    assert abs(sum(w.values()) - 1) < 1e-9
    g = BM.group_shares(w)
    assert g == {"multi_env": 0.35, "grounding": 0.30, "macbook_601": 0.25, "general_replay": 0.10}


def test_weights_renormalize_and_override():
    w = BM.resolve_weights("wave2", '{"general_replay": 0.2}', [s for s in BM.SOURCES if s != "webclones_train"])
    assert w["webclones_train"] == 0.0
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w["general_replay"] > BM.PRESETS["wave2"]["general_replay"]
    with pytest.raises(ValueError):
        BM.resolve_weights("wave2", '{"screenspot_pro": 0.1}', BM.SOURCES)
    assert BM.resolve_weights("stage1", None, BM.SOURCES)["code_review"] == 0.0


def test_public_grounding_allowed_sources_widened():
    assert {"screenspot_v2", "ui_vision", "groundcua", "os_atlas_macos"} == set(BM.PUBLIC_SOURCES)
    assert not set(BM.PUBLIC_SOURCES) & set(BM.HELDOUT_GROUNDING)
    assert set(BM.DENSE_SOURCES) == {"groundcua", "os_atlas_macos"}


# --------------------------------------------------------------------------- build_mix: episode admission


def ep(env, family, split="train", seed=1000, meta=None, reward=1.0):
    return {"env": env, "family": family, "split": split, "seed": seed, "meta": meta or {}, "reward": reward,
            "goal": "g", "time_line": "Current time: 9:00 am",
            "turns": [{"raw": "[[ACTION:cua_click:Chromium|OK]] Clicking OK.", "image": None,
                       "steps": [{"action": "cua_click", "target": "OK", "tier": "L1_AX", "ok": True, "detail": "", "observation": ""}]},
                      {"raw": "Done.", "image": None, "steps": []}]}


def test_miniwob_train_tasks_admitted_heldout_rejected():
    for t in M.ORACLES:
        assert BM.admit_episode(ep("miniwob", t)) == "miniwob_46"
    for t in sorted(M.HELDOUT_TASKS):
        with pytest.raises(BM.LeakageError):
            BM.admit_episode(ep("miniwob", t))
    with pytest.raises(BM.LeakageError):
        BM.admit_episode(ep("miniwob", "not-a-task"))
    assert len(M.ORACLES) == 46 and len(M.HELDOUT_TASKS) == 30


def test_webclones_only_train_apps():
    for app in W.TRAIN_APPS:
        assert BM.admit_episode(ep("webclones", app)) == "webclones_train"
    for app in W.HELDOUT_APPS + W.EXCLUDED_APPS:
        with pytest.raises(BM.LeakageError):
            BM.admit_episode(ep("webclones", app))
    assert set(W.HELDOUT_APPS) == {"gocalendar", "opendining", "staynb"}


def test_synth_envs_train_split_and_theme_only():
    th = BM.synth_themes()
    for env in ("code_review", "cal_board"):
        name = sorted(th[env])[0]
        assert BM.admit_episode(ep(env, "x", meta={"meta": {"theme": name}})) == env
        with pytest.raises(BM.LeakageError):
            BM.admit_episode(ep(env, "x", split="val"))
        with pytest.raises(BM.LeakageError):
            BM.admit_episode(ep(env, "x", meta={"meta": {"theme": sorted(th[env + "_ood"])[0]}}))
    assert not th["code_review"] & th["code_review_ood"] and not th["cal_board"] & th["cal_board_ood"]


def write_traj(d: Path, eps):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "trajectories.jsonl", "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")


def test_load_episodes_skips_eval_splits_and_raises_on_leak(tmp_path):
    good = [ep("code_review", "safe_review_action"), ep("miniwob", "click-test"), ep("webclones", "gomail"),
            ep("feed_arena", "most_viral")]
    evals = [ep("code_review", "x", split="ood_eval"), ep("miniwob", "click-button", split="heldout"),
             ep("webclones", "staynb", split="heldout"), ep("cal_board", "x", split="test")]
    write_traj(tmp_path / "a", good + evals + [ep("miniwob", "click-test", reward=0.0)])
    eps, skipped = BM.load_episodes([(str(tmp_path / "a"), set())], rewarded_only=True)
    assert sorted(e["_src"] for e in eps) == ["code_review", "feed_arena", "miniwob_46", "webclones_train"]
    assert skipped["heldout_split"] == 4 and skipped["unrewarded"] == 1
    eps, skipped = BM.load_episodes([(str(tmp_path / "a"), {"miniwob_46", "webclones"})], rewarded_only=True)
    assert sorted(e["_src"] for e in eps) == ["code_review", "feed_arena"]
    write_traj(tmp_path / "b", [ep("miniwob", "click-button", split="train")])
    with pytest.raises(BM.LeakageError):
        BM.load_episodes([(str(tmp_path / "b"), set())], rewarded_only=True)
    write_traj(tmp_path / "c", [ep("webclones", "gocalendar", split="train")])
    with pytest.raises(BM.LeakageError):
        BM.load_episodes([(str(tmp_path / "c"), set())], rewarded_only=True)


# --------------------------------------------------------------------------- build_mix: grounding gates


def test_grounding_gate_rejects_heldout_sources_and_images():
    keys = {"basename": {"EVAL.png"}, "path": set(), "sha256": {"abc"}}
    ok = {"id": "1", "source": "groundcua", "image_path": "/x/a.png", "image_sha256": "zzz"}
    assert BM.assert_no_grounding_leak([ok], BM.DENSE_SOURCES, keys, "t") == {"groundcua": 1}
    for bad in ({**ok, "source": "screenspot_pro"}, {**ok, "source": "osworld_g"}, {**ok, "image_path": "/y/EVAL.png"},
                {**ok, "image_sha256": "abc"}, {**ok, "source": "mystery"}):
        with pytest.raises(BM.LeakageError):
            BM.assert_no_grounding_leak([bad], BM.DENSE_SOURCES, keys, "t")


@pytest.mark.skipif(not (DENSE / "dense_train.jsonl").exists(), reason="dense grounding not staged")
def test_real_dense_rows_pass_gate_with_zero_eval_leakage():
    keys = BM.eval_image_keys([str(CU / "grounding/screenspot_pro.jsonl"), str(CU / "grounding/osworld_g.jsonl")],
                              str(CU / "dense_grounding/eval_index.json"))
    assert len(keys["basename"]) > 1500 and len(keys["sha256"]) > 1500
    n = 0
    for name in ("dense_train.jsonl", "dense_val.jsonl"):
        rows = [json.loads(l) for l in open(DENSE / name)]
        counts = BM.assert_no_grounding_leak(rows, BM.DENSE_SOURCES, keys, name)
        assert set(counts) <= {"groundcua", "os_atlas_macos"}
        n += len(rows)
    assert n == 45364
    tr, va, st = BM.select_grounding([str(DENSE / "dense_train.jsonl")], str(DENSE / "dense_val.jsonl"),
                                     BM.DENSE_SOURCES, 64, keys, "dense")
    assert len(tr) == 40973 and len(va) == 64 and not {r["id"] for r in tr} & {r["id"] for r in va}


# --------------------------------------------------------------------------- build_mix: trainer fold


def test_trainer_sources_parsed_from_train_py():
    assert BM.trainer_sources() == ("feed_arena", "tab_forge", "inbox_desk", "public_grounding", "mac_grounding", "general_replay")


def _fake(src, n):
    return [{"src": src, "id": f"{src}{k}", "n_tokens": 1, "completion_ids": [1]} for k in range(n)]


def test_fold_reproduces_member_weights():
    sizes = {"feed_arena": 300, "tab_forge": 120, "inbox_desk": 90, "code_review": 240, "cal_board": 200, "miniwob_46": 400,
             "webclones_train": 15, "public_grounding": 4000, "dense_grounding": 40000, "mac_grounding": 540, "general_replay": 700}
    data = {"train": {s: _fake(s, n) for s, n in sizes.items()}, "val": {s: _fake(s, 3) for s in sizes}}
    w = BM.resolve_weights("wave2", None, sizes)
    known = BM.trainer_sources()
    out, ow, fold = BM.fold_for_trainer(data, w, known)
    assert set(out["train"]) == set(known) and abs(sum(ow.values()) - 1) < 1e-9
    assert abs(ow["public_grounding"] - 0.30) < 1e-9 and abs(ow["mac_grounding"] - 0.25) < 1e-9
    # P(src) = P(bucket) * share inside the bucket must equal the native weight
    for host, f in fold.items():
        for s, share in f.get("realized_share", {}).items():
            assert abs(ow[host] * share - w[s]) < 2e-3, (host, s)
    for s, n in sizes.items():  # every sample survives at least once
        host = s if s in known else BM.FOLD[s]
        assert {x["id"] for x in out["train"][host] if x["src"] == s} == {f"{s}{k}" for k in range(n)}
    assert sum(len(v) for v in out["val"].values()) == 3 * len(sizes)
    same, w2, f2 = BM.fold_for_trainer(data, w, BM.SOURCES)
    assert same is data and f2 == {}


# --------------------------------------------------------------------------- build_mix: end to end on fixtures


def _jpeg(path: Path, w=64, h=48):
    from PIL import Image

    Image.new("RGB", (w, h), (30, 60, 90)).save(path, "JPEG", quality=80)


@pytest.fixture
def fake_tokenizer(monkeypatch):
    def render_ids(msgs):
        n_img = sum(1 for m in msgs if isinstance(m.get("content"), list)
                    for c in m["content"] if c.get("type") == "image_url")
        return [1, 2, 3] + [-101] * n_img + SF.FORCED_PREFIX

    monkeypatch.setattr(BM.sf, "render_ids", render_ids)
    monkeypatch.setattr(BM, "encode", lambda text: [7] * max(1, len(text) // 4))


def _fixtures(root: Path):
    img = root / "img_src"
    img.mkdir()
    for k in range(4):
        _jpeg(img / f"g{k}.jpg")
    _jpeg(img / "EVAL_IMG.jpg")
    g = lambda i, src, split: {"id": f"{src}_{i}", "image_path": str(img / f"g{i % 4}.jpg"), "instruction": "the OK button",  # noqa: E731
                               "center_1000": [500, 500], "bbox_1000": [400, 400, 600, 600], "source": src, "split": split}
    def wj(p, rows):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    wj(root / "grounding/grounding_train.jsonl", [g(i, "ui_vision", "train") for i in range(6)])
    wj(root / "grounding/grounding_val.jsonl", [g(i + 10, "screenspot_v2", "val") for i in range(4)])
    wj(root / "dense/dense_train.jsonl", [g(i + 20, "groundcua", "train") for i in range(8)])
    wj(root / "dense/dense_val.jsonl", [g(i + 40, "os_atlas_macos", "val") for i in range(3)])
    wj(root / "grounding/screenspot_pro.jsonl", [{**g(0, "screenspot_pro", "heldout_eval"), "image_path": str(img / "EVAL_IMG.jpg")}])
    wj(root / "grounding/osworld_g.jsonl", [{**g(1, "osworld_g", "heldout_eval"), "image_path": str(img / "EVAL_IMG.jpg")}])
    mac = []
    for i in range(10):
        r = g(i + 60, "mac", "train")
        r.update(ts=i, grounder={"resize_max_width": 1536, "prompt": SF.grounder_prompt(r["instruction"]), "target": '{"point":[1,2]}'})
        mac.append(r)
    wj(root / "mac/mac_train.jsonl", mac)
    wj(root / "replay.jsonl", [{"id": f"r{i}", "prompt_ids": [1, 2], "completion_ids": [3, 4]} for i in range(10)])
    tdir = root / "traj"
    (tdir / "frames").mkdir(parents=True)
    _jpeg(tdir / "frames/t1.jpg", 1024, 768)
    eps = []
    for s in range(1000, 1010):
        for env, fam in (("code_review", "safe_review_action"), ("cal_board", "filter_and_report"),
                         ("miniwob", "click-test"), ("webclones", "gomail")):
            e = ep(env, fam, seed=s)
            e["turns"][1]["image"] = "frames/t1.jpg"
            eps.append(e)
    eps.append(ep("miniwob", "click-button", split="heldout"))
    eps.append(ep("code_review", "x", split="ood_eval"))
    write_traj(tdir, eps)
    return root


def _args(root: Path, out: Path):
    return ["--daytona", str(root / "traj"), "--replay", str(root / "replay.jsonl"), "--out", str(out), "--workers", "2",
            "--grounding-train", str(root / "grounding/grounding_train.jsonl"),
            "--grounding-val", str(root / "grounding/grounding_val.jsonl"),
            "--dense-train", str(root / "dense/dense_train.jsonl"), "--dense-val", str(root / "dense/dense_val.jsonl"),
            "--eval-grounding", f"{root}/grounding/screenspot_pro.jsonl,{root}/grounding/osworld_g.jsonl",
            "--eval-index", str(root / "none.json"), "--mac", str(root / "mac/mac_train.jsonl")]


def test_build_mix_end_to_end(tmp_path, fake_tokenizer):
    root = _fixtures(tmp_path)
    out = tmp_path / "mix"
    man = BM.main(_args(root, out))
    native = man["counts"]["train"]
    for s in ("code_review", "cal_board", "miniwob_46", "webclones_train", "public_grounding", "dense_grounding",
              "mac_grounding", "general_replay"):
        assert native[s] + man["counts"]["val"][s] > 0, s
    assert man["group_shares"]["macbook_601"] > 0 and abs(sum(man["native_weights"].values()) - 1) < 1e-9
    assert man["info"]["daytona"]["skipped"]["heldout_split"] == 2
    data = pickle.load(open(out / "samples.pkl", "rb"))
    assert set(data["train"]) == set(BM.trainer_sources())  # folded for today's train.py
    ids = {x["id"] for sp in data.values() for v in sp.values() for x in v}
    assert not any("click-button" in i or "ood_eval" in i for i in ids)
    assert man["trainer_fold"]["public_grounding"]["members"] == ["public_grounding", "dense_grounding"]
    man2 = BM.main(_args(root, tmp_path / "mix_native") + ["--no-fold"])
    d2 = pickle.load(open(tmp_path / "mix_native/samples.pkl", "rb"))
    assert set(d2["train"]) == set(BM.SOURCES) and man2["trainer_fold"] == {}


def test_stage1_preset_reproduces_old_mix(tmp_path, fake_tokenizer):
    root = _fixtures(tmp_path)
    man = BM.main(_args(root, tmp_path / "mix1") + ["--preset", "stage1"])
    assert set(man["zero_weight_sources"]) == {"code_review", "cal_board", "miniwob_46", "webclones_train", "dense_grounding"}
    for s in man["zero_weight_sources"]:
        assert man["counts"]["train"][s] == 0
    # fixture has no stage-1 env episodes: the stage-1 weights renormalize over grounding + replay (25:25:10)
    w = man["weights"]
    assert man["trainer_fold"] == {} and abs(w["mac_grounding"] - w["public_grounding"]) < 1e-9
    assert abs(w["general_replay"] / w["mac_grounding"] - 0.4) < 1e-9


def test_legacy_env_override_still_works(tmp_path, monkeypatch):
    """build_stage4_data swaps DAYTONA_ENVS for its own env names and calls daytona()/load_episodes."""
    write_traj(tmp_path, [ep("miniwob_forms", "enter-text"), ep("feed_arena", "most_viral"), ep("code_review", "x")])
    monkeypatch.setattr(BM, "DAYTONA_ENVS", ("feed_arena", "miniwob_forms"))
    eps, skipped = BM.load_episodes([(str(tmp_path), set())], rewarded_only=True)
    assert sorted(e["_src"] for e in eps) == ["feed_arena", "miniwob_forms"] and skipped["env"] == 1


def test_build_mix_refuses_leaky_grounding(tmp_path, fake_tokenizer):
    root = _fixtures(tmp_path)
    p = root / "dense/dense_train.jsonl"
    rows = [json.loads(l) for l in open(p)]
    rows[0]["image_path"] = str(root / "img_src/EVAL_IMG.jpg")
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(BM.LeakageError):
        BM.main(_args(root, tmp_path / "mix"))


def test_build_mix_refuses_without_eval_files(tmp_path, fake_tokenizer):
    root = _fixtures(tmp_path)
    args = _args(root, tmp_path / "mix")
    args[args.index("--eval-grounding") + 1] = str(root / "missing.jsonl")
    with pytest.raises(SystemExit):
        BM.main(args)


# --------------------------------------------------------------------------- sync_datasets


def test_sync_gate_rejects_leaks():
    keys = {"basename": {"E.png"}, "path": set(), "sha256": {"s"}}
    SD.gate_train_rows("x", [{"id": 1, "source": "groundcua", "image_path": "/a/b.png"}], keys)
    for bad in ({"id": 1, "source": "osworld_g", "image_path": "/a/b.png"},
                {"id": 1, "source": "groundcua", "image_path": "/a/b.png", "split": "heldout_eval"},
                {"id": 1, "source": "groundcua", "image_path": "/z/E.png"},
                {"id": 1, "source": "groundcua", "image_path": "/a/b.png", "image_sha256": "s"}):
        with pytest.raises(SD.LeakageError):
            SD.gate_train_rows("x", [bad], keys)


def test_sync_files_and_remote_paths():
    assert SD.FILES["dense_grounding/grounding/dense_train.jsonl"] == (None, "train")
    assert SD.FILES["dense_grounding/grounding/dense_val.jsonl"] == (None, "train")
    assert SD.FILES["grounding/osworld_g.jsonl"][1] == "eval" and SD.FILES["grounding/screenspot_pro.jsonl"] == (200, "eval")
    assert SD.remote_path(f"{SD.DS}/dense_grounding/crops/a.jpg", "/home/u") == "/home/u/cua_datasets/dense_grounding/crops/a.jpg"
    assert SD.remote_path(f"{SD.TRACES}/s/f.jpg", "/home/u") == "/home/u/cua_datasets/mac/traces/s/f.jpg"
    with pytest.raises(ValueError):
        SD.remote_path("/tmp/x.png", "/home/u")


@pytest.mark.skipif(not (DENSE / "dense_train.jsonl").exists(), reason="datasets not staged")
def test_sync_dry_run_real(capsys):
    assert SD.main(["--dry-run", "--remote-home", "/home/u"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rows"]["dense_grounding/grounding/dense_train.jsonl"] == 40973
    assert out["rows"]["grounding/screenspot_pro_200.jsonl"] == 200 and out["rows"]["grounding/osworld_g.jsonl"] == 510
    assert out["rows"]["mac/mac_train.jsonl"] == 601
    assert "miniwob/miniwob/html" in out["trees"] and "webclones/agisdk" in out["trees"]


# --------------------------------------------------------------------------- collect_wave2


def test_collect_jobs_train_only():
    jobs, st = C.jobs_for(list(C.ENVS))
    mw = {t for e, _, t in jobs if e == "miniwob"}
    assert mw == set(M.ORACLES) and not mw & M.HELDOUT_TASKS and mw <= set(M.TRAIN_TASKS)
    assert len({M.TASK_SKILLS[t] for t in mw}) == 9
    assert all(s < C.EVAL_SEED_MIN for _, s, _ in jobs)
    assert st["by_env"]["code_review"] + len([k for k in st["skipped_ambiguous"] if k.startswith("code_review")]) == 240
    with pytest.raises(ValueError):
        C.jobs_for(["code_review"], {"code_review": range(4990, 5001)})
    with pytest.raises(ValueError):
        C.jobs_for(["miniwob"], tasks=["click-button"])
    lim, _ = C.jobs_for(list(C.ENVS), limit=2)
    assert len(lim) == 6


@pytest.mark.parametrize("env", ["code_review", "cal_board"])
def test_experts_reach_reward_1_on_twin(env):
    mod = C.PLANNERS[env][0]
    n = 0
    for seed in range(1000, 1030):
        for fam in mod.FAMILIES:
            if "skip" in C.PLANNERS[env][1](mod.generate(seed, family=fam)):
                continue
            r = C.twin_episode(env, seed, fam)
            assert r["reward"] == 1.0, (env, fam, seed, r["info"], r["turns"])
            assert len(r["turns"]) <= SF.MAX_TURNS
            for t in r["turns"]:
                assert C.trainable_turn(t["raw"], t["steps"]), t
            n += 1
    assert n >= 80


def test_twin_catches_a_wrong_expert():
    sc = CR.generate(1000, family="safe_review_action")
    twin = C.PageTwin("code_review", sc)
    wrong = "approve" if sc.mode == "request_changes" else "request_changes"
    other = next(p for p in sc.prs if p.number != sc.target)
    twin.ops.append((wrong, {"pr": sc.target, "ref": str(sc.target), "body": ""}))
    twin.ops.append(("merge", {"pr": other.number, "ref": str(other.number)}))
    r, _ = CR.score(sc, twin.state(), "")
    assert r < 1.0


def test_expert_replies_are_studio_valid():
    for env, fam in (("code_review", "find_failing_check"), ("cal_board", "resolve_conflict")):
        r = C.twin_episode(env, 1001, fam)
        for t in r["turns"]:
            acts = [a for a in parse_policy_output(t["raw"]).actions if a.name.startswith("cua_")]
            for a in acts:
                c = parse_cua_action(a.name, a.arg)
                assert c.valid and (a.arg.startswith("Chromium") or a.name == "cua_observe"), t["raw"]
        assert "[[" not in r["answer"]


def test_studioize_and_point_turns_masked():
    for raw in (M.click_action("Submit"), M.type_action("Username", "kanna"), M.scroll_action("Item 3", -2),
                M.hotkey_action("ctrl+a")):
        s = C.studioize(raw)
        a = parse_studio_actions(s)[0]
        assert parse_cua_action(a.name, a.arg).valid and "Chromium" in a.arg
    pt = C.studioize(M.click_action("circle", (512, 300)))
    assert C.has_point(pt) and not C.trainable_turn(pt, [{"ok": True}])
    assert C.trainable_turn(C.studioize(M.click_action("Submit")), [{"ok": True}])
    assert not C.trainable_turn(C.studioize(M.click_action("Submit")), [{"ok": False}])


def test_episode_record_round_trips_into_build_mix(tmp_path):
    (tmp_path / "frames").mkdir()
    buf = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(buf, "JPEG")
    run = {"env": "miniwob", "seed": 1001, "family": "click-test", "split": "train", "goal": "g", "time_line": "t",
           "reward": 1.0, "info": {}, "answer": "Done.", "finished": True, "bad_actions": 0, "timing": {}, "reset_info": {},
           "turns": [{"raw": C.studioize(M.click_action("Click Me!")) + " Clicking.", "image": None,
                      "steps": [{"action": "cua_click", "ok": True, "observation": ""}]},
                     {"raw": "Done.", "image": buf.getvalue(), "steps": []}]}
    rec = C.episode_record(run, str(tmp_path), {"meta": {"task": "click-test", "split": "train"}})
    assert rec["turns"][1]["image"] == "frames/miniwob_1001_click-test_t1.jpg" and (tmp_path / rec["turns"][1]["image"]).exists()
    assert [t["train"] for t in rec["turns"]] == [True, True] and rec["source"] == "oracle_wave2"
    write_traj(tmp_path, [rec])
    eps, _ = BM.load_episodes([(str(tmp_path), set())], rewarded_only=True)
    assert [e["_src"] for e in eps] == ["miniwob_46"]


def test_collect_dry_run_cli(tmp_path, capsys):
    assert C.main(["--dry-run", "--out", str(tmp_path), "--seeds", '{"code_review":[1000,1006],"cal_board":[1000,1006]}']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["twin_failures"] == 0 and out["miniwob"]["heldout_overlap"] == []
    rep = json.loads((tmp_path / "collect_wave2_dryrun.json").read_text())
    assert rep["studio_format_turn0"]["roles"][0] == "system"


# --------------------------------------------------------------------------- eval_wave2


def test_eval_plan_counts_and_heldout():
    ids = EV.webclones_heldout_ids()
    assert len(ids) == 46
    jobs = EV.plan_jobs(list(EV.INTERACTIVE), ids)
    g = EV.assert_heldout(jobs)
    assert g["episodes"] == {"code_review_ood": 36, "cal_board_ood": 36, "miniwob_heldout": 60, "webclones_heldout": 46}
    assert {j[3] for j in jobs if j[1] == "miniwob"} == set(M.HELDOUT_TASKS)
    train_jobs, _ = C.jobs_for(list(C.ENVS))
    assert not {(e, s, f) for _, e, s, f, _ in jobs} & set(train_jobs)
    assert not {j[3] for j in jobs if j[1] == "miniwob"} & {t for e, _, t in train_jobs if e == "miniwob"}


@pytest.mark.parametrize("bad", [("x", "miniwob", 9000, "click-test", "heldout"),
                                 ("x", "code_review", 5000, "safe_review_action", "train"),
                                 ("x", "cal_board", 1000, "filter_and_report", "ood_eval"),
                                 ("x", "webclones", 0, "v2.gomail-1", "heldout")])
def test_eval_gate_rejects_train_items(bad):
    with pytest.raises(AssertionError):
        EV.assert_heldout([bad])


def test_eval_dry_run_cli(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert EV.main(["--dry-run", "--out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert set(plan["suites"]) == set(EV.SUITES)
    assert plan["interactive"]["train_overlap"] == 0
    assert set(plan["static"]) == {"screenspot_pro_200", "osworld_g", "macbook_151"}
    assert plan["workers"] <= 12
    with pytest.raises(SystemExit):
        EV.main(["--dry-run", "--suites", "screenspot_pro_full"])


def test_eval_summarize_groups():
    res = [{"suite": "miniwob_heldout", "env": "miniwob", "family": "click-button", "seed": 9000, "reward": 1.0, "turns": 2},
           {"suite": "miniwob_heldout", "env": "miniwob", "family": "drag-items", "seed": 9000, "reward": 0.0, "turns": 8},
           {"suite": "webclones_heldout", "env": "webclones", "family": "v1.gocalendar-1", "seed": 0, "reward": 1.0, "turns": 3},
           {"suite": "code_review_ood", "env": "code_review", "family": "safe_review_action", "seed": 5000, "reward": -1.0, "turns": 4}]
    s = EV.summarize(res)
    assert s["miniwob_heldout"]["pass"] == 1 and s["miniwob_heldout"]["by_group"] == {"click": "1/1", "drag": "0/1"}
    assert s["webclones_heldout"]["by_group"] == {"gocalendar": "1/1"}
    assert s["code_review_ood"]["unsafe_negative"] == 1
