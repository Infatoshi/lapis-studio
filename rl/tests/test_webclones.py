"""Offline tests for envs.webclones: frozen app split, deterministic-only task filter, Studio
action/observation formatting, verifier parsing, and the trainer adapter. No network, no sandboxes."""

from __future__ import annotations

import json

import pytest

from envs import webclones as W
from studio_contract import format_cdp_ax_lines, observation_prompt, parse_cua_action, parse_policy_output

HAVE_DATA = W.webclones_src().exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run envs.webclones.stage_data() first")


@pytest.fixture(scope="module")
def idx() -> W.TaskIndex:
    if not HAVE_DATA:
        pytest.skip("agisdk not staged")
    return W.load_tasks()


# --------------------------------------------------------------------------- split


def test_split_disjoint_and_sized():
    assert set(W.TRAIN_APPS).isdisjoint(W.HELDOUT_APPS)
    assert len(W.TRAIN_APPS) == 8 and len(W.HELDOUT_APPS) == 3
    buckets = list(W.TRAIN_APPS) + list(W.HELDOUT_APPS) + list(W.EXCLUDED_APPS)
    assert sorted(buckets) == sorted(W.ALL_APPS) and len(set(buckets)) == len(buckets)
    assert isinstance(W.TRAIN_APPS, tuple) and isinstance(W.HELDOUT_APPS, tuple)


def test_split_is_pinned():
    assert W.split_fingerprint() == W.SPLIT_SHA256
    swapped = list(W.TRAIN_APPS[1:]) + ["gocalendar"]
    assert W.split_fingerprint(swapped, ["dashdish", "opendining", "staynb"]) != W.SPLIT_SHA256


@needs_data
def test_split_covers_every_upstream_app():
    apps = {t.app for t in W.load_all_tasks()}
    assert apps == set(W.ALL_APPS)


def test_app_split_lookup():
    assert W.app_split("gomail") == "train"
    assert W.app_split("staynb") == "heldout"
    assert W.app_split("zilloft") == "train"
    assert W.app_split("marrisuite") == "excluded"


# --------------------------------------------------------------------------- deterministic filter


@needs_data
def test_only_deterministic_graders_kept(idx):
    assert idx.kept, "no tasks loaded"
    for t in idx.kept:
        assert set(t.eval_kinds) <= {"jmespath", "script"}, t.id
        assert "llm_boolean" not in t.eval_kinds
        assert t.possible and t.split in ("train", "heldout")
        for ev in t.evals:
            if ev.type == "script":
                assert W.audit_script(W.script_path(t, ev).read_text()) == [], t.id


@needs_data
def test_known_drops(idx):
    d = idx.dropped
    assert d["v1.gomail-1"] == "llm_judge"
    assert d["v2.staynb-9"] == "script_run_id"  # run_id-seconds parity fallback
    assert d["v2.omnizon-25"].startswith("script_reads_path")  # folder-timestamp fallback
    assert d["v2.networkin-25"].startswith("script_reads_path")  # hard-coded folder name
    assert d["v2.gomail-21"].startswith("banned:passes_on_empty_state")
    assert d["v2.marrisuite-8"] == "excluded_app"
    assert d["v2.zilloft-3"] == d["v2.zilloft-14"] == "dead_deployment"  # 410 GONE
    assert d["v1.omnizon-2"] == "dead_deployment"  # DMCA takedown page
    assert sum(r == "dead_deployment" for r in d.values()) == 20
    assert sum(r == "llm_judge" for r in d.values()) == 55


@needs_data
def test_impure_scripts_flagged_even_on_dead_deployments():
    tasks = {t.id: t for t in W.load_all_tasks()}
    t = tasks["v2.zilloft-14"]
    assert W.audit_script(W.script_path(t, t.evals[0]).read_text()) == ["wall_clock"]
    t = tasks["v2.zilloft-3"]
    assert W.audit_script(W.script_path(t, t.evals[0]).read_text()) == ["run_id"]


@needs_data
def test_counts_snapshot(idx):
    c = idx.counts()
    assert c["kept"] == 152 and c["kept"] + c["dropped"] == 233
    assert c["train"] == 106 and c["heldout"] == 46
    assert set(c["per_app"]) == set(W.TRAIN_APPS) | set(W.HELDOUT_APPS)
    assert {a: c["per_app"][a]["total"] for a in W.HELDOUT_APPS} == {"gocalendar": 14, "opendining": 16, "staynb": 16}
    assert all(v["total"] >= 2 for v in c["per_app"].values())


def test_audit_script_patterns():
    assert W.audit_script("import json\n# avoid false positives like 'random'\nprint('SUCCESS')\n") == []
    assert W.audit_script("from datetime import datetime\nx = datetime.now()\n") == ["wall_clock"]
    assert W.audit_script("import os\nfolder = os.path.basename(p)\n") == ["reads_path_or_fs"]
    assert W.audit_script("rid = cfg.get('run_id')\n") == ["run_id"]
    assert W.audit_script("import random\nrandom.random()\n") == ["random"]
    # a '#' inside a string is not a comment: the code after it is still audited
    assert W.audit_script("s = 'a#b'; t = datetime.now()\n") == ["wall_clock"]


# --------------------------------------------------------------------------- task draws


@needs_data
def test_pick_task_respects_split(idx):
    for seed in range(300):
        assert W.pick_task(idx, seed, "train").app in W.TRAIN_APPS
        assert W.pick_task(idx, seed, "heldout").app in W.HELDOUT_APPS
        assert W.pick_task(idx, seed, "ood_eval").app in W.HELDOUT_APPS
    assert W.pick_task(idx, 7, "train") == W.pick_task(idx, 7, "train")
    assert W.pick_task(idx, 3, "train", "gomail").app == "gomail"
    with pytest.raises(ValueError):
        W.pick_task(idx, 0, "train", "gocalendar")
    with pytest.raises(ValueError):
        W.pick_task(idx, 0, "train", "v2.gocalendar-4")
    with pytest.raises(ValueError):
        W.pick_task(idx, 0, "heldout", "v1.gomail-2")
    with pytest.raises(ValueError):
        W.pick_task(idx, 0, "bogus")


@needs_data
def test_task_urls(idx):
    t = idx.get("v2.gomail-1")
    assert t.config_url == "https://real-gomail-1glk4rtaf-real-sites.vercel.app/config?run_id=0&task_id=v2.gomail-1&latency=0"
    assert t.finish_url.endswith(".vercel.app/finish") and "//finish" not in t.finish_url
    assert t.origin == "https://real-gomail-1glk4rtaf-real-sites.vercel.app"


# --------------------------------------------------------------------------- verifier


def test_parse_finish_text():
    assert W.parse_finish_text(' {"differences": {}} \n') == {"differences": {}}
    for bad in ("", "   ", "<html>", "[1,2]", "null"):
        with pytest.raises(W.VerifierError):
            W.parse_finish_text(bad)


@needs_data
def test_jmespath_verifier(idx):
    t = idx.get("v1.gomail-2")  # differences.emails.updated[0].isRead == True
    ok = {"differences": {"emails": {"updated": [{"id": "1", "isRead": True}]}}}
    assert W.verify(t, ok)[0] == 1.0
    assert W.verify(t, {"differences": {"emails": {"updated": [{"id": "1", "isRead": False}]}}})[0] == 0.0
    assert W.verify(t, {"differences": {"emails": {"updated": [{"id": "1", "isRead": "true"}]}}})[0] == 0.0
    r, info = W.verify(t, {})
    assert r == 0.0 and info["results"][0]["actual"] is None


@needs_data
def test_jmespath_functions(idx):
    t = idx.get("v1.gomail-8")  # updated[0].id == '3', updated[0].trash, length(updated) == 1
    one = {"differences": {"emails": {"updated": [{"id": "3", "trash": True}]}}}
    two = {"differences": {"emails": {"updated": [{"id": "3", "trash": True}, {"id": "4", "trash": True}]}}}
    wrong = {"differences": {"emails": {"updated": [{"id": 3, "trash": True}]}}}  # int id != '3'
    assert W.verify(t, one)[0] == 1.0
    assert W.verify(t, two)[0] == 0.0
    r, info = W.verify(t, wrong)
    assert r == 0.0 and [x["ok"] for x in info["results"]] == [False, True, True]


@needs_data
def test_script_verifier(idx):
    t = idx.get("v2.gomail-1")
    r, info = W.verify(t, {})
    assert r == 0.0 and info["results"][0]["type"] == "script" and "FAILURE" in info["results"][0]["output"]


@needs_data
def test_verify_refuses_llm_tasks():
    t = next(x for x in W.load_all_tasks() if x.id == "v1.gomail-1")
    with pytest.raises(W.VerifierError):
        W.verify(t, {})


@needs_data
def test_no_task_rewards_an_empty_state(idx):
    empties = ({}, {"initialfinaldiff": {"added": {}, "updated": {}, "deleted": {}}, "differences": {}})
    for t in idx.kept:
        for st in empties:
            assert W.verify(t, st)[0] == 0.0, t.id


@needs_data
def test_live_noop_states_score_zero(idx):
    """Replays /finish states captured live right after reset (noop_sweep). Skips until captured."""
    d = W.DATA_ROOT / "noop_states"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        pytest.skip("no live no-op states captured yet")
    kept = {t.id: t for t in idx.kept}
    n = 0
    for f in files:
        t = kept.get(f.stem)
        if t is None:
            continue  # banned after capture
        assert W.verify(t, json.loads(f.read_text()))[0] == 0.0, t.id
        n += 1
    assert n > 0


# --------------------------------------------------------------------------- Studio formatting


ORACLE = W.ORACLES["v1.gomail-2"]


def test_oracle_reply_parses_as_studio_action():
    out = parse_policy_output(ORACLE[0])
    assert [a.name for a in out.actions] == ["cua_click"]
    cmd = parse_cua_action(out.actions[0].name, out.actions[0].arg)
    assert cmd.valid and cmd.kind == "click"
    assert cmd.target == "Chromium|Email from Support, Your Account Statement is Ready"
    assert out.clean_text == "Opening the first inbox email marks it read."
    final = parse_policy_output(ORACLE[1])
    assert final.actions == [] and final.clean_text.startswith("Done")


def test_cdp_row_formats_like_studio():
    nodes = [
        {"nodeId": "1", "role": {"value": "RootWebArea"}, "name": {"value": "GoMail"}, "childIds": ["2"],
         "bounds": [0, 0, 991, 638]},
        {"nodeId": "2", "parentId": "1", "role": {"value": "row"},
         "name": {"value": "Email from Support, Your Account Statement is Ready"}, "childIds": ["3"],
         "bounds": [256, 314, 633, 40]},
        {"nodeId": "3", "parentId": "2", "role": {"value": "checkbox"}, "name": {"value": "Select email from Support"},
         "bounds": [273, 327, 16, 16]},
    ]
    text = format_cdp_ax_lines(nodes, origin=(16, 114), app_name="Chromium")
    assert text.splitlines() == [
        "[Chromium pid 0]",
        'WebArea "GoMail" @16,114 991x638',
        'Row "Email from Support, Your Account Statement is Ready" @272,428 633x40',
        'CheckBox "Select email from Support" @289,441 16x16 {Press}',
    ]


def test_stage1_target_strip_and_observation_header():
    E = W._stage1_envs()
    assert E.strip_app("Chromium|Email from Support, Your Account Statement is Ready") == \
        "Email from Support, Your Account Statement is Ready"
    head = observation_prompt("cua_click:Email from Support", "L1_AX", True, 'AXRow "Email from Support"', "x")
    assert head.startswith("[CUA observation after cua_click:Email from Support (L1_AX, ok=1,")


def test_env_class_is_a_studio_env():
    E = W._stage1_envs()
    cls = W._env_class()
    assert issubclass(cls, E.StudioEnv) and cls.name == "webclones"


# --------------------------------------------------------------------------- trainer adapter


def _episode(app: str = "gomail", split: str = "train", reward: float = 1.0) -> dict:
    step = {"action": "cua_click", "target": "Chromium|Email from Support", "tier": "L1_AX", "ok": True,
            "detail": 'AXRow "Email from Support" (Chromium)', "observation": 'Row "Email from Support" @1,1 9x9'}
    return {"env": "webclones", "seed": 0, "family": app, "split": split, "goal": "Mark the first email as read.",
            "time_line": "Current time: 9:41 pm", "reward": reward, "reset_info": {"task_id": "v1.gomail-2"},
            "turns": [{"raw": ORACLE[0], "image": None, "steps": [step]},
                      {"raw": ORACLE[1], "image": b"\xff\xd8fakejpeg", "steps": []}]}


def test_episode_samples_shape():
    SF = W._studio_format()
    s = W.episode_samples(_episode())
    assert [x["turn"] for x in s] == [0, 1]
    assert s[0]["completion_text"] == ORACLE[0] and s[1]["completion_text"] == ORACLE[1]
    assert [m["role"] for m in s[0]["messages"]] == ["system", "user"]
    assert s[0]["messages"][0]["content"] == SF.STUDIO_SYSTEM_PROMPT
    assert s[0]["messages"][-1]["content"].endswith("Question: Mark the first email as read.")
    m1 = s[1]["messages"]
    assert [m["role"] for m in m1] == ["system", "user", "assistant", "user"]
    leaf = m1[-1]["content"]
    assert leaf[0]["type"] == "image_url" and leaf[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "[CUA observation after cua_click:Chromium|Email from Support (L1_AX, ok=1" in leaf[1]["text"]
    assert s[0]["src"] == "webclones" and s[0]["task_id"] == "v1.gomail-2" and s[0]["split"] == "train"


def test_episode_samples_refuse_heldout_and_failed():
    with pytest.raises(ValueError):
        W.episode_samples(_episode(app="staynb", split="train"))
    with pytest.raises(ValueError):
        W.episode_samples(_episode(app="gomail", split="heldout"))
    assert W.episode_samples(_episode(reward=0.0)) == []
    assert len(W.episode_samples(_episode(reward=0.0), include_failed=True)) == 2


def test_scripted_policy():
    p = W.scripted_policy(["a", "b"])
    assert [p([], {"turn": k}) for k in range(3)] == ["a", "b", "Done."]
