"""Offline tests + 10k-seed shortcut audits for TabForge and InboxDesk.

Live Daytona smoke (2x daytona-small, always deleted):
  LAPIS_LIVE_DAYTONA=1 uv run pytest tests/test_tab_inbox.py -k live -s
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import difflib
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean

import pytest

from synth import inbox_desk as I
from synth import tab_forge as T

N_AUDIT = 10_000
HERE = Path(__file__).resolve().parent.parent


# ============================================================================ TabForge


def test_tabforge_deterministic_and_seed_sensitive():
    for s in range(50):
        assert T.generate(s).to_json() == T.generate(s).to_json()
    assert len({T.generate(s).to_json() for s in range(200)}) == 200


def test_tabforge_knobs_and_invariants():
    for s in range(2000):
        sc = T.generate(s)
        n = len(sc.tabs)
        assert 4 <= n <= 12 and n == sc.meta["n_tabs"]
        assert len({t.slug for t in sc.tabs}) == n and len({t.code for t in sc.tabs}) == n
        assert sum(t.is_form for t in sc.tabs) == 1
        tgt = sc.target
        assert tgt.group >= 0 and sc.meta["group_size"] >= 2
        group = [t for t in sc.tabs if t.group == tgt.group]
        assert len({t.title for t in group}) == 1 and len({t.attr_value for t in group}) == len(group)
        assert sc.active_tab_idx != sc.target_idx
        assert sc.protected_idx, "at least one protected tab"
        if sc.family == "close_only_requested":
            assert not tgt.protected
        if sc.family == "cross_tab_lookup":
            assert sc.tabs[sc.dest_idx].is_form and sc.mode in ("submit", "report")
        assert "ABC-1234" not in {t.code for t in sc.tabs}
    sc = T.generate(1, family="switch_to_tab", n_tabs=12, n_lookalikes=3, style_seed=4)
    assert len(sc.tabs) == 12 and sc.meta["n_lookalikes"] == 3 and sc.meta["ood_style"]
    sc = T.generate(1, family="switch_to_tab", n_tabs=4, n_lookalikes=3)
    assert sc.meta["n_lookalikes"] == 1  # clamped: 4 tabs fit one group + form
    for bad in ({"n_tabs": 3}, {"n_tabs": 13}, {"n_lookalikes": 0}, {"family": "nope"}):
        with pytest.raises(ValueError):
            T.generate(0, **bad)
    sc = T.generate(3, family="close_only_requested")
    with pytest.raises(ValueError):
        T.generate(3, family="close_only_requested", protected_tabs=[sc.target_idx])
    sc2 = T.generate(3, family="switch_to_tab", active_tab_idx=0 if sc.target_idx else 1, protected_tabs=[])
    assert sc2.protected_idx == []


def test_tabforge_styles_split():
    assert {T.is_ood_style(s) for s in range(3)} == {False}
    assert {T.is_ood_style(s) for s in (3, 4)} == {True}
    assert all(not T.generate(s).meta["ood_style"] for s in range(300))  # default style is train


def test_tabforge_render_contains_disambiguators():
    sc = T.generate(11, family="cross_tab_lookup", mode="submit")
    for t in sc.tabs:
        page = T.render_tab(sc, t)
        assert t.code in page and f"{t.memory_mb} MB" in page
        assert (t.attr_value in page) if t.attr_value else True
        assert ("PROTECTED" in page) == t.protected
        assert ('id="code"' in page) == t.is_form
    compile(T.SERVER_PY, "server.py", "exec")
    compile(T.PROBE_PY, "probe.py", "exec")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_tabforge_server_logs_submissions(tmp_path):
    sc = T.generate(5, family="cross_tab_lookup", mode="submit")
    (tmp_path / "t").mkdir()
    form = sc.tabs[sc.dest_idx]
    (tmp_path / "t" / f"{form.slug}.html").write_text(T.render_tab(sc, form))
    srv = tmp_path / "server.py"
    srv.write_text(T.SERVER_PY)
    log = tmp_path / "subs.jsonl"
    port = _free_port()
    p = subprocess.Popen([sys.executable, str(srv), str(tmp_path), str(port), str(log)])
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                assert b"Submit" in urllib.request.urlopen(f"{base}/t/{form.slug}.html", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        urllib.request.urlopen(f"{base}/submit?tab={form.slug}&value={sc.target.code}", timeout=2).read()
        subs = [json.loads(x) for x in log.read_text().splitlines()]
        assert subs[-1]["tab"] == form.slug and subs[-1]["value"] == sc.target.code
        assert T.score(sc, T.simulate_probe(sc, active=sc.dest_idx), subs, None)[0] == 1.0
    finally:
        p.kill()


def test_tabforge_score_oracle_and_penalties():
    for s in range(600):
        sc = T.generate(s)
        tgt, others = sc.target, [t for t in sc.tabs if t.idx != sc.target_idx]
        prot = sc.protected_idx
        unprot = [t.idx for t in others if not t.protected]
        sub_ok = [{"tab": sc.tabs[sc.dest_idx].slug, "value": tgt.code}] if sc.dest_idx >= 0 else []
        ans_ok = f"The code is {tgt.code}"
        if sc.family == "switch_to_tab":
            assert T.score(sc, T.simulate_probe(sc, active=tgt.idx), [], None)[0] == 1.0
            assert T.score(sc, T.simulate_probe(sc), [], None)[0] == 0.0  # stayed put
            assert T.score(sc, T.simulate_probe(sc, closed={prot[0]}, active=tgt.idx), [], None)[0] == -1.0
            if unprot:
                assert T.score(sc, T.simulate_probe(sc, closed={unprot[0]}, active=tgt.idx), [], None)[0] == 0.0
        elif sc.family == "close_only_requested":
            assert T.score(sc, T.simulate_probe(sc, closed={tgt.idx}), [], None)[0] == 1.0
            assert T.score(sc, T.simulate_probe(sc), [], None)[0] == 0.0
            assert T.score(sc, T.simulate_probe(sc, closed={tgt.idx, prot[0]}), [], None)[0] == -1.0
            if unprot:
                assert T.score(sc, T.simulate_probe(sc, closed={tgt.idx, unprot[0]}), [], None)[0] == 0.0
        else:
            probe = T.simulate_probe(sc, active=sc.dest_idx)
            if sc.mode == "submit":
                assert T.score(sc, probe, sub_ok, None)[0] == 1.0
                assert T.score(sc, probe, [], ans_ok)[0] == 0.0  # must submit, not just say it
                wrong = [{"tab": sc.tabs[sc.dest_idx].slug, "value": others[0].code}]
                assert T.score(sc, probe, wrong, None)[0] == 0.0
                assert T.score(sc, probe, sub_ok + wrong, None)[0] == 0.0  # last submission counts
                assert T.score(sc, probe, wrong + sub_ok, None)[0] == 1.0
                assert T.score(sc, probe, [{"tab": tgt.slug, "value": tgt.code}], None)[0] == 0.0
            else:
                assert T.score(sc, probe, [], ans_ok.lower())[0] == 1.0
                assert T.score(sc, probe, [], f"{tgt.code} or {others[0].code}")[0] == 0.0
            assert T.score(sc, T.simulate_probe(sc, closed={prot[0]}), sub_ok, ans_ok)[0] == -1.0
        assert T.score(sc, {"ok": False, "tabs": []}, sub_ok, ans_ok)[0] in (0.0,)


def test_tabforge_probe_parsing():
    sc = T.generate(2, family="switch_to_tab")
    p = T.simulate_probe(sc, active=sc.target_idx)
    p["tabs"].append({"url": "chrome://newtab/", "title": "New Tab", "value": "hidden"})
    assert T.active_slug(p) == sc.target.slug
    assert T.slug_of("http://127.0.0.1:8771/t/ab12cd34.html#x") == "ab12cd34"
    assert T.slug_of("chrome://newtab/") is None


def _tab_heuristics(sc: T.Scenario, rng: random.Random) -> dict[str, float]:
    tgt = sc.target
    first_title = next(t for t in sc.tabs if t.title == tgt.title)
    rand = rng.randrange(len(sc.tabs))
    out: dict[str, float] = {}
    if sc.family == "switch_to_tab":
        sw = lambda i: T.score(sc, T.simulate_probe(sc, active=i), [], None)[0]
        out = {"stay_put": sw(sc.active_tab_idx), "first_tab": sw(0), "last_tab": sw(len(sc.tabs) - 1),
               "random_tab": sw(rand), "first_title_match": sw(first_title.idx)}
    elif sc.family == "close_only_requested":
        cl = lambda i: T.score(sc, T.simulate_probe(sc, closed={i}), [], None)[0]
        out = {"close_active(ctrl+w)": cl(sc.active_tab_idx), "close_first_tab": cl(0),
               "close_random_tab": cl(rand), "close_first_title_match": cl(first_title.idx)}
    else:
        probe = T.simulate_probe(sc, active=sc.dest_idx)
        dest = sc.tabs[sc.dest_idx].slug

        def give(code: str) -> float:
            subs = [{"tab": dest, "value": code}] if sc.mode == "submit" else []
            return T.score(sc, probe, subs, code)[0]

        out = {"code_of_active_tab": give(sc.tabs[sc.active_tab_idx].code), "code_of_first_tab": give(sc.tabs[0].code),
               "code_of_random_tab": give(sc.tabs[rand].code), "code_of_first_title_match": give(first_title.code),
               "code_of_form_tab": give(sc.tabs[sc.dest_idx].code)}
    return out


def _audit(gen, heur, families, n=N_AUDIT):
    res: dict[str, dict[str, list[float]]] = {f: {} for f in families}
    chance: dict[str, list[float]] = {f: [] for f in families}
    for fam in families:
        rng = random.Random(f"audit:{fam}")
        for s in range(n):
            sc = gen(s, fam)
            for k, v in heur(sc, rng).items():
                res[fam].setdefault(k, []).append(v)
            chance[fam].append(sc._chance)
    return ({f: {k: mean(v) for k, v in d.items()} for f, d in res.items()},
            {f: mean(v) for f, v in chance.items()})


def _print_audit(name, res, chance):
    print(f"\n[{name}] shortcut audit, {N_AUDIT} seeds per family (mean reward; oracle = 1.0)")
    for fam, d in res.items():
        print(f"  {fam}: chance~{chance[fam]:.3f}  " + "  ".join(f"{k}={v:+.3f}" for k, v in d.items()))


def test_tabforge_shortcut_audit_10k():
    def gen(s, fam):
        sc = T.generate(s, family=fam)
        # chance = picking uniformly among the k same-title look-alikes (the best a title-reader can do)
        sc._chance = 1.0 / sc.meta["group_size"]
        return sc

    res, chance = _audit(gen, _tab_heuristics, T.FAMILIES)
    _print_audit("TabForge", res, chance)
    sw, cl, cx = res["switch_to_tab"], res["close_only_requested"], res["cross_tab_lookup"]
    assert sw["stay_put"] == 0.0
    assert sw["first_tab"] < 0.15 and sw["last_tab"] < 0.15 and sw["random_tab"] < 0.15
    assert abs(sw["first_title_match"] - chance["switch_to_tab"]) < 0.03  # title alone = coin flip among look-alikes
    assert cl["close_active(ctrl+w)"] < -0.15  # active tab is often protected; never the target
    assert cl["close_first_tab"] < 0.0 and cl["close_random_tab"] < 0.0
    assert abs(cl["close_first_title_match"] - chance["close_only_requested"]) < 0.05 or cl["close_first_title_match"] < chance["close_only_requested"]
    assert cx["code_of_active_tab"] == 0.0 and cx["code_of_form_tab"] == 0.0
    assert cx["code_of_first_tab"] < 0.15 and cx["code_of_random_tab"] < 0.15
    assert abs(cx["code_of_first_title_match"] - chance["cross_tab_lookup"]) < 0.03


# ============================================================================ InboxDesk


def test_inbox_deterministic_and_seed_sensitive():
    for s in range(50):
        assert I.generate(s).to_json() == I.generate(s).to_json()
    assert len({I.generate(s).to_json() for s in range(200)}) == 200


def test_inbox_knobs_and_invariants():
    for s in range(2000):
        sc = I.generate(s)
        n = len(sc.threads)
        assert 12 <= n <= 30 and len({t.id for t in sc.threads}) == n
        assert sum(t.unread for t in sc.threads) == sc.meta["n_unread"]
        assert len({t.fact for t in sc.threads}) == n
        tgt = sc.target
        assert tgt.cc, "reply-all must differ from reply"
        assert tgt.fact_type == sc.fact_type and tgt.fact in tgt.body
        assert I.extract_facts(sc.fact_type, tgt.body) == {tgt.fact}
        same = [t for t in sc.threads if t.sender_name == tgt.sender_name and t.subject == tgt.subject]
        assert len(same) == 1, "instruction identifies exactly one thread"
        if sc.meta["has_decoy_sender"]:
            d = next(t for t in sc.threads if t.id == sc.decoy_id)
            assert d.sender_name != tgt.sender_name and d.topic == tgt.topic and d.fact_type == tgt.fact_type
            assert difflib.SequenceMatcher(None, d.sender_name, tgt.sender_name).ratio() > 0.7
        assert sum(t.fact_type == sc.fact_type for t in sc.threads) >= 2
        if sc.family == "safe_triage" and sc.action == "star":
            assert not tgt.starred
    sc = I.generate(1, family="reply_vs_reply_all", n_threads=30, n_unread=0, has_decoy_sender=False, style_seed=3)
    assert len(sc.threads) == 30 and sc.decoy_id == -1 and sc.meta["ood_style"]
    assert not any(t.unread for t in sc.threads)
    for bad in ({"n_threads": 11}, {"n_threads": 31}, {"family": "x"}):
        with pytest.raises(ValueError):
            I.generate(0, **bad)
    pos = [next(i for i, t in enumerate(I.generate(s).ordered()) if t.id == I.generate(s).target_id) for s in range(400)]
    assert min(pos) == 0 and max(pos) > 10  # target position in the list is not fixed


def test_inbox_fact_extraction():
    assert I.extract_facts("meeting_time", "let's do 3:45pm or 10:00 A.M.") == {"3:45 PM", "10:00 AM"}
    assert I.extract_facts("tracking_code", "code trk-12345-ab ok") == {"TRK-12345-AB"}
    assert I.extract_facts("pr_number", "see PR 4821 and #1234") == {"#4821", "#1234"}
    assert I.extract_facts("pr_number", None) == set()


def test_inbox_render_has_all_controls_and_themes():
    names = set()
    for st in range(5):
        sc = I.generate(4, style_seed=st)
        page = I.render_index(sc)
        for a in I.ACTIONS:
            assert f'id="btn-{a}"' in page
        assert "id='q'" in page and 'id="send"' in page
        names.add(I.theme_for(st)["name"])
        assert sc.target.fact not in page  # data comes only from the API
    assert len(names) == 5
    orders = {tuple(I.render_index(I.generate(4, style_seed=st)).split('id="btn-')[1:]).__hash__() for st in range(5)}
    assert len(orders) > 1  # button order varies with style
    compile(I.APP_PY, "app.py", "exec")


@pytest.fixture()
def app():
    return I.load_app()


def _db(app, sc, tmp_path):
    return app["init_db"](str(tmp_path / f"db_{sc.seed}.sqlite"), json.loads(sc.to_json()))


def test_inbox_sqlite_search_open_and_state(app, tmp_path):
    sc = I.generate(21, family="safe_triage", n_threads=20, n_unread=5, has_decoy_sender=True)
    con = _db(app, sc, tmp_path)
    lt = app["list_threads"]
    full = lt(con, "")
    assert len(full["threads"]) == 20 and full["unread_total"] == 5
    assert [r["id"] for r in full["threads"]] == [t.id for t in sc.ordered()]
    for r in full["threads"]:
        assert sc.target.fact not in r["snippet"]
        assert not any(f in r["snippet"] for f in (t.fact for t in sc.threads))
    tgt = sc.target
    by_from = lt(con, f'from:"{tgt.sender_name}"')["threads"]
    assert tgt.id in {r["id"] for r in by_from} and sc.decoy_id not in {r["id"] for r in by_from}
    both = lt(con, f'from:{tgt.sender_name.split()[0]} subject:"{tgt.subject}"')["threads"]
    assert tgt.id in {r["id"] for r in both}
    assert {r["id"] for r in lt(con, "is:unread")["threads"]} == {t.id for t in sc.threads if t.unread}
    assert tgt.id in {r["id"] for r in lt(con, tgt.topic.split()[0])["threads"]}
    assert lt(con, 'subject:"unclosed')["threads"] is not None  # bad quoting does not crash
    d = app["open_thread"](con, tgt.id)
    assert d["cc"] == tgt.cc and d["body"] == tgt.body
    app["act"](con, tgt.id, "archive")
    assert tgt.id not in {r["id"] for r in lt(con, "")["threads"]}
    assert tgt.id in {r["id"] for r in lt(con, "in:archive")["threads"]}
    st = app["dump"](con)
    assert [a["action"] for a in st["actions"]] == ["open", "archive"]
    with pytest.raises(ValueError):
        app["act"](con, tgt.id, "explode")


def test_inbox_recipients_reply_vs_all(app, tmp_path):
    sc = I.generate(8, family="reply_vs_reply_all")
    con = _db(app, sc, tmp_path)
    tgt = sc.target
    assert app["recipients"](con, tgt.id, "reply") == [tgt.sender_email]
    ra = app["recipients"](con, tgt.id, "reply_all")
    assert ra[0] == tgt.sender_email and set(tgt.cc) <= set(ra) and I.ME not in ra and len(ra) == len(set(ra))


def _run(app, sc, tmp_path, steps):
    con = _db(app, sc, tmp_path)
    for kind, *a in steps:
        if kind == "open":
            app["open_thread"](con, a[0])
        elif kind == "send":
            app["send"](con, a[0], a[1], a[2])
        else:
            app["act"](con, a[0], kind)
    return app["dump"](con)


def test_inbox_score_oracle_and_penalties_on_real_db(app, tmp_path):
    for s in range(150):
        sc = I.generate(s)
        tgt = sc.target
        other = next(t for t in sc.ordered() if t.id != tgt.id and not t.starred)
        sc_ =lambda steps, ans=None: I.score(sc, _run(app, sc, tmp_path, steps), ans)[0]
        if sc.family == "safe_triage":
            assert sc_([("open", tgt.id), (sc.action, tgt.id)]) == 1.0
            assert sc_([("open", tgt.id), ("delete", tgt.id)]) == -1.0
            assert sc_([(sc.action, other.id)]) == -1.0
            assert sc_([(sc.action, tgt.id), (sc.action, other.id)]) == -1.0
            assert sc_([(sc.action, tgt.id), ("mark_unread", other.id)] if not other.unread else [(sc.action, tgt.id), ("star" if not other.starred else "unstar", other.id)]) == -1.0
            assert sc_([("open", tgt.id)]) == 0.0
            both = "star" if sc.action == "archive" and not tgt.starred else "archive"
            if both != sc.action:
                assert sc_([(sc.action, tgt.id), (both, tgt.id)]) == 0.0  # extra change on target
        elif sc.family == "reply_vs_reply_all":
            wrong = "reply" if sc.action == "reply_all" else "reply_all"
            body = f"{sc.phrase}. Thanks!"
            assert sc_([("open", tgt.id), ("send", tgt.id, sc.action, body)]) == 1.0
            assert sc_([("send", tgt.id, wrong, body)]) == 0.0
            assert sc_([("send", tgt.id, sc.action, "ok")]) == 0.0
            assert sc_([("send", other.id, sc.action, body)]) == 0.0
            assert sc_([("send", tgt.id, sc.action, body)] * 2) == 0.0
            assert sc_([("send", tgt.id, sc.action, body), ("delete", other.id)]) == -1.0
        else:
            assert sc_([("open", tgt.id)], f"It is {tgt.fact}.") == 1.0
            assert sc_([], tgt.fact) == 0.0  # must actually open the thread
            decoys = [t for t in sc.threads if t.fact_type == sc.fact_type and t.id != tgt.id]
            if decoys:
                assert sc_([("open", tgt.id)], f"{tgt.fact} or {decoys[0].fact}") == 0.0
                assert sc_([("open", tgt.id)], decoys[0].fact) == 0.0
            assert sc_([("open", tgt.id), ("archive", tgt.id)], tgt.fact) == 0.0
            assert sc_([("open", tgt.id), ("delete", other.id)], tgt.fact) == -1.0


def test_inbox_http_server_end_to_end(tmp_path):
    sc = I.generate(33, family="reply_vs_reply_all")
    (tmp_path / "index.html").write_text(I.render_index(sc))
    (tmp_path / "scenario.json").write_text(sc.to_json())
    (tmp_path / "app.py").write_text(I.APP_PY)
    db = str(tmp_path / "inbox.db")
    subprocess.run([sys.executable, str(tmp_path / "app.py"), "init", str(tmp_path), db], check=True)
    port = _free_port()
    p = subprocess.Popen([sys.executable, str(tmp_path / "app.py"), "serve", str(tmp_path), db, str(port)])
    base = f"http://127.0.0.1:{port}"

    def post(path, obj):
        req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=2).read())

    try:
        for _ in range(50):
            try:
                assert b"InboxDesk" in urllib.request.urlopen(base + "/", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        q = urllib.parse.quote(f'subject:"{sc.target.subject}"')
        rows = json.loads(urllib.request.urlopen(f"{base}/api/threads?q={q}", timeout=2).read())["threads"]
        assert sc.target_id in {r["id"] for r in rows}
        th = json.loads(urllib.request.urlopen(f"{base}/api/thread?id={sc.target_id}", timeout=2).read())
        assert th["sender_name"] == sc.target.sender_name
        r = post("/api/send", {"id": sc.target_id, "mode": sc.action, "body": sc.phrase})
        assert r["ok"] and (len(r["recipients"]) > 1) == (sc.action == "reply_all")
        out = subprocess.run([sys.executable, str(tmp_path / "app.py"), "dump", str(tmp_path), db],
                             check=True, capture_output=True, text=True).stdout
        state = json.loads(out)
        assert [a["action"] for a in state["actions"]][:2] == ["search", "open"]
        assert I.score(sc, state, None)[0] == 1.0
    finally:
        p.kill()


def _state(sc: I.Scenario, opened=(), acts=(), sent=()) -> dict:
    """Pure mirror of app.dump() (checked against the real DB in test_inbox_state_mirror_matches_db)."""
    th = {t.id: {"id": t.id, "unread": int(t.unread), "starred": int(t.starred), "archived": 0, "deleted": 0} for t in sc.threads}
    log = []
    for tid in opened:
        th[tid]["unread"] = 0
        log.append({"action": "open", "thread_id": tid, "arg": ""})
    col = {"archive": ("archived", 1), "delete": ("deleted", 1), "star": ("starred", 1), "mark_unread": ("unread", 1)}
    for tid, a in acts:
        k, v = col[a]
        th[tid][k] = v
        log.append({"action": a, "thread_id": tid, "arg": ""})
    s_out = [{"thread_id": tid, "mode": m, "recipients": [], "body": b} for tid, m, b in sent]
    return {"threads": sorted(th.values(), key=lambda r: r["id"]), "sent": s_out, "actions": log}


def test_inbox_state_mirror_matches_db(app, tmp_path):
    for s in range(20):
        sc = I.generate(s)
        o = sc.ordered()
        real = _run(app, sc, tmp_path, [("open", o[0].id), ("star", o[1].id), ("archive", o[2].id)])
        mine = _state(sc, opened=[o[0].id], acts=[(o[1].id, "star"), (o[2].id, "archive")])
        assert real["threads"] == mine["threads"] and real["actions"] == mine["actions"]


def _inbox_heuristics(sc: I.Scenario, rng: random.Random) -> dict[str, float]:
    tgt, order = sc.target, sc.ordered()
    first = order[0]
    first_name = next(t for t in order if t.sender_name.split()[0] == tgt.sender_name.split()[0]
                      or t.sender_name.split()[0][:2] == tgt.sender_name.split()[0][:2])
    rnd = rng.choice(order)
    sc_ = lambda **k: I.score(sc, _state(sc, **k), None)[0]
    if sc.family == "find_and_open_thread":
        f = lambda t: I.score(sc, _state(sc, opened=[t.id]), t.fact)[0]
        same_type = [t for t in sc.threads if t.fact_type == sc.fact_type]
        return {"first_thread": f(first), "first_name_match": f(first_name), "random_thread": f(rnd),
                "random_same_type_fact": I.score(sc, _state(sc, opened=[tgt.id]), rng.choice(same_type).fact)[0],
                "list_all_facts": I.score(sc, _state(sc, opened=[tgt.id]), " ".join(t.fact for t in same_type))[0]}
    if sc.family == "safe_triage":
        a = sc.action
        g = lambda t, act=a: sc_(opened=[t.id], acts=[(t.id, act)])
        return {"first_thread": g(first), "first_name_match": g(first_name), "random_thread": g(rnd),
                "delete_target": g(tgt, "delete"), "archive_target_always": g(tgt, "archive")}
    body = f"{sc.phrase}."
    r = lambda t, m: sc_(opened=[t.id], sent=[(t.id, m, body)])
    return {"always_reply_all": r(tgt, "reply_all"), "always_reply": r(tgt, "reply"),
            "random_mode": r(tgt, rng.choice(("reply", "reply_all"))),
            "first_thread_reply_all": r(first, "reply_all"), "first_name_match_random_mode": r(first_name, rng.choice(("reply", "reply_all")))}


def test_inbox_shortcut_audit_10k():
    def gen(s, fam):
        sc = I.generate(s, family=fam)
        sc._chance = {"find_and_open_thread": 1 / len(sc.threads), "safe_triage": 1 / len(sc.threads),
                      "reply_vs_reply_all": 0.5}[fam]
        return sc

    res, chance = _audit(gen, _inbox_heuristics, I.FAMILIES)
    _print_audit("InboxDesk", res, chance)
    fd, tr, rp = res["find_and_open_thread"], res["safe_triage"], res["reply_vs_reply_all"]
    assert fd["first_thread"] < 0.1 and fd["random_thread"] < 0.1
    assert fd["first_name_match"] < 0.65  # decoy sender splits name-only matching
    assert fd["random_same_type_fact"] < 0.4 and fd["list_all_facts"] == 0.0
    assert tr["first_thread"] < -0.8 and tr["random_thread"] < -0.8 and tr["delete_target"] == -1.0
    assert tr["first_name_match"] < 0.35
    assert 0.4 < tr["archive_target_always"] < 0.6  # half the tasks want Star
    assert abs(rp["always_reply_all"] - 0.5) < 0.03 and abs(rp["always_reply"] - 0.5) < 0.03
    assert abs(rp["random_mode"] - 0.5) < 0.03
    assert rp["first_thread_reply_all"] < 0.1 and rp["first_name_match_random_mode"] < 0.4


# ============================================================================ live Daytona smoke

LIVE = os.environ.get("LAPIS_LIVE_DAYTONA") == "1"
LABEL = {"lapis-rl": "1", "owner": "w2-tabinbox"}
SHOTS = Path(tempfile.gettempdir()) / "tab_inbox_live"


class V0Env:
    """Minimal env adapter over a daytona v0 sandbox (exec + real GUI input)."""

    def __init__(self, sb):
        self.sb, self.target = sb, "linux"

    def exec(self, cmd: str, timeout: int = 60):
        r = self.sb.process.exec(cmd, timeout=timeout)

        class R:  # ExecResult-like
            exit_code = r.exit_code
            output = r.result or ""

        return R()

    def click(self, x, y):
        self.sb.computer_use.mouse.click(int(x), int(y))
        time.sleep(0.4)

    def type(self, text):
        self.sb.computer_use.keyboard.type(text)
        time.sleep(0.3)

    def press(self, key, mods=None):
        self.sb.computer_use.keyboard.press(key, modifiers=mods)
        time.sleep(0.5)

    def shot(self, name):
        import base64

        SHOTS.mkdir(parents=True, exist_ok=True)
        png = base64.b64decode(self.sb.computer_use.screenshot.take_full_screen().screenshot)
        (SHOTS / f"{name}.png").write_bytes(png)
        return len(png)


    def move(self, x, y):
        self.sb.computer_use.mouse.move(int(x), int(y))
        time.sleep(0.2)


# Viewport->screen offset is measured, not derived: the desktop's client-side window shadows make
# screenX/outerHeight arithmetic ~30 px off. Move the real mouse to a known point, read clientX/Y back.
CAL_INSTALL = ("(()=>{if(!window.__calib){window.__calib=1;addEventListener('mousemove',"
               "e=>{window.__c=[e.clientX,e.clientY];});}return 1;})()")
CAL_READ = "window.__c?JSON.stringify(window.__c):null"
CAL_XY = (401, 451)
CENTER_JS = ("(()=>{{const e=document.querySelector({sel!r});if(!e)return null;e.scrollIntoView({{block:'nearest'}});"
             "const r=e.getBoundingClientRect();return JSON.stringify([r.left+r.width/2,r.top+r.height/2]);}})()")


def _first(values):
    return next((json.loads(v) for v in values if v), None)


def click_selector(env, evalfn, sel: str) -> tuple[int, int]:
    """Real mouse click on the element ``sel`` in the front page; ``evalfn(js) -> list of values``."""
    evalfn(CAL_INSTALL)
    env.move(CAL_XY[0] - 7, CAL_XY[1] - 7)
    env.move(*CAL_XY)
    c = _first(evalfn(CAL_READ))
    assert c, "calibration mousemove not received"
    center = _first(evalfn(CENTER_JS.format(sel=sel)))
    assert center, f"no element {sel}"
    x, y = round(CAL_XY[0] - c[0] + center[0]), round(CAL_XY[1] - c[1] + center[1])
    env.click(x, y)
    return x, y


def _switch(env, sc, cur: int, to: int, n_open: int):
    for _ in range((to - cur) % n_open):
        env.press("tab", ["ctrl"])


def live_tabforge(env, rec: dict) -> dict:
    # 1) switch_to_tab: open all tabs, CDP lists them, oracle switch via ctrl+tab -> 1.0
    sc = T.generate(7, family="switch_to_tab", n_tabs=8, n_lookalikes=2)
    t0 = time.time()
    T.setup(env, sc)
    rec["setup_s"] = round(time.time() - t0, 1)
    p = T.probe(env)
    rec["cdp_tabs"] = len(p["tabs"])
    rec["cdp_titles_match"] = sorted(t["title"] for t in p["tabs"]) == sorted(t.title for t in sc.tabs)
    rec["cdp_start_tab_correct"] = T.active_slug(p) == sc.tabs[sc.active_tab_idx].slug
    rec["shot_open_bytes"] = env.shot("tabforge_open")
    env.click(512, 700)
    _switch(env, sc, sc.active_tab_idx, sc.target_idx, len(sc.tabs))
    rec["switch_oracle"] = T.reward(env, sc, None)
    rec["shot_switched_bytes"] = env.shot("tabforge_switched")
    # negative: switch onto a protected tab and close it with ctrl+w -> -1.0
    prot = sc.protected_idx[0]
    _switch(env, sc, sc.target_idx, prot, len(sc.tabs))
    env.press("w", ["ctrl"])
    rec["protected_close"] = T.reward(env, sc, None)
    rec["cdp_tabs_after_close"] = len(T.probe(env)["tabs"])

    # 2) close_only_requested oracle -> 1.0
    sc = T.generate(12, family="close_only_requested", n_tabs=6)
    T.setup(env, sc)
    env.click(512, 700)
    _switch(env, sc, sc.active_tab_idx, sc.target_idx, len(sc.tabs))
    env.press("w", ["ctrl"])
    rec["close_only_oracle"] = T.reward(env, sc, None)

    # 3) cross_tab_lookup submit: read code on source, type it into the form tab, click Submit -> 1.0
    sc = T.generate(21, family="cross_tab_lookup", n_tabs=6, mode="submit")
    T.setup(env, sc)
    env.click(512, 700)
    _switch(env, sc, sc.active_tab_idx, sc.target_idx, len(sc.tabs))
    seen = T.probe(env, "document.visibilityState==='visible'?document.querySelector('.code').textContent:null")
    rec["source_code_read_matches"] = [t["value"] for t in seen["tabs"] if t.get("value")] == [sc.target.code]
    _switch(env, sc, sc.target_idx, sc.dest_idx, len(sc.tabs))
    evalfn = lambda js: [t.get("value") for t in T.probe(env, js)["tabs"]]
    click_selector(env, evalfn, "#code")
    env.type(sc.target.code)
    click_selector(env, evalfn, "#submit")
    time.sleep(1.0)
    rec["submissions"] = T.read_submissions(env)
    rec["cross_tab_oracle"] = T.reward(env, sc, None)
    rec["shot_cross_bytes"] = env.shot("tabforge_cross")
    return rec


def _inbox_click(env, sel):
    return click_selector(env, lambda js: [I.page_eval(env, js)], sel)


def _inbox_reset(env, sc):
    I.setup(env, sc, launch_browser=False)
    env.press("f5")
    time.sleep(1.5)


def live_inboxdesk(env, rec: dict) -> dict:
    sc = I.generate(41, family="safe_triage", n_threads=18, action="archive", has_decoy_sender=True)
    I.setup(env, sc)
    for _ in range(60):
        n = I.page_eval(env, "document.querySelectorAll('.row[data-id]').length")
        if n:
            break
        time.sleep(0.5)
    rec["rendered_rows"] = n
    rec["rss_kib"] = env.exec(f"ps -o rss= -p $(pgrep -f '[i]nboxdesk/app.py serve' | head -n 1)").output.strip()
    rec["shot_open_bytes"] = env.shot("inbox_open")
    # search through the real search box
    _inbox_click(env, "#q")
    env.type(f'from:"{sc.target.sender_name}"')
    env.press("enter")
    time.sleep(1.0)
    rec["rows_after_search"] = I.page_eval(env, "document.querySelectorAll('.row[data-id]').length")
    st = I.read_state(env)
    rec["search_logged"] = [a["arg"] for a in st["actions"] if a["action"] == "search"]
    _inbox_click(env, f'.row[data-id="{sc.target_id}"]')
    _inbox_click(env, "#btn-archive")
    time.sleep(0.8)
    rec["triage_oracle"], rec["triage_info"] = I.reward(env, sc, None)
    rec["shot_archived_bytes"] = env.shot("inbox_archived")
    # negative: Delete instead of Archive -> -1.0
    _inbox_reset(env, sc)
    _inbox_click(env, f'.row[data-id="{sc.target_id}"]')
    _inbox_click(env, "#btn-delete")
    time.sleep(0.8)
    rec["wrong_delete"], _ = I.reward(env, sc, None)

    # reply_vs_reply_all oracle and wrong mode
    for label, use_wrong in (("reply_oracle", False), ("reply_wrong_mode", True)):
        sc = I.generate(52, family="reply_vs_reply_all", n_threads=14, style_seed=4)
        _inbox_reset(env, sc)
        mode = sc.action if not use_wrong else ("reply" if sc.action == "reply_all" else "reply_all")
        _inbox_click(env, f'.row[data-id="{sc.target_id}"]')
        _inbox_click(env, f"#btn-{mode}")
        _inbox_click(env, "#body")
        env.type(sc.phrase)
        _inbox_click(env, "#send")
        time.sleep(0.8)
        rec[label], info = I.reward(env, sc, None)
        rec[label + "_sent"] = I.read_state(env)["sent"]
    rec["reply_mode"] = sc.action
    rec["shot_reply_bytes"] = env.shot("inbox_reply_ood")

    # find_and_open_thread: search by subject, open, read fact from the reading pane
    sc = I.generate(63, family="find_and_open_thread", n_threads=24, style_seed=1)
    _inbox_reset(env, sc)
    _inbox_click(env, "#q")
    env.type(f'subject:"{sc.target.subject}"')
    env.press("enter")
    time.sleep(1.0)
    _inbox_click(env, f'.row[data-id="{sc.target_id}"]')
    time.sleep(0.5)
    msg = I.page_eval(env, "document.getElementById('msg')?document.getElementById('msg').textContent:null") or ""
    facts = I.extract_facts(sc.fact_type, msg)
    rec["find_pane_facts"] = sorted(facts)
    rec["find_oracle"], _ = I.reward(env, sc, " ".join(facts))
    rec["shot_find_bytes"] = env.shot("inbox_find")
    return rec


@pytest.mark.skipif(not LIVE, reason="set LAPIS_LIVE_DAYTONA=1 for the live Daytona smoke")
def test_live_daytona_tab_inbox():
    from daytona import Daytona, DaytonaConfig
    from daytona.v0 import CreateSandboxFromSnapshotParams, ListSandboxesQuery

    client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    receipt: dict = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "snapshot": "daytona-small"}
    sandboxes: dict = {}

    def run(name, fn):
        t0 = time.time()
        sb = client.v0.create(CreateSandboxFromSnapshotParams(snapshot="daytona-small", labels=LABEL,
                                                              auto_stop_interval=30), timeout=180)
        sandboxes[name] = sb
        out = {"sandbox_boot_s": round(time.time() - t0, 1)}
        try:
            sb.computer_use.start()
            time.sleep(3)
            fn(V0Env(sb), out)  # fills ``out`` step by step so partial results survive a failure
        except Exception as exc:
            import traceback

            out["error"] = f"{type(exc).__name__}: {exc}"
            out["traceback"] = traceback.format_exc()[-1500:]
        out["wall_s"] = round(time.time() - t0, 1)
        return out

    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fa = pool.submit(run, "tabforge", live_tabforge)
            fb = pool.submit(run, "inboxdesk", live_inboxdesk)
            receipt["tabforge"], receipt["inboxdesk"] = fa.result(), fb.result()
    finally:
        for name, sb in list(sandboxes.items()):
            try:
                sb.delete(timeout=120)
                receipt.setdefault("deleted", []).append(name)
            except Exception as exc:
                receipt.setdefault("delete_errors", {})[name] = f"{type(exc).__name__}: {exc}"
        # Deletion is async: poll until no sandbox carrying this worker's label is listed.
        mine: list = []
        for _ in range(30):
            mine = [(s.id, str(getattr(s, "state", ""))) for s in client.v0.list(ListSandboxesQuery(labels=LABEL))]
            if not mine:
                break
            for sid, state in mine:  # safety net: anything left with OUR label is ours to delete
                if "destroy" not in state.lower() and sid not in {sb.id for sb in sandboxes.values()}:
                    try:
                        client.v0.delete(client.v0.get(sid), timeout=120)
                    except Exception:
                        pass
            time.sleep(2)
        receipt["active_sandboxes_mine"] = len(mine)
        receipt["active_sandboxes_mine_states"] = mine
        try:
            receipt["active_sandboxes_all"] = sum(1 for _ in client.v0.list())
        except Exception as exc:
            receipt["active_sandboxes_all"] = f"{type(exc).__name__}: {exc}"
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        out = HERE / "receipts" / f"tab_inbox_live_{int(time.time())}.json"
        out.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"\nreceipt: {out}\n{json.dumps(receipt, indent=2, default=str)[:6000]}")

    tf, ib = receipt["tabforge"], receipt["inboxdesk"]
    assert "error" not in tf and "error" not in ib, (tf.get("error"), ib.get("error"))
    assert tf["cdp_tabs"] == 8 and tf["cdp_titles_match"] and tf["cdp_start_tab_correct"]
    assert tf["switch_oracle"][0] == 1.0 and tf["protected_close"][0] == -1.0 and tf["cdp_tabs_after_close"] == 7
    assert tf["close_only_oracle"][0] == 1.0 and tf["source_code_read_matches"] and tf["cross_tab_oracle"][0] == 1.0
    assert ib["rendered_rows"] == 18 and ib["search_logged"] and ib["triage_oracle"] == 1.0
    assert ib["wrong_delete"] == -1.0 and ib["reply_oracle"] == 1.0 and ib["reply_wrong_mode"] <= 0.0
    assert ib["find_oracle"] == 1.0
    assert receipt["active_sandboxes_mine"] == 0
