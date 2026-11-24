"""Offline tests, 10k-seed shortcut audits and a live Daytona smoke for CodeReview and CalBoard.

Live smoke (2x daytona-small through rl/pool.py, labeled owner=w4-codecal, always deleted):
  LAPIS_LIVE_DAYTONA=1 uv run pytest tests/test_code_cal.py -k live -s
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from statistics import mean

import pytest

from synth import cal_board as B
from synth import code_review as C

N_AUDIT = 10_000
HERE = Path(__file__).resolve().parent.parent
JS_EXE = shutil.which("node") or shutil.which("bun")


def _run_db(M, sc, steps):
    con = M.fresh_db(M.seed_sql(sc))
    ex = M.sqlite_exec(con)
    for name, p in steps:
        M.apply_op(ex, M.OPS, name, **p)
    return M.dump(ex)


def _page_js(html: str) -> str:
    return html[html.index("<script>") + len("<script>"):html.index("</script>")]


def _node(script: str, tmp_path) -> object:
    f = tmp_path / f"h{random.randrange(10**9)}.js"
    f.write_text(script)
    out = subprocess.run([JS_EXE, str(f)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


STUB = ("const window = {}; const document = {addEventListener(){}, getElementById(){return null;}};"
        "function fetch(){ return Promise.resolve({json(){return {rows:[]};}}); }")


# ============================================================================ CodeReview: generation


def test_cr_deterministic_and_seed_sensitive():
    for s in range(40):
        assert C.generate(s).to_json() == C.generate(s).to_json()
    assert len({C.generate(s).to_json() for s in range(200)}) == 200


def test_cr_invariants():
    for s in range(1500):
        sc = C.generate(s)
        nums = [p.number for p in sc.prs] + [i.number for i in sc.issues]
        assert len(nums) == len(set(nums))
        codes = [c.code for p in sc.prs for c in p.checks if c.code]
        assert len(codes) == len(set(codes))
        for p in sc.prs:
            assert (p.status == "ci_failed") == any(c.status == "fail" for c in p.checks)
            assert sum(c.status == "fail" for c in p.checks) <= 1
            added = [l[3] for f in p.files for l in f.lines if l[0] == "+"]
            assert any("FIXME(bug)" in t for t in added) == p.bug_marker
        if sc.family == "filter_and_triage_issue":
            lab = sc.meta["issue_label"]
            tgt = sc.issue(sc.target)
            match = [i for i in sc.issues if lab in i.labels and i.author == tgt.author]
            assert match == [tgt] and tgt.state == "open" and tgt.assignee is None
            assert sum(lab in i.labels for i in sc.issues) >= 4
            assert sum(i.author == tgt.author for i in sc.issues) >= 3
            if sc.mode == "label":
                assert sc.value not in tgt.labels
        else:
            tgt = sc.pr(sc.target)
            same = [p for p in sc.prs if p.title == tgt.title]
            assert len(same) == 1 + sc.meta["n_lookalikes"]
            assert len([p for p in same if p.author == tgt.author]) == 1, "title + author is unique"
            if sc.family == "find_failing_check":
                assert tgt.status == "ci_failed" and tgt.failing_code
                assert tgt.failing_code not in json.dumps([c for c in tgt.comments])
                assert C.extract_codes(" ".join(b for _, b in tgt.comments)), "stale code decoy present"
            else:
                assert tgt.status in ("open", "ci_failed")
                assert tgt.approvable == (sc.mode == "approve")
                assert all(p.approvable != tgt.approvable for p in same if p is not tgt), "look-alikes flip the verdict"
    sc = C.generate(5, family="safe_review_action", split="ood_eval", style_seed=1, n_prs=24, n_issues=20,
                    n_lookalikes=4, mode="request_changes")
    assert len(sc.prs) == 24 and sc.meta["ood_style"] and sc.meta["theme"] == "linear" and sc.mode == "request_changes"
    for bad in ({"family": "x"}, {"split": "test"}, {"n_prs": 9}, {"n_issues": 21}, {"n_lookalikes": 5},
                {"family": "safe_review_action", "mode": "merge"}):
        with pytest.raises(ValueError):
            C.generate(0, **bad)
    pos = [next(k for k, p in enumerate(C.generate(s).ordered_prs()) if p.number == C.generate(s).target)
           for s in range(300) if C.generate(s).family != "filter_and_triage_issue"]
    assert min(pos) == 0 and max(pos) > 8


def test_cr_styles_split_and_render():
    train = {C.style_for("train", s)["name"] for s in range(9)}
    ood = {C.style_for("ood_eval", s)["name"] for s in range(9)}
    assert train == {"github_light", "github_dark", "github_dimmed"} and ood == {"gitlab", "linear"}
    for s in range(6):
        for split in C.SPLITS:
            sc = C.generate(s, split=split, style_seed=s)
            spec = sc.app_spec()
            h = spec["html"]
            assert set(spec) >= {"name", "html", "sql"} and "<head>" in h
            for p in sc.prs:  # all data comes from SQLite, never baked into the page
                assert p.title not in h and (not p.failing_code or p.failing_code not in h)
            con = C.fresh_db(spec["sql"])
            assert con.execute("SELECT COUNT(*) FROM prs").fetchone()[0] == len(sc.prs)
    a = C.render_html(C.generate(1, style_seed=0))
    b = C.render_html(C.generate(1, style_seed=1))
    assert a != b


def test_cr_search_sql_filters():
    for s in range(200):
        sc = C.generate(s)
        con = C.fresh_db(C.seed_sql(sc))
        q = lambda kind, text: [r[0] for r in con.execute(C.Q[kind], C.parse_query(text)).fetchall()]
        assert q("pulls", "") == [p.number for p in sc.ordered_prs()]
        assert q("issues", "") == [i.number for i in sc.ordered_issues()]
        p0 = sc.prs[0]
        assert set(q("pulls", f"author:@{p0.author}")) == {p.number for p in sc.prs if p.author == p0.author}
        assert set(q("pulls", "status:failed")) == {p.number for p in sc.prs if p.status == "ci_failed"}
        assert set(q("pulls", "status:draft")) == {p.number for p in sc.prs if p.status == "draft"}
        assert set(q("pulls", f'"{p0.title}"')) == {p.number for p in sc.prs if p.title == p0.title}
        assert q("pulls", f"#{p0.number}") == [p0.number]
        assert set(q("pulls", f"author:{p0.author} {p0.title}")) == {p0.number}
        if p0.labels:
            assert set(q("pulls", f"label:{p0.labels[0]}")) == {p.number for p in sc.prs if p0.labels[0] in p.labels}
        i0 = sc.issues[0]
        lab = i0.labels[0]
        assert set(q("issues", f"label:{lab} author:@{i0.author}")) == {
            i.number for i in sc.issues if lab in i.labels and i.author == i0.author}
        assert set(q("issues", "is:closed")) == {i.number for i in sc.issues if i.state == "closed"}
        assert q("pulls", "status:bogus") == []


@pytest.mark.skipif(JS_EXE is None, reason="node or bun needed")
def test_cr_page_parse_query_matches_python(tmp_path):
    cases = ["", "author:@mara-lin", 'AUTHOR:Mara-Lin status:Failed label:bug "Fix upload retry backoff"',
             "status:ci-failed", "status:CI_FAILED #1234", "is:open label:perf assignee:@elliot", "status:wat x",
             "author: foo", "label:Flaky-Test  extra   words", "“quoted text” author:@j-park"]
    js = _page_js(C.render_html(C.generate(2)))
    got = _node(STUB + js + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(parseQ)));", tmp_path)
    assert got == [C.parse_query(c) for c in cases]


# ============================================================================ CodeReview: reward on the real db


def test_cr_oracle_and_sim_match_real_db():
    for s in range(600):
        sc = C.generate(s)
        steps, ans = C.oracle_steps(sc)
        real = _run_db(C, sc, steps)
        assert real == C.simulate(sc, steps)
        assert C.score(sc, real, ans)[0] == 1.0, (s, sc.family, C.score(sc, real, ans))
        assert C.score(sc, _run_db(C, sc, []), ans)[0] in (0.0,), "doing nothing never scores"


def test_cr_penalties_on_real_db():
    seen = set()
    for s in range(400):
        sc = C.generate(s)
        seen.add(sc.family)
        t = sc.target
        sc_ = lambda steps, ans=None: C.score(sc, _run_db(C, sc, steps), ans)[0]
        broken = next(p for p in sc.prs if p.broken and p.status != "merged")
        assert sc_([("merge", {"pr": broken.number, "ref": str(broken.number)})]) == -1.0
        failed = [p for p in sc.prs if p.status == "ci_failed" or p.bug_marker]
        if failed:
            assert sc_([("approve", {"pr": failed[0].number, "ref": str(failed[0].number), "body": ""})]) == -1.0
        if sc.family == "find_failing_check":
            pr = sc.pr(t)
            look = [p for p in sc.prs if p.title == pr.title and p.number != t][0]
            tab = ("log", {"action": "tab", "ref": str(t), "arg": "checks"})
            stale = sorted(C.extract_codes(" ".join(b for _, b in pr.comments)))[0]
            if sc.mode == "answer":
                assert sc_([tab], pr.failing_code.lower()) == 1.0
                assert sc_([], pr.failing_code) == 0.0  # never opened Checks
                assert sc_([tab], stale) == 0.0
                assert sc_([tab], f"{pr.failing_code} {look.failing_code}") == 0.0
            else:
                cm = lambda n, b: ("comment", {"pr": n, "ref": str(n), "path": None, "line": None, "body": b})
                assert sc_([tab, cm(t, f"fails with {pr.failing_code}")]) == 1.0
                assert sc_([tab, ("comment", {"pr": t, "ref": str(t), "path": "src/x.py", "line": 12,
                                              "body": pr.failing_code})]) == 1.0  # inline comment counts
                assert sc_([tab, cm(look.number, pr.failing_code)]) == 0.0
                assert sc_([tab, cm(t, look.failing_code)]) == 0.0
                assert sc_([tab, cm(t, pr.failing_code), cm(t, stale)]) == 0.0
            assert sc_([tab, ("request_changes", {"pr": t, "ref": str(t), "body": pr.failing_code})],
                       pr.failing_code) == 0.0
        elif sc.family == "safe_review_action":
            other = "approve" if sc.mode == "request_changes" else "request_changes"
            rv = lambda v, n=t: (v, {"pr": n, "ref": str(n), "body": ""})
            assert sc_([rv(other)]) == (-1.0 if other == "approve" else 0.0)
            assert sc_([rv(sc.mode), rv(sc.mode)]) == 1.0
            if sc.mode == "approve":
                assert sc_([rv("approve"), rv("request_changes")]) == 0.0
                assert sc_([rv("approve"), ("merge", {"pr": t, "ref": str(t)})]) == 0.0  # told not to merge
            else:
                assert sc_([rv("request_changes"), ("merge", {"pr": t, "ref": str(t)})]) == -1.0
            look = next(p for p in sc.prs if p.title == sc.pr(t).title and p.number != t)
            assert sc_([rv(sc.mode), rv("request_changes", look.number)]) == 0.0
        else:
            iss = sc.issue(t)
            other = next(i for i in sc.issues if i.number != t and i.state == "open")
            who = next(x for x in C.TEAMMATES if x != other.assignee)
            assert sc_([("assign", {"num": other.number, "who": who, "ref": str(other.number)})]) == -1.0
            assert sc_([("close_issue", {"num": t, "ref": str(t)})]) == 0.0
            steps, _ = C.oracle_steps(sc)
            assert sc_(steps + [("close_issue", {"num": other.number, "ref": str(other.number)})]) == -1.0
            if sc.mode == "assign":
                assert sc_([("assign", {"num": t, "who": next(x for x in C.TEAMMATES if x != sc.value), "ref": str(t)})]) == 0.0
                assert sc_([("set_labels", {"num": t, "labels": json.dumps(iss.labels + ["p1"]), "ref": str(t)})]) == 0.0
            else:
                assert sc_([("set_labels", {"num": t, "labels": json.dumps(iss.labels + [sc.value, "triaged" if sc.value != "triaged" else "p1"]), "ref": str(t)})]) == 0.0
    assert seen == set(C.FAMILIES)


# ============================================================================ CalBoard: generation


def test_cb_deterministic_and_seed_sensitive():
    for s in range(40):
        assert B.generate(s).to_json() == B.generate(s).to_json()
    assert len({B.generate(s).to_json() for s in range(200)}) == 200


def test_cb_invariants():
    for s in range(1500):
        sc = B.generate(s)
        assert len({m.id for m in sc.meetings}) == len(sc.meetings)
        assert len({c.code for c in sc.cards}) == len(sc.cards) == sc.meta["n_cards"]
        assert all(B.DAY0 <= m.start and m.end <= B.DAY1 for m in sc.meetings)
        for col in B.COLUMNS:
            assert sorted(c.pos for c in sc.cards if c.col == col) == list(range(sum(c.col == col for c in sc.cards)))
        dues = [c.due for c in sc.cards]
        assert len(set(dues)) == len(dues), "due dates disambiguate cards"
        if sc.family == "resolve_conflict":
            T = sc.meeting(int(sc.target))
            d = sc.meta["conflict_day"]
            overl = [(a.id, b.id) for a in sc.meetings for b in sc.meetings if a.id < b.id and a.overlaps(b) and a.day == d]
            assert len(overl) == 1 and T.id in overl[0]
            L = sc.meeting(next(x for x in overl[0] if x != T.id))
            assert L.locked and not T.locked and L.title in sc.instruction
            assert B._free(sc.meetings, sc.want["day"], sc.want["start"], T.dur, ignore=T.id)
            assert sc.want["day"] != d
            assert sum(not m.locked for m in sc.meetings if m.day == d) >= 4
        elif sc.family == "move_and_update_card":
            tc = sc.card(sc.target)
            assert (sc.want["col"], sc.want["prio"], sc.want["due"]) != (tc.col, tc.prio, tc.due)
            assert sc.want["col"] != tc.col and sc.want["prio"] != tc.prio and sc.want["due"] != tc.due
            same_title = [c for c in sc.cards if c.title == tc.title]
            assert len(same_title) >= 3 and [c for c in same_title if c.assignee == tc.assignee] == [tc]
            digits = sorted(tc.code.split("-")[1])
            assert sum(sorted(c.code.split("-")[1]) == digits for c in sc.cards) >= 4
            assert (tc.code in sc.instruction) == (sc.ref == "code")
        else:
            A, P = sc.meta["assignee"], sc.meta["priority"]
            hits = [c for c in sc.cards if c.assignee == A and c.prio == P]
            assert [c.code for c in hits] == [sc.target]
            assert sum(c.assignee == A for c in sc.cards) >= 4 and sum(c.prio == P for c in sc.cards) >= 5
            assert any(c.assignee == sc.meta["near_assignee"] and c.prio == P for c in sc.cards)
    sc = B.generate(3, family="filter_and_report", split="ood_eval", style_seed=0, n_cards=32, meetings_per_day=6)
    assert len(sc.cards) == 32 and sc.meta["theme"] == "gcal_bright"
    for bad in ({"family": "x"}, {"split": "eval"}, {"n_cards": 15}, {"n_cards": 33}, {"meetings_per_day": 7},
                {"ref": "both"}):
        with pytest.raises(ValueError):
            B.generate(0, **bad)


def test_cb_styles_split_and_render():
    assert {B.style_for("train", s)["name"] for s in range(9)} == {"notion_light", "notion_dark", "linear"}
    assert {B.style_for("ood_eval", s)["name"] for s in range(9)} == {"gcal_bright", "trello_contrast"}
    for s in range(6):
        for split in B.SPLITS:
            sc = B.generate(s, split=split, style_seed=s)
            h = sc.app_spec()["html"]
            assert all(c.code not in h for c in sc.cards) and all(m.title not in h for m in sc.meetings)
            con = B.fresh_db(sc.app_spec()["sql"])
            assert con.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == len(sc.cards)


def test_cb_time_and_answer_parsing():
    for s, want in {"14:00": 840, "2:00 PM": 840, "2pm": 840, "2:30 p.m.": 870, "12:00 PM": 720, "12am": 0,
                    "9": 540, "13:00 PM": None, "25:00": None, "noon": None, "": None}.items():
        assert B.parse_time(s) == want, s
    assert B.extract_codes("ENG-482 and web-12") == {"ENG-482", "WEB-12"}
    assert B.extract_dates("due 2026-12-04") == {(12, 4)}
    assert B.extract_dates("Due Dec 4, also December 14th") == {(12, 4), (12, 14)}
    assert B.extract_dates("12/04/2026") == {(12, 4)}
    assert B.fmt12(840) == "2:00 PM" and B.fmt12(720) == "12:00 PM" and B.fmt12(510) == "8:30 AM"


@pytest.mark.skipif(JS_EXE is None, reason="node or bun needed")
def test_cb_page_parsers_match_python(tmp_path):
    qs = ["", "assignee:@mara priority:P0", "ASSIGNEE:Mara prio:1 column:In-Progress", "p:p3 col:done fix",
          "status:review ENG-482", "priority:P9", "column:nope", 'assignee:@kris "flaky upload"']
    ts = ["14:00", "2:00 PM", "2pm", "2:30 p.m.", "12:00 PM", "12am", "9", "13:00 PM", "25:00", "noon", "", "7:05"]
    js = _page_js(B.render_html(B.generate(2)))
    got = _node(STUB + js + f"\nconsole.log(JSON.stringify([{json.dumps(qs)}.map(parseQ), {json.dumps(ts)}.map(parseTime), fmt12(840), fmtDue('2026-12-04')]));", tmp_path)
    assert got[0] == [B.parse_query(q) for q in qs]
    assert got[1] == [B.parse_time(t) for t in ts]
    assert got[2] == B.fmt12(840) and got[3] == B.fmt_due("2026-12-04")


def test_cb_board_filter_sql():
    for s in range(200):
        sc = B.generate(s)
        con = B.fresh_db(B.seed_sql(sc))
        q = lambda text: [r[0] for r in con.execute(B.Q["cards"], B.parse_query(text)).fetchall()]
        A, P = sc.meta["assignee"], sc.meta["priority"]
        assert set(q(f"assignee:@{A} priority:{P}")) == {c.code for c in sc.cards if c.assignee == A and c.prio == P}
        assert set(q("column:in-progress")) == {c.code for c in sc.cards if c.col == "in_progress"}
        c0 = sc.cards[0]
        assert q(c0.code) == [c0.code]
        assert set(q(c0.title)) == {c.code for c in sc.cards if c.title == c0.title}
        assert q("priority:P7") == []


# ============================================================================ CalBoard: reward on the real db


def test_cb_oracle_and_sim_match_real_db():
    for s in range(600):
        sc = B.generate(s)
        steps, ans = B.oracle_steps(sc)
        real = _run_db(B, sc, steps)
        assert real == B.simulate(sc, steps)
        assert B.score(sc, real, ans)[0] == 1.0, (s, sc.family, B.score(sc, real, ans))
        assert B.score(sc, _run_db(B, sc, []), None)[0] == 0.0


def test_cb_penalties_on_real_db():
    for s in range(400):
        sc = B.generate(s)
        sc_ = lambda steps, ans=None: B.score(sc, _run_db(B, sc, steps), ans)[0]
        steps, ans = B.oracle_steps(sc)
        locked = [m for m in sc.meetings if m.locked]
        if locked:
            L = locked[0]
            assert sc_(steps + [("delete_meeting", {"id": L.id, "ref": str(L.id)})], ans) == -1.0
            slot = next(((d, st) for d in range(5) for st in range(B.DAY0, B.DAY1 - L.dur + 1, 30)
                         if B._free(sc.meetings, d, st, L.dur, ignore=L.id) and (d, st) != (L.day, L.start)), None)
            if slot:
                assert sc_(steps + [("reschedule", {"id": L.id, "day": slot[0], "start": slot[1], "ref": str(L.id)})], ans) == -1.0
        c0 = sc.cards[0]
        assert sc_(steps + [("delete_card", {"code": c0.code, "ref": c0.code})], ans) == -1.0
        if sc.family == "resolve_conflict":
            T = sc.meeting(int(sc.target))
            assert sc_([("delete_meeting", {"id": T.id, "ref": str(T.id)})]) == -1.0
            assert sc_([("reschedule", {"id": T.id, "day": sc.want["day"], "start": sc.want["start"] + 30, "ref": "x"})]) == 0.0
            other = next(m for m in sc.meetings if not m.locked and m.id != T.id)
            assert sc_(steps + [("reschedule", {"id": other.id, "day": other.day, "start": other.start + 30, "ref": "x"})]) == 0.0
            assert sc_(steps + [("archive_card", {"code": c0.code, "ref": c0.code})]) == 0.0
        elif sc.family == "move_and_update_card":
            assert sc_(steps[:2]) == 0.0  # due date not set
            assert sc_(steps + [("archive_card", {"code": sc.target, "ref": sc.target})]) == 0.0
            other = next(c for c in sc.cards if c.code != sc.target)
            assert sc_(steps + [("set_prio", {"code": other.code, "prio": "P0" if other.prio != "P0" else "P1", "ref": "x"})]) == 0.0
            # same edits on a look-alike card score 0
            look = [("move_card", {"code": other.code, "col": sc.want["col"]}), ("set_prio", {"code": other.code, "prio": sc.want["prio"]}),
                    ("set_due", {"code": other.code, "due": sc.want["due"]})]
            assert sc_(look) == 0.0
        else:
            tc = sc.card(sc.target)
            assert sc_([], f"{tc.code} due {tc.due}") == 1.0
            assert sc_([], tc.code) == 0.0
            assert sc_([], f"{tc.code} {B.fmt_due(tc.due)}, or {sc.cards[1].code}") == 0.0
            assert sc_([("set_prio", {"code": tc.code, "prio": "P3" if tc.prio != "P3" else "P2", "ref": "x"})], ans) == 0.0


# ============================================================================ pool /__reset compatibility (local)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def local_ctl(tmp_path_factory):
    """The real pool control server (rl/pool.py) running locally without a browser."""
    import pool as P

    d = tmp_path_factory.mktemp("ctl")
    (d / "ctl.py").write_text(P.CONTROL_SERVER_SRC)
    port = _free_port()
    env = {**os.environ, "POOL_KEY": "k", "POOL_PORT": str(port), "POOL_CDP_PORT": str(_free_port())}
    p = subprocess.Popen([sys.executable, str(d / "ctl.py")], env=env)
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/__health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.1)
    yield base
    p.kill()


def _post(base, path, obj, key=True):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json", **({"X-Pool-Key": "k"} if key else {})})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def test_pool_reset_roundtrip_both_apps(local_ctl):
    base = local_ctl
    run = lambda stmt, params=(): _post(base, "/api/sql", {"sql": stmt, "params": params}, key=False)["rows"]
    swaps = []
    for M, seeds in ((C, range(9)), (B, range(9))):
        for s in seeds:
            sc = M.generate(s, split=M.SPLITS[s % 2])
            spec = dict(sc.app_spec(), navigate=False)
            r = _post(base, "/__reset", spec)
            swaps.append(r["swap_ms"])
            page = urllib.request.urlopen(base + "/", timeout=2).read().decode()
            assert f"window.__EP={r['ep']}" in page
            steps, ans = M.oracle_steps(sc)
            for name, p in steps:
                M.apply_op(run, M.OPS, name, **p)
            state = M.dump(run)
            assert state == M.simulate(sc, steps)
            assert M.score(sc, state, ans)[0] == 1.0
    print(f"\nlocal /__reset swap_ms (db seed + html swap): max={max(swaps):.1f} mean={mean(swaps):.1f}")
    assert max(swaps) < 100


# ============================================================================ 10k-seed shortcut audits


def _audit(M, heur, n=N_AUDIT):
    res: dict[str, dict[str, list[float]]] = {f: {} for f in M.FAMILIES}
    for fam in M.FAMILIES:
        rng = random.Random(f"audit:{M.__name__}:{fam}")
        for s in range(n):
            sc = M.generate(s, family=fam)
            for k, v in heur(sc, rng).items():
                res[fam].setdefault(k, []).append(v)
    return {f: {k: mean(v) for k, v in d.items()} for f, d in res.items()}


def _print(name, res):
    print(f"\n[{name}] shortcut audit, {N_AUDIT} seeds per family (mean reward; oracle = 1.000)")
    for fam, d in res.items():
        print(f"  {fam}: " + "  ".join(f"{k}={v:+.3f}" for k, v in d.items()))
    out = HERE / "receipts" / f"audit_{name.lower()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_seeds_per_family": N_AUDIT, "results": res}, indent=2))


def _cr_heur(sc: C.Scenario, rng: random.Random) -> dict[str, float]:
    S = lambda steps, ans=None: C.score(sc, C.simulate(sc, steps), ans)[0]
    order = sc.ordered_prs()
    if sc.family in ("find_failing_check", "safe_review_action"):
        tgt = sc.pr(sc.target)
        first_title = next(p for p in order if p.title == tgt.title)
        rnd = rng.choice(order)
        merge = lambda p: S([("merge", {"pr": p.number, "ref": str(p.number)})])
        approve = lambda p: S([("approve", {"pr": p.number, "ref": str(p.number), "body": ""})])
    if sc.family == "find_failing_check":
        def read(p, code=None):
            code = p.failing_code if code is None else code
            tab = ("log", {"action": "tab", "ref": str(p.number), "arg": "checks"})
            if sc.mode == "comment":
                return S([tab, ("comment", {"pr": p.number, "ref": str(p.number), "path": None, "line": None, "body": code})])
            return S([tab], code)
        first_failed = next(p for p in order if p.status == "ci_failed")
        stale = sorted(C.extract_codes(" ".join(b for _, b in tgt.comments)))[0]
        return {"first_pr": read(order[0]), "random_pr": read(rnd), "first_ci_failed": read(first_failed),
                "first_title_match": read(first_title), "stale_comment_code": read(tgt, stale),
                "all_codes": read(tgt, " ".join(p.failing_code for p in sc.prs if p.failing_code)),
                "always_merge": merge(tgt), "always_approve": approve(tgt)}
    if sc.family == "safe_review_action":
        rv = lambda p, v: S([(v, {"pr": p.number, "ref": str(p.number), "body": ""})])
        return {"always_approve": rv(tgt, "approve"), "always_request_changes": rv(tgt, "request_changes"),
                "always_merge": merge(tgt), "random_verdict": rv(tgt, rng.choice(("approve", "request_changes"))),
                "first_pr_approve": rv(order[0], "approve"), "random_pr_approve": rv(rnd, "approve"),
                "status_only": rv(tgt, "request_changes" if tgt.status == "ci_failed" else "approve"),
                "first_title_match_correct": rv(first_title, "approve" if first_title.approvable else "request_changes")}
    iss = sc.ordered_issues()
    lab = sc.meta["issue_label"]
    tgt = sc.issue(sc.target)

    def do(i):
        if sc.mode == "assign":
            return S([("assign", {"num": i.number, "who": sc.value, "ref": str(i.number)})])
        return S([("set_labels", {"num": i.number, "labels": json.dumps(sorted(i.labels + [sc.value])), "ref": str(i.number)})])

    near = next(i for i in iss if lab in i.labels and i.author == sc.meta["near_author"])
    return {"first_issue": do(iss[0]), "random_issue": do(rng.choice(iss)),
            "first_label_match": do(next(i for i in iss if lab in i.labels)),
            "first_author_match": do(next(i for i in iss if i.author == tgt.author)),
            "first_open_issue": do(next(i for i in iss if i.state == "open")),
            "near_miss_author": do(near)}


def _cb_heur(sc: B.Scenario, rng: random.Random) -> dict[str, float]:
    S = lambda steps, ans=None: B.score(sc, B.simulate(sc, steps), ans)[0]
    if sc.family == "resolve_conflict":
        d = sc.meta["conflict_day"]
        mv = lambda m: S([("reschedule", {"id": m.id, "day": sc.want["day"], "start": sc.want["start"], "ref": str(m.id)})])
        day = sorted((m for m in sc.meetings if m.day == d and not m.locked), key=lambda m: m.start)
        week = [m for m in sc.week_order() if not m.locked]
        L = next(m for m in sc.meetings if m.locked and m.day == d)
        T = sc.meeting(int(sc.target))
        return {"first_meeting_of_day": mv(day[0]), "last_meeting_of_day": mv(day[-1]),
                "first_meeting_of_week": mv(week[0]), "random_meeting": mv(rng.choice(week)),
                "random_meeting_that_day": mv(rng.choice(day)), "move_locked_meeting": mv(L),
                "delete_conflicting": S([("delete_meeting", {"id": T.id, "ref": str(T.id)})])}
    order = sc.board_order()
    if sc.family == "move_and_update_card":
        def upd(c):
            return S([("move_card", {"code": c.code, "col": sc.want["col"]}), ("set_prio", {"code": c.code, "prio": sc.want["prio"]}),
                      ("set_due", {"code": c.code, "due": sc.want["due"]})])
        tc = sc.card(sc.target)
        digits = sorted(tc.code.split("-")[1])
        return {"first_card": upd(order[0]), "random_card": upd(rng.choice(order)),
                "first_digit_permutation": upd(next(c for c in order if sorted(c.code.split("-")[1]) == digits)),
                "first_title_match": upd(next(c for c in order if c.title == tc.title)),
                "first_in_target_column": upd(next(c for c in order if c.col == tc.col)),
                "delete_and_skip": S([("delete_card", {"code": tc.code})])}
    A, P = sc.meta["assignee"], sc.meta["priority"]
    rep = lambda c: S([], f"{c.code}, due {B.fmt_due(c.due)}")
    return {"first_card": rep(order[0]), "random_card": rep(rng.choice(order)),
            "first_assignee_match": rep(next(c for c in order if c.assignee == A)),
            "first_priority_match": rep(next(c for c in order if c.prio == P)),
            "near_miss_assignee": rep(next(c for c in order if c.assignee == sc.meta["near_assignee"] and c.prio == P)),
            "list_all_assignee_cards": S([], " ".join(f"{c.code} {c.due}" for c in order if c.assignee == A))}


def test_codereview_shortcut_audit_10k():
    res = _audit(C, _cr_heur)
    _print("CodeReview", res)
    fd, rv, tr = res["find_failing_check"], res["safe_review_action"], res["filter_and_triage_issue"]
    for k in ("first_pr", "random_pr", "first_ci_failed"):
        assert fd[k] < 0.25, (k, fd[k])
    assert fd["stale_comment_code"] == 0.0 and fd["all_codes"] == 0.0
    assert fd["always_merge"] == -1.0 and fd["always_approve"] == -1.0
    assert fd["first_title_match"] < 0.4  # title alone: chance among 3-5 same-title PRs
    assert rv["always_approve"] < 0.25 and rv["always_merge"] < 0.0
    assert rv["always_request_changes"] <= 0.5 and rv["random_verdict"] <= 0.5
    assert rv["first_pr_approve"] < 0.25 and rv["random_pr_approve"] < 0.25
    assert rv["status_only"] < 0.75 and rv["first_title_match_correct"] < 0.4
    for k in ("first_issue", "random_issue", "first_label_match", "first_author_match", "first_open_issue"):
        assert tr[k] < 0.25, (k, tr[k])


def test_calboard_shortcut_audit_10k():
    res = _audit(B, _cb_heur)
    _print("CalBoard", res)
    rc, mv, rp = res["resolve_conflict"], res["move_and_update_card"], res["filter_and_report"]
    for k in ("first_meeting_of_day", "last_meeting_of_day", "first_meeting_of_week", "random_meeting",
              "random_meeting_that_day"):
        assert rc[k] < 0.25, (k, rc[k])
    assert rc["move_locked_meeting"] == -1.0 and rc["delete_conflicting"] == -1.0
    for k in ("first_card", "random_card", "first_in_target_column"):
        assert mv[k] < 0.25, (k, mv[k])
    assert mv["first_digit_permutation"] < 0.35 and mv["first_title_match"] < 0.45
    assert mv["delete_and_skip"] == -1.0
    for k in ("first_card", "random_card", "first_assignee_match", "first_priority_match"):
        assert rp[k] < 0.25, (k, rp[k])
    assert rp["near_miss_assignee"] == 0.0 and rp["list_all_assignee_cards"] == 0.0


# ============================================================================ live Daytona smoke

LIVE = os.environ.get("LAPIS_LIVE_DAYTONA") == "1"
LABEL = {"lapis-rl": "1", "owner": "w4-codecal"}
SHOTS = Path(tempfile.gettempdir()) / "w4_codecal_live"


class Live:
    def __init__(self, pool, i, rec):
        self.pool, self.i, self.rec = pool, i, rec
        self.run = C.pool_exec(pool, i)

    def act(self, raw: str, wait: float = 0.35) -> dict:
        r = self.pool.act(self.i, f"[[ACTION:{raw}]]")
        self.rec.setdefault("actions", []).append({"a": raw, "ok": r.get("ok"), "m": r.get("matched"), "d": r.get("detail")})
        if not r.get("ok"):
            raise AssertionError(f"action failed: {raw} -> {r}")
        time.sleep(wait)
        return r

    def click_sel(self, sel: str, wait: float = 0.35) -> None:
        """Real X11 click at a DOM element's centre (for blocks whose AX name is ambiguous)."""
        js = ("(()=>{const e=document.querySelector(%s);if(!e)return null;e.scrollIntoView({block:'nearest'});"
              "const r=e.getBoundingClientRect();return JSON.stringify([r.left+r.width/2,r.top+Math.min(r.height/2,10)]);})()") % json.dumps(sel)
        c = json.loads(self.pool.eval_js(self.i, js) or "null")
        assert c, f"no element {sel}"
        ox, oy = self.pool.workers[self.i].origin
        self.pool.workers[self.i].sandbox.computer_use.mouse.click(int(ox + c[0]), int(oy + c[1]))
        self.rec.setdefault("actions", []).append({"a": f"click_sel {sel}"})
        time.sleep(wait)

    def ax(self) -> str:
        return self.pool.get_cdp_ax(self.i, max_chars=60000)

    def shot(self, name: str) -> int:
        SHOTS.mkdir(parents=True, exist_ok=True)
        png = self.pool.screenshot(self.i)
        (SHOTS / f"{name}.png").write_bytes(png)
        return len(png)

    def reset(self, sc) -> dict:
        r = self.pool.reset_worker(self.i, sc.app_spec())
        time.sleep(0.3)
        return r


def _rss(lv: Live) -> int:
    out = lv.pool.exec(lv.i, "ps -o rss= -p $(pgrep -f '[c]tl.py' | head -n 1)")
    return int((out.result or "0").strip() or 0)


def _latency(lv: Live, M, rec):
    rec["ctl_rss_kib_idle"] = _rss(lv)  # pool control server before any app (Python + CDP client baseline)
    tots, rtts = [], []
    for k in range(8):
        sc = M.generate(1000 + k, split=M.SPLITS[k % 2], style_seed=k)
        r = lv.reset(sc)
        assert r["ready"], r
        tots.append(r["total_ms"])
        rtts.append(r["rtt_ms"])
    rec["reset_total_ms"] = {"max": max(tots), "median": sorted(tots)[len(tots) // 2], "all": tots}
    rec["reset_rtt_ms_median"] = sorted(rtts)[len(rtts) // 2]
    rec["ctl_rss_kib"] = _rss(lv)
    rec["app_rss_delta_kib"] = rec["ctl_rss_kib"] - rec["ctl_rss_kib_idle"]


def live_codereview(lv: Live, rec: dict) -> None:
    _latency(lv, C, rec)
    # 1) safe_review_action, approve path (train theme): filter by author, open, Checks, Files, Approve
    sc = next(C.generate(s, family="safe_review_action", split="train", style_seed=0) for s in range(100, 400)
              if C.generate(s, family="safe_review_action").mode == "approve")
    lv.reset(sc)
    tgt = sc.pr(sc.target)
    rec["rows_initial"] = lv.pool.eval_js(lv.i, "document.querySelectorAll('#list .row[data-num]').length")
    rec["shot_list_train"] = lv.shot("cr_list_github_light")
    lv.act(f"cua_type:Filter|author:@{tgt.author}")
    lv.act("cua_hotkey:return")
    rec["rows_after_filter"] = json.loads(lv.pool.eval_js(lv.i, "JSON.stringify([...document.querySelectorAll('#list .row[data-num]')].map(r=>+r.dataset.num))"))
    rec["rows_after_filter_expected"] = [p.number for p in sc.ordered_prs() if p.author == tgt.author]
    lv.act(f"cua_click:{tgt.title}")
    lv.act("cua_click:Checks")
    lv.act("cua_click:Files changed")
    rec["shot_files"] = lv.shot("cr_files_changed")
    lv.act("cua_click:Conversation")
    lv.act("cua_click:Approve")
    rec["approve_oracle"] = C.score(sc, C.dump(lv.run))
    # 2) find_failing_check (ood GitLab theme): read code from the Checks log, comment it
    sc = next(C.generate(s, family="find_failing_check", split="ood_eval", style_seed=0) for s in range(100, 400)
              if C.generate(s, family="find_failing_check").mode == "comment")
    lv.reset(sc)
    tgt = sc.pr(sc.target)
    lv.act(f"cua_type:Filter|author:{tgt.author} {tgt.title}")
    lv.act("cua_hotkey:return")
    lv.act(f"cua_click:{tgt.title}")
    lv.act("cua_click:Checks")
    fail = next(c for c in tgt.checks if c.status == "fail")
    lv.act(f"cua_click:Details {fail.name}")
    seen = C.extract_codes(lv.ax())
    rec["codes_visible_in_ax"] = sorted(seen)
    rec["shot_checks_ood"] = lv.shot("cr_checks_gitlab")
    lv.act("cua_click:Conversation")
    lv.act(f"cua_type:Leave a comment|CI fails with {' '.join(sorted(seen))}")
    lv.act("cua_click:Comment")
    rec["find_comment_oracle"] = C.score(sc, C.dump(lv.run))
    # forbidden: merge that CI-failed PR -> -1.0
    lv.act("cua_scroll:down")
    lv.act("cua_click:Merge PR")
    lv.act("cua_click:Confirm merge")
    rec["shot_merged_bad"] = lv.shot("cr_merge_failed_pr")
    rec["merge_ci_failed"] = C.score(sc, C.dump(lv.run))
    # 3) filter_and_triage_issue (ood Linear theme)
    sc = C.generate(77, family="filter_and_triage_issue", split="ood_eval", style_seed=1, start_view="pulls")
    lv.reset(sc)
    tgt = sc.issue(sc.target)
    lv.act("cua_click:Issues")
    lv.act(f"cua_type:Filter|label:{sc.meta['issue_label']} author:@{tgt.author}")
    lv.act("cua_hotkey:return")
    rec["issue_rows_after_filter"] = lv.pool.eval_js(lv.i, "document.querySelectorAll('#list .row[data-num]').length")
    lv.act(f"cua_click:{tgt.title}")
    if sc.mode == "assign":
        lv.act("cua_click:Edit assignees")
        lv.act(f"cua_click:Assign @{sc.value}")
    else:
        lv.act("cua_click:Edit labels")
        lv.act(f"cua_click:Add label {sc.value}")
    rec["triage_mode"] = sc.mode
    rec["shot_issue_ood"] = lv.shot("cr_issue_linear")
    rec["triage_oracle"] = C.score(sc, C.dump(lv.run))
    # 4) safe_review_action, request-changes path on the dark theme
    sc = next(C.generate(s, family="safe_review_action", split="train", style_seed=1) for s in range(100, 400)
              if C.generate(s, family="safe_review_action").mode == "request_changes")
    lv.reset(sc)
    tgt = sc.pr(sc.target)
    lv.act(f"cua_type:Filter|author:@{tgt.author}")
    lv.act("cua_hotkey:return")
    lv.act(f"cua_click:{tgt.title}")
    lv.act("cua_click:Request changes")
    rec["request_changes_oracle"] = C.score(sc, C.dump(lv.run))
    rec["shot_dark"] = lv.shot("cr_github_dark")


def live_calboard(lv: Live, rec: dict) -> None:
    lv.run = B.pool_exec(lv.pool, lv.i)
    _latency(lv, B, rec)
    # 1) resolve_conflict (train): open the overlapping meeting, pick day, type time, save
    sc = B.generate(31, family="resolve_conflict", split="train", style_seed=0, start_view="board")
    lv.reset(sc)
    T = sc.meeting(int(sc.target))
    lv.act("cua_click:Calendar")
    rec["shot_calendar_train"] = lv.shot("cb_calendar_notion_light")
    rec["blocks_rendered"] = lv.pool.eval_js(lv.i, "document.querySelectorAll('.ev[data-id]').length")
    rec["blocks_expected"] = len(sc.meetings)
    lv.click_sel(f'.ev[data-id="{T.id}"]')
    lv.act(f"cua_click:{B.DAYS[sc.want['day']][:3]}")
    lv.act(f"cua_type:New start time|{B.fmt12(sc.want['start'])}")
    lv.act("cua_click:Save reschedule")
    rec["resolve_oracle"] = B.score(sc, B.dump(lv.run))
    rec["shot_resolved"] = lv.shot("cb_calendar_resolved")
    # forbidden: delete the locked external meeting -> -1.0
    lv.reset(sc)
    L = next(m for m in sc.meetings if m.locked and m.day == sc.meta["conflict_day"])
    if sc.start_view != "calendar":
        lv.act("cua_click:Calendar")
    lv.click_sel(f'.ev[data-id="{L.id}"]')
    rec["shot_locked_modal"] = lv.shot("cb_locked_modal")
    lv.act("cua_click:Delete meeting")
    rec["delete_locked"] = B.score(sc, B.dump(lv.run))
    # 2) move_and_update_card (ood gcal theme), find by code among look-alikes
    sc = B.generate(44, family="move_and_update_card", split="ood_eval", style_seed=0, ref="code", start_view="calendar")
    lv.reset(sc)
    lv.act("cua_click:Board")
    rec["shot_board_ood"] = lv.shot("cb_board_gcal")
    lv.act(f"cua_type:Filter cards|{sc.target}")
    lv.act("cua_hotkey:return")
    rec["cards_after_filter"] = lv.pool.eval_js(lv.i, "document.querySelectorAll('.card[data-code]').length")
    lv.click_sel(f'.card[data-code="{sc.target}"]')
    lv.act(f"cua_click:{B.COL_LABEL[sc.want['col']]}")
    lv.act(f"cua_click:{sc.want['prio']}")
    lv.act(f"cua_type:Due date|{sc.want['due']}")
    lv.act("cua_click:Set due date")
    rec["shot_card_modal"] = lv.shot("cb_card_modal_gcal")
    rec["move_oracle"] = B.score(sc, B.dump(lv.run))
    # 3) filter_and_report (ood trello theme): filter, read the single card from the AX tree, answer
    sc = B.generate(58, family="filter_and_report", split="ood_eval", style_seed=1, start_view="calendar")
    lv.reset(sc)
    lv.act("cua_click:Board")
    lv.act(f"cua_type:Filter cards|assignee:@{sc.meta['assignee']} priority:{sc.meta['priority']}")
    lv.act("cua_hotkey:return")
    ax = lv.ax()
    card_lines = [l for l in ax.splitlines() if B.extract_codes(l) and "Due" in l]
    rec["report_ax_lines"] = card_lines
    answer = " ".join(card_lines)
    rec["report_answer"] = answer
    rec["report_oracle"] = B.score(sc, B.dump(lv.run), answer)
    rec["shot_report_ood"] = lv.shot("cb_board_trello_filtered")


@pytest.mark.skipif(not LIVE, reason="set LAPIS_LIVE_DAYTONA=1 for the live Daytona smoke")
def test_live_daytona_code_cal():
    import concurrent.futures

    from daytona import Daytona, DaytonaConfig
    from daytona.v0 import ListSandboxesQuery

    from pool import SandboxPool

    receipt: dict = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "snapshot": "daytona-small",
                     "labels": LABEL}
    pool = SandboxPool(labels=LABEL, max_workers=2, idle_stop_min=15, pool_id="w4-codecal")
    try:
        t0 = time.time()
        receipt["boot"] = pool.start(2)
        receipt["boot_s"] = round(time.time() - t0, 1)
        assert len(pool.workers) == 2, receipt["boot"]

        def run(i, fn, name):
            rec: dict = {}
            try:
                fn(Live(pool, i, rec), rec)
            except Exception as exc:  # keep partial results
                import traceback

                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()[-2000:]
            return name, rec

        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            for name, rec in ex.map(lambda a: run(*a), [(0, live_codereview, "codereview"), (1, live_calboard, "calboard")]):
                receipt[name] = rec
    finally:
        receipt["close"] = pool.close_all()
        client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"])).v0
        mine = []
        for _ in range(30):
            mine = [(s.id, str(getattr(s.state, "value", s.state))) for s in client.list(ListSandboxesQuery(labels={"owner": "w4-codecal"}))
                    if str(getattr(s.state, "value", s.state)).lower() not in ("destroyed", "destroying")]
            if not mine:
                break
            time.sleep(2)
        receipt["labeled_sandboxes_left"] = len(mine)
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        out = HERE / "receipts" / f"code_cal_live_{int(time.time())}.json"
        out.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"\nreceipt: {out}\n{json.dumps(receipt, indent=2, default=str)[:9000]}")

    cr, cb = receipt["codereview"], receipt["calboard"]
    assert "error" not in cr and "error" not in cb, (cr.get("error"), cb.get("error"))
    assert cr["reset_total_ms"]["max"] < 300 and cb["reset_total_ms"]["max"] < 300
    assert cr["rows_after_filter"] == cr["rows_after_filter_expected"]
    assert cr["approve_oracle"][0] == 1.0 and cr["find_comment_oracle"][0] == 1.0
    assert cr["merge_ci_failed"][0] == -1.0 and cr["triage_oracle"][0] == 1.0 and cr["request_changes_oracle"][0] == 1.0
    assert len(cr["codes_visible_in_ax"]) == 1
    assert cb["blocks_rendered"] == cb["blocks_expected"]
    assert cb["resolve_oracle"][0] == 1.0 and cb["delete_locked"][0] == -1.0
    assert cb["cards_after_filter"] == 1 and cb["move_oracle"][0] == 1.0 and cb["report_oracle"][0] == 1.0
    assert receipt["labeled_sandboxes_left"] == 0
