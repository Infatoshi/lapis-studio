"""Offline tests, 10k-seed shortcut audits and a live Daytona smoke for TerminalLogs and DocWiki.

Live smoke (2x daytona-small through rl/pool.py, labeled owner=w3-termwiki, always deleted):
  LAPIS_LIVE_DAYTONA=1 uv run pytest tests/test_term_wiki.py -k live -s
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
from collections import Counter
from pathlib import Path
from statistics import mean

import pytest

from synth import doc_wiki as W
from synth import terminal_logs as T

N_AUDIT = 10_000
HERE = Path(__file__).resolve().parent.parent
JS_EXE = shutil.which("node") or shutil.which("bun")


def _run_db(M, sc, steps):
    con = M.fresh_db(M.seed_sql(sc))
    ex = M.sqlite_exec(con)
    for name, p in steps:
        M.apply_op(ex, M.OPS, name, **p)
    return M.dump(ex), ex


def _page_js(html: str) -> str:
    return html[html.index("<script>") + len("<script>"):html.index("</script>")]


def _node(script: str, tmp_path) -> object:
    f = tmp_path / f"h{random.randrange(10**9)}.js"
    f.write_text(script)
    out = subprocess.run([JS_EXE, str(f)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


STUB = ("const window = {}; const document = {addEventListener(){}, getElementById(){return null;}};"
        "function fetch(){ return Promise.resolve({json(){return {rows:[]};}}); }")


def _roots_from_trace(trace: str) -> str:
    """The code on the last 'Caused by:' line (what the instruction asks for)."""
    cb = [l for l in trace.splitlines() if l.startswith("Caused by:")]
    return T.extract_codes(cb[-1]).pop() if cb else ""


# ============================================================================ TerminalLogs: generation


def test_tl_deterministic_and_seed_sensitive():
    for s in range(40):
        assert T.generate(s).to_json() == T.generate(s).to_json()
    assert len({T.generate(s).to_json() for s in range(200)}) == 200


def test_tl_invariants():
    for s in range(1500):
        for fam in T.FAMILIES:
            sc = T.generate(s, family=fam)
            names = [x.name for x in sc.services]
            assert len(names) == len(set(names)) and 9 <= len(names) <= 15
            assert len({l.id for l in sc.logs}) == len(sc.logs)
            assert all(sc.now - 1800 < l.ts <= sc.now for l in sc.logs)
            assert not any(x.status == "crashloop" and x.protected for x in sc.services)
            assert sum(x.status == "crashloop" for x in sc.services) <= 1
            assert sum(x.status == "healthy" and not x.protected for x in sc.services) >= 2
            assert any(x.protected for x in sc.services)
            for nm in names:  # no code-like tokens outside codes
                assert not T.extract_codes(nm)
            for l in sc.logs:
                if l.level == "ERROR":
                    # each trace line is its own AX text node; Studio caps a node at 280 chars (TEXT_CAP)
                    assert max(len(x) for x in l.trace.splitlines()) < 200
                    assert l.trace and T.extract_codes(l.trace.splitlines()[0]) == {l.code}
                    assert _roots_from_trace(l.trace) == l.root
            if fam == "diagnose_error_code":
                tgt = [l for l in sc.logs if l.service == sc.target]
                since = sc.now - sc.window * 60 if sc.window else 0
                errs = [l for l in tgt if l.level == "ERROR" and l.ts >= since]
                assert len(errs) >= 2 and {l.root for l in errs} == {sc.want["root"]}
                assert all(l.code != sc.want["root"] for l in errs)  # surface code differs
                assert all(len([x for x in l.trace.splitlines() if x.startswith("Caused by:")]) == 2 for l in errs)
                old = [l for l in tgt if l.level == "ERROR" and l.ts < since]
                assert bool(old) == bool(sc.window) and all(l.root != sc.want["root"] for l in old)
                warns = Counter(l.code for l in tgt if l.level == "WARN")
                assert sc.want["root"] not in warns
                assert sum(warns.values()) > len([l for l in tgt if l.level == "ERROR"])
                sib = [l for l in sc.logs if l.service == sc.meta["sibling"] and l.level == "ERROR"]
                assert sib and sc.want["root"] not in {l.root for l in sib}
                assert sc.meta["sibling"].split("-")[0] == sc.target.split("-")[0]
                assert sc.mode in ("report", "file")
                all_roots = [l.root for l in sc.logs if l.root]
                assert all_roots.count(sc.want["root"]) == len(errs)
            elif fam == "safe_restart_service":
                cl = [x for x in sc.services if x.status == "crashloop"]
                assert [x.name for x in cl] == [sc.target]
                assert sc.service(sc.meta["sibling"]).protected
                assert sum(x.status == "degraded" for x in sc.services) >= 2
            else:
                S, R = sc.meta["sev"], sc.meta["region"]
                match = [a for a in sc.alerts if a.sev == S and a.region == R and not a.acked]
                assert [str(a.id) for a in match] == [sc.target]
                assert any(a.sev == S and a.region == sc.meta["near_region"] for a in sc.alerts)
                assert len({a.id for a in sc.alerts}) == len(sc.alerts)


def test_tl_styles_split_and_render():
    tr = {t["name"] for t in T.THEMES["train"]}
    ood = {t["name"] for t in T.THEMES["ood_eval"]}
    assert tr.isdisjoint(ood)
    for split in T.SPLITS:
        for k in range(4):
            sc = T.generate(k, split=split, style_seed=k)
            h = sc.app_spec()["html"]
            th = T.style_for(split, k)
            assert th["bg"] in h and th["name"] == sc.meta["theme"]
            assert "__" not in _page_js(h).replace("__EP", "")  # all placeholders substituted
    with pytest.raises(ValueError):
        T.generate(1, split="test")


def test_tl_page_parsers_match_python(tmp_path):
    if not JS_EXE:
        pytest.skip("no node/bun")
    sc = T.generate(3)
    qs = ["service:billing-api level:ERROR since:15m", "svc:x lvl:warning timeout", "level:err last:2h", "since:abc",
          "level:debug foo bar", "", "SERVICE:Ingest-Worker LEVEL:Info", 'since:10 "quoted text"']
    aq = ["sev:SEV2 region:eu-west-1 status:open", "severity:1 region:us-east-2", "sev:sev3 status:acked disk",
          "sev:9 status:closed", "", "Region:AP-SOUTH-1 latency"]
    js = (STUB + _page_js(sc.app_spec()["html"]) +
          f"console.log(JSON.stringify([{json.dumps(qs)}.map(parseLogQ), {json.dumps(aq)}.map(parseAlertQ)]));")
    lq, al = _node(js, tmp_path)
    assert lq == [T.parse_log_query(q, sc.now) for q in qs]
    assert al == [T.parse_alert_query(q) for q in aq]


def test_tl_filter_sql():
    for s in range(150):
        sc = T.generate(s, family="diagnose_error_code")
        _, ex = _run_db(T, sc, [])
        flt = f"service:{sc.target} level:ERROR" + (f" since:{sc.window}m" if sc.window else "")
        rows = ex(T.Q["logs"], T.parse_log_query(flt, sc.now))
        since = sc.now - sc.window * 60 if sc.window else 0
        want = [l.id for l in sc.log_order() if l.service == sc.target and l.level == "ERROR" and l.ts >= since]
        assert [r[0] for r in rows] == want
        assert {_roots_from_trace(r[6]) for r in rows} == {sc.want["root"]}
        assert len(ex(T.Q["logs"], T.parse_log_query("", sc.now))) == len(sc.logs)
        sc = T.generate(s, family="ack_matching_alert")
        _, ex = _run_db(T, sc, [])
        rows = ex(T.Q["alerts"], T.parse_alert_query(f"sev:{sc.meta['sev']} region:{sc.meta['region']} status:open"))
        assert [str(r[0]) for r in rows] == [sc.target]


def test_tl_oracle_and_sim_match_real_db():
    for fam in T.FAMILIES:
        for s in range(300):
            sc = T.generate(s, family=fam, split=T.SPLITS[s % 2])
            steps, ans = T.oracle_steps(sc)
            st, _ = _run_db(T, sc, steps)
            assert st == T.simulate(sc, steps)
            assert T.score(sc, st, ans)[0] == 1.0, (fam, s, T.score(sc, st, ans))


def test_tl_penalties_on_real_db():
    for s in range(200):
        for fam in T.FAMILIES:
            sc = T.generate(s, family=fam)
            steps, ans = T.oracle_steps(sc)
            healthy = next(x for x in sc.services if x.status == "healthy" and not x.protected)
            prot = next(x for x in sc.services if x.protected)
            for bad in (("restart_service", healthy), ("drain_service", healthy), ("restart_service", prot),
                        ("drain_service", prot)):
                st, _ = _run_db(T, sc, steps + [(bad[0], {"svc": bad[1].name})])
                assert T.score(sc, st, ans)[0] == -1.0
        sc = T.generate(s, family="safe_restart_service")
        deg = next(x for x in sc.services if x.status == "degraded" and not x.protected) if any(
            x.status == "degraded" and not x.protected for x in sc.services) else None
        if deg:
            st, _ = _run_db(T, sc, [("restart_service", {"svc": deg.name})])
            assert T.score(sc, st)[0] == 0.0
        st, _ = _run_db(T, sc, [("restart_service", {"svc": sc.target}), ("drain_service", {"svc": sc.target})])
        assert T.score(sc, st)[0] == 0.0  # drained the crash-looper too
        sc = T.generate(s, family="diagnose_error_code", mode="file")
        st, _ = _run_db(T, sc, [("file_incident", {"svc": sc.target, "code": sc.want["root"].lower(), "note": ""})])
        assert T.score(sc, st)[0] == 1.0  # page upper-cases; scorer normalizes too
        st, _ = _run_db(T, sc, [("file_incident", {"svc": sc.meta["sibling"], "code": sc.want["root"], "note": ""})])
        assert T.score(sc, st)[0] == 0.0
        sc = T.generate(s, family="diagnose_error_code", mode="report")
        assert T.score(sc, T.simulate(sc, []), f"root cause is {sc.want['root'].lower()}")[0] == 1.0
        st = T.simulate(sc, [("file_incident", {"svc": sc.target, "code": sc.want["root"], "note": ""})])
        assert T.score(sc, st, sc.want["root"])[0] == 0.0


# ============================================================================ DocWiki: generation


def test_dw_deterministic_and_seed_sensitive():
    for s in range(40):
        assert W.generate(s).to_json() == W.generate(s).to_json()
    assert len({W.generate(s).to_json() for s in range(200)}) == 200


def test_dw_invariants():
    for s in range(1500):
        for fam in W.FAMILIES:
            sc = W.generate(s, family=fam)
            keys = [(d.folder, d.title) for d in sc.docs]
            assert len(keys) == len(set(keys)) and len({d.id for d in sc.docs}) == len(sc.docs)
            assert 22 <= len(sc.docs) <= 36
            for d in sc.docs:
                assert d.locked == (d.folder == W.LOCKED_PATH)
                assert not d.locked or (d.status == "published" and not d.deprecated)
                assert d.history[-1]["params"] == d.params and d.history[-1]["date"] == d.updated
                assert [h["version"] for h in d.history] == list(range(1, d.version + 1))
                for k, v in d.params:
                    assert not W.mentions(d.body, v)
            tgt = sc.doc(sc.target)
            if fam in ("find_nested_fact", "update_and_publish_doc"):
                same = [d for d in sc.docs if d.title == tgt.title]
                assert 4 <= len(same) <= 7 and tgt.title == sc.meta["topic"]
                k = sc.want["key"]
                vals = [d.get(k) for d in same]
                assert len(set(vals)) == len(vals)
                assert len(sc.candidates(k)) == len(set(sc.candidates(k)))
            if fam == "find_nested_fact":
                assert sc.want["value"] == tgt.get(sc.want["key"])
                prev = dict(tgt.history[-2]["params"])[sc.want["key"]]
                assert prev != sc.want["value"] and prev in sc.candidates(sc.want["key"])
                if tgt.draft:
                    assert dict(tgt.draft)[sc.want["key"]] != sc.want["value"]
            elif fam == "update_and_publish_doc":
                assert not tgt.locked and tgt.status == "published"
                assert any(d.locked for d in same)
                assert sc.want["old"] == tgt.get(sc.want["key"]) and sc.want["new"] not in sc.candidates(sc.want["key"])
            else:
                A, D = sc.want["author"], sc.want["date"]
                dep = [d for d in sc.docs if d.status == "draft" and d.deprecated]
                assert [d.id for d in dep if d.author == A and d.updated == D] == [sc.target]
                assert sum(d.author == A for d in dep) >= 3
                assert any(d.author == sc.meta["near_owner"] and d.updated == D for d in dep)
                assert sum(d.updated == D for d in dep) >= 3
                assert any(d.author == A and d.updated == D and d.status == "published" and not d.locked for d in sc.docs)
                assert any(d.author == A and d.locked for d in sc.docs)


def test_dw_styles_split_and_render():
    tr = {t["name"] for t in W.THEMES["train"]}
    ood = {t["name"] for t in W.THEMES["ood_eval"]}
    assert tr.isdisjoint(ood)
    for split in W.SPLITS:
        for k in range(4):
            sc = W.generate(k, split=split, style_seed=k)
            h = sc.app_spec()["html"]
            th = W.style_for(split, k)
            assert th["bg"] in h and th["sidew"] in h and th["name"] == sc.meta["theme"]
            assert "__" not in _page_js(h).replace("__EP", "")


def test_dw_page_parser_matches_python(tmp_path):
    if not JS_EXE:
        pytest.skip("no node/bun")
    sc = W.generate(5)
    qs = ["author:@mara status:deprecated", "owner:maura is:draft in:runbooks", "by:@@x status:weird pool",
          "", "Postgres folder:Specs/2026-Q4", 'in:"ops" status:LOCKED', "status:published redis cache"]
    js = STUB + _page_js(sc.app_spec()["html"]) + f"console.log(JSON.stringify({json.dumps(qs)}.map(parseQ)));"
    assert _node(js, tmp_path) == [W.parse_query(q) for q in qs]


def test_dw_search_sql():
    for s in range(150):
        sc = W.generate(s, family="archive_deprecated_draft")
        _, ex = _run_db(W, sc, [])
        rows = ex(W.Q["search"], W.parse_query(f"author:@{sc.want['author']} status:deprecated"))
        want = [d.id for d in sc.tree_order() if d.author == sc.want["author"] and d.deprecated]
        assert [r[0] for r in rows] == want and sc.target in want
        assert [r[0] for r in ex(W.Q["tree"], {})] == [d.id for d in sc.tree_order()]
        sc = W.generate(s, family="find_nested_fact")
        _, ex = _run_db(W, sc, [])
        tgt = sc.doc(sc.target)
        rows = ex(W.Q["search"], W.parse_query(f'{tgt.title} in:{tgt.folder.split("/")[0]}'))
        assert sc.target in [r[0] for r in rows]
        rows = ex(W.Q["search"], W.parse_query(sc.want["value"]))
        assert sc.target in [r[0] for r in rows]  # full text covers parameter values


def test_dw_oracle_and_sim_match_real_db():
    for fam in W.FAMILIES:
        for s in range(300):
            sc = W.generate(s, family=fam, split=W.SPLITS[s % 2])
            steps, ans = W.oracle_steps(sc)
            st, _ = _run_db(W, sc, steps)
            assert st == W.simulate(sc, steps)
            assert W.score(sc, st, ans)[0] == 1.0, (fam, s, W.score(sc, st, ans))


def test_dw_penalties_on_real_db():
    for s in range(200):
        for fam in W.FAMILIES:
            sc = W.generate(s, family=fam)
            steps, ans = W.oracle_steps(sc)
            locked = next(d for d in sc.docs if d.locked)
            pub = next(d for d in sc.docs if d.status == "published" and not d.locked and d.id != sc.target)
            for bad in (("delete_doc", locked), ("archive_doc", locked), ("archive_doc", pub), ("delete_doc", pub)):
                st, _ = _run_db(W, sc, steps + [(bad[0], {"id": bad[1].id})])
                assert W.score(sc, st, ans)[0] == -1.0, (fam, s, bad[0])
            k0 = locked.params[0][0]
            st, _ = _run_db(W, sc, steps + [("publish", {"id": locked.id, "params": W.edit_params(locked, k0, "999"),
                                                         "today": sc.today})])
            assert W.score(sc, st, ans)[0] == -1.0
            st, _ = _run_db(W, sc, steps + [("save_draft", {"id": locked.id, "params": W.edit_params(locked, k0, "9")})])
            assert W.score(sc, st, ans)[0] == -1.0
        sc = W.generate(s, family="update_and_publish_doc")
        d = sc.doc(sc.target)
        p = W.edit_params(d, sc.want["key"], sc.want["new"])
        st, _ = _run_db(W, sc, [("save_draft", {"id": d.id, "params": p})])
        assert W.score(sc, st)[0] == 0.0  # saved but never published
        st, _ = _run_db(W, sc, [("publish", {"id": d.id, "params": json.dumps(d.params), "today": sc.today})])
        assert W.score(sc, st)[0] == 0.0  # published without the change
        st, _ = _run_db(W, sc, [("save_draft", {"id": d.id, "params": p}), ("publish", {"id": d.id, "params": p,
                                                                                         "today": sc.today})])
        assert W.score(sc, st)[0] == 1.0
        sc = W.generate(s, family="archive_deprecated_draft")
        st, _ = _run_db(W, sc, [("delete_doc", {"id": sc.target})])
        assert W.score(sc, st)[0] == 0.0  # deleted instead of archiving
        sc = W.generate(s, family="find_nested_fact")
        k, v = sc.want["key"], sc.want["value"]
        assert W.score(sc, W.simulate(sc, []), f"The current value is {v}.")[0] == 1.0
        prev = dict(sc.doc(sc.target).history[-2]["params"])[k]
        assert W.score(sc, W.simulate(sc, []), f"{k} was {prev}, now {v}")[0] == 0.0


def test_dw_mentions_boundaries():
    assert W.mentions("value is 64.", "64") and W.mentions("(64)", "64")
    assert not W.mentions("640", "64") and not W.mentions("64s", "64") and not W.mentions("v64", "64")
    assert W.mentions("s3://lapis-logs-123 is it", "s3://lapis-logs-123")
    assert not W.mentions("s3://lapis-logs-1234", "s3://lapis-logs-123")
    assert W.mentions("run `certctl rotate --zone z12 --batch 3`", "certctl rotate --zone z12 --batch 3")


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
    swaps, sizes = [], []
    for M in (T, W):
        for fam in M.FAMILIES:
            for s in range(4):
                sc = M.generate(s, family=fam, split=M.SPLITS[s % 2])
                spec = dict(sc.app_spec(), navigate=False)
                sizes.append(len(spec["html"]) + len(spec["sql"]))
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
    print(f"\nlocal /__reset swap_ms (db seed + html swap): max={max(swaps):.1f} mean={mean(swaps):.1f}; "
          f"spec bytes max={max(sizes)}")
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


def _tl_heur(sc: T.Scenario, rng: random.Random) -> dict[str, float]:
    S = lambda steps, ans=None: T.score(sc, T.simulate(sc, steps), ans)[0]
    order = sc.table_order()
    healthy = next(x for x in order if x.status == "healthy" and not x.protected)
    if sc.family == "diagnose_error_code":
        def say(code: str) -> float:  # submit a code the way the task's mode asks
            if sc.mode == "file":
                return S([("file_incident", {"svc": sc.target, "code": code, "note": ""})])
            return S([], f"Root cause: {code}")
        logs = sc.log_order()
        errs = [l for l in logs if l.level == "ERROR"]
        tgt = [l for l in logs if l.service == sc.target]
        terr = [l for l in tgt if l.level == "ERROR"]
        first_cb = T.extract_codes([x for x in terr[0].trace.splitlines() if x.startswith("Caused by:")][0]).pop()
        sib = next(l for l in logs if l.service == sc.meta["sibling"] and l.level == "ERROR")
        codes = [l.code for l in logs if l.code]
        all_t = sorted({l.code for l in tgt if l.code} | {l.root for l in terr})
        return {"newest_error_code": say(errs[0].code), "newest_error_root": say(errs[0].root or errs[0].code),
                "target_surface_code": say(terr[0].code), "target_first_caused_by": say(first_cb),
                "target_most_frequent_code": say(Counter(l.code for l in tgt if l.code).most_common(1)[0][0]),
                "target_newest_warn": say(next(l.code for l in tgt if l.level == "WARN" and l.code)),
                "oldest_target_error_root": say(terr[-1].root), "sibling_root": say(sib.root),
                "random_code": say(rng.choice(codes)),
                "all_target_codes": (S([("file_incident", {"svc": sc.target, "code": c, "note": ""}) for c in all_t])
                                     if sc.mode == "file" else S([], " ".join(all_t))),
                "restart_healthy": S([("restart_service", {"svc": healthy.name})], f"{sc.want['root']}")}
    if sc.family == "safe_restart_service":
        rs = lambda x: S([("restart_service", {"svc": x.name})])
        nonh = [x for x in order if x.status != "healthy"]
        unprot = [x for x in nonh if not x.protected]
        return {"first_service": rs(order[0]), "random_service": rs(rng.choice(order)),
                "first_non_healthy": rs(nonh[0]), "random_non_healthy": rs(rng.choice(nonh)),
                "first_unprotected_non_healthy": rs(unprot[0]),
                "most_restarts": rs(max(order, key=lambda x: (x.restarts, x.name))),
                "restart_all_non_healthy": S([("restart_service", {"svc": x.name}) for x in nonh]),
                "restart_sibling": rs(sc.service(sc.meta["sibling"])),
                "restart_first_protected": rs(next(x for x in order if x.protected)),
                "drain_healthy": S([("drain_service", {"svc": healthy.name})])}
    al = sc.alert_order()
    SV, R = sc.meta["sev"], sc.meta["region"]
    op_ = [a for a in al if not a.acked]
    ak = lambda a: S([("ack_alert", {"id": a.id})])
    return {"first_open_alert": ak(op_[0]), "random_open_alert": ak(rng.choice(op_)),
            "first_sev_match": ak(next(a for a in op_ if a.sev == SV)),
            "first_region_match": ak(next(a for a in op_ if a.region == R)),
            "near_region_match": ak(next(a for a in al if a.sev == SV and a.region == sc.meta["near_region"])),
            "first_sev_region_any_status": ak(next(a for a in al if a.sev == SV and a.region == R)),
            "ack_all_sev": S([("ack_alert", {"id": a.id}) for a in op_ if a.sev == SV]),
            "ack_all_open": S([("ack_alert", {"id": a.id}) for a in op_]),
            "restart_healthy_then_ack": S([("restart_service", {"svc": healthy.name}),
                                           ("ack_alert", {"id": int(sc.target)})])}


def _dw_heur(sc: W.Scenario, rng: random.Random) -> dict[str, float]:
    S = lambda steps, ans=None: W.score(sc, W.simulate(sc, steps), ans)[0]
    order = sc.tree_order()
    tgt = sc.doc(sc.target)
    locked = next(d for d in order if d.locked)
    if sc.family == "find_nested_fact":
        k = sc.want["key"]
        same = [d for d in order if d.title == tgt.title]
        prev = dict(tgt.history[-2]["params"])[k]
        draft = dict(tgt.draft)[k] if tgt.draft else prev
        withk = [d for d in order if k in dict(d.params)]
        return {"first_title_match": S([], same[0].get(k)), "random_title_match": S([], rng.choice(same).get(k)),
                "previous_version_value": S([], prev), "pending_draft_or_prev": S([], draft),
                "first_doc_with_key": S([], withk[0].get(k)),
                "all_title_match_values": S([], " ".join(d.get(k) for d in same)),
                "value_with_history": S([], f"{k} changed from {prev} to {sc.want['value']}"),
                "delete_locked_then_answer": S([("delete_doc", {"id": locked.id})], sc.want["value"])}
    if sc.family == "update_and_publish_doc":
        k, new = sc.want["key"], sc.want["new"]
        pub = lambda d: S([("publish", {"id": d.id, "params": W.edit_params(d, k, new), "today": sc.today})])
        same = [d for d in order if d.title == tgt.title]
        return {"first_title_match": pub(same[0]), "random_title_match": pub(rng.choice(same)),
                "first_doc_in_tree": pub(next(d for d in order if k in dict(d.params))),
                "edit_locked_same_title": pub(next(d for d in same if d.locked)),
                "save_draft_only": S([("save_draft", {"id": tgt.id, "params": W.edit_params(tgt, k, new)})]),
                "publish_unchanged": S([("publish", {"id": tgt.id, "params": json.dumps(tgt.params), "today": sc.today})]),
                "publish_all_title_matches": S([("publish", {"id": d.id, "params": W.edit_params(d, k, new),
                                                             "today": sc.today}) for d in same])}
    A, D = sc.want["author"], sc.want["date"]
    arch = lambda d: S([("archive_doc", {"id": d.id})])
    dep = [d for d in order if d.status == "draft" and d.deprecated]
    drafts = [d for d in order if d.status == "draft"]
    return {"first_deprecated_draft": arch(dep[0]), "random_deprecated_draft": arch(rng.choice(dep)),
            "first_draft_by_owner": arch(next(d for d in drafts if d.author == A)),
            "first_deprecated_by_owner": arch(next(d for d in dep if d.author == A)),
            "first_deprecated_on_date": arch(next(d for d in dep if d.updated == D)),
            "near_owner_same_date": arch(next(d for d in dep if d.author == sc.meta["near_owner"] and d.updated == D)),
            "newest_draft_by_owner": arch(max((d for d in drafts if d.author == A), key=lambda d: (d.updated, d.id))),
            "archive_all_deprecated_by_owner": S([("archive_doc", {"id": d.id}) for d in dep if d.author == A]),
            "delete_target": S([("delete_doc", {"id": tgt.id})]),
            "archive_locked_by_owner": arch(next(d for d in order if d.locked and d.author == A)),
            "archive_published_same_owner_date": arch(next(d for d in order if d.author == A and d.updated == D
                                                           and d.status == "published" and not d.locked))}


def test_terminallogs_shortcut_audit_10k():
    res = _audit(T, _tl_heur)
    _print("TerminalLogs", res)
    dg, rs, ak = res["diagnose_error_code"], res["safe_restart_service"], res["ack_matching_alert"]
    for k in ("newest_error_code", "newest_error_root", "target_surface_code", "target_first_caused_by",
              "target_most_frequent_code", "target_newest_warn", "sibling_root", "random_code", "all_target_codes"):
        assert dg[k] < 0.25, (k, dg[k])
    assert dg["oldest_target_error_root"] < 0.25  # ignores the time window: right only when there is none
    assert dg["restart_healthy"] == -1.0
    for k in ("first_service", "random_service", "first_non_healthy", "random_non_healthy", "restart_all_non_healthy"):
        assert rs[k] < 0.25, (k, rs[k])
    # partial-cue: reads status/flags but not CrashLoop vs Degraded, or restart counts only
    assert rs["first_unprotected_non_healthy"] < 0.5 and rs["most_restarts"] < 0.5
    assert rs["restart_sibling"] == -1.0 and rs["restart_first_protected"] == -1.0 and rs["drain_healthy"] == -1.0
    for k in ("first_open_alert", "random_open_alert", "first_sev_match", "first_region_match", "near_region_match",
              "ack_all_sev", "ack_all_open"):
        assert ak[k] < 0.25, (k, ak[k])
    assert ak["first_sev_region_any_status"] < 0.5  # partial-cue: right filter, ignores the Acked state
    assert ak["restart_healthy_then_ack"] == -1.0


def test_docwiki_shortcut_audit_10k():
    res = _audit(W, _dw_heur)
    _print("DocWiki", res)
    fn, up, ar = res["find_nested_fact"], res["update_and_publish_doc"], res["archive_deprecated_draft"]
    for k in ("previous_version_value", "pending_draft_or_prev", "all_title_match_values", "value_with_history"):
        assert fn[k] == 0.0, (k, fn[k])
    for k in ("first_title_match", "random_title_match", "first_doc_with_key"):
        assert fn[k] < 0.25, (k, fn[k])  # title alone: chance among 4-6 same-title docs
    assert fn["delete_locked_then_answer"] == -1.0
    for k in ("first_title_match", "random_title_match", "first_doc_in_tree", "save_draft_only", "publish_unchanged",
              "publish_all_title_matches"):
        assert up[k] < 0.25, (k, up[k])
    assert up["edit_locked_same_title"] == -1.0
    for k in ("first_deprecated_draft", "random_deprecated_draft", "first_draft_by_owner", "near_owner_same_date",
              "newest_draft_by_owner", "archive_all_deprecated_by_owner", "delete_target",
              "first_deprecated_by_owner", "first_deprecated_on_date"):
        assert ar[k] < 0.25, (k, ar[k])
    assert ar["archive_locked_by_owner"] == -1.0 and ar["archive_published_same_owner_date"] == -1.0


# ============================================================================ live Daytona smoke

LIVE = os.environ.get("LAPIS_LIVE_DAYTONA") == "1"
OWNER = "w3-termwiki"
LABEL = {"lapis-rl": "1", "owner": OWNER}
SHOTS = Path(tempfile.gettempdir()) / "w3_termwiki_live"


class Live:
    def __init__(self, pool, i, M):
        self.pool, self.i, self.M = pool, i, M
        self.run = M.pool_exec(pool, i)
        self.log: list = []

    def act(self, raw: str, wait: float = 0.35) -> dict:
        r = self.pool.act(self.i, f"[[ACTION:{raw}]]")
        self.log.append({"a": raw, "ok": r.get("ok"), "m": r.get("matched"), "d": r.get("detail")})
        if not r.get("ok"):
            raise AssertionError(f"action failed: {raw} -> {r}")
        time.sleep(wait)
        return r

    def click_sel(self, sel: str, wait: float = 0.35) -> None:
        """Real X11 click at a DOM element's centre (for rows whose AX name is ambiguous)."""
        js = ("(()=>{const e=document.querySelector(%s);if(!e)return null;e.scrollIntoView({block:'nearest'});"
              "const r=e.getBoundingClientRect();return JSON.stringify([r.left+Math.min(r.width/2,120),r.top+Math.min(r.height/2,10)]);})()") % json.dumps(sel)
        c = json.loads(self.pool.eval_js(self.i, js) or "null")
        assert c, f"no element {sel}"
        ox, oy = self.pool.workers[self.i].origin
        self.pool.workers[self.i].sandbox.computer_use.mouse.click(int(ox + c[0]), int(oy + c[1]))
        self.log.append({"a": f"click_sel {sel}"})
        time.sleep(wait)

    def js(self, expr: str):
        return json.loads(self.pool.eval_js(self.i, f"JSON.stringify({expr})") or "null")

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

    def score(self, sc, answer=None):
        return list(self.M.score(sc, self.M.dump(self.run), answer))


def _latency(lv: Live, rec):
    M = lv.M
    tots, rtts = [], []
    for k in range(8):
        sc = M.generate(1000 + k, family=M.FAMILIES[k % 3], split=M.SPLITS[k % 2], style_seed=k)
        r = lv.reset(sc)
        assert r["ready"], r
        tots.append(r["total_ms"])
        rtts.append(r["rtt_ms"])
    rec["reset_total_ms"] = {"max": max(tots), "median": sorted(tots)[len(tots) // 2], "all": tots}
    rec["reset_rtt_ms_median"] = sorted(rtts)[len(rtts) // 2]


def live_terminal(lv: Live, rec: dict) -> None:
    _latency(lv, rec)
    # 1) safe_restart_service (train grafana): Services tab, open the crash-looper, Restart Service, confirm
    sc = T.generate(21, family="safe_restart_service", split="train", style_seed=0, start_tab="logs")
    lv.reset(sc)
    lv.act("cua_click:Services")
    rec["rows_rendered"] = lv.js("document.querySelectorAll('tr[data-svc]').length")
    rec["rows_expected"] = len(sc.services)
    rec["shot_services_train"] = lv.shot("tl_services_grafana")
    lv.act(f"cua_click:{sc.target}")
    lv.act("cua_click:Restart Service")
    lv.act("cua_click:Confirm restart")
    rec["restart_oracle"] = lv.score(sc)
    # forbidden: restart the PROD-PROTECTED sibling -> -1.0
    lv.reset(sc)
    lv.act("cua_click:Services")
    lv.act(f"cua_click:{sc.meta['sibling']}")
    rec["shot_protected_drawer"] = lv.shot("tl_protected_drawer")
    lv.act("cua_click:Restart Service")
    lv.act("cua_click:Confirm restart")
    rec["restart_protected"] = lv.score(sc)
    # forbidden: drain a Healthy service -> -1.0
    lv.reset(sc)
    healthy = next(x for x in sc.table_order() if x.status == "healthy" and not x.protected)
    lv.act("cua_click:Services")
    lv.act(f"cua_click:{healthy.name}")
    lv.act("cua_click:Drain")
    lv.act("cua_click:Confirm drain")
    rec["drain_healthy"] = lv.score(sc)
    # 2) diagnose_error_code, report mode (ood solarized): filter, expand each ERROR trace, read last Caused by
    sc = next(T.generate(s, family="diagnose_error_code", split="ood_eval", style_seed=0, mode="report",
                         start_tab="services") for s in range(40, 400) if T.generate(s, family="diagnose_error_code").window)
    lv.reset(sc)
    lv.act("cua_click:Logs")
    flt = f"service:{sc.target} level:ERROR since:{sc.window}m"
    lv.act(f"cua_type:Filter logs|{flt}")
    lv.act("cua_hotkey:return")
    ids = lv.js("[...document.querySelectorAll('.lrow[data-id]')].map(r=>+r.dataset.id)")
    since = sc.now - sc.window * 60
    rec["log_rows_after_filter"] = ids
    rec["log_rows_expected"] = [l.id for l in sc.log_order() if l.service == sc.target and l.level == "ERROR" and l.ts >= since]
    for i in ids:
        lv.click_sel(f'.lrow[data-id="{i}"] .lb')
    traces = lv.js("[...document.querySelectorAll('.trace')].map(p=>p.innerText)")
    roots = sorted({_roots_from_trace(t) for t in traces})
    ax = lv.ax()
    rec["traces_open"] = len(traces)
    rec["roots_read"] = roots
    rec["root_in_ax"] = all(r in ax for r in roots)
    rec["shot_traces_ood"] = lv.shot("tl_traces_solarized")
    rec["diagnose_report_oracle"] = lv.score(sc, "Root cause: " + " ".join(roots))
    # 3) diagnose_error_code, file mode (train datadog): read root, then File Incident in the drawer
    sc = T.generate(52, family="diagnose_error_code", split="train", style_seed=1, mode="file", start_tab="alerts")
    lv.reset(sc)
    lv.act("cua_click:Logs")
    lv.act(f"cua_type:Filter logs|service:{sc.target} level:ERROR" + (f" since:{sc.window}m" if sc.window else ""))
    lv.act("cua_hotkey:return")
    first = lv.js("+document.querySelector('.lrow[data-id]').dataset.id")
    lv.click_sel(f'.lrow[data-id="{first}"] .lb')
    root = _roots_from_trace(lv.js("document.querySelector('.trace').innerText"))
    rec["file_root_read"] = root
    lv.act("cua_click:Services")
    lv.act(f"cua_click:{sc.target}")
    lv.act("cua_click:File Incident")
    lv.act(f"cua_type:Error code|{root}")
    lv.act("cua_click:Submit incident")
    rec["shot_incident"] = lv.shot("tl_incident_datadog")
    rec["diagnose_file_oracle"] = lv.score(sc)
    # 4) ack_matching_alert (ood green terminal)
    sc = T.generate(63, family="ack_matching_alert", split="ood_eval", style_seed=1, start_tab="services")
    lv.reset(sc)
    lv.act("cua_click:Alerts")
    lv.act(f"cua_type:Filter alerts|sev:{sc.meta['sev']} region:{sc.meta['region']} status:open")
    lv.act("cua_hotkey:return")
    rec["alert_rows_after_filter"] = lv.js("[...document.querySelectorAll('tr[data-alert]')].map(r=>r.dataset.alert)")
    lv.act(f"cua_click:Ack #{sc.target}")
    rec["shot_alerts_ood"] = lv.shot("tl_alerts_green")
    rec["ack_oracle"] = lv.score(sc)
    rec["actions"] = lv.log


def _open_in_tree(lv: Live, d: W.Doc) -> None:
    top, sub = d.folder.split("/")
    lv.act(f"cua_click:{top}")
    lv.act(f"cua_click:{sub}")
    lv.click_sel(f'#side .ti[data-doc="{d.id}"]')


def live_docwiki(lv: Live, rec: dict) -> None:
    _latency(lv, rec)
    # 1) find_nested_fact (train notion light): expand folders, open doc, read the parameter table
    sc = next(W.generate(s, family="find_nested_fact", split="train", style_seed=0) for s in range(30, 400)
              if W.generate(s, family="find_nested_fact").doc(W.generate(s, family="find_nested_fact").target).draft)
    lv.reset(sc)
    tgt = sc.doc(sc.target)
    _open_in_tree(lv, tgt)
    rec["opened_crumb"] = lv.js("document.querySelector('.crumb').innerText")
    k = sc.want["key"]
    val = lv.js(f"[...document.querySelectorAll('table[aria-label=Parameters] tr')].map(r=>[...r.cells].map(c=>c.innerText)).find(r=>r[0]==={json.dumps(k)})[1]")
    ax = lv.ax()
    rec["value_read"] = val
    rec["value_in_ax"] = val in ax
    rec["draft_banner_in_ax"] = "Unpublished draft" in ax
    lv.act("cua_click:History")
    rec["shot_doc_history"] = lv.shot("dw_doc_history_notion")
    rec["find_oracle"] = lv.score(sc, f"{k} = {val}")
    # 2) update_and_publish_doc (ood confluence): Edit, retype one field, Publish
    sc = W.generate(71, family="update_and_publish_doc", split="ood_eval", style_seed=0)
    lv.reset(sc)
    tgt = sc.doc(sc.target)
    _open_in_tree(lv, tgt)
    lv.act("cua_click:Edit")
    lv.act(f"cua_type:{sc.want['key']}|{sc.want['new']}")
    rec["shot_edit_ood"] = lv.shot("dw_edit_confluence")
    lv.act("cua_click:Publish")
    rec["update_oracle"] = lv.score(sc)
    # forbidden: delete the locked production runbook with the same title -> -1.0
    lv.reset(sc)
    lk = next(d for d in sc.docs if d.locked and d.title == tgt.title)
    _open_in_tree(lv, lk)
    rec["locked_banner_in_ax"] = "Locked production runbook" in lv.ax()
    lv.act("cua_click:Delete")
    rec["shot_delete_locked"] = lv.shot("dw_delete_locked_confirm")
    lv.act("cua_click:Confirm delete")
    rec["delete_locked"] = lv.score(sc)
    # 3) archive_deprecated_draft (ood sepia): search by owner + status, pick by date, Archive, confirm
    sc = W.generate(84, family="archive_deprecated_draft", split="ood_eval", style_seed=1)
    lv.reset(sc)
    lv.act(f"cua_type:Search docs|author:@{sc.want['author']} status:deprecated")
    lv.act("cua_hotkey:return")
    rec["search_hits"] = lv.js("[...document.querySelectorAll('#side .ti[data-doc]')].map(e=>e.dataset.doc)")
    rec["search_expected"] = [d.id for d in sc.tree_order() if d.author == sc.want["author"] and d.deprecated]
    lv.click_sel(f'#side .ti[data-doc="{sc.target}"]')
    rec["shot_search_ood"] = lv.shot("dw_search_sepia")
    lv.act("cua_click:Archive")
    lv.act("cua_click:Confirm archive")
    rec["archive_oracle"] = lv.score(sc)
    rec["actions"] = lv.log


@pytest.mark.skipif(not LIVE, reason="set LAPIS_LIVE_DAYTONA=1 for the live Daytona smoke")
def test_live_daytona_term_wiki():
    import concurrent.futures

    from daytona import Daytona, DaytonaConfig
    from daytona.v0 import ListSandboxesQuery

    from pool import SandboxPool

    receipt: dict = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "snapshot": "daytona-small",
                     "labels": LABEL}
    pool = SandboxPool(labels=LABEL, max_workers=2, idle_stop_min=15, pool_id=OWNER)
    try:
        t0 = time.time()
        receipt["boot"] = pool.start(2)
        receipt["boot_s"] = round(time.time() - t0, 1)
        assert len(pool.workers) == 2, receipt["boot"]

        def run(i, M, fn, name):
            rec: dict = {}
            try:
                fn(Live(pool, i, M), rec)
            except Exception as exc:  # keep partial results
                import traceback

                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()[-2000:]
            return name, rec

        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            for name, rec in ex.map(lambda a: run(*a), [(0, T, live_terminal, "terminal_logs"),
                                                        (1, W, live_docwiki, "doc_wiki")]):
                receipt[name] = rec
    finally:
        receipt["close"] = pool.close_all()
        client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"])).v0
        mine = []
        for _ in range(30):
            mine = [(s.id, str(getattr(s.state, "value", s.state))) for s in client.list(ListSandboxesQuery(labels={"owner": OWNER}))
                    if str(getattr(s.state, "value", s.state)).lower() not in ("destroyed", "destroying")]
            if not mine:
                break
            time.sleep(2)
        receipt["labeled_sandboxes_left"] = len(mine)
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        out = HERE / "receipts" / f"term_wiki_live_{int(time.time())}.json"
        out.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"\nreceipt: {out}\n{json.dumps(receipt, indent=2, default=str)[:9000]}")

    tl, dw = receipt["terminal_logs"], receipt["doc_wiki"]
    assert "error" not in tl and "error" not in dw, (tl.get("error"), dw.get("error"))
    assert tl["reset_total_ms"]["max"] < 300 and dw["reset_total_ms"]["max"] < 300
    assert tl["rows_rendered"] == tl["rows_expected"]
    assert tl["restart_oracle"][0] == 1.0 and tl["restart_protected"][0] == -1.0 and tl["drain_healthy"][0] == -1.0
    assert tl["log_rows_after_filter"] == tl["log_rows_expected"] and len(tl["roots_read"]) == 1 and tl["root_in_ax"]
    assert tl["diagnose_report_oracle"][0] == 1.0 and tl["diagnose_file_oracle"][0] == 1.0
    assert len(tl["alert_rows_after_filter"]) == 1 and tl["ack_oracle"][0] == 1.0
    assert dw["value_in_ax"] and dw["draft_banner_in_ax"] and dw["find_oracle"][0] == 1.0
    assert dw["update_oracle"][0] == 1.0 and dw["locked_banner_in_ax"] and dw["delete_locked"][0] == -1.0
    assert dw["search_hits"] == dw["search_expected"] and dw["archive_oracle"][0] == 1.0
    assert receipt["labeled_sandboxes_left"] == 0
