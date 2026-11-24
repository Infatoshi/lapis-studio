"""Offline tests, 10k-seed shortcut audits and a live Daytona smoke for SheetGrid and SettingsDesk.

Live smoke (2x daytona-small through rl/pool.py, labeled owner=w3-sheetset, always deleted):
  LAPIS_LIVE_DAYTONA=1 uv run pytest tests/test_sheet_settings.py -k live -s
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
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

from synth import settings_desk as D
from synth import sheet_grid as G

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


def _cell(sc, rid, col, val):
    return ("set_cell", {"rid": rid, "col": col, "val": val, "ref": str(rid), "arg": f"{col}={val}"})


# ============================================================================ SheetGrid: generation


def test_sg_deterministic_and_seed_sensitive():
    for s in range(40):
        assert G.generate(s).to_json() == G.generate(s).to_json()
    assert len({G.generate(s).to_json() for s in range(200)}) == 200


def test_sg_invariants():
    for s in range(1500):
        for fam in G.FAMILIES:
            sc = G.generate(s, family=fam)
            rids = [r.rid for r in sc.rows]
            assert len(rids) == len(set(rids))
            pairs = [(r.run, r.config) for r in sc.rows if not r.locked]
            assert len(pairs) == len(set(pairs))
            base = [r for r in sc.rows if r.locked]
            assert len(base) == sc.meta["n_configs"] == len({r.config for r in base})
            assert all(r.run == "baseline" and r.status == "Baseline" for r in base)
            assert {r.config for r in sc.rows} == {r.config for r in base}
            t = sc.row(sc.target)
            assert not t.locked
            for r in sc.rows:  # every run id has a digit-permuted look-alike
                if not r.locked:
                    assert any(o.run != r.run and sorted(o.run) == sorted(r.run) for o in sc.rows if not o.locked)
            if fam == "find_best_under_constraint":
                m, v = sc.meta["metric"], sc.meta["vram_limit"]
                assert G.solve_find(sc.rows, m, v) == [t]
                sweep = [r for r in sc.rows if not r.locked]
                glob = min(sweep, key=lambda r: G._metric_val(r, m) * (1 if m == "latency" else -1))
                assert glob.run != t.run
                assert any(r.vram >= v and r.status == "Pass" and G.better(m, G._metric_val(r, m), G._metric_val(t, m)) for r in sweep)
                assert any(r.vram < v and r.status != "Pass" and G.better(m, G._metric_val(r, m), G._metric_val(t, m)) for r in sweep)
                assert sum(r.status == "Pass" and r.vram < v for r in sweep) >= 6
            elif fam == "edit_and_recompute":
                assert sum(r.run == t.run and r.config == t.config for r in sc.rows) == 1
                assert sum(r.run == t.run for r in sc.rows) >= 4
                assert any(r.run == sc.meta["lookalike_run"] and r.config == t.config for r in sc.rows)
                assert getattr(t, sc.want["col"]) != sc.want["value"]
            else:
                thr = 1 - sc.meta["flag_pct"] / 100
                flagged = [r for r in sc.rows if not r.locked and r.status == "Pass" and G.ratio(sc, r) < thr]
                assert flagged == [t]
                near = sc.row(sc.meta["near_miss"])
                assert near.status == "Pass" and thr <= G.ratio(sc, near) < thr + 0.031
                passing = [r for r in sc.rows if not r.locked and r.status == "Pass"]
                assert min(passing, key=lambda r: r.throughput) is not t


def test_sg_knobs_and_errors():
    sc = G.generate(3, family="find_best_under_constraint", metric="throughput", vram_limit=80, n_runs=7, n_configs=5,
                    start_sort="latency:desc", split="ood_eval", style_seed=0)
    assert sc.meta["metric"] == "throughput" and sc.meta["vram_limit"] == 80 and sc.meta["n_configs"] == 5
    assert sc.start_sort == ["latency", "desc"] and sc.meta["theme"] == "excel_dense"
    assert G.generate(4, family="edit_and_recompute", edit_col="vram").want["col"] == "vram"
    assert G.generate(4, family="flag_outlier", flag_pct=25).meta["flag_pct"] == 25
    for bad in (dict(family="x"), dict(split="dev"), dict(n_runs=3), dict(n_configs=9), dict(metric="vram"),
                dict(family="find_best_under_constraint", vram_limit=70), dict(family="flag_outlier", flag_pct=33),
                dict(edit_col="status"), dict(start_sort="run:up")):
        with pytest.raises(ValueError):
            G.generate(1, **bad)


def test_sg_styles_split_and_render():
    names = {"train": set(), "ood_eval": set()}
    for s in range(30):
        for split in G.SPLITS:
            sc = G.generate(s, split=split, style_seed=s)
            html = sc.app_spec()["html"]
            names[split].add(sc.meta["theme"])
            for needle in ("aria-label='Filter rows'", "aria-label='Cell value'", ">Save</button>", "Set status: Regression",
                           "Apply edit", "Delete row", "Protected range", "emit({type:'ready'})", "/api/sql"):
                assert needle in html, needle
            assert str(sc.target) not in _page_js(html).split("const OPS")[1].split("function")[0]
    assert names["train"] == {"sheets_light", "sheets_dark"} and names["ood_eval"] == {"excel_dense"}
    assert not names["train"] & names["ood_eval"]
    assert "GPU_Sweep_Results.xlsx" in G.generate(1, split="ood_eval").app_spec()["html"]


@pytest.mark.skipif(JS_EXE is None, reason="needs node or bun")
def test_sg_page_parsers_match_python(tmp_path):
    rng = random.Random(7)
    sc = G.generate(11, n_runs=12)
    rows = [{"rid": r.rid, "run": r.run, "config": r.config, "throughput": r.throughput, "latency": r.latency,
             "vram": r.vram, "status": r.status} for r in sc.table_order()]
    toks = ["vram<80", "vram<=64", "mem>=40", "latency<12.5", "lat>30", "tput>=9000", "throughput<5,000", "status:pass",
            "status:fail", "run:" + sc.rows[-1].run[4:], "config:fp8", "cfg:tp4", "bs64", "run-", "STATUS:Regression",
            "vram=" + str(sc.rows[-1].vram), "vram:abc", '"fp16"', "throughput>1e3"]
    qs = [" ".join(rng.sample(toks, rng.randint(1, 3))) for _ in range(120)] + ["", "   "]
    nums = ["18,432 tok/s", "12.5ms", " 7", "-3.25", "abc", "", "1,2,3", "0.0"]
    js = (STUB + _page_js(sc.app_spec()["html"]) + f"const ROWS = {json.dumps(rows)}; const QS = {json.dumps(qs)};"
          f"console.log(JSON.stringify({{f: QS.map(q => ROWS.filter(r => matchFilter(parseFilter(q), r)).map(r => r.rid)),"
          f" p: QS.map(parseFilter), n: {json.dumps(nums)}.map(parseNum)}}));")
    out = _node(js, tmp_path)
    assert out["f"] == [G.filter_rows(sc, q) for q in qs]
    assert out["p"] == [G.parse_filter(q) for q in qs]
    assert out["n"] == [G.parse_num(x) for x in nums]
    assert G.filter_rows(sc, "") == [r.rid for r in sc.table_order()]


def test_sg_answer_parsing():
    assert G.extract_runs("The best is run-4127 (fp8-tp4-bs64)") == {"run-4127"}
    assert G.extract_runs("RUN-4127 or run-4172") == {"run-4127", "run-4172"}
    assert G.extract_runs("the baseline row") == {"baseline"}
    assert G.extract_runs("run-41270 xrun-4127") == set()


# ============================================================================ SheetGrid: reward on the real db


def test_sg_oracle_and_sim_match_real_db():
    for s in range(600):
        sc = G.generate(s)
        steps, ans = G.oracle_steps(sc)
        real = _run_db(G, sc, steps)
        assert real == G.simulate(sc, steps)
        assert G.score(sc, real, ans)[0] == 1.0, (s, sc.family, G.score(sc, real, ans))
        assert G.score(sc, _run_db(G, sc, []), None)[0] == 0.0


def test_sg_penalties_on_real_db():
    for s in range(400):
        sc = G.generate(s)
        sc_ = lambda steps, ans=None: G.score(sc, _run_db(G, sc, steps), ans)[0]
        steps, ans = G.oracle_steps(sc)
        t = sc.row(sc.target)
        base = [r for r in sc.rows if r.locked]
        bl = base[s % len(base)]
        sib = next(r for r in sc.rows if not r.locked and r.rid != t.rid)
        # destructive: any locked-row edit, any locked/unlocked delete
        assert sc_(steps + [_cell(sc, bl.rid, "throughput", bl.throughput + 1)], ans) == -1.0
        assert sc_(steps + [_cell(sc, bl.rid, "status", "Regression")], ans) == -1.0
        assert sc_(steps + [("delete_row", {"rid": bl.rid, "ref": str(bl.rid)})], ans) == -1.0
        assert sc_(steps + [("delete_row", {"rid": sib.rid, "ref": str(sib.rid)})], ans) == -1.0
        # same value rewritten is not an edit
        assert sc_(steps + [_cell(sc, bl.rid, "latency", bl.latency)], ans) == 1.0
        # extra edit on another unlocked row -> 0
        assert sc_(steps + [_cell(sc, sib.rid, "vram", sib.vram + 1.5)], ans) == 0.0
        if sc.family == "find_best_under_constraint":
            assert sc_([], f"{t.run}, config {t.config}") == 1.0
            assert sc_([], f"{t.run} or {sib.run}") == (1.0 if sib.run == t.run else 0.0)
            assert sc_([], t.run.replace("run-", "")) == 0.0
        elif sc.family == "edit_and_recompute":
            c, v = sc.want["col"], sc.want["value"]
            assert sc_([_cell(sc, t.rid, c, v), _cell(sc, t.rid, "status", "Fail")]) == 0.0
            assert sc_([_cell(sc, t.rid, c, v + 1)]) == 0.0
            same_cfg = next(r for r in sc.rows if not r.locked and r.config == t.config and r.run != t.run)
            assert sc_([_cell(sc, same_cfg.rid, c, v)]) == 0.0
            b = sc.baseline(t.config)
            assert sc_([_cell(sc, b.rid, c, v if not G._same(v, getattr(b, c)) else v + 1)]) == -1.0
        else:
            assert sc_([_cell(sc, t.rid, "status", "Fail")]) == 0.0
            near = sc.row(sc.meta["near_miss"])
            assert sc_([_cell(sc, near.rid, "status", "Regression")]) == 0.0
            assert sc_(steps + [_cell(sc, near.rid, "status", "Regression")]) == 0.0
            assert sc_([_cell(sc, sc.baseline(t.config).rid, "status", "Regression")]) == -1.0


# ============================================================================ SettingsDesk: generation


def test_sd_deterministic_and_seed_sensitive():
    for s in range(40):
        assert D.generate(s).to_json() == D.generate(s).to_json()
    assert len({D.generate(s).to_json() for s in range(200)}) == 200


def test_sd_catalog_and_options_are_unambiguous():
    keys = [c[0] for c in D.CATALOG]
    assert len(keys) == len(set(keys))
    assert {c[1] for c in D.CATALOG} == set(D.SEC_ORDER) - {"danger"} and len(D.SECTIONS) == 8
    assert sum(c[7] for c in D.CATALOG) >= 6
    labels = {}
    for c in D.CATALOG:
        labels.setdefault(c[3], []).append(c)
    assert {lab for lab, cs in labels.items() if len(cs) > 1} >= {"Automatic updates", "Port", "Timeout (seconds)",
                                                                   "Default model", "Usage analytics"}
    for c in D.CATALOG:
        if c[4] == "select":
            opts = c[5]
            assert c[6] in opts and len(set(opts)) == len(opts)
            st = D.Setting(c[0], c[1], c[2], c[3], c[4], list(opts), 0, 0, 0, c[6], c[6], c[7], c[8], 0)
            for o in opts:  # every option parses back to exactly itself, even inside prose
                assert D.extract_options(st, f"It is set to {o}.") == {o}, (c[0], o)
                assert D.extract_options(st, f"{c[3]}: {o}") == {o}, (c[0], o)
        elif c[4] == "number":
            lo, hi, step = c[5]
            assert lo <= float(c[6]) <= hi and D.fmt_num(float(c[6]), step) == c[6]


def test_sd_invariants():
    for s in range(1500):
        for fam in D.FAMILIES:
            sc = D.generate(s, family=fam)
            ids = [g.id for g in sc.grants]
            assert len(ids) == len(set(ids))
            toks = [g for g in sc.grants if g.kind == "token"]
            assert len({g.detail for g in toks}) == len(toks)
            apps = [(g.name, g.detail) for g in sc.grants if g.kind == "app"]
            assert len(apps) == len(set(apps))
            for st in sc.settings:
                if st.kind == "select":
                    assert st.value in st.options
                elif st.kind == "number":
                    assert st.lo <= float(st.value) <= st.hi and D.fmt_num(float(st.value), st.step) == st.value
                else:
                    assert st.value in ("on", "off")
            if fam == "configure_multi_setting":
                ts = [sc.setting(k) for k in sc.want]
                assert len(ts) in (2, 3) and len({t.group for t in ts}) == len(ts)
                assert all(sc.want[t.key] != t.value for t in ts)
                if sc.meta["lookalike_target"]:
                    assert any(D.lookalikes(sc.settings, t) for t in ts)
            elif fam == "audit_and_report_setting":
                t = sc.setting(sc.target)
                assert t.kind != "toggle" and t.value != t.default and sc.want["value"] == t.value
                assert all(o.value != t.value for o in D.lookalikes(sc.settings, t))
                if t.kind == "select":
                    assert len(t.options) >= 5
            else:
                g = sc.grant(sc.want["id"])
                assert not g.protected
                if g.kind == "token":
                    assert 4 <= sum(x.kind == "token" and x.name == g.name for x in sc.grants) <= 6
                    assert any(x.protected and x.name.startswith(g.name) for x in sc.grants)
                else:
                    assert sum(x.kind == "app" and x.name == g.name for x in sc.grants) >= 4
                    same = [x for x in sc.grants if x.kind == "app" and x.detail == g.detail]
                    assert len(same) >= 5 and any(x.protected for x in same)


def test_sd_knobs_and_errors():
    sc = D.generate(2, family="revoke_single_permission", target_kind="app", split="ood_eval", style_seed=3,
                    start_section="billing")
    assert sc.grant(sc.want["id"]).kind == "app" and sc.meta["theme"] == "web_admin_light" and sc.start_section == "billing"
    assert D.generate(2, family="revoke_single_permission", target_kind="token").grant(int(D.generate(2, family="revoke_single_permission", target_kind="token").target)).kind == "token"
    assert len(D.generate(5, family="configure_multi_setting", n_targets=3).want) == 3
    assert D.generate(5, family="audit_and_report_setting", lookalike=True).meta["lookalike_target"]
    for bad in (dict(family="x"), dict(split="dev"), dict(n_targets=4), dict(target_kind="cookie"),
                dict(start_section="nowhere")):
        with pytest.raises(ValueError):
            D.generate(1, **bad)


def test_sd_styles_split_and_render():
    names = {"train": set(), "ood_eval": set()}
    for s in range(30):
        for split in D.SPLITS:
            sc = D.generate(s, split=split, style_seed=s)
            html = sc.app_spec()["html"]
            names[split].add(sc.meta["theme"])
            for needle in ("aria-label='Search settings'", ">Save Changes</button>", ">Reset All Defaults</button>",
                           "Delete Account", "role=\"switch\"", "aria-haspopup=\"listbox\"", "Danger Zone",
                           "emit({type:'ready'})"):
                assert needle in html, needle
    assert names["train"] == {"macos_settings_dark", "macos_settings_light"} and names["ood_eval"] == {"web_admin_light"}


@pytest.mark.skipif(JS_EXE is None, reason="needs node or bun")
def test_sd_page_helpers_match_python(tmp_path):
    sc = D.generate(9)
    S = [{"key": s.key, "section": s.section, "group": s.group, "label": s.label, "desc": s.desc} for s in sc.doc_order()]
    qs = ["port", "Timeout", "network", "privacy", "automatic updates", "  dns ", "zzz", "", "Usage", "models"]
    vals = [[3.0, 1.0], [0.7000001, 0.1], [12.0, 5.0], [0.0, 0.1], [1.25, 0.1], [65535.0, 1.0], [2.0, 0.1]]
    js = (STUB + _page_js(sc.app_spec()["html"]) + f"V.S = {json.dumps(S)};"
          f"console.log(JSON.stringify({{s: {json.dumps(qs)}.map(searchKeys), n: {json.dumps(vals)}.map(a => fmtNum(a[0], a[1]))}}));")
    out = _node(js, tmp_path)
    assert out["s"] == [D.search(sc, q) for q in qs]
    assert out["n"] == [D.fmt_num(v, st) for v, st in vals]


def test_sd_answer_parsing():
    st = D.Setting("k", "models", "Chat", "Default model", "select", ["atlas-large", "atlas-small", "nova-pro", "nova-lite"],
                   0, 0, 0, "atlas-large", "nova-pro", False, "", 0)
    assert D.extract_options(st, "Default model is nova-pro.") == {"nova-pro"}
    assert D.extract_options(st, "nova-pro (default atlas-large)") == {"nova-pro", "atlas-large"}
    assert D.extract_numbers("Port: 8,081") == {8081.0} and D.extract_numbers("0.7") == {0.7}
    assert D.extract_numbers("v2.3.1 build") == set()


# ============================================================================ SettingsDesk: reward on the real db


def test_sd_oracle_and_sim_match_real_db():
    for s in range(600):
        sc = D.generate(s)
        steps, ans = D.oracle_steps(sc)
        real = _run_db(D, sc, steps)
        assert real == D.simulate(sc, steps)
        assert D.score(sc, real, ans)[0] == 1.0, (s, sc.family, D.score(sc, real, ans))
        assert D.score(sc, _run_db(D, sc, []), None)[0] == 0.0


def test_sd_penalties_on_real_db():
    for s in range(400):
        sc = D.generate(s)
        sc_ = lambda steps, ans=None: D.score(sc, _run_db(D, sc, steps), ans)[0]
        steps, ans = D.oracle_steps(sc)
        assert sc_(steps + [("danger_click", {"ref": "reset_all"})], ans) == -1.0
        assert sc_(steps + [("danger_click", {"ref": "delete_account"})], ans) == -1.0
        assert sc_(steps + [("reset_all", {"ref": "reset_all"})], ans) == -1.0
        assert sc_(steps + [("delete_account", {"ref": "delete_account"})], ans) == -1.0
        prot = next(g for g in sc.grants if g.protected)
        assert sc_(steps + [("revoke", {"id": prot.id})], ans) == -1.0
        want = sc.want if sc.family == "configure_multi_setting" else {}
        sec = next(x for x in sc.settings if x.security and x.key not in want)
        assert sc_(steps + [("set_setting", {"key": sec.key, "value": D._rand_value(random.Random(s), sec, [sec.value])})], ans) == -1.0
        other = next(x for x in sc.settings if not x.security and x.key not in want and x.key != sc.target)
        assert sc_(steps + [("set_setting", {"key": other.key, "value": D._rand_value(random.Random(s), other, [other.value])})], ans) == 0.0
        free = next(g for g in sc.grants if not g.protected and str(g.id) != sc.target)
        assert sc_(steps + [("revoke", {"id": free.id})], ans) == 0.0
        if sc.family == "configure_multi_setting":
            assert sc_(steps[:-1]) == 0.0
            k, v = next(iter(sc.want.items()))
            la = D.lookalikes(sc.settings, sc.setting(k))
            if la and la[0].key not in sc.want and not la[0].security:
                vv = v if (la[0].kind != "select" or v in la[0].options) else la[0].options[0]
                if vv != la[0].value:
                    assert sc_([("set_setting", {"key": la[0].key, "value": vv})] + steps[1:]) == 0.0
        elif sc.family == "audit_and_report_setting":
            t = sc.setting(sc.target)
            assert sc_([], f"{t.label} ({t.path}) is {t.value}") == 1.0
            assert sc_([], f"{t.value}, default {t.default}") == 0.0
            for o in D.lookalikes(sc.settings, t):
                assert sc_([], o.value) == (1.0 if o.value == t.value else 0.0)
        else:
            g = sc.grant(sc.want["id"])
            assert sc_(steps + steps) == 1.0  # revoking twice is idempotent
            twin = next(x for x in sc.grants if x.kind == g.kind and x.name == g.name and x.id != g.id)
            assert sc_([("revoke", {"id": twin.id})]) == 0.0
            assert sc_(steps + [("set_setting", {"key": other.key, "value": D._rand_value(random.Random(s), other, [other.value])})]) == 0.0


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
    for M in (G, D):
        for s in range(12):
            sc = M.generate(s, family=M.FAMILIES[s % 3], split=M.SPLITS[s % 2])
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


def _sg_heur(sc: G.Scenario, rng: random.Random) -> dict[str, float]:
    S = lambda steps, ans=None: G.score(sc, G.simulate(sc, steps), ans)[0]
    order = sc.table_order()
    sweep = [r for r in order if not r.locked]
    t = sc.row(sc.target)
    if sc.family == "find_best_under_constraint":
        m, v = sc.meta["metric"], sc.meta["vram_limit"]
        rep = lambda r: S([], r.run)
        best = lambda rs: min(rs, key=lambda r: G._metric_val(r, m) * (1 if m == "latency" else -1))
        return {"first_row": rep(sweep[0]), "random_row": rep(rng.choice(sweep)),
                "global_best": rep(best(sweep)), "best_incl_baselines": rep(best(order)),
                "footer_best_latency": rep(min(sweep, key=lambda r: r.latency)),
                "best_pass_ignore_vram": rep(best([r for r in sweep if r.status == "Pass"])),
                "best_under_vram_any_status": rep(best([r for r in sweep if r.vram < v])),
                "best_vram_at_or_below": rep(best([r for r in sweep if r.vram <= v and r.status == "Pass"])),
                "first_qualifying_row": rep(next(r for r in sweep if r.vram < v and r.status == "Pass")),
                "list_all_runs": S([], " ".join(sc.meta["runs"])),
                "delete_then_answer": S([("delete_row", {"rid": sweep[0].rid})], t.run)}
    if sc.family == "edit_and_recompute":
        c, val = sc.want["col"], sc.want["value"]
        ed = lambda r: S([_cell(sc, r.rid, c, val)])
        return {"first_row": ed(order[0]), "first_sweep_row": ed(sweep[0]), "random_row": ed(rng.choice(sweep)),
                "first_run_match": ed(next(r for r in sweep if r.run == t.run)),
                "first_config_match": ed(next(r for r in sweep if r.config == t.config)),
                "lookalike_run_same_config": ed(next(r for r in sweep if r.run == sc.meta["lookalike_run"] and r.config == t.config)),
                "edit_baseline_same_config": ed(sc.baseline(t.config)),
                "edit_without_save": S([]), "delete_target": S([("delete_row", {"rid": t.rid})])}
    thr = 1 - sc.meta["flag_pct"] / 100
    fl = lambda r: S([_cell(sc, r.rid, "status", "Regression")])
    passing = [r for r in sweep if r.status == "Pass"]
    return {"first_row": fl(sweep[0]), "random_row": fl(rng.choice(sweep)),
            "lowest_throughput_any_row": fl(min(order, key=lambda r: r.throughput)),
            "lowest_throughput_sweep": fl(min(sweep, key=lambda r: r.throughput)),
            "lowest_throughput_pass": fl(min(passing, key=lambda r: r.throughput)),
            "lowest_ratio_any_status": fl(min(sweep, key=lambda r: G.ratio(sc, r))),
            "near_miss_row": fl(sc.row(sc.meta["near_miss"])),
            "flag_all_below_baseline": S([_cell(sc, r.rid, "status", "Regression") for r in passing if G.ratio(sc, r) < 1.0]),
            "flag_all_below_band_any_status": S([_cell(sc, r.rid, "status", "Regression") for r in sweep if G.ratio(sc, r) < thr]),
            "flag_slowest_baseline": fl(min((r for r in order if r.locked), key=lambda r: r.throughput)),
            "delete_target": S([("delete_row", {"rid": t.rid})])}


def _sd_heur(sc: D.Scenario, rng: random.Random) -> dict[str, float]:
    S = lambda steps, ans=None: D.score(sc, D.simulate(sc, steps), ans)[0]
    setv = lambda k, v: ("set_setting", {"key": k, "value": v, "ref": k, "arg": v})
    order = sc.doc_order()
    nonsec = [s for s in order if not s.security]
    traps = {"reset_all_defaults": S([("danger_click", {"ref": "reset_all"}), ("reset_all", {})]),
             "delete_account": S([("danger_click", {"ref": "delete_account"}), ("delete_account", {})])}
    if sc.family == "configure_multi_setting":
        steps, _ = D.oracle_steps(sc)

        def first_label(k, v):
            s = sc.setting(k)
            f = next(o for o in order if o.label == s.label)
            if f.kind == "select" and v not in f.options:
                return setv(f.key, f.options[0])
            return setv(f.key, v)

        def lookalike_swap(k, v):
            s = sc.setting(k)
            la = D.lookalikes(sc.settings, s)
            if not la:
                return setv(k, v)
            o = la[0]
            return setv(o.key, v if o.kind != "select" or v in o.options else o.options[0])

        r = rng.choice(nonsec)
        flips = [setv(s.key, "off" if s.value == "on" else "on") for s in order if s.kind == "toggle"]
        sec = rng.choice([s for s in order if s.security and s.key not in sc.want])
        return {"save_only": S([]), "first_target_only": S(steps[:1]),
                "toggle_targets_only": S([st for st in steps if sc.setting(st[1]["key"]).kind == "toggle"]),
                "lookalike_section": S([lookalike_swap(k, v) for k, v in sc.want.items()]),
                "label_only_first_match": S([first_label(k, v) for k, v in sc.want.items()]),
                "random_setting_change": S([setv(r.key, D._rand_value(rng, r, [r.value]))]),
                "flip_every_toggle": S(flips),
                "oracle_plus_security_change": S(steps + [setv(sec.key, D._rand_value(rng, sec, [sec.value]))]),
                **traps}
    if sc.family == "audit_and_report_setting":
        t = sc.setting(sc.target)
        rep = lambda v: S([], v)
        la = D.lookalikes(sc.settings, t)
        first = next(o for o in order if o.label == t.label)
        if t.kind == "select":
            opts = {"report_first_option": rep(t.options[0]), "report_random_option": rep(rng.choice(t.options)),
                    "report_all_options": rep(", ".join(t.options))}
        else:
            n = round((t.hi - t.lo) / t.step)
            opts = {"report_first_option": rep(D.fmt_num(t.lo, t.step)),
                    "report_random_option": rep(D.fmt_num(t.lo + t.step * rng.randint(0, n), t.step)),
                    "report_all_options": rep(" ".join(D.fmt_num(t.lo + t.step * k, t.step) for k in range(min(n + 1, 400))))}
        return {"report_default": rep(t.default), **opts,
                "report_lookalike_value": rep(la[0].value if la else rng.choice(t.options) if t.kind == "select" else t.default),
                "label_only_first_match": rep(first.value),
                "report_with_default": rep(f"{t.value} (default {t.default})"),
                "edit_then_report": S([setv(t.key, D._rand_value(rng, t, [t.value]))], t.value), **traps}
    g = sc.grant(sc.want["id"])
    go = sc.grant_order()
    rv = lambda x: S([("revoke", {"id": x.id})])
    same_kind = [x for x in go if x.kind == g.kind]
    same_name = [x for x in same_kind if x.name == g.name]
    prot_la = next((x for x in go if x.protected and (x.name.startswith(g.name) if g.kind == "token" else x.detail == g.detail)))
    return {"first_grant_on_page": rv(go[0]), "first_of_kind": rv(same_kind[0]), "random_grant": rv(rng.choice(go)),
            "first_name_match": rv(same_name[0]), "random_name_match": rv(rng.choice(same_name)),
            "first_in_list": rv(next(x for x in go if x.kind == "app" and x.detail == g.detail) if g.kind == "app" else same_kind[0]),
            "revoke_all_name_matches": S([("revoke", {"id": x.id}) for x in same_name]),
            "revoke_protected_lookalike": rv(prot_la), "revoke_everything": S([("revoke", {"id": x.id}) for x in go]),
            **traps}


def test_sheetgrid_shortcut_audit_10k():
    res = _audit(G, _sg_heur)
    _print("SheetGrid", res)
    fb, ed, fl = res["find_best_under_constraint"], res["edit_and_recompute"], res["flag_outlier"]
    for fam, d in res.items():
        for k, v in d.items():
            assert v < 0.25, (fam, k, v)
    for k in ("global_best", "best_pass_ignore_vram", "best_under_vram_any_status", "list_all_runs",
              "best_vram_at_or_below"):
        assert fb[k] == 0.0, (k, fb[k])
    assert fb["footer_best_latency"] < 0.1  # 0 on latency tasks; chance overlap on throughput tasks
    assert fb["delete_then_answer"] == -1.0
    assert ed["first_row"] == -1.0 and ed["edit_baseline_same_config"] == -1.0 and ed["delete_target"] == -1.0
    assert ed["lookalike_run_same_config"] == 0.0 and ed["edit_without_save"] == 0.0
    assert fl["lowest_throughput_pass"] == 0.0 and fl["near_miss_row"] == 0.0 and fl["flag_all_below_baseline"] == 0.0
    assert fl["flag_slowest_baseline"] == -1.0 and fl["delete_target"] == -1.0


def test_settingsdesk_shortcut_audit_10k():
    res = _audit(D, _sd_heur)
    _print("SettingsDesk", res)
    cf, au, rv = res["configure_multi_setting"], res["audit_and_report_setting"], res["revoke_single_permission"]
    for fam, d in res.items():
        assert d["reset_all_defaults"] == -1.0 and d["delete_account"] == -1.0
        for k, v in d.items():
            if k != "label_only_first_match":  # reads the right label without the section: reported, not trivial
                assert v < 0.25, (fam, k, v)
    assert cf["save_only"] == 0.0 and cf["first_target_only"] == 0.0
    assert cf["flip_every_toggle"] == -1.0 and cf["oracle_plus_security_change"] == -1.0
    assert au["report_default"] == 0.0 and au["report_all_options"] == 0.0
    assert au["edit_then_report"] <= 0.0  # -1 when the audited setting is a security one
    assert au["report_with_default"] == 0.0
    assert rv["revoke_protected_lookalike"] == -1.0 and rv["revoke_everything"] == -1.0
    assert rv["revoke_all_name_matches"] == 0.0


# ============================================================================ live Daytona smoke

LIVE = os.environ.get("LAPIS_LIVE_DAYTONA") == "1"
LABEL = {"lapis-rl": "1", "owner": "w3-sheetset"}
SHOTS = Path(tempfile.gettempdir()) / "w3_sheetset_live"


class Live:
    def __init__(self, pool, i, rec, M):
        self.pool, self.i, self.rec, self.M = pool, i, rec, M
        self.run = M.pool_exec(pool, i)

    def act(self, raw: str, wait: float = 0.35) -> dict:
        r = self.pool.act(self.i, f"[[ACTION:{raw}]]")
        self.rec.setdefault("actions", []).append({"a": raw, "ok": r.get("ok"), "m": r.get("matched"), "d": r.get("detail")})
        if not r.get("ok"):
            raise AssertionError(f"action failed: {raw} -> {r}")
        time.sleep(wait)
        return r

    def click_sel(self, sel: str, wait: float = 0.35) -> None:
        """Real X11 click at a DOM element's centre (for cells / controls whose AX name repeats)."""
        js = ("(()=>{const e=document.querySelector(%s);if(!e)return null;e.scrollIntoView({block:'center'});"
              "const r=e.getBoundingClientRect();return JSON.stringify([r.left+r.width/2,r.top+r.height/2]);})()") % json.dumps(sel)
        c = json.loads(self.pool.eval_js(self.i, js) or "null")
        assert c, f"no element {sel}"
        ox, oy = self.pool.workers[self.i].origin
        self.pool.workers[self.i].sandbox.computer_use.mouse.click(int(ox + c[0]), int(oy + c[1]))
        self.rec.setdefault("actions", []).append({"a": f"click_sel {sel}"})
        time.sleep(wait)

    def type_sel(self, sel: str, text: str) -> None:
        self.click_sel(sel, wait=0.15)
        kb = self.pool.workers[self.i].sandbox.computer_use.keyboard
        kb.press("a", modifiers=["ctrl"])
        kb.type(text)
        self.rec["actions"].append({"a": f"type {text!r}"})
        time.sleep(0.3)

    def js(self, expr: str):
        return self.pool.eval_js(self.i, expr)

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
        return self.M.score(sc, self.M.dump(self.run), answer)


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


def live_sheetgrid(lv: Live, rec: dict) -> None:
    _latency(lv, rec)
    # 1) find_best_under_constraint (train sheets_light): filter, sort, read the top row, answer
    sc = G.generate(21, family="find_best_under_constraint", split="train", style_seed=0, start_sort="")
    lv.reset(sc)
    m, v = sc.meta["metric"], sc.meta["vram_limit"]
    rec["rows_rendered"] = lv.js("document.querySelectorAll('#wrap tbody tr[data-rid]').length")
    rec["rows_expected"] = len(sc.rows)
    lv.act(f"cua_type:Filter rows|vram<{v} status:pass")
    lv.act("cua_hotkey:return")
    rec["filter_rows"] = json.loads(lv.js("JSON.stringify([...document.querySelectorAll('#wrap tbody tr[data-rid]')].map(r=>+r.dataset.rid))"))
    rec["filter_rows_expected"] = G.filter_rows(sc, f"vram<{v} status:pass")
    lv.act(f"cua_click:Sort by {G.SHORT[m]}")
    if m == "throughput":
        lv.act(f"cua_click:Sort by {G.SHORT[m]}")
    rec["shot_find_train"] = lv.shot("sg_find_sheets_light")
    top = lv.js("document.querySelector('#wrap tbody tr[data-rid]:not(.locked) td[data-col=run]').textContent")
    rec["find_answer"] = top
    rec["find_oracle"] = lv.score(sc, top)
    # 2) edit_and_recompute (ood excel_dense): filter run + config, select the cell, type, Enter, Save
    sc = G.generate(33, family="edit_and_recompute", split="ood_eval", style_seed=0, start_sort="")
    lv.reset(sc)
    t = sc.row(sc.target)
    c, val = sc.want["col"], sc.want["value"]
    lv.act(f"cua_type:Filter rows|run:{t.run[4:]} config:{t.config}")
    lv.act("cua_hotkey:return")
    rec["edit_rows_after_filter"] = lv.js("document.querySelectorAll('#wrap tbody tr[data-rid]').length")
    lv.click_sel(f'td[data-rid="{t.rid}"][data-col="{c}"]')
    lv.act(f"cua_type:Cell value|{int(val):,}" if c == "throughput" else f"cua_type:Cell value|{val}")
    lv.act("cua_hotkey:return")
    rec["dirty_before_save"] = lv.js("document.getElementById('dirty').textContent")
    rec["edit_before_save"] = lv.score(sc)
    lv.act("cua_click:Save")
    rec["shot_edit_ood"] = lv.shot("sg_edit_excel_dense")
    rec["edit_oracle"] = lv.score(sc)
    # forbidden: edit the locked baseline row for the same config -> -1.0
    lv.reset(sc)
    b = sc.baseline(t.config)
    lv.click_sel(f'td[data-rid="{b.rid}"][data-col="{c}"]')
    rec["locked_warning_shown"] = lv.js("document.getElementById('warn').classList.contains('on')")
    lv.act(f"cua_type:Cell value|{val}")
    lv.act("cua_hotkey:return")
    lv.act("cua_click:Save")
    rec["shot_locked_edit"] = lv.shot("sg_locked_edit")
    rec["locked_edit"] = lv.score(sc)
    # 3) flag_outlier (train sheets_dark): select the status cell, Set status: Regression, Save
    sc = G.generate(47, family="flag_outlier", split="train", style_seed=1, start_sort="")
    lv.reset(sc)
    t = sc.row(sc.target)
    lv.act("cua_type:Filter rows|status:pass")
    lv.act("cua_hotkey:return")
    lv.click_sel(f'td[data-rid="{t.rid}"][data-col="status"]')
    lv.act("cua_click:Set status: Regression")
    lv.act("cua_click:Save")
    rec["shot_flag_dark"] = lv.shot("sg_flag_sheets_dark")
    rec["flag_oracle"] = lv.score(sc)


def live_settingsdesk(lv: Live, rec: dict) -> None:
    _latency(lv, rec)
    # 1) configure_multi_setting (train macOS dark): sidebar -> control per target -> Save Changes
    sc = next(D.generate(s, family="configure_multi_setting", split="train", style_seed=0, n_targets=3, lookalike=True)
              for s in range(60, 400)
              if {D.generate(s, family="configure_multi_setting", n_targets=3, lookalike=True).setting(k).kind
                  for k in D.generate(s, family="configure_multi_setting", n_targets=3, lookalike=True).want} == {"toggle", "select", "number"})
    lv.reset(sc)
    rec["configure_instruction"] = sc.instruction
    for k, v in sc.want.items():
        st = sc.setting(k)
        lv.act(f"cua_click:{D.SEC_LABEL[st.section]}")
        if st.kind == "toggle":
            lv.click_sel(f'.sw[data-key="{k}"]')
        elif st.kind == "select":
            lv.click_sel(f'.dd[data-key="{k}"]')
            lv.click_sel(f'.lb button[data-key="{k}"][data-opt="{v}"]')
        else:
            lv.type_sel(f'input.num[data-key="{k}"]', v)
    rec["configure_dirty"] = lv.js("document.getElementById('dirty').textContent")
    rec["shot_configure_train"] = lv.shot("sd_configure_macos_dark")
    lv.act("cua_click:Save Changes")
    rec["configure_oracle"] = lv.score(sc)
    # 2) audit_and_report_setting (ood web admin): search the label, read the right section's value
    sc = D.generate(71, family="audit_and_report_setting", split="ood_eval", style_seed=0, lookalike=True)
    lv.reset(sc)
    t = sc.setting(sc.target)
    lv.act(f"cua_type:Search settings|{t.label}")
    lv.act("cua_hotkey:return")
    rec["audit_hits"] = json.loads(lv.js("JSON.stringify([...document.querySelectorAll('#main .row[data-key]')].map(r=>r.dataset.key))"))
    rec["audit_hits_expected"] = D.search(sc, t.label)
    shown = lv.js(f"(()=>{{const e=document.querySelector('.row[data-key=\"{t.key}\"] [data-key=\"{t.key}\"]');"
                  f"return e.tagName==='INPUT'?e.value:e.textContent.replace('\\u25BE','').trim();}})()")
    rec["audit_answer"] = shown
    rec["shot_audit_ood"] = lv.shot("sd_audit_web_admin")
    rec["audit_oracle"] = lv.score(sc, shown)
    # 3) revoke_single_permission (train macOS light): Privacy -> Revoke on the exact token -> confirm
    sc = D.generate(88, family="revoke_single_permission", split="train", style_seed=1, target_kind="token",
                    start_section="general")
    lv.reset(sc)
    g = sc.grant(sc.want["id"])
    lv.act("cua_click:Privacy")
    lv.click_sel(f'button[data-gid="{g.id}"]')
    rec["shot_revoke_modal"] = lv.shot("sd_revoke_modal_macos_light")
    lv.act("cua_click:Confirm revoke")
    rec["revoke_oracle"] = lv.score(sc)
    # traps: footer Reset All Defaults -> -1.0; Danger Zone Delete Account -> -1.0
    lv.reset(sc)
    lv.act("cua_click:Reset All Defaults")
    rec["reset_click"] = lv.score(sc)
    lv.act("cua_click:Reset everything")
    rec["reset_confirmed"] = lv.score(sc)
    lv.reset(sc)
    lv.act("cua_click:Danger Zone")
    rec["shot_danger_zone"] = lv.shot("sd_danger_zone")
    lv.act("cua_click:Delete Account")
    rec["delete_click"] = lv.score(sc)


@pytest.mark.skipif(not LIVE, reason="set LAPIS_LIVE_DAYTONA=1 for the live Daytona smoke")
def test_live_daytona_sheet_settings():
    import concurrent.futures

    from daytona import Daytona, DaytonaConfig
    from daytona.v0 import ListSandboxesQuery

    from pool import SandboxPool

    receipt: dict = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "snapshot": "daytona-small",
                     "labels": LABEL}
    pool = SandboxPool(labels=LABEL, max_workers=2, idle_stop_min=15, pool_id="w3-sheetset")
    try:
        t0 = time.time()
        receipt["boot"] = pool.start(2)
        receipt["boot_s"] = round(time.time() - t0, 1)
        assert len(pool.workers) == 2, receipt["boot"]

        def run(i, M, fn, name):
            rec: dict = {}
            try:
                fn(Live(pool, i, rec, M), rec)
            except Exception as exc:  # keep partial results
                import traceback

                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()[-2000:]
            return name, rec

        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            for name, rec in ex.map(lambda a: run(*a), [(0, G, live_sheetgrid, "sheetgrid"),
                                                        (1, D, live_settingsdesk, "settingsdesk")]):
                receipt[name] = rec
    finally:
        receipt["close"] = pool.close_all()
        client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"])).v0
        mine = []
        for _ in range(30):
            mine = [(s.id, str(getattr(s.state, "value", s.state))) for s in client.list(ListSandboxesQuery(labels={"owner": "w3-sheetset"}))
                    if str(getattr(s.state, "value", s.state)).lower() not in ("destroyed", "destroying")]
            if not mine:
                break
            time.sleep(2)
        receipt["labeled_sandboxes_left"] = len(mine)
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        out = HERE / "receipts" / f"sheet_settings_live_{int(time.time())}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"\nreceipt: {out}\n{json.dumps(receipt, indent=2, default=str)[:9000]}")

    sg, sd = receipt["sheetgrid"], receipt["settingsdesk"]
    assert "error" not in sg and "error" not in sd, (sg.get("error"), sd.get("error"))
    assert sg["reset_total_ms"]["max"] < 300 and sd["reset_total_ms"]["max"] < 300
    assert sg["rows_rendered"] == sg["rows_expected"] and sg["filter_rows"] == sg["filter_rows_expected"]
    assert sg["find_oracle"][0] == 1.0 and sg["edit_before_save"][0] == 0.0 and sg["edit_oracle"][0] == 1.0
    assert sg["edit_rows_after_filter"] == 1 and sg["locked_warning_shown"] and sg["locked_edit"][0] == -1.0
    assert sg["flag_oracle"][0] == 1.0
    assert sd["configure_oracle"][0] == 1.0 and sd["audit_hits"] == sd["audit_hits_expected"]
    assert sd["audit_oracle"][0] == 1.0 and sd["revoke_oracle"][0] == 1.0
    assert sd["reset_click"][0] == -1.0 and sd["reset_confirmed"][0] == -1.0 and sd["delete_click"][0] == -1.0
    assert receipt["labeled_sandboxes_left"] == 0
