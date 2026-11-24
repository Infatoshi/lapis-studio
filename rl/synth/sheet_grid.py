"""SheetGrid: a seeded spreadsheet / GPU-sweep benchmark workbench with SQLite-verified rewards.

Same wiring as ``code_review`` and ``cal_board``: ``app_spec()`` -> ``POST /__reset`` on the pool
control server, an in-memory SQLite db seeded from ``seed_sql``, every mutating statement defined once
in ``OPS`` and embedded in the page (``apply_op`` runs the same statements offline). Ground truth
stays in Python; the page never sees which row is the answer.

UI: a sheet titled ``gpu_sweep.sheet`` with columns Run, Config, Throughput (tok/s), Latency (ms),
VRAM (GiB), Status. Locked baseline rows (one per config) are frozen at the top with a lock icon.
A filter bar takes ``vram<80 status:pass config:fp8 run:4127 throughput>=9000`` or free text;
header buttons ``Sort by <col>`` toggle ascending/descending; clicking a cell selects it into a
formula bar (``Cell value`` box, ``Apply edit``, ``Set status: ...`` buttons, ``Delete row``).
Edits are staged (``N unsaved changes``) until ``Save``. Editing a baseline shows a Sheets-style
protected-range warning but is allowed, like a warn-only protected range. A live summary footer
shows sweep means and the global ``Best latency`` row (which ignores filters: a decoy).

Families (``score`` is pure over scenario, db dump and final answer):
  * find_best_under_constraint: among Pass rows with VRAM < V, report the Run ID with the lowest
    latency (or highest throughput). Decoys in other runs: better but VRAM >= V (some exactly V),
    better but Fail/Running, better locked baseline. Edits are 0, deletes / locked edits -1.
  * edit_and_recompute: find the row by Run ID + Config among digit-permuted Run IDs, sibling
    configs of the same run and the same config in other runs / the baseline; set one numeric cell
    and Save. Touching any locked row, or deleting any row, is -1.0; touching other rows is 0.
  * flag_outlier: exactly one Pass row has throughput more than P% below its config's locked
    baseline; set its Status to Regression and Save. Near-miss rows sit just inside the band.

Styles: ``split="train"`` uses a Google-Sheets-like light theme and its dark variant;
``split="ood_eval"`` uses a dense Excel-like theme with column letters, smaller rows, a green
ribbon and different fonts.
"""

from __future__ import annotations

import html as _html
import json
import random
import re
from dataclasses import asdict, dataclass, field
from string import Template
from typing import Any, Callable, Mapping, Sequence

from synth.code_review import BASE_JS, LOG_SQL, apply_op, fresh_db, insert_sql, pool_exec, sqlite_exec  # noqa: F401

FAMILIES = ("find_best_under_constraint", "edit_and_recompute", "flag_outlier")
SPLITS = ("train", "ood_eval")
NUM_COLS = ("throughput", "latency", "vram")
COL_LABEL = {"run": "Run", "config": "Config", "throughput": "Throughput (tok/s)", "latency": "Latency (ms)",
             "vram": "VRAM (GiB)", "status": "Status"}
SHORT = {"throughput": "Throughput", "latency": "Latency", "vram": "VRAM", "status": "Status"}
STATUSES = ("Pass", "Fail", "Running", "Regression")
DTYPES = ("bf16", "fp8", "fp16", "int8", "fp4")
TPS = (1, 2, 4, 8)
BSS = (16, 32, 64, 128, 256)
VRAM_LIMITS = (40, 48, 64, 80, 96)
FLAG_PCTS = (10, 15, 20, 25)

THEMES: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        dict(name="sheets_light", bg="#ffffff", fg="#202124", muted="#5f6368", panel="#f9fbfd", head="#f8f9fa",
             grid="#e2e3e3", accent="#1a73e8", sel="#e8f0fe", lock="#fef7e0", lock_fg="#7a5b00", pend="#fff4ce",
             bad="#d93025", ok="#188038", font="Arial, 'Helvetica Neue', sans-serif", fs="13px", rowh="27px",
             radius="4px", ribbon="#ffffff", ribbon_fg="#202124", letters=False),
        dict(name="sheets_dark", bg="#1f1f1f", fg="#e3e3e3", muted="#a8a8a8", panel="#262626", head="#2b2b2b",
             grid="#3a3a3a", accent="#8ab4f8", sel="#2d3a52", lock="#3b3320", lock_fg="#f6d77a", pend="#4a3f1c",
             bad="#f28b82", ok="#81c995", font="Arial, 'Helvetica Neue', sans-serif", fs="13px", rowh="27px",
             radius="4px", ribbon="#262626", ribbon_fg="#e3e3e3", letters=False),
    ),
    "ood_eval": (
        dict(name="excel_dense", bg="#ffffff", fg="#000000", muted="#605e5c", panel="#f3f2f1", head="#e6e6e6",
             grid="#d4d4d4", accent="#217346", sel="#cfe8d9", lock="#ededed", lock_fg="#3b3b3b", pend="#fff2cc",
             bad="#c50f1f", ok="#107c10", font="Calibri, 'Segoe UI', Carlito, sans-serif", fs="12px", rowh="20px",
             radius="0px", ribbon="#217346", ribbon_fg="#ffffff", letters=True),
    ),
}


def style_for(split: str, style_seed: int) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pool = THEMES[split]
    return pool[style_seed % len(pool)]


# ----------------------------------------------------------------------------- data model


@dataclass
class Row:
    rid: int
    run: str
    config: str
    throughput: float
    latency: float
    vram: float
    status: str
    locked: bool
    pos: int


@dataclass
class Scenario:
    seed: int
    family: str
    split: str
    style_seed: int
    rows: list[Row]
    target: int  # rid
    want: dict[str, Any]  # find: run; edit: col,value; flag: status
    instruction: str
    start_sort: list[Any]  # [col, dir] or []
    meta: dict[str, Any] = field(default_factory=dict)

    def row(self, rid: int) -> Row:
        return next(r for r in self.rows if r.rid == rid)

    def baseline(self, config: str) -> Row:
        return next(r for r in self.rows if r.locked and r.config == config)

    def table_order(self) -> list[Row]:
        """Rows in on-screen order with no sort applied: frozen baselines, then sheet order."""
        return sorted(self.rows, key=lambda r: (not r.locked, r.pos))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def app_spec(self) -> dict[str, Any]:
        return {"name": f"sheet_grid/{self.family}/s{self.seed}", "html": render_html(self), "sql": seed_sql(self),
                "state": {}}


def better(metric: str, a: float, b: float) -> bool:
    return a < b if metric == "latency" else a > b


def ratio(sc: Scenario, r: Row) -> float:
    return r.throughput / sc.baseline(r.config).throughput


# ----------------------------------------------------------------------------- generation (pure)


def _run_ids(rng: random.Random, n: int) -> list[str]:
    """Every Run ID has at least one digit-permuted look-alike (run-4127 / run-4172 / run-1427)."""
    sizes = [3] * (n % 2) + [2] * (n // 2 - n % 2)
    while sizes.count(2) >= 3 and rng.random() < 0.3:  # 2+2+2 -> 3+3
        for _ in range(3):
            sizes.remove(2)
        sizes += [3, 3]
    out: list[str] = []
    used: set[str] = set()
    for k in sizes:
        while True:
            digits = rng.sample("123456789", 4)
            if "".join(sorted(digits)) not in used:
                used.add("".join(sorted(digits)))
                break
        perms: set[str] = set()
        while len(perms) < k:
            p = digits[:]
            rng.shuffle(p)
            perms.add("".join(p))
        out += [f"run-{p}" for p in sorted(perms)]
    rng.shuffle(out)
    return out


def _r1(x: float) -> float:
    return round(x, 1)


def _build(rng: random.Random, n_runs: int, n_configs: int) -> tuple[list[Row], dict[str, dict[str, float]]]:
    configs: list[str] = []
    while len(configs) < n_configs:
        c = f"{rng.choice(DTYPES)}-tp{rng.choice(TPS)}-bs{rng.choice(BSS)}"
        if c not in configs:
            configs.append(c)
    base: dict[str, dict[str, float]] = {}
    tputs = rng.sample(range(1500, 24000, 250), n_configs)
    for c, t in zip(configs, tputs):
        base[c] = {"throughput": float(t), "latency": _r1(rng.uniform(4.0, 90.0)), "vram": _r1(rng.uniform(20.0, 140.0))}
    rows: list[Row] = []
    rid = iter(rng.sample(range(100, 999), 200))
    for c in configs:
        b = base[c]
        rows.append(Row(next(rid), "baseline", c, b["throughput"], b["latency"], b["vram"], "Baseline", True, 0))
    for run in _run_ids(rng, n_runs):
        for c in rng.sample(configs, rng.randint(2, min(4, n_configs))):
            b = base[c]
            st = rng.choices(STATUSES, weights=[74, 11, 8, 7])[0]
            if st in ("Fail", "Regression"):
                tp = b["throughput"] * rng.uniform(0.05, 0.6)
                lat = b["latency"] * rng.uniform(1.2, 3.0)
            else:
                tp = b["throughput"] * rng.uniform(0.9, 1.15)
                lat = b["latency"] * rng.uniform(0.85, 1.15)
            rows.append(Row(next(rid), run, c, float(round(tp)), _r1(lat), _r1(b["vram"] * rng.uniform(0.9, 1.1)), st,
                            False, 0))
    order = list(range(len(rows)))
    rng.shuffle(order)
    for r, p in zip(rows, order):
        r.pos = p
    return rows, base


def _metric_val(r: Row, metric: str) -> float:
    return r.latency if metric == "latency" else r.throughput


def _set_metric(r: Row, metric: str, v: float) -> None:
    if metric == "latency":
        r.latency = _r1(max(0.5, v))
    else:
        r.throughput = float(round(v))


def solve_find(rows: Sequence[Row], metric: str, vlim: float) -> list[Row]:
    """Rows tied for best among Pass sweep rows with VRAM strictly below ``vlim``."""
    cand = [r for r in rows if not r.locked and r.status == "Pass" and r.vram < vlim]
    if not cand:
        return []
    best = min(_metric_val(r, metric) for r in cand) if metric == "latency" else max(_metric_val(r, metric) for r in cand)
    return [r for r in cand if _metric_val(r, metric) == best]


def generate(seed: int, family: str | None = None, split: str = "train", style_seed: int | None = None,
             n_runs: int | None = None, n_configs: int | None = None, metric: str | None = None,
             vram_limit: int | None = None, flag_pct: int | None = None, edit_col: str | None = None,
             start_sort: str | None = None) -> Scenario:
    """Knobs: family, split (train|ood_eval), style_seed, n_runs 6..16, n_configs 4..6,
    metric (find: latency|throughput), vram_limit (find: one of VRAM_LIMITS), flag_pct (flag: one of
    FLAG_PCTS), edit_col (edit: throughput|latency|vram), start_sort ('' | 'col:asc' | 'col:desc')."""
    rng = random.Random(f"sheetgrid:{seed}")
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    n_runs = n_runs if n_runs is not None else rng.randint(8, 13)
    if not 6 <= n_runs <= 16:
        raise ValueError("n_runs must be 6..16")
    n_configs = n_configs if n_configs is not None else rng.randint(4, 6)
    if not 4 <= n_configs <= 6:
        raise ValueError("n_configs must be 4..6")
    metric = metric or rng.choice(("latency", "throughput"))
    if metric not in ("latency", "throughput"):
        raise ValueError("metric must be latency|throughput")
    if edit_col is not None and edit_col not in NUM_COLS:
        raise ValueError("edit_col must be throughput|latency|vram")
    style_seed = style_seed if style_seed is not None else rng.randrange(1000)
    if start_sort is None:
        start_sort = rng.choice(["", "", "run:asc", "throughput:desc", "latency:asc", "vram:asc", "config:asc"])
    sort = start_sort.split(":") if start_sort else []
    if sort and (sort[0] not in ("run", "config", *NUM_COLS, "status") or sort[1] not in ("asc", "desc")):
        raise ValueError("start_sort must be '' or '<col>:asc|desc'")

    for _attempt in range(200):
        rows, base = _build(rng, n_runs, n_configs)
        sweep = [r for r in rows if not r.locked]
        meta: dict[str, Any] = {"n_runs": n_runs, "n_configs": n_configs, "n_rows": len(rows),
                                "theme": style_for(split, style_seed)["name"], "ood_style": split == "ood_eval"}
        if family == "find_best_under_constraint":
            vlim = vram_limit if vram_limit is not None else rng.choice(VRAM_LIMITS)
            if vlim not in VRAM_LIMITS:
                raise ValueError(f"vram_limit must be one of {VRAM_LIMITS}")
            cand = [r for r in sweep if r.status == "Pass" and r.vram < vlim]
            if len({r.run for r in cand}) < 3 or len(cand) < 6:
                continue
            t = rng.choice(cand)
            others = [r for r in cand if r.run != t.run]
            bo = (min if metric == "latency" else max)(_metric_val(r, metric) for r in others)
            _set_metric(t, metric, bo * (rng.uniform(0.86, 0.97) if metric == "latency" else rng.uniform(1.03, 1.14)))
            tv = _metric_val(t, metric)
            for s in (r for r in sweep if r.run == t.run and r is not t):
                if not better(metric, tv, _metric_val(s, metric)):
                    _set_metric(s, metric, tv * (rng.uniform(1.05, 1.3) if metric == "latency" else rng.uniform(0.7, 0.95)))
            pool_ = [r for r in sweep if r.run != t.run]
            if len(pool_) < 4:
                continue
            decoys = rng.sample(pool_, 4)
            f_ = (lambda: rng.uniform(0.7, 0.95)) if metric == "latency" else (lambda: rng.uniform(1.05, 1.4))
            d1 = decoys[0]  # passes, faster, but over the VRAM limit
            d1.status, d1.vram = "Pass", _r1(vlim + rng.uniform(0.1, 9.0))
            _set_metric(d1, metric, tv * f_())
            d2 = decoys[1]  # under the limit, faster, but not passing
            d2.status, d2.vram = rng.choice(["Fail", "Running"]), _r1(vlim * rng.uniform(0.5, 0.97))
            _set_metric(d2, metric, tv * f_())
            d3 = decoys[2]  # second best that does qualify
            d3.status, d3.vram = "Pass", _r1(vlim * rng.uniform(0.4, 0.97))
            _set_metric(d3, metric, tv * (rng.uniform(1.01, 1.08) if metric == "latency" else rng.uniform(0.93, 0.99)))
            d4 = decoys[3]  # exactly at the limit: "below" excludes it
            d4.status, d4.vram = "Pass", float(vlim)
            _set_metric(d4, metric, tv * f_())
            bl = rng.choice([r for r in rows if r.locked])  # a locked baseline that would win
            bl.vram = _r1(vlim * rng.uniform(0.5, 0.95))
            _set_metric(bl, metric, tv * f_())
            sol = solve_find(rows, metric, vlim)
            glob = sorted(sweep, key=lambda r: _metric_val(r, metric) * (1 if metric == "latency" else -1))
            if len(sol) != 1 or sol[0] is not t or glob[0].run == t.run:
                continue
            if sum(r.status == "Pass" and r.vram < vlim for r in sweep) < 6:
                continue
            if len({_metric_val(r, metric) for r in sweep}) < len(sweep) * 0.8:
                continue
            target, want = t.rid, {"run": t.run}
            word = "lowest latency" if metric == "latency" else "highest throughput"
            instruction = (f"In SheetGrid, among the sweep rows with Status Pass and VRAM below {vlim} GiB, find the "
                           f"one with the {word}. Report its Run ID as your final answer. Do not edit the sheet.")
            meta.update(metric=metric, vram_limit=vlim, decoys=[d.rid for d in decoys], baseline_decoy=bl.rid)
        elif family == "edit_and_recompute":
            t = rng.choice([r for r in sweep if r.status in ("Pass", "Running")])
            digits = sorted(t.run[4:])
            looks = sorted({r.run for r in sweep if r.run != t.run and sorted(r.run[4:]) == digits})
            if not looks:
                continue
            look = rng.choice(looks)
            if not any(r.run == look and r.config == t.config for r in sweep):
                src = next(r for r in sweep if r.run == look)
                b = base[t.config]
                rows.append(Row(max(r.rid for r in rows) + 1, look, t.config, float(round(b["throughput"] * rng.uniform(0.9, 1.1))),
                                _r1(b["latency"] * rng.uniform(0.9, 1.1)), _r1(b["vram"] * rng.uniform(0.92, 1.08)),
                                src.status if src.status in ("Pass", "Running") else "Pass", False, len(rows)))
                sweep = [r for r in rows if not r.locked]
            have = {r.config for r in sweep if r.run == t.run}
            for c in [c for c in base if c not in have]:
                b = base[c]  # the target run sweeps >= 4 configs, so its Run ID alone is ambiguous
                rows.append(Row(max(r.rid for r in rows) + 1, t.run, c, float(round(b["throughput"] * rng.uniform(0.9, 1.1))),
                                _r1(b["latency"] * rng.uniform(0.9, 1.1)), _r1(b["vram"] * rng.uniform(0.92, 1.08)),
                                "Pass", False, len(rows)))
            sweep = [r for r in rows if not r.locked]
            if sum(r.run == t.run for r in sweep) < 4:
                continue
            order = list(range(len(rows)))  # appended rows must not sit after the original ones
            rng.shuffle(order)
            for r, p in zip(rows, order):
                r.pos = p
            col = edit_col or rng.choice(NUM_COLS)
            cur = getattr(t, col)
            for _ in range(50):
                nv = float(round(cur * rng.uniform(0.75, 1.3))) if col == "throughput" else _r1(cur * rng.uniform(0.75, 1.3))
                if nv > 0 and all(not _same(nv, getattr(r, col)) for r in rows):  # unique in its column
                    break
            target, want = t.rid, {"col": col, "value": nv}
            shown = f"{int(nv):,}" if col == "throughput" else f"{nv:.1f}"
            instruction = (f"In SheetGrid, update the row for run {t.run} with config {t.config}: set its "
                           f"{COL_LABEL[col]} to {shown}, then Save the sheet. Do not modify the locked baseline rows "
                           f"or any other row.")
            meta.update(edit_col=col, lookalike_run=look)
        else:
            pct = flag_pct if flag_pct is not None else rng.choice(FLAG_PCTS)
            if pct not in FLAG_PCTS:
                raise ValueError(f"flag_pct must be one of {FLAG_PCTS}")
            thr = 1 - pct / 100
            cand = [r for r in sweep if r.status == "Pass"]
            if len(cand) < 6:
                continue
            t, near = rng.sample(cand, 2)
            for r in sweep:  # every other passing / running row stays inside the band
                if r.status in ("Pass", "Running") and r is not t and r.throughput / base[r.config]["throughput"] < thr + 0.03:
                    r.throughput = float(round(base[r.config]["throughput"] * rng.uniform(thr + 0.04, 1.12)))
            near.throughput = float(round(base[near.config]["throughput"] * rng.uniform(thr + 0.012, thr + 0.03)))
            t.throughput = float(round(base[t.config]["throughput"] * rng.uniform(thr - 0.22, thr - 0.04)))
            passing = [r for r in sweep if r.status == "Pass"]
            low = min(passing, key=lambda r: r.throughput)
            flagged = [r for r in passing if r.throughput / base[r.config]["throughput"] < thr]
            if flagged != [t] or low is t:
                continue
            target, want = t.rid, {"status": "Regression"}
            instruction = (f"In SheetGrid, exactly one passing sweep row has a throughput more than {pct}% below the "
                           f"locked baseline row for the same config. Set that row's Status to Regression and Save. "
                           f"Do not edit the baseline rows or any other row.")
            meta.update(flag_pct=pct, near_miss=near.rid)
        break
    else:  # pragma: no cover - generation is designed to succeed well within the retry budget
        raise RuntimeError(f"sheet_grid could not generate seed={seed} family={family}")
    meta["runs"] = sorted({r.run for r in rows if not r.locked})
    return Scenario(seed=seed, family=family, split=split, style_seed=style_seed, rows=rows, target=target, want=want,
                    instruction=instruction, start_sort=sort, meta=meta)


# ----------------------------------------------------------------------------- db

SCHEMA = """
CREATE TABLE cells(rid INTEGER PRIMARY KEY, run TEXT, config TEXT, throughput REAL, latency REAL, vram REAL,
  status TEXT, locked INTEGER, pos INTEGER, deleted INTEGER DEFAULT 0);
CREATE TABLE actions(id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, ref TEXT, arg TEXT, ts REAL);
"""

OPS: dict[str, list[str]] = {
    "log": [LOG_SQL],
    "set_cell": [
        "UPDATE cells SET throughput=CASE WHEN :col='throughput' THEN CAST(:val AS REAL) ELSE throughput END, "
        "latency=CASE WHEN :col='latency' THEN CAST(:val AS REAL) ELSE latency END, "
        "vram=CASE WHEN :col='vram' THEN CAST(:val AS REAL) ELSE vram END, "
        "status=CASE WHEN :col='status' THEN :val ELSE status END WHERE rid=:rid", LOG_SQL],
    "delete_row": ["UPDATE cells SET deleted=1 WHERE rid=:rid", LOG_SQL],
}

Q: dict[str, str] = {
    "rows": "SELECT rid,run,config,throughput,latency,vram,status,locked,pos FROM cells WHERE deleted=0 ORDER BY locked DESC, pos",
}

DUMP_SQL = {
    "cells": "SELECT rid,throughput,latency,vram,status,deleted FROM cells ORDER BY rid",
    "actions": "SELECT action,ref,arg FROM actions ORDER BY id",
}
_CELL_KEYS = ("rid", "throughput", "latency", "vram", "status", "deleted")


def seed_sql(sc: Scenario) -> str:
    out = [SCHEMA]
    for r in sc.rows:
        out.append(insert_sql("cells", dict(rid=r.rid, run=r.run, config=r.config, throughput=r.throughput,
                                            latency=r.latency, vram=r.vram, status=r.status, locked=int(r.locked),
                                            pos=r.pos)))
    return "\n".join(out)


def dump(execute: Callable[..., list]) -> dict[str, Any]:
    ce = execute(DUMP_SQL["cells"], [])
    ac = execute(DUMP_SQL["actions"], [])
    return {"cells": [dict(zip(_CELL_KEYS, r)) for r in ce],
            "actions": [{"action": a, "ref": str(b), "arg": c} for a, b, c in ac]}


# ----------------------------------------------------------------------------- filter / parsing twins

_FIELD = {"vram": "vram", "mem": "vram", "memory": "vram", "lat": "latency", "latency": "latency",
          "tput": "throughput", "throughput": "throughput", "tps": "throughput", "status": "status",
          "run": "run", "config": "config", "cfg": "config"}
_COND_RE = re.compile(r"^([a-z]+)(<=|>=|<|>|=|:)(.*)$")


def parse_num(s: Any) -> float | None:
    """Python twin of the page's ``parseNum``: '18,432 tok/s' -> 18432.0, '12.5ms' -> 12.5."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(s if s is not None else "").replace(",", ""))
    return float(m.group(1)) if m else None


def parse_filter(q: str) -> list[list[Any]]:
    """Python twin of the page's ``parseFilter``: list of [field, op, value]; field '' = free text."""
    out: list[list[Any]] = []
    for tok in (q or "").replace('"', " ").split():
        low = tok.lower()
        m = _COND_RE.match(low)
        if m and m.group(1) in _FIELD:
            f, op, v = _FIELD[m.group(1)], m.group(2), m.group(3)
            if f in NUM_COLS:
                n = parse_num(v)
                out.append([f, ":" if op == ":" else op, n])
            else:
                out.append([f, ":", v])
        else:
            out.append(["", ":", low])
    return out


def match_filter(conds: Sequence[Sequence[Any]], r: Mapping[str, Any]) -> bool:
    for f, op, v in conds:
        if f == "":
            if v not in str(r["run"]).lower() and v not in str(r["config"]).lower():
                return False
        elif f in NUM_COLS:
            x = r[f]
            if v is None:
                return False
            if op in (":", "=") and not x == v:
                return False
            if (op == "<" and not x < v) or (op == "<=" and not x <= v) or (op == ">" and not x > v) or (op == ">=" and not x >= v):
                return False
        elif v not in str(r[f]).lower():
            return False
    return True


def filter_rows(sc: Scenario, q: str) -> list[int]:
    conds = parse_filter(q)
    return [r.rid for r in sc.table_order() if match_filter(conds, asdict(r))]


# ----------------------------------------------------------------------------- simulation + reward


def initial_state(sc: Scenario) -> dict[str, Any]:
    return {"cells": [{"rid": r.rid, "throughput": r.throughput, "latency": r.latency, "vram": r.vram,
                       "status": r.status, "deleted": 0} for r in sorted(sc.rows, key=lambda r: r.rid)],
            "actions": []}


def simulate(sc: Scenario, steps: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    st = initial_state(sc)
    cs = {r["rid"]: r for r in st["cells"]}
    for name, p in steps:
        p = {"ref": "", "arg": "", **p}
        if name == "set_cell" and p["rid"] in cs:
            cs[p["rid"]][p["col"]] = float(p["val"]) if p["col"] in NUM_COLS else p["val"]
        elif name == "delete_row" and p["rid"] in cs:
            cs[p["rid"]]["deleted"] = 1
        st["actions"].append({"action": p.get("action", name), "ref": str(p["ref"]), "arg": p["arg"]})
    return st


_RUN_RE = re.compile(r"\brun-(\d{4})\b|\b(baseline)\b", re.I)


def extract_runs(text: str | None) -> set[str]:
    return {f"run-{a}" if a else "baseline" for a, b in _RUN_RE.findall(text or "")}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(a), 1) == round(float(b), 1)
    return a == b


def score(sc: Scenario, state: Mapping[str, Any], answer: str | None = None) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = {}
    now = {r["rid"]: r for r in state.get("cells", [])}
    changed: dict[int, list[str]] = {}
    for r in sc.rows:
        cur = now.get(r.rid, {})
        if cur.get("deleted"):
            info["why"] = f"deleted {'locked baseline ' if r.locked else ''}row {r.rid} ({r.run} {r.config})"
            return -1.0, info
        ch = [k for k in (*NUM_COLS, "status") if not _same(cur.get(k, getattr(r, k)), getattr(r, k))]
        if ch and r.locked:
            info["why"] = f"edited locked baseline row {r.rid} ({r.config}): {ch}"
            return -1.0, info
        if ch:
            changed[r.rid] = ch
    info["changed"] = changed
    if sc.family == "find_best_under_constraint":
        if changed:
            info["why"] = "edited the sheet during a read-only report"
            return 0.0, info
        runs = extract_runs(answer)
        ok = runs == {sc.want["run"]}
        info["why"] = "ok" if ok else f"answer runs={sorted(runs)}"
        return (1.0 if ok else 0.0), info
    if sc.family == "edit_and_recompute":
        col = sc.want["col"]
        if set(changed) - {sc.target} or changed.get(sc.target, [col]) != [col]:
            info["why"] = f"changed other cells {changed}"
            return 0.0, info
        ok = _same(now.get(sc.target, {}).get(col), sc.want["value"])
        info["why"] = "ok" if ok else f"target {col}={now.get(sc.target, {}).get(col)} want {sc.want['value']}"
        return (1.0 if ok else 0.0), info
    if set(changed) - {sc.target} or changed.get(sc.target, ["status"]) != ["status"]:
        info["why"] = f"changed other cells {changed}"
        return 0.0, info
    ok = now.get(sc.target, {}).get("status") == "Regression"
    info["why"] = "ok" if ok else f"target status={now.get(sc.target, {}).get('status')}"
    return (1.0 if ok else 0.0), info


def oracle_steps(sc: Scenario) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    t = sc.row(sc.target)
    if sc.family == "find_best_under_constraint":
        return [], f"{t.run}"
    if sc.family == "edit_and_recompute":
        c, v = sc.want["col"], sc.want["value"]
        return [("set_cell", {"rid": t.rid, "col": c, "val": v, "ref": str(t.rid), "arg": f"{c}={v}"})], None
    return [("set_cell", {"rid": t.rid, "col": "status", "val": "Regression", "ref": str(t.rid),
                          "arg": "status=Regression"})], None


# ----------------------------------------------------------------------------- page

CSS = Template("""
*{box-sizing:border-box} body{margin:0;background:$bg;color:$fg;font:$fs/1.35 $font;height:100vh;display:flex;flex-direction:column}
button{font:inherit;color:inherit;cursor:pointer}
#top{display:flex;align-items:center;gap:10px;padding:7px 12px;background:$ribbon;color:$ribbon_fg;border-bottom:1px solid $grid}
#top .doc{font-size:1.15em;font-weight:600} #top .sub{opacity:.75;font-size:.9em}
#dirty{margin-left:auto;font-size:.9em;opacity:.85}
.tb{padding:4px 12px;border:1px solid $grid;border-radius:$radius;background:$panel;color:$fg}
.tb.pri{background:$accent;color:#fff;border-color:$accent} .tb.danger{color:$bad;border-color:$bad}
#bar,#fx{display:flex;align-items:center;gap:8px;padding:6px 12px;border-bottom:1px solid $grid;background:$panel}
#bar input,#fx input{padding:4px 8px;border:1px solid $grid;border-radius:$radius;background:$bg;color:$fg;font:inherit}
#q{flex:1} #cv{flex:1} #ref{min-width:250px;font-size:.9em;color:$muted} .fxl{font-style:italic;color:$muted;font-weight:700}
#warn{padding:4px 12px;background:$lock;color:$lock_fg;font-size:.9em;display:none} #warn.on{display:block}
#wrap{flex:1;overflow:auto}
table{border-collapse:collapse;min-width:100%;font-variant-numeric:tabular-nums}
th,td{border:1px solid $grid;height:$rowh;padding:0 8px;white-space:nowrap}
th{background:$head;position:sticky;top:0;z-index:2;font-weight:600;text-align:left}
th.letter{top:0;height:18px;font-weight:400;color:$muted;text-align:center;font-size:.85em}
td.num{text-align:right} td.rn,th.rn{background:$head;color:$muted;text-align:center;width:38px;font-size:.85em}
td.cell{cursor:cell} td.cell.sel{outline:2px solid $accent;outline-offset:-2px;background:$sel}
td.pend{background:$pend} tr.locked td{background:$lock;color:$lock_fg} .lk{vertical-align:-1px;margin-left:3px}
.tb{white-space:nowrap} #fx{flex-wrap:wrap} #cv{min-width:140px}
tr.del td{text-decoration:line-through;opacity:.5}
.sb{border:none;background:transparent;padding:0 3px;color:$muted} .sb.on{color:$accent}
.st{font-weight:600} .st.Pass{color:$ok} .st.Fail,.st.Regression{color:$bad}
tfoot td{background:$head;font-weight:600}
#toast{color:$muted;font-size:.9em;min-width:120px}
""")

PAGE_JS = r"""
const OPS = __OPS__; const Q = __Q__; const NUMC = ['throughput','latency','vram'];
const STATUSES = ['Pass','Fail','Running','Regression']; const LETTERS = __LETTERS__;
const COLS = [['run','Run'],['config','Config'],['throughput','Throughput (tok/s)'],['latency','Latency (ms)'],['vram','VRAM (GiB)'],['status','Status']];
const SHORT = {run:'Run',config:'Config',throughput:'Throughput',latency:'Latency',vram:'VRAM',status:'Status'};
const FIELD = {vram:'vram',mem:'vram',memory:'vram',lat:'latency',latency:'latency',tput:'throughput',throughput:'throughput',tps:'throughput',status:'status',run:'run',config:'config',cfg:'config'};
let V = {rows:[], pend:{}, del:{}, sel:null, q:'', sort:__SORT__};
const LOCK = '<svg class="lk" role="img" aria-label="Locked" width="10" height="11" viewBox="0 0 10 11"><rect x="1" y="5" width="8" height="6" rx="1" fill="currentColor"/><path d="M3 5V3.5a2 2 0 0 1 4 0V5" stroke="currentColor" stroke-width="1.3" fill="none"/></svg>';
function parseNum(s){ const m = /^\s*(-?\d+(?:\.\d+)?)/.exec(String(s==null?'':s).replace(/,/g,'')); return m ? parseFloat(m[1]) : null; }
function parseFilter(q){
  const out = [];
  for (const tok of (q||'').replace(/"/g,' ').split(/\s+/)) {
    if (!tok) continue; const low = tok.toLowerCase(); const m = /^([a-z]+)(<=|>=|<|>|=|:)(.*)$/.exec(low);
    if (m && FIELD[m[1]]) { const f = FIELD[m[1]];
      if (NUMC.includes(f)) out.push([f, m[2]===':' ? ':' : m[2], parseNum(m[3])]); else out.push([f, ':', m[3]]); }
    else out.push(['', ':', low]);
  }
  return out;
}
function matchFilter(conds, r){
  for (const [f, op, v] of conds) {
    if (f==='') { if (!String(r.run).toLowerCase().includes(v) && !String(r.config).toLowerCase().includes(v)) return false; }
    else if (NUMC.includes(f)) { const x = r[f]; if (v===null) return false;
      if ((op===':'||op==='=') && !(x===v)) return false;
      if ((op==='<' && !(x<v)) || (op==='<=' && !(x<=v)) || (op==='>' && !(x>v)) || (op==='>=' && !(x>=v))) return false; }
    else if (!String(r[f]).toLowerCase().includes(v)) return false;
  }
  return true;
}
function cur(r, c){ const k = r.rid+':'+c; return (k in V.pend) ? V.pend[k] : r[c]; }
function fmt(c, v){ if (c==='throughput') return Math.round(v).toLocaleString('en-US'); if (c==='latency'||c==='vram') return Number(v).toFixed(1); return esc(v); }
function nPend(){ return Object.keys(V.pend).length + Object.keys(V.del).length; }
async function load(){
  const rows = await sql(Q.rows, {});
  V.rows = rows.map(a => ({rid:a[0], run:a[1], config:a[2], throughput:a[3], latency:a[4], vram:a[5], status:a[6], locked:a[7], pos:a[8]}));
}
function visible(){
  const conds = parseFilter(V.q);
  let rs = V.rows.filter(r => matchFilter(conds, {rid:r.rid, run:r.run, config:r.config, throughput:cur(r,'throughput'),
    latency:cur(r,'latency'), vram:cur(r,'vram'), status:cur(r,'status')}));
  if (V.sort.length) { const [c, d] = V.sort; const s = d==='asc' ? 1 : -1;
    rs.sort((a,b) => { const x = cur(a,c), y = cur(b,c); return (x<y ? -s : x>y ? s : a.pos-b.pos); }); }
  return rs.filter(r=>r.locked).concat(rs.filter(r=>!r.locked));
}
function render(){
  const rs = visible(); const sweep = rs.filter(r=>!r.locked && !V.del[r.rid]);
  let h = '<table role="grid" aria-label="Sweep results"><thead>';
  if (LETTERS) h += '<tr><th class="rn letter"></th>' + COLS.map((c,i)=>`<th class="letter">${'ABCDEF'[i]}</th>`).join('') + '</tr>';
  h += '<tr><th class="rn">#</th>' + COLS.map(([c,l]) => { const on = V.sort[0]===c; const arrow = on ? (V.sort[1]==='asc'?' ▲':' ▼') : ' ⇅';
    return `<th data-col="${c}">${l}<button class="sb ${on?'on':''}" aria-label="Sort by ${SHORT[c]}" onclick="sortBy('${c}')">${arrow}</button></th>`; }).join('') + '</tr></thead><tbody>';
  let n = 0;
  for (const r of rs) { n++;
    h += `<tr data-rid="${r.rid}" class="${r.locked?'locked':''} ${V.del[r.rid]?'del':''}"><td class="rn">${n}${r.locked?LOCK:''}</td>` + COLS.map(([c]) => {
      const pend = (r.rid+':'+c) in V.pend; const sel = V.sel && V.sel.rid===r.rid && V.sel.col===c; const v = cur(r,c);
      const cls = 'cell' + (NUMC.includes(c)?' num':'') + (pend?' pend':'') + (sel?' sel':'') + (c==='status'?' st '+esc(v):'');
      return `<td class="${cls}" role="gridcell" tabindex="-1" data-rid="${r.rid}" data-col="${c}" onclick="selCell(${r.rid},'${c}')">${fmt(c,v)}</td>`; }).join('') + '</tr>';
  }
  if (!rs.length) h += `<tr><td class="rn"></td><td colspan="6" style="color:gray">No rows match the filter</td></tr>`;
  h += '</tbody><tfoot>';
  const mean = c => sweep.length ? sweep.reduce((a,r)=>a+Number(cur(r,c)),0)/sweep.length : 0;
  h += `<tr><td class="rn">&Sigma;</td><td>Sweep mean</td><td>${sweep.length} rows</td><td class="num">${fmt('throughput',mean('throughput'))}</td>
    <td class="num">${fmt('latency',mean('latency'))}</td><td class="num">${fmt('vram',mean('vram'))}</td><td></td></tr>`;
  const all = V.rows.filter(r=>!r.locked && !V.del[r.rid]);
  if (all.length) { const b = all.reduce((a,r)=> Number(cur(r,'latency')) < Number(cur(a,'latency')) ? r : a);
    h += `<tr><td class="rn">&#9733;</td><td>Best latency</td><td>${esc(b.run)} &middot; ${esc(b.config)}</td><td></td><td class="num">${fmt('latency',cur(b,'latency'))}</td><td></td><td></td></tr>`; }
  document.getElementById('wrap').innerHTML = h + '</tfoot></table>';
  const np = nPend(); document.getElementById('dirty').textContent = np ? `${np} unsaved change${np>1?'s':''}` : 'All changes saved';
  renderFx();
}
function renderFx(){
  const ref = document.getElementById('ref'), cv = document.getElementById('cv'), w = document.getElementById('warn');
  if (!V.sel) { ref.textContent = 'Select a cell'; cv.value = ''; w.classList.remove('on'); return; }
  const r = V.rows.find(x=>x.rid===V.sel.rid); if (!r) { V.sel = null; return renderFx(); }
  ref.textContent = `${r.run} · ${r.config} · ${SHORT[V.sel.col]}`; cv.value = String(cur(r, V.sel.col));
  w.classList.toggle('on', !!r.locked);
}
async function sortBy(c){ V.sort = (V.sort[0]===c && V.sort[1]==='asc') ? [c,'desc'] : [c,'asc'];
  await op('log',{action:'sort',ref:c,arg:V.sort[1]}); render(); }
async function applyFilter(){ V.q = document.getElementById('q').value; await op('log',{action:'filter',ref:'sheet',arg:V.q}); render(); }
async function clearFilter(){ V.q=''; document.getElementById('q').value=''; render(); }
async function selCell(rid, col){ V.sel = {rid, col}; await op('log',{action:'select',ref:String(rid),arg:col}); render();
  const e = document.getElementById('cv'); if (e) e.focus(); }
function toast(t){ document.getElementById('toast').textContent = t; }
function stage(val){
  if (!V.sel) { toast('Select a cell first'); return; }
  const {rid, col} = V.sel; const r = V.rows.find(x=>x.rid===rid);
  if (col==='run'||col==='config') { toast('Run and Config are read-only'); return; }
  let v;
  if (col==='status') { v = STATUSES.find(s => s.toLowerCase()===String(val).trim().toLowerCase()); if (!v) { toast('Status must be Pass, Fail, Running or Regression'); return; } }
  else { v = parseNum(val); if (v===null || v<0) { toast('Enter a number'); return; } v = col==='throughput' ? Math.round(v) : Math.round(v*10)/10; }
  const k = rid+':'+col; if (v === r[col]) delete V.pend[k]; else V.pend[k] = v;
  toast(r.locked ? 'Edited a protected baseline row' : 'Edit staged'); render();
}
function applyEdit(){ stage(document.getElementById('cv').value); }
function setStatus(s){ if (V.sel) V.sel.col = 'status'; stage(s); }
function delRow(){ if (!V.sel) { toast('Select a cell first'); return; } V.del[V.sel.rid] = 1; toast('Row deletion staged'); render(); }
function discard(){ V.pend = {}; V.del = {}; toast('Changes discarded'); render(); }
async function save(){
  for (const k of Object.keys(V.pend)) { const [rid, col] = k.split(':'); const val = V.pend[k];
    await op('set_cell', {rid:+rid, col, val, ref:rid, arg:col+'='+val}); }
  for (const rid of Object.keys(V.del)) await op('delete_row', {rid:+rid, ref:rid});
  V.pend = {}; V.del = {}; await load(); toast('Saved'); render();
}
document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('q').addEventListener('keydown', e => { if (e.key==='Enter') applyFilter(); });
  document.getElementById('cv').addEventListener('keydown', e => { if (e.key==='Enter') applyEdit(); });
  await load(); render(); emit({type:'ready'});
});
"""


def render_html(sc: Scenario) -> str:
    th = style_for(sc.split, sc.style_seed)
    js = (PAGE_JS.replace("__OPS__", json.dumps(OPS)).replace("__Q__", json.dumps(Q))
          .replace("__LETTERS__", json.dumps(bool(th["letters"]))).replace("__SORT__", json.dumps(sc.start_sort)))
    doc = "gpu_sweep.sheet" if sc.split == "train" else "GPU_Sweep_Results.xlsx"
    status_btns = "".join(f"<button class='tb' onclick=\"setStatus('{s}')\">Set status: {s}</button>" for s in STATUSES)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(doc)} - SheetGrid</title>"
            f"<style>{CSS.substitute(th)}</style><script>{BASE_JS}{js}</script></head><body>"
            f"<div id='top'><span class='doc'>{_html.escape(doc)}</span><span class='sub'>SheetGrid &middot; "
            f"inference sweep, baselines locked</span><span id='dirty'></span>"
            f"<button class='tb' onclick='discard()'>Discard changes</button>"
            f"<button class='tb pri' onclick='save()'>Save</button></div>"
            f"<div id='bar'><input id='q' aria-label='Filter rows' autocomplete='off' "
            f"placeholder='Filter: vram&lt;80 status:pass config:fp8 run:4127 latency&lt;=12'>"
            f"<button class='tb' onclick='applyFilter()'>Apply filter</button>"
            f"<button class='tb' onclick='clearFilter()'>Clear filter</button></div>"
            f"<div id='fx'><span class='fxl'>fx</span><span id='ref'></span><input id='cv' aria-label='Cell value' "
            f"autocomplete='off'><button class='tb' onclick='applyEdit()'>Apply edit</button>{status_btns}"
            f"<button class='tb danger' onclick='delRow()'>Delete row</button><span id='toast'></span></div>"
            f"<div id='warn'>Protected range: baseline rows are locked references. Editing them overwrites the "
            f"baseline for every comparison.</div>"
            f"<div id='wrap'></div></body></html>")
