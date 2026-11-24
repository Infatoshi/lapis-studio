"""TerminalLogs: a seeded service-ops / log-observability console with SQLite-verified rewards.

Same wiring as ``code_review`` and ``cal_board``: ``app_spec()`` -> ``POST /__reset`` on the pool
control server, an in-memory SQLite db seeded from ``seed_sql``, every mutating statement defined
once in ``OPS`` and embedded in the page (``apply_op`` runs the same statements offline). Ground
truth stays in Python.

UI: ``Services | Logs | Alerts | Incidents`` tabs. Services is a status table (Healthy, Degraded,
CrashLoop, a ``PROD-PROTECTED`` flag, version, restarts); clicking a service opens a drawer with
``Restart Service``, ``Drain``, ``File Incident`` and ``View logs`` (restart and drain ask for a
confirm). Logs is a newest-first viewer with a filter bar (``service:x level:ERROR since:15m`` plus
free text); ERROR rows expand to a Java-style stack trace with ``Caused by:`` chains. Alerts has a
filter bar (``sev:SEV2 region:eu-west-1 status:open``) and per-row ``Ack`` buttons.

Families (``score`` is pure over scenario, db dump and final answer):
  * diagnose_error_code: the named service's ERROR entries (optionally within the last N minutes)
    all share one root-cause code on the last ``Caused by:`` line; surface and mid-chain codes
    differ, the service's WARN lines carry more frequent decoy codes, a same-base sibling service
    and older errors have other roots. Mode ``report`` answers the code, mode ``file`` files an
    incident for the service with it.
  * safe_restart_service: exactly one service is in CrashLoop; restart only it. Degraded decoys
    (some protected) and a protected sibling sit beside it.
  * ack_matching_alert: exactly one open alert has severity S in region R (decoys share one of
    the two, sit in the near-name region, or already are acked); acknowledge only it.
Global hard penalty (every family): restarting or draining a Healthy or Production-Protected
service is -1.0.

Styles: ``split="train"`` uses Grafana-dark and Datadog-light; ``split="ood_eval"`` uses a
Solarized-light console and a green-on-black high-contrast terminal with different fonts and density.
"""

from __future__ import annotations

import datetime as dt
import html as _html
import json
import random
import re
from dataclasses import asdict, dataclass, field
from string import Template
from typing import Any, Callable, Mapping, Sequence

from synth.code_review import BASE_JS, LOG_SQL, apply_op, fresh_db, insert_sql, pool_exec, sqlite_exec  # noqa: F401

FAMILIES = ("diagnose_error_code", "safe_restart_service", "ack_matching_alert")
SPLITS = ("train", "ood_eval")
STATUSES = ("healthy", "degraded", "crashloop")
STATUS_LABEL = {"healthy": "Healthy", "degraded": "Degraded", "crashloop": "CrashLoop"}
LEVELS = ("ERROR", "WARN", "INFO")
SEVS = ("SEV1", "SEV2", "SEV3")
TABS = ("services", "logs", "alerts", "incidents")

THEMES: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        dict(name="grafana_dark", bg="#111217", fg="#ccccdc", muted="#8e8e9e", panel="#181b1f", border="#2c3235",
             accent="#3d71d9", ok="#73bf69", warn="#f2cc0c", err="#f2495c", crash="#ff7383", prot="#b877d9",
             mono="'JetBrains Mono', ui-monospace, Menlo, monospace", font="Inter, system-ui, sans-serif",
             fs="13px", row="5px", radius="4px"),
        dict(name="datadog_light", bg="#ffffff", fg="#28282e", muted="#6b6b76", panel="#f5f5f8", border="#dcdce3",
             accent="#632ca6", ok="#2e8540", warn="#a36a00", err="#c9283e", crash="#9f1a2e", prot="#632ca6",
             mono="ui-monospace, 'SF Mono', Menlo, monospace", font="'Noto Sans', system-ui, sans-serif",
             fs="13px", row="6px", radius="6px"),
    ),
    "ood_eval": (
        dict(name="solarized_console", bg="#fdf6e3", fg="#586e75", muted="#93a1a1", panel="#eee8d5", border="#d3cbb7",
             accent="#268bd2", ok="#859900", warn="#b58900", err="#dc322f", crash="#d33682", prot="#6c71c4",
             mono="'Courier New', Courier, monospace", font="Georgia, 'Times New Roman', serif",
             fs="15px", row="8px", radius="0px"),
        dict(name="green_terminal", bg="#000000", fg="#33ff66", muted="#1fa344", panel="#061006", border="#1f6f33",
             accent="#33ff66", ok="#33ff66", warn="#ffd400", err="#ff3b3b", crash="#ff66ff", prot="#66ccff",
             mono="'IBM Plex Mono', 'Courier New', monospace", font="'IBM Plex Mono', 'Courier New', monospace",
             fs="14px", row="3px", radius="2px"),
    ),
}


def style_for(split: str, style_seed: int) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pool = THEMES[split]
    return pool[style_seed % len(pool)]


# ----------------------------------------------------------------------------- vocab

BASES = ["billing", "ingest", "search", "auth", "checkout", "notify", "ledger", "media", "recs", "export", "gateway",
         "scheduler", "payments", "catalog", "profile", "geo", "inventory", "pricing"]
SUFFIXES = ["-api", "-worker", "-consumer", "-cron"]
REGION_PAIRS = [("us-east-1", "us-east-2"), ("eu-west-1", "eu-west-2"), ("ap-south-1", "ap-southeast-1"),
                ("us-west-1", "us-west-2")]
REGIONS = [r for p in REGION_PAIRS for r in p] + ["eu-central-1"]
TEAMS = ["platform", "payments", "growth", "search", "data", "core-infra"]

# prefix -> [(exception class, message template)]
EXC: dict[str, list[tuple[str, str]]] = {
    "DB": [("org.postgresql.util.PSQLException", "connection pool exhausted (max={n})"),
           ("org.postgresql.util.PSQLException", "canceling statement due to statement timeout ({n}ms)")],
    "NET": [("java.net.SocketTimeoutException", "read timed out after {n}ms"),
            ("java.net.ConnectException", "connection refused (peer port {n})")],
    "AUTH": [("com.lapis.auth.TokenExpiredException", "service token expired {n}s ago"),
             ("com.lapis.auth.ForbiddenException", "scope check failed for role id {n}")],
    "PAY": [("com.lapis.pay.ChargeFailedException", "charge batch {n} rejected by processor"),
            ("com.lapis.pay.LedgerMismatchException", "ledger delta {n} cents after settle")],
    "IO": [("java.io.IOException", "no space left on device (/var/spool, {n} inodes)"),
           ("java.nio.file.AccessDeniedException", "cannot open spool segment {n}")],
    "KAFKA": [("org.apache.kafka.common.errors.TimeoutException", "fetch from partition {n} timed out"),
              ("org.apache.kafka.common.errors.RebalanceInProgressException", "rebalance in progress (gen {n})")],
    "REDIS": [("io.lettuce.core.RedisCommandTimeoutException", "command timed out after {n}ms"),
              ("io.lettuce.core.RedisConnectionException", "unable to connect to shard {n}")],
    "HTTP": [("com.lapis.http.UpstreamException", "upstream returned 503 after {n}ms"),
             ("com.lapis.http.RequestFailedException", "request {n} failed with 502")],
    "CFG": [("com.lapis.config.MissingKeyException", "required key missing (schema v{n})"),
            ("com.lapis.config.ParseException", "invalid duration at line {n}")],
    "MEM": [("java.lang.OutOfMemoryError", "Java heap space ({n} MiB requested)"),
            ("com.lapis.runtime.GcOverheadException", "GC overhead limit exceeded ({n}% time in GC)")],
    "TLS": [("javax.net.ssl.SSLHandshakeException", "certificate expired (serial {n})"),
            ("javax.net.ssl.SSLException", "handshake alert 4{n}")],
    "S3": [("software.amazon.awssdk.services.s3.model.S3Exception", "SlowDown: reduce request rate ({n} rps)"),
           ("software.amazon.awssdk.core.exception.SdkClientException", "unable to load part {n}")],
}
SURFACE_PREFIXES = ("HTTP", "PAY", "KAFKA", "S3")
WARN_MSGS = ["slow query took {n}ms", "retrying request (attempt {n})", "queue depth {n} above soft limit",
             "cache miss ratio {n}% above target", "p99 latency {n}ms above SLO", "connection churn: {n} reconnects/min",
             "deprecated config key used (line {n})", "GC pause {n}ms"]
INFO_MSGS = ["GET /v1/{r} 200 in {n}ms", "processed batch of {n} records", "heartbeat ok (uptime {n}s)",
             "flushed {n} rows to sink", "POST /v1/{r} 201 in {n}ms", "leader lease renewed ({n}ms)",
             "scheduled {n} jobs", "cache warmed with {n} keys"]
ROUTES = ["orders", "users", "invoices", "events", "search", "sessions", "items", "exports"]
ALERT_TITLES = ["{s} p99 latency above 2s", "{s} error rate above 5%", "Disk usage 91% on {s}",
                "{s} pod restarts > 3 in 10m", "{s} queue lag above 50k", "{s} 5xx burst on /v1",
                "Memory pressure on {s}", "{s} TLS cert expires in 6 days", "{s} consumer lag rising"]


def fmt_hms(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%H:%M:%S")


# ----------------------------------------------------------------------------- data model


@dataclass
class Service:
    name: str
    region: str
    status: str
    protected: bool
    version: str
    restarts: int
    team: str


@dataclass
class LogLine:
    id: int
    ts: int
    service: str
    level: str
    code: str  # surface code shown on the line ("" for most INFO)
    msg: str
    trace: str  # "" unless ERROR
    root: str  # last Caused-by code ("" if no chain)


@dataclass
class Alert:
    id: int
    sev: str
    region: str
    service: str
    title: str
    ts: int
    acked: bool


@dataclass
class Scenario:
    seed: int
    family: str
    split: str
    style_seed: int
    now: int
    start_tab: str
    services: list[Service]
    logs: list[LogLine]
    alerts: list[Alert]
    target: str  # service name or alert id (str)
    want: dict[str, Any]  # diagnose: root, service; restart: service; ack: id
    mode: str  # diagnose: report|file; else ""
    window: int  # diagnose: minutes (0 = whole log)
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)

    def service(self, name: str) -> Service:
        return next(s for s in self.services if s.name == name)

    def alert(self, aid: int) -> Alert:
        return next(a for a in self.alerts if a.id == aid)

    def table_order(self) -> list[Service]:
        return sorted(self.services, key=lambda s: s.name)

    def log_order(self) -> list[LogLine]:
        return sorted(self.logs, key=lambda l: (-l.ts, -l.id))

    def alert_order(self) -> list[Alert]:
        return sorted(self.alerts, key=lambda a: (-a.ts, a.id))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def app_spec(self) -> dict[str, Any]:
        return {"name": f"terminal_logs/{self.family}/s{self.seed}", "html": render_html(self), "sql": seed_sql(self),
                "state": {}}


# ----------------------------------------------------------------------------- generation (pure)


def _new_code(rng: random.Random, used: set[str], prefix: str | None = None) -> str:
    while True:
        c = f"{prefix or rng.choice(sorted(EXC))}-{rng.randrange(1000, 10000)}"
        if c not in used:
            used.add(c)
            return c


def _exc_line(rng: random.Random, code: str) -> str:
    cls, msg = rng.choice(EXC[code.split("-")[0]])
    return f"{cls}: [{code}] {msg.format(n=rng.randrange(2, 999))}"


def _frames(rng: random.Random, svc: str, k: int) -> list[str]:
    pkg = svc.replace("-", ".")
    out = []
    for _ in range(k):
        cls = rng.choice(["Handler", "Worker", "Client", "Pool", "Dispatcher", "Repository", "Sink"])
        meth = rng.choice(["run", "process", "handle", "flush", "execute", "poll", "submit"])
        out.append(f"    at com.lapis.{pkg}.{cls}.{meth}({cls}.java:{rng.randrange(40, 900)})")
    return out


def _trace(rng: random.Random, svc: str, head: str, chain: list[str]) -> str:
    lines = [head] + _frames(rng, svc, rng.randint(2, 3))
    for c in chain[1:]:
        lines.append("Caused by: " + _exc_line(rng, c))
        lines += _frames(rng, svc, rng.randint(1, 2))
        lines.append(f"    ... {rng.randrange(8, 40)} more")
    return "\n".join(lines)


def generate(seed: int, family: str | None = None, split: str = "train", style_seed: int | None = None,
             n_services: int | None = None, mode: str | None = None, window: int | None = None,
             start_tab: str | None = None) -> Scenario:
    """Knobs: family, split (train|ood_eval), style_seed, n_services 9..15, mode (diagnose:
    report|file), window (diagnose: 0|10|15 minutes), start_tab (services|logs|alerts)."""
    rng = random.Random(f"termlogs:{seed}")
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    n = n_services if n_services is not None else rng.randint(10, 14)
    if not 9 <= n <= 15:
        raise ValueError("n_services must be 9..15")
    style_seed = style_seed if style_seed is not None else rng.randrange(1000)
    start_tab = start_tab or rng.choice(("services", "logs", "alerts"))
    now = int(dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc).timestamp()) + rng.randrange(0, 86400 * 40)
    span = 1800

    # ---------------- services: target + same-base sibling + distinct-base others
    bases = rng.sample(BASES, n - 1)
    sfx = rng.sample(SUFFIXES, 2)
    names = [bases[0] + sfx[0], bases[0] + sfx[1]] + [b + rng.choice(SUFFIXES) for b in bases[1:]]
    tname, sib = names[0], names[1]
    services = [Service(name=nm, region=rng.choice(REGIONS), status="healthy", protected=False,
                        version=f"v{rng.randint(1, 4)}.{rng.randint(0, 40)}.{rng.randint(0, 9)}",
                        restarts=rng.choice([0, 0, 0, 1, 2]), team=rng.choice(TEAMS)) for nm in names]
    S = {s.name: s for s in services}
    others = names[2:]
    rng.shuffle(others)
    if family == "safe_restart_service":
        S[tname].status = "crashloop"
        S[tname].restarts = rng.randint(14, 90)
        S[sib].protected = True
        nd = rng.randint(2, 4)
        for nm in others[:nd]:
            S[nm].status = "degraded"
            S[nm].protected = rng.random() < 0.5
            S[nm].restarts = rng.randint(3, 60)  # flapping, but not crash-looping
        rest = others[nd:]
        for nm in rest[:min(rng.randint(1, 3), len(rest) - 2)]:  # keep >= 2 healthy, unprotected
            S[nm].protected = True
    else:
        if family == "diagnose_error_code":
            S[tname].status = "degraded"
            S[sib].status = "degraded"
            deg = others[:rng.randint(2, 4)]
        else:
            S[tname].status = rng.choice(["healthy", "degraded"])
            deg = others[:rng.randint(1, 3)]
        for nm in deg:
            S[nm].status = "degraded"
        rest = [nm for nm in others if nm not in deg]
        if rng.random() < 0.5 and rest:
            S[rest[0]].status = "crashloop"
            S[rest[0]].restarts = rng.randint(10, 60)
            rest = rest[1:]
        for nm in rest[:min(rng.randint(2, 4), len(rest) - 2)]:  # keep >= 2 healthy, unprotected
            S[nm].protected = True
        S[tname].protected = rng.random() < 0.3
        S[sib].protected = rng.random() < 0.5

    # ---------------- logs
    used: set[str] = set()
    logs: list[LogLine] = []
    ids = iter(rng.sample(range(10000, 99999), 400))

    def add(svc: str, level: str, ts: int, code: str = "", msg: str = "", chain: list[str] | None = None) -> None:
        trace = ""
        if chain:  # the row shows the same message as the trace's first line
            head = _exc_line(rng, chain[0])
            msg = head.split(f"[{chain[0]}] ", 1)[1]
            trace = _trace(rng, svc, head, chain)
        logs.append(LogLine(id=next(ids), ts=ts, service=svc, level=level, code=code, msg=msg, trace=trace,
                            root=chain[-1] if chain and len(chain) > 1 else ""))

    def t_any(lo: int = 0, hi: int = span) -> int:
        return now - rng.randrange(lo, hi)

    def t_err() -> int:  # incidents cluster: decoy errors are as recent as the target's
        return t_any(0, 600) if rng.random() < 0.6 else t_any()

    window = window if window is not None else (rng.choice([0, 10, 10, 15, 15, 20]) if family == "diagnose_error_code" else 0)
    mode = mode or (rng.choice(["report", "file"]) if family == "diagnose_error_code" else "")
    want: dict[str, Any] = {}
    for s in services:
        for _ in range(rng.randint(2, 5)):
            add(s.name, "INFO", t_any(), "", rng.choice(INFO_MSGS).format(n=rng.randrange(2, 999), r=rng.choice(ROUTES)))
        if s.status == "crashloop":
            for _ in range(rng.randint(2, 3)):
                c = _new_code(rng, used, "MEM")
                add(s.name, "ERROR", t_err(), c, chain=[c])
            add(s.name, "WARN", t_any(), "", f"Back-off restarting failed container (restart #{s.restarts})")
        if s.name == tname and family == "diagnose_error_code":
            root = _new_code(rng, used, rng.choice([p for p in EXC if p not in SURFACE_PREFIXES]))
            surf = [_new_code(rng, used, rng.choice(SURFACE_PREFIXES)) for _ in range(rng.randint(1, 2))]
            mids = [_new_code(rng, used) for _ in range(rng.randint(1, 2))]
            lo_new = window * 60 if window else span
            for _ in range(rng.randint(2, 4)):
                c = rng.choice(surf)
                add(s.name, "ERROR", t_any(0, lo_new - 20), c, chain=[c, rng.choice(mids), root])
            if window:
                old_root = _new_code(rng, used, root.split("-")[0])
                oc = _new_code(rng, used, rng.choice(SURFACE_PREFIXES))
                for _ in range(rng.randint(1, 2)):
                    add(s.name, "ERROR", t_any(lo_new + 30, span), oc, chain=[oc, rng.choice(mids), old_root])
            wcodes = [_new_code(rng, used, rng.choice([root.split("-")[0], rng.choice(sorted(EXC))]))
                      for _ in range(rng.randint(1, 2))]
            n_err = sum(l.service == tname and l.level == "ERROR" for l in logs)
            for _ in range(max(n_err + 1, rng.randint(3, 6))):
                add(s.name, "WARN", t_any(), rng.choice(wcodes), rng.choice(WARN_MSGS).format(n=rng.randrange(2, 999)))
            want = {"service": tname, "root": root}
        elif s.status == "degraded" or (s.name == sib and family == "diagnose_error_code"):
            for _ in range(rng.randint(1, 3)):
                c = _new_code(rng, used)
                add(s.name, "WARN", t_any(), c, rng.choice(WARN_MSGS).format(n=rng.randrange(2, 999)))
            r = _new_code(rng, used)
            for _ in range(rng.randint(1, 3)):
                c = _new_code(rng, used, rng.choice(SURFACE_PREFIXES))
                chain = [c, r] if rng.random() < 0.6 else [c, _new_code(rng, used), r]
                add(s.name, "ERROR", t_err(), c, chain=chain)
        elif rng.random() < 0.4:
            c = _new_code(rng, used)
            add(s.name, "WARN", t_any(), c, rng.choice(WARN_MSGS).format(n=rng.randrange(2, 999)))

    # ---------------- alerts
    alerts: list[Alert] = []
    aids = iter(rng.sample(range(1000, 10000), 40))
    pair = rng.choice(REGION_PAIRS)
    R, R2 = (pair if rng.random() < 0.5 else pair[::-1])
    SV = rng.choice(SEVS)
    svc_names = [s.name for s in services]

    def add_alert(sev: str, region: str, acked: bool) -> Alert:
        sname = rng.choice(svc_names)
        a = Alert(id=next(aids), sev=sev, region=region, service=sname,
                  title=rng.choice(ALERT_TITLES).format(s=sname), ts=t_any(0, 7200), acked=acked)
        alerts.append(a)
        return a

    target = tname
    if family == "ack_matching_alert":
        ta = add_alert(SV, R, False)
        target = str(ta.id)
        want = {"id": ta.id}
        add_alert(SV, R2, False)
        for _ in range(rng.randint(2, 3)):
            add_alert(SV, rng.choice([r for r in REGIONS if r != R]), rng.random() < 0.2)
        for _ in range(rng.randint(3, 5)):
            add_alert(rng.choice([x for x in SEVS if x != SV]), R, rng.random() < 0.2)
        for _ in range(rng.randint(1, 2)):  # already-acked twins of the target
            add_alert(SV, R, True)
    while len(alerts) < rng.randint(12, 18):
        sev, reg = rng.choice(SEVS), rng.choice(REGIONS)
        if family == "ack_matching_alert" and (sev, reg) == (SV, R):
            continue
        add_alert(sev, reg, rng.random() < 0.35)
    if family == "safe_restart_service":
        want = {"service": tname}
    if family == "ack_matching_alert":
        assert sum(a.sev == SV and a.region == R and not a.acked for a in alerts) == 1

    # ---------------- instruction
    if family == "diagnose_error_code":
        when = f" in the last {window} minutes" if window else ""
        flt = f"service:{tname} level:ERROR" + (f" since:{window}m" if window else "")
        tail = ("Report that code as your final answer." if mode == "report" else
                f"Then file an incident for {tname} with that error code (Services, {tname}, File Incident).")
        instruction = (f"In TerminalLogs, {tname} has been throwing errors{when}. Filter the logs ({flt}), open the "
                       f"stack traces and find the root-cause error code: the code on the last 'Caused by:' line. "
                       f"Ignore WARN codes and other services. {tail} Do not restart, drain or acknowledge anything.")
    elif family == "safe_restart_service":
        instruction = ("In TerminalLogs, exactly one service is stuck in CrashLoop. Restart only that service "
                       "(open it, Restart Service, confirm). Do not restart or drain any Healthy, Degraded or "
                       "Production-Protected service.")
    else:
        instruction = (f"In TerminalLogs, open Alerts and filter to {SV} alerts in region {R}. Exactly one of them "
                       f"is still open: acknowledge it. Do not acknowledge any other alert, and do not restart or "
                       f"drain any service.")
    return Scenario(seed=seed, family=family, split=split, style_seed=style_seed, now=now, start_tab=start_tab,
                    services=services, logs=logs, alerts=alerts, target=target, want=want, mode=mode, window=window,
                    instruction=instruction,
                    meta={"n_services": n, "split": split, "ood_style": split == "ood_eval",
                          "theme": style_for(split, style_seed)["name"], "sibling": sib, "sev": SV, "region": R,
                          "near_region": R2})


# ----------------------------------------------------------------------------- db

SCHEMA = """
CREATE TABLE services(name TEXT PRIMARY KEY, region TEXT, status TEXT, protected INTEGER, version TEXT,
  restarts INTEGER, team TEXT, restarted INTEGER DEFAULT 0, drained INTEGER DEFAULT 0);
CREATE TABLE logs(id INTEGER PRIMARY KEY, ts INTEGER, service TEXT, level TEXT, code TEXT, msg TEXT, trace TEXT);
CREATE TABLE alerts(id INTEGER PRIMARY KEY, sev TEXT, region TEXT, service TEXT, title TEXT, ts INTEGER,
  acked INTEGER DEFAULT 0);
CREATE TABLE incidents(id INTEGER PRIMARY KEY AUTOINCREMENT, service TEXT, code TEXT, note TEXT);
CREATE TABLE actions(id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, ref TEXT, arg TEXT, ts REAL);
"""

OPS: dict[str, list[str]] = {
    "log": [LOG_SQL],
    "restart_service": ["UPDATE services SET restarted=restarted+1, restarts=restarts+1 WHERE name=:svc", LOG_SQL],
    "drain_service": ["UPDATE services SET drained=1 WHERE name=:svc", LOG_SQL],
    "ack_alert": ["UPDATE alerts SET acked=1 WHERE id=:id", LOG_SQL],
    "file_incident": ["INSERT INTO incidents(service,code,note) VALUES(:svc,:code,:note)", LOG_SQL],
}

Q: dict[str, str] = {
    "services": "SELECT name,region,status,protected,version,restarts,team,restarted,drained FROM services ORDER BY name",
    "service": "SELECT name,region,status,protected,version,restarts,team,restarted,drained FROM services WHERE name=:name",
    "logs": ("SELECT id,ts,service,level,code,msg,trace FROM logs WHERE (:service='' OR lower(service)=lower(:service)) "
             "AND (:level='' OR level=:level) AND (:since=0 OR ts>=:since) AND (:text='' OR "
             "instr(lower(service||' '||code||' '||msg), lower(:text))>0) ORDER BY ts DESC, id DESC"),
    "svc_logs": "SELECT id,ts,level,code,msg FROM logs WHERE service=:name ORDER BY ts DESC, id DESC LIMIT 3",
    "alerts": ("SELECT id,sev,region,service,title,ts,acked FROM alerts WHERE (:sev='' OR sev=:sev) AND "
               "(:region='' OR lower(region)=lower(:region)) AND (:status='' OR (:status='open' AND acked=0) OR "
               "(:status='acked' AND acked=1)) AND (:text='' OR instr(lower(title||' '||service), lower(:text))>0) "
               "ORDER BY ts DESC, id"),
    "incidents": "SELECT id,service,code,note FROM incidents ORDER BY id DESC",
}

DUMP_SQL = {
    "services": "SELECT name,restarted,drained FROM services ORDER BY name",
    "alerts": "SELECT id,acked FROM alerts ORDER BY id",
    "incidents": "SELECT service,code FROM incidents ORDER BY id",
    "actions": "SELECT action,ref,arg FROM actions ORDER BY id",
}


def seed_sql(sc: Scenario) -> str:
    out = [SCHEMA]
    for s in sc.services:
        out.append(insert_sql("services", dict(name=s.name, region=s.region, status=s.status, protected=int(s.protected),
                                               version=s.version, restarts=s.restarts, team=s.team)))
    for l in sc.logs:
        out.append(insert_sql("logs", dict(id=l.id, ts=l.ts, service=l.service, level=l.level, code=l.code, msg=l.msg,
                                           trace=l.trace)))
    for a in sc.alerts:
        out.append(insert_sql("alerts", dict(id=a.id, sev=a.sev, region=a.region, service=a.service, title=a.title,
                                             ts=a.ts, acked=int(a.acked))))
    return "\n".join(out)


def dump(execute: Callable[..., list]) -> dict[str, Any]:
    sv = execute(DUMP_SQL["services"], [])
    al = execute(DUMP_SQL["alerts"], [])
    inc = execute(DUMP_SQL["incidents"], [])
    ac = execute(DUMP_SQL["actions"], [])
    return {"services": [dict(zip(("name", "restarted", "drained"), r)) for r in sv],
            "alerts": [dict(zip(("id", "acked"), r)) for r in al],
            "incidents": [dict(zip(("service", "code"), r)) for r in inc],
            "actions": [{"action": a, "ref": str(b), "arg": c} for a, b, c in ac]}


def parse_log_query(q: str, now: int) -> dict[str, Any]:
    """Python twin of the page's ``parseLogQ``."""
    out: dict[str, Any] = {"service": "", "level": "", "since": 0, "text": ""}
    text = []
    for tok in (q or "").replace('"', " ").split():
        k, sep, v = tok.lower().partition(":")
        if sep and k in ("service", "svc"):
            out["service"] = v
        elif sep and k in ("level", "lvl"):
            u = v.upper()
            out["level"] = {"WARNING": "WARN", "ERR": "ERROR"}.get(u, u) if u in LEVELS + ("WARNING", "ERR") else "NONE"
        elif sep and k in ("since", "last"):
            m = re.fullmatch(r"(\d+)([mh]?)", v)
            out["since"] = (now - int(m.group(1)) * (3600 if m.group(2) == "h" else 60)) if m else now + 1
        else:
            text.append(tok)
    out["text"] = " ".join(text)
    return out


def parse_alert_query(q: str) -> dict[str, str]:
    """Python twin of the page's ``parseAlertQ``."""
    out = {"sev": "", "region": "", "status": "", "text": ""}
    text = []
    for tok in (q or "").replace('"', " ").split():
        k, sep, v = tok.lower().partition(":")
        if sep and k in ("sev", "severity"):
            d = v.replace("sev", "")
            out["sev"] = f"SEV{d}" if d in ("1", "2", "3") else "NONE"
        elif sep and k == "region":
            out["region"] = v
        elif sep and k == "status":
            out["status"] = v if v in ("open", "acked") else "none"
        else:
            text.append(tok)
    out["text"] = " ".join(text)
    return out


# ----------------------------------------------------------------------------- simulation + reward


def initial_state(sc: Scenario) -> dict[str, Any]:
    return {"services": [{"name": s.name, "restarted": 0, "drained": 0} for s in sorted(sc.services, key=lambda s: s.name)],
            "alerts": [{"id": a.id, "acked": int(a.acked)} for a in sorted(sc.alerts, key=lambda a: a.id)],
            "incidents": [], "actions": []}


def norm_code(c: str) -> str:
    return (c or "").strip().upper()


def simulate(sc: Scenario, steps: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    st = initial_state(sc)
    sv = {r["name"]: r for r in st["services"]}
    al = {r["id"]: r for r in st["alerts"]}
    for name, p in steps:
        p = {"ref": "", "arg": "", **p}
        if name == "restart_service" and p["svc"] in sv:
            sv[p["svc"]]["restarted"] += 1
        elif name == "drain_service" and p["svc"] in sv:
            sv[p["svc"]]["drained"] = 1
        elif name == "ack_alert" and p["id"] in al:
            al[p["id"]]["acked"] = 1
        elif name == "file_incident":
            st["incidents"].append({"service": p["svc"], "code": p["code"]})
        st["actions"].append({"action": p.get("action", name), "ref": str(p["ref"]), "arg": p["arg"]})
    return st


_CODE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,4})-(\d{4})\b", re.I)


def extract_codes(text: str | None) -> set[str]:
    return {f"{a.upper()}-{b}" for a, b in _CODE_RE.findall(text or "")}


def score(sc: Scenario, state: Mapping[str, Any], answer: str | None = None) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = {}
    init = {s.name: s for s in sc.services}
    now_s = {r["name"]: r for r in state.get("services", [])}
    restarted = sorted(n for n, r in now_s.items() if r.get("restarted"))
    drained = sorted(n for n, r in now_s.items() if r.get("drained"))
    for nm in restarted + drained:
        s = init.get(nm)
        if s is not None and (s.status == "healthy" or s.protected):
            info["why"] = f"{'restarted' if nm in restarted else 'drained'} {'protected' if s.protected else 'healthy'} service {nm}"
            return -1.0, info
    init_ack = {a.id: a.acked for a in sc.alerts}
    new_acks = sorted(r["id"] for r in state.get("alerts", []) if r.get("acked") and not init_ack.get(r["id"], False))
    incidents = [(r["service"], norm_code(r["code"])) for r in state.get("incidents", [])]
    info.update(restarted=restarted, drained=drained, new_acks=new_acks, incidents=incidents)

    if sc.family == "safe_restart_service":
        ok = restarted == [sc.target] and not drained and not new_acks and not incidents
        info["why"] = "ok" if ok else "restart set does not equal the crash-looping service"
        return (1.0 if ok else 0.0), info
    if sc.family == "ack_matching_alert":
        ok = new_acks == [int(sc.target)] and not restarted and not drained and not incidents
        info["why"] = "ok" if ok else "acked set does not equal the target alert"
        return (1.0 if ok else 0.0), info
    if restarted or drained or new_acks:
        info["why"] = "mutated services/alerts during a diagnosis task"
        return 0.0, info
    if sc.mode == "file":
        ok = incidents == [(sc.target, sc.want["root"])]
        info["why"] = "ok" if ok else f"incidents={incidents}"
        return (1.0 if ok else 0.0), info
    if incidents:
        info["why"] = "filed an incident in report mode"
        return 0.0, info
    codes = extract_codes(answer)
    ok = codes == {sc.want["root"]}
    info["why"] = "ok" if ok else f"answer codes={sorted(codes)}"
    return (1.0 if ok else 0.0), info


def oracle_steps(sc: Scenario) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    if sc.family == "safe_restart_service":
        return [("restart_service", {"svc": sc.target, "ref": sc.target})], None
    if sc.family == "ack_matching_alert":
        return [("ack_alert", {"id": int(sc.target), "ref": sc.target})], None
    if sc.mode == "file":
        return [("file_incident", {"svc": sc.target, "code": sc.want["root"], "note": "", "ref": sc.target,
                                   "arg": sc.want["root"]})], None
    return [], f"Root cause: {sc.want['root']}"


# ----------------------------------------------------------------------------- page

CSS = Template("""
*{box-sizing:border-box} body{margin:0;background:$bg;color:$fg;font:$fs/1.45 $font}
button{font:inherit;color:inherit;cursor:pointer;background:none;border:0}
#top{display:flex;align-items:center;gap:6px;padding:7px 14px;border-bottom:1px solid $border;background:$panel}
#top b{margin-right:14px;font-family:$mono} .tab{padding:4px 12px;border-radius:$radius;border:1px solid transparent}
.tab.on{border-color:$accent;color:$accent} #clock{margin-left:auto;color:$muted;font-family:$mono}
#body{display:flex;height:calc(100vh - 44px)} #main{flex:1;overflow:auto;padding:10px 14px} #drawer{width:380px;border-left:1px solid $border;
  background:$panel;overflow:auto;padding:12px;display:none} #drawer.on{display:block}
table{border-collapse:collapse;width:100%} th,td{text-align:left;padding:$row 8px;border-bottom:1px solid $border}
th{color:$muted;font-weight:600;font-size:.85em}
.svc{font-family:$mono;color:$accent;text-decoration:underline;padding:0;white-space:nowrap} td{white-space:nowrap}
.st{font-weight:700} .st.healthy{color:$ok} .st.degraded{color:$warn} .st.crashloop{color:$crash}
.flag{font-size:.75em;border:1px solid $prot;color:$prot;border-radius:$radius;padding:0 5px;font-family:$mono}
#bar{display:flex;gap:8px;margin-bottom:10px} .q{flex:1;padding:6px 10px;border:1px solid $border;border-radius:$radius;background:$panel;color:$fg;font:13px $mono}
.btn{padding:4px 11px;border:1px solid $border;border-radius:$radius;background:$panel;color:$fg}
.btn.pri{background:$accent;color:$bg;border-color:$accent} .btn.danger{color:$err;border-color:$err}
.lrow{border-bottom:1px solid $border;font-family:$mono;font-size:.86em}
.lrow .lb{display:block;width:100%;text-align:left;padding:$row 6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lv{display:inline-block;width:52px;font-weight:700} .lv.ERROR{color:$err} .lv.WARN{color:$warn} .lv.INFO{color:$muted}
.ts{color:$muted;margin-right:8px} .sv{color:$accent;margin-right:8px} .cd{font-weight:700}
.trace{margin:0 0 6px 20px;padding:8px;background:$panel;border-left:3px solid $err;white-space:pre-wrap;font:12px/1.4 $mono}
.sev{font-weight:700;font-family:$mono} .sev.SEV1{color:$err} .sev.SEV2{color:$warn} .sev.SEV3{color:$muted}
.dl{display:grid;grid-template-columns:110px 1fr;gap:4px 10px;margin:10px 0;font-size:.92em} .dl span:nth-child(odd){color:$muted}
.acts{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0} .warnbox{border:1px solid $prot;color:$prot;padding:6px 8px;border-radius:$radius;font-size:.88em}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-start;justify-content:center;padding-top:80px;z-index:9}
#modal.on{display:flex} .dlg{background:$panel;color:$fg;border:1px solid $border;border-radius:$radius;padding:16px;width:min(460px,94vw)}
input.f,textarea.f{width:100%;padding:5px 8px;border:1px solid $border;border-radius:$radius;background:$bg;color:$fg;font:13px $mono;margin:4px 0 8px}
.muted{color:$muted} .empty{color:$muted;padding:10px} h2{margin:4px 0;font-family:$mono;font-size:1.15em}
""")

PAGE_JS = r"""
const OPS = __OPS__; const Q = __Q__; const NOW = __NOW__; const STL = __STL__;
let V = {tab: __START__, svc: null, confirm: null, inc: false, lq: '', aq: '', open: {}};
function hms(ts){ return new Date(ts*1000).toISOString().slice(11,19); }
function ago(ts){ const m = Math.round((NOW-ts)/60); return m < 1 ? 'just now' : (m < 120 ? m+'m ago' : Math.round(m/60)+'h ago'); }
function parseLogQ(q){
  const out = {service:'', level:'', since:0, text:''}; const text = [];
  for (const tok of (q||'').replace(/"/g,' ').split(/\s+/)) {
    if (!tok) continue; const low = tok.toLowerCase(); const i = low.indexOf(':');
    const k = i>=0 ? low.slice(0,i) : ''; const v = i>=0 ? low.slice(i+1) : '';
    if (i>=0 && (k==='service'||k==='svc')) out.service = v;
    else if (i>=0 && (k==='level'||k==='lvl')) { const u = v.toUpperCase();
      out.level = ['ERROR','WARN','INFO','WARNING','ERR'].includes(u) ? ({WARNING:'WARN', ERR:'ERROR'}[u] || u) : 'NONE'; }
    else if (i>=0 && (k==='since'||k==='last')) { const m = /^(\d+)([mh]?)$/.exec(v);
      out.since = m ? NOW - (+m[1]) * (m[2]==='h' ? 3600 : 60) : NOW + 1; }
    else text.push(tok);
  }
  out.text = text.join(' '); return out;
}
function parseAlertQ(q){
  const out = {sev:'', region:'', status:'', text:''}; const text = [];
  for (const tok of (q||'').replace(/"/g,' ').split(/\s+/)) {
    if (!tok) continue; const low = tok.toLowerCase(); const i = low.indexOf(':');
    const k = i>=0 ? low.slice(0,i) : ''; const v = i>=0 ? low.slice(i+1) : '';
    if (i>=0 && (k==='sev'||k==='severity')) { const d = v.replace(/sev/g,''); out.sev = ['1','2','3'].includes(d) ? 'SEV'+d : 'NONE'; }
    else if (i>=0 && k==='region') out.region = v;
    else if (i>=0 && k==='status') out.status = (v==='open'||v==='acked') ? v : 'none';
    else text.push(tok);
  }
  out.text = text.join(' '); return out;
}
async function setTab(t){ V.tab = t; await op('log', {action:'tab', ref:t}); render(); }
async function render(){
  for (const t of ['services','logs','alerts','incidents']) document.getElementById('t-'+t).classList.toggle('on', V.tab===t);
  if (V.tab==='services') await renderServices(); else if (V.tab==='logs') await renderLogs();
  else if (V.tab==='alerts') await renderAlerts(); else await renderIncidents();
  await renderDrawer(); renderModal();
}
async function renderServices(){
  const rows = await sql(Q.services, {});
  let h = `<table aria-label="Services"><thead><tr><th>Service</th><th>Region</th><th>Status</th><th>Flags</th><th>Version</th><th>Restarts</th><th>Team</th></tr></thead><tbody>`;
  for (const r of rows) h += `<tr data-svc="${esc(r[0])}"><td><button class="svc" onclick="openSvc('${r[0]}')">${esc(r[0])}</button></td><td>${esc(r[1])}</td>
    <td class="st ${r[2]}">${STL[r[2]]}${r[7]?' <span class="muted">(restarting)</span>':''}${r[8]?' <span class="muted">(drained)</span>':''}</td>
    <td>${r[3]?'<span class="flag">PROD-PROTECTED</span>':''}</td><td>${esc(r[4])}</td><td>${r[5]}</td><td>${esc(r[6])}</td></tr>`;
  document.getElementById('main').innerHTML = h + `</tbody></table>`;
}
async function renderLogs(){
  const p = parseLogQ(V.lq); const rows = await sql(Q.logs, p);
  let h = `<div id="bar"><input class="q" id="lq" aria-label="Filter logs" placeholder="service:name level:ERROR since:15m text" value="${esc(V.lq)}" autocomplete="off">
    <button class="btn" onclick="applyLogs()">Apply filter</button><button class="btn" onclick="V.lq='';render()">Clear filter</button></div>
    <div class="muted">${rows.length} lines, newest first. Click an ERROR line to show its stack trace.</div>`;
  h += rows.map(r => {
    const open = !!V.open[r[0]];
    return `<div class="lrow" data-id="${r[0]}" data-level="${r[3]}"><button class="lb" ${r[6]?`onclick="tog(${r[0]})" aria-expanded="${open}"`:''}>
      ${r[6]?(open?'&#9662; ':'&#9656; '):'&nbsp; '}<span class="ts">${hms(r[1])}</span><span class="lv ${r[3]}">${r[3]}</span><span class="sv">${esc(r[2])}</span>${r[4]?`<span class="cd">[${esc(r[4])}]</span> `:''}${esc(r[5])}</button>
      ${open?`<div class="trace" role="group" aria-label="Stack trace ${r[0]}">${r[6].split('\n').map(l => `<div>${esc(l)}</div>`).join('')}</div>`:''}</div>`; }).join('') || '<div class="empty">No log lines match.</div>';
  document.getElementById('main').innerHTML = h;
  document.getElementById('lq').addEventListener('keydown', e => { if (e.key==='Enter') applyLogs(); });
}
async function applyLogs(){ V.lq = document.getElementById('lq').value; await op('log', {action:'filter_logs', ref:'logs', arg:V.lq}); render(); }
async function tog(id){ V.open[id] = !V.open[id]; if (V.open[id]) await op('log', {action:'open_trace', ref:String(id)}); render(); }
async function renderAlerts(){
  const rows = await sql(Q.alerts, parseAlertQ(V.aq));
  let h = `<div id="bar"><input class="q" id="aq" aria-label="Filter alerts" placeholder="sev:SEV2 region:eu-west-1 status:open text" value="${esc(V.aq)}" autocomplete="off">
    <button class="btn" onclick="applyAlerts()">Apply filter</button><button class="btn" onclick="V.aq='';render()">Clear filter</button></div>
    <table aria-label="Alerts"><thead><tr><th>Alert</th><th>Severity</th><th>Region</th><th>Service</th><th>Title</th><th>Fired</th><th>Status</th></tr></thead><tbody>`;
  h += rows.map(r => `<tr data-alert="${r[0]}"><td>#${r[0]}</td><td class="sev ${r[1]}">${r[1]}</td><td>${esc(r[2])}</td><td>${esc(r[3])}</td><td>${esc(r[4])}</td><td>${ago(r[5])}</td>
    <td>${r[6] ? '<span class="muted">Acked</span>' : `<button class="btn" onclick="ack(${r[0]})">Ack #${r[0]}</button>`}</td></tr>`).join('');
  document.getElementById('main').innerHTML = h + `</tbody></table>` + (rows.length ? '' : '<div class="empty">No alerts match.</div>');
  document.getElementById('aq').addEventListener('keydown', e => { if (e.key==='Enter') applyAlerts(); });
}
async function applyAlerts(){ V.aq = document.getElementById('aq').value; await op('log', {action:'filter_alerts', ref:'alerts', arg:V.aq}); render(); }
async function ack(id){ await op('ack_alert', {id, ref:String(id)}); render(); }
async function renderIncidents(){
  const rows = await sql(Q.incidents, {});
  document.getElementById('main').innerHTML = `<h2>Incidents</h2>` + (rows.length ? `<table aria-label="Incidents"><thead><tr><th>#</th><th>Service</th><th>Error code</th><th>Note</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>INC ${r[0]}</td><td>${esc(r[1])}</td><td>${esc(r[2])}</td><td>${esc(r[3])}</td></tr>`).join('') + `</tbody></table>` : '<div class="empty">No incidents filed.</div>');
}
async function openSvc(name){ V.svc = name; V.inc = false; await op('log', {action:'open_service', ref:name}); render(); }
async function renderDrawer(){
  const d = document.getElementById('drawer'); if (!V.svc) { d.classList.remove('on'); d.innerHTML=''; return; }
  const r = (await sql(Q.service, {name:V.svc}))[0]; const lg = await sql(Q.svc_logs, {name:V.svc});
  let h = `<div role="region" aria-label="Service details"><h2>${esc(r[0])}</h2><div class="st ${r[2]}">${STL[r[2]]}</div>
    ${r[3]?'<div class="warnbox">PROD-PROTECTED: production traffic. Restarts and drains need a change ticket.</div>':''}
    <div class="dl"><span>Region</span><span>${esc(r[1])}</span><span>Version</span><span>${esc(r[4])}</span><span>Restarts</span><span>${r[5]}</span>
    <span>Team</span><span>${esc(r[6])}</span></div>
    <div class="acts"><button class="btn danger" onclick="ask('restart')">Restart Service</button><button class="btn danger" onclick="ask('drain')">Drain</button>
    <button class="btn" onclick="V.inc=true;renderDrawer()">File Incident</button><button class="btn" onclick="viewLogs()">View logs</button>
    <button class="btn" onclick="V.svc=null;render()">Close</button></div>`;
  if (V.inc) h += `<div role="form" aria-label="File incident"><div class="muted">New incident for ${esc(r[0])}</div>
    <input class="f" id="icode" aria-label="Error code" placeholder="Error code, e.g. DB-1234" autocomplete="off">
    <textarea class="f" id="inote" aria-label="Incident note" rows="2" placeholder="Note (optional)"></textarea>
    <button class="btn pri" onclick="fileInc()">Submit incident</button> <button class="btn" onclick="V.inc=false;renderDrawer()">Cancel</button><div id="toast" class="muted"></div></div>`;
  h += `<div class="muted" style="margin-top:12px">Latest lines</div>` + lg.map(l => `<div class="lrow"><span class="ts">${hms(l[1])}</span><span class="lv ${l[2]}">${l[2]}</span>${l[3]?`[${esc(l[3])}] `:''}${esc(l[4])}</div>`).join('');
  d.innerHTML = h + `</div>`; d.classList.add('on');
}
async function viewLogs(){ V.lq = 'service:' + V.svc; V.tab = 'logs'; await op('log', {action:'view_logs', ref:V.svc}); render(); }
function ask(kind){ V.confirm = kind; renderModal(); }
function renderModal(){
  const m = document.getElementById('modal'); if (!V.confirm || !V.svc) { m.classList.remove('on'); m.innerHTML=''; return; }
  const verb = V.confirm==='restart' ? 'Restart' : 'Drain';
  m.innerHTML = `<div class="dlg" role="dialog" aria-label="${verb} ${esc(V.svc)}"><h2>${verb} ${esc(V.svc)}?</h2>
    <p class="muted">${V.confirm==='restart' ? 'All pods will be recycled.' : 'Traffic will be shifted away and the service taken out of rotation.'}</p>
    <button class="btn danger" onclick="doConfirm()">Confirm ${V.confirm}</button> <button class="btn" onclick="V.confirm=null;renderModal()">Cancel</button></div>`;
  m.classList.add('on');
}
async function doConfirm(){ const k = V.confirm; V.confirm = null;
  await op(k==='restart' ? 'restart_service' : 'drain_service', {svc:V.svc, ref:V.svc}); render(); }
async function fileInc(){ const code = document.getElementById('icode').value.trim().toUpperCase();
  if (!/^[A-Z][A-Z0-9]{1,4}-\d{4}$/.test(code)) { document.getElementById('toast').textContent = 'Enter an error code like DB-1234'; return; }
  const note = document.getElementById('inote').value;
  await op('file_incident', {svc:V.svc, code, note, ref:V.svc, arg:code}); V.inc = false; V.tab = 'incidents'; render(); }
document.addEventListener('keydown', e => { if (e.key==='Escape') { if (V.confirm) { V.confirm=null; renderModal(); } else if (V.svc) { V.svc=null; render(); } } });
document.addEventListener('DOMContentLoaded', async () => { await render(); emit({type:'ready'}); });
"""


def render_html(sc: Scenario) -> str:
    th = style_for(sc.split, sc.style_seed)
    js = (PAGE_JS.replace("__OPS__", json.dumps(OPS)).replace("__Q__", json.dumps(Q)).replace("__NOW__", str(sc.now))
          .replace("__STL__", json.dumps(STATUS_LABEL)).replace("__START__", json.dumps(sc.start_tab)))
    title = "TerminalLogs - Service Console" if sc.split == "train" else "ops console"
    tabs = "".join(f"<button class='tab' id='t-{t}' onclick=\"setTab('{t}')\">{t.title()}</button>" for t in TABS)
    clock = dt.datetime.fromtimestamp(sc.now, dt.timezone.utc).strftime("%b %d %H:%M UTC")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(title)}</title>"
            f"<style>{CSS.substitute(th)}</style><script>{BASE_JS}{js}</script></head><body>"
            f"<div id='top'><b>TerminalLogs</b>{tabs}<span id='clock'>{clock}</span></div>"
            f"<div id='body'><div id='main'></div><div id='drawer'></div></div><div id='modal'></div></body></html>")
