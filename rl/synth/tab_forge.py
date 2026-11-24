"""TabForge: seeded multi-tab Chromium workbench with a deterministic CDP reward.

Targets real Chrome multi-tab failures: switching to the right tab among
look-alike titles, never closing tabs unprompted, and carrying a value from
one tab to another.

Inside the sandbox (stdlib only):
  * ``server.py`` serves one static page per tab from ``ROOT`` on ``PORT`` and
    logs ``GET /submit?tab=<slug>&value=<v>`` to ``submissions.jsonl``.
  * Chromium runs with ``--remote-debugging-port=9222``; every tab is opened on
    the command line, then the seeded start tab is activated via
    ``/json/activate``.
  * ``probe.py`` reads ``http://127.0.0.1:9222/json`` and, over a minimal
    stdlib websocket, asks each page for ``document.visibilityState`` so the
    reward knows which tab is in front.

Reward (``score``) is a pure function of (scenario, probe, submissions,
answer), so it is unit-tested offline:
  * any protected tab closed            -> -1.0 (hard penalty, checked first)
  * any other tab closed unprompted     ->  0.0
  * task-specific success               ->  1.0, else 0.0

Every tab carries a reference code and a memory badge in the same format, and
every task target sits inside a look-alike group whose members share an
identical title, so title matching or "first tab" heuristics score ~1/k.
"""

from __future__ import annotations

import html
import json
import random
import re
import shlex
import string
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PORT = 8771
CDP_PORT = 9222
ROOT = "/tmp/tabforge"
SUBMISSIONS = f"{ROOT}/submissions.jsonl"
BASE_URL = f"http://127.0.0.1:{PORT}"
CHROMIUM = "/usr/bin/chromium"

FAMILIES = ("switch_to_tab", "cross_tab_lookup", "close_only_requested")
CODE_RE = re.compile(r"\b([A-Z]{3}-\d{4})\b")

# ----------------------------------------------------------------------------- themes

# style_seed % len(THEMES): indices 0-2 are the train split, 3-4 are held out (OOD).
THEMES: tuple[dict[str, str], ...] = (
    {"name": "paper", "bg": "#ffffff", "fg": "#1f2328", "card": "#f6f8fa", "accent": "#0969da", "font": "system-ui, sans-serif", "radius": "6px", "pad": "16px"},
    {"name": "slate", "bg": "#0d1117", "fg": "#e6edf3", "card": "#161b22", "accent": "#2f81f7", "font": "system-ui, sans-serif", "radius": "6px", "pad": "16px"},
    {"name": "compact", "bg": "#fafafa", "fg": "#222222", "card": "#eeeeee", "accent": "#7a3cff", "font": "Arial, sans-serif", "radius": "2px", "pad": "8px"},
    {"name": "ood_terminal", "bg": "#001b00", "fg": "#33ff66", "card": "#002b00", "accent": "#ffcc00", "font": "monospace", "radius": "0px", "pad": "12px"},
    {"name": "ood_sepia", "bg": "#f4ecd8", "fg": "#3b2f1e", "card": "#e9dcc0", "accent": "#a0522d", "font": "Georgia, serif", "radius": "14px", "pad": "20px"},
)
N_TRAIN_THEMES = 3


def theme_for(style_seed: int) -> dict[str, str]:
    return THEMES[style_seed % len(THEMES)]


def is_ood_style(style_seed: int) -> bool:
    return style_seed % len(THEMES) >= N_TRAIN_THEMES


# ----------------------------------------------------------------------------- tab kinds

FIRST = ["Priya", "Marcus", "Elena", "Tomasz", "Aiko", "Diego", "Fatima", "Noah", "Ingrid", "Kwame", "Lena", "Ravi"]
LAST = ["Raman", "Okafor", "Lindqvist", "Nowak", "Tanaka", "Alvarez", "Haddad", "Becker", "Solberg", "Mensah", "Fischer", "Iyer"]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1", "ap-northeast-1", "sa-east-1", "ca-central-1"]
HANDLES = ["spacex", "nasa", "esa", "blueorigin", "rocketlab", "ulalaunch", "isro", "jaxa_en"]
DOC_PAGES = ["Installation", "Authentication", "Rate limits", "Webhooks", "Pagination", "Errors", "Streaming", "Batching"]
SETTINGS_PAGES = ["Notifications", "Privacy", "Security", "Billing", "Appearance", "Accessibility", "Downloads", "Sync"]

# Look-alike capable kinds: every member of a group shares one identical title.
LOOKALIKE_KINDS: dict[str, dict[str, Any]] = {
    "linkedin_post": {"title": "Post | LinkedIn - High memory usage", "attr": "author", "noun": "LinkedIn post"},
    "x_profile": {"title": "Profile / X", "attr": "handle", "noun": "X profile"},
    "grafana": {"title": "Dashboard | Grafana", "attr": "region", "noun": "Grafana dashboard"},
    "docs": {"title": "API Reference - Docs", "attr": "section", "noun": "API docs"},
    "settings": {"title": "Settings", "attr": "page", "noun": "Settings"},
}
SINGLETON_TITLES = [
    ("x_elon", "Elon Musk (@elonmusk) / X"),
    ("x_spacex", "SpaceX on X"),
    ("gmail", "Inbox (3) - Gmail"),
    ("calendar", "Google Calendar - Week of Nov 23"),
    ("github", "Pull requests · lapis-studio"),
    ("youtube", "YouTube"),
    ("hn", "Hacker News"),
    ("wiki", "Chromium - Wikipedia"),
    ("slack", "general (Channel) - Lapis - Slack"),
    ("drive", "My Drive - Google Drive"),
    ("notion", "Roadmap | Notion"),
]
FORM_TITLE = "Submit | Ops Portal"


@dataclass
class Tab:
    idx: int
    slug: str
    kind: str
    title: str
    group: int  # look-alike group id, -1 when unique
    attr_name: str
    attr_value: str
    memory_mb: int
    code: str
    protected: bool = False
    is_form: bool = False

    @property
    def url(self) -> str:
        return f"{BASE_URL}/t/{self.slug}.html"


@dataclass
class Scenario:
    seed: int
    family: str
    style_seed: int
    tabs: list[Tab]
    active_tab_idx: int
    target_idx: int  # switch/close target, or cross-tab source
    dest_idx: int  # cross-tab destination (form tab); -1 otherwise
    mode: str  # cross_tab_lookup: "submit" | "report"; else ""
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> Tab:
        return self.tabs[self.target_idx]

    @property
    def protected_idx(self) -> list[int]:
        return [t.idx for t in self.tabs if t.protected]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _code(rng: random.Random, used: set[str]) -> str:
    while True:
        c = "".join(rng.choices(string.ascii_uppercase, k=3)) + "-" + f"{rng.randrange(10000):04d}"
        if c not in used:
            used.add(c)
            return c


def _slug(rng: random.Random, used: set[str]) -> str:
    while True:
        s = "".join(rng.choices(string.ascii_lowercase + string.digits, k=8))
        if s not in used:
            used.add(s)
            return s


def _attr_values(kind: str, rng: random.Random, k: int) -> list[str]:
    if kind == "linkedin_post":
        firsts = rng.sample(FIRST, k)
        lasts = rng.sample(LAST, k)
        return [f"{f} {l}" for f, l in zip(firsts, lasts)]
    pool = {"x_profile": HANDLES, "grafana": REGIONS, "docs": DOC_PAGES, "settings": SETTINGS_PAGES}[kind]
    vals = rng.sample(pool, k)
    return [f"@{v}" for v in vals] if kind == "x_profile" else vals


def generate(
    seed: int,
    family: str | None = None,
    n_tabs: int | None = None,
    n_lookalikes: int | None = None,
    active_tab_idx: int | None = None,
    protected_tabs: list[int] | None = None,
    style_seed: int | None = None,
    mode: str | None = None,
) -> Scenario:
    """Deterministic scenario for ``seed``; every knob is optional and seeded when None."""
    rng = random.Random(f"tabforge:{seed}")
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    n_tabs = n_tabs if n_tabs is not None else rng.randint(4, 12)
    if not 4 <= n_tabs <= 12:
        raise ValueError("n_tabs must be in 4..12")
    max_groups = max(1, min(3, (n_tabs - 1) // 2))
    n_lookalikes = n_lookalikes if n_lookalikes is not None else rng.randint(1, max_groups)
    if not 1 <= n_lookalikes <= 3:
        raise ValueError("n_lookalikes must be in 1..3")
    n_lookalikes = min(n_lookalikes, max_groups)
    style_seed = style_seed if style_seed is not None else rng.randrange(N_TRAIN_THEMES)

    used_codes: set[str] = {"ABC-1234"}  # the format example in instructions
    used_slugs: set[str] = set()
    mems = rng.sample(range(180, 2400, 7), n_tabs)

    # Look-alike groups: sizes 2..3 while leaving room for the form tab.
    specs: list[tuple[str, str, int, str, str]] = []  # kind, title, group, attr_name, attr_value
    budget = n_tabs - 1
    kinds = rng.sample(list(LOOKALIKE_KINDS), n_lookalikes)
    for g, kind in enumerate(kinds):
        remaining_groups = n_lookalikes - g - 1
        size = rng.randint(2, 3)
        size = max(2, min(size, budget - 2 * remaining_groups))
        budget -= size
        info = LOOKALIKE_KINDS[kind]
        for v in _attr_values(kind, rng, size):
            specs.append((kind, info["title"], g, info["attr"], v))
    singles = rng.sample(SINGLETON_TITLES, len(SINGLETON_TITLES))
    while len(specs) < n_tabs - 1:
        kind, title = singles.pop()
        specs.append((kind, title, -1, "", ""))
    specs.append(("form", FORM_TITLE, -1, "", ""))
    rng.shuffle(specs)

    tabs = [
        Tab(
            idx=i, slug=_slug(rng, used_slugs), kind=k, title=t, group=g, attr_name=an, attr_value=av,
            memory_mb=mems[i], code=_code(rng, used_codes), is_form=(k == "form"),
        )
        for i, (k, t, g, an, av) in enumerate(specs)
    ]
    form_idx = next(t.idx for t in tabs if t.is_form)
    look = [t for t in tabs if t.group >= 0]

    target = rng.choice(look)
    if active_tab_idx is None:
        active_tab_idx = rng.choice([t.idx for t in tabs if t.idx != target.idx])
    if not 0 <= active_tab_idx < n_tabs:
        raise ValueError("active_tab_idx out of range")
    if active_tab_idx == target.idx:
        raise ValueError("active_tab_idx must differ from the target tab")

    if protected_tabs is None:
        cand = [t.idx for t in tabs if t.idx != target.idx]
        k = rng.randint(1, max(1, len(cand) // 2))
        # The start tab and same-title siblings are protected often: the tempting wrong closes.
        protected_tabs = sorted(set(rng.sample(cand, k)) | ({active_tab_idx} if rng.random() < 0.5 else set()))
    if family == "close_only_requested" and target.idx in protected_tabs:
        raise ValueError("close target cannot be protected")
    for i in protected_tabs:
        tabs[i].protected = True

    dest_idx = -1
    mode = mode or ""
    if family == "cross_tab_lookup":
        dest_idx = form_idx
        mode = mode or rng.choice(("submit", "report"))
    instruction = _instruction(family, tabs, target, dest_idx, mode, rng)
    return Scenario(
        seed=seed, family=family, style_seed=style_seed, tabs=tabs, active_tab_idx=active_tab_idx,
        target_idx=target.idx, dest_idx=dest_idx, mode=mode, instruction=instruction,
        meta={"n_tabs": n_tabs, "n_lookalikes": n_lookalikes, "group_size": sum(1 for t in tabs if t.group == target.group),
              "ood_style": is_ood_style(style_seed)},
    )


def _describe(t: Tab, rng: random.Random) -> str:
    noun = LOOKALIKE_KINDS[t.kind]["noun"]
    if rng.random() < 0.6:
        return f'the "{t.title}" tab for the {noun} whose {t.attr_name} is {t.attr_value}'
    return f'the "{t.title}" tab whose memory badge reads {t.memory_mb} MB'


def _instruction(family: str, tabs: list[Tab], target: Tab, dest_idx: int, mode: str, rng: random.Random) -> str:
    keep = "Do not close any tab; tabs marked PROTECTED hold unsaved work."
    desc = _describe(target, rng)
    if family == "switch_to_tab":
        return f"Several tabs share the same title. Switch Chromium to {desc} and leave it in front. {keep}"
    if family == "close_only_requested":
        return (f"Close only {desc}. Keep every other tab open, especially the PROTECTED ones, "
                "and do not close the window.")
    dest = tabs[dest_idx]
    if mode == "submit":
        return (f"Open {desc}, read its reference code (format ABC-1234), then switch to the "
                f'"{dest.title}" tab, enter the code in the form and press Submit. {keep}')
    return (f"Open {desc} and report its reference code (format ABC-1234) as your final answer. {keep}")


# ----------------------------------------------------------------------------- rendering


def css(style_seed: int) -> str:
    th = theme_for(style_seed)
    return (
        f"body{{margin:0;background:{th['bg']};color:{th['fg']};font-family:{th['font']};}}"
        f".wrap{{max-width:860px;margin:0 auto;padding:{th['pad']};}}"
        f".card{{background:{th['card']};border-radius:{th['radius']};padding:{th['pad']};margin:12px 0;}}"
        f".badge{{display:inline-block;border:1px solid {th['accent']};color:{th['accent']};"
        f"border-radius:{th['radius']};padding:2px 8px;margin-right:8px;font-size:13px;}}"
        f".prot{{background:#b42318;color:#fff;padding:6px 10px;border-radius:{th['radius']};font-weight:700;}}"
        f"h1{{font-size:22px;margin:8px 0;}} .code{{font-family:monospace;font-size:18px;font-weight:700;}}"
        f"input{{font-size:18px;padding:6px;width:220px;}} button{{font-size:16px;padding:6px 16px;"
        f"background:{th['accent']};color:#fff;border:0;border-radius:{th['radius']};}}"
    )


LOREM = [
    "Shipping the new rollout plan this week.", "Latency regressions traced to a cold cache.",
    "Quarterly numbers are in, thread below.", "We are hiring across infra and research.",
    "Thoughts on agent reliability in production.", "The launch window moved by two days.",
]


def render_tab(sc: Scenario, t: Tab) -> str:
    rng = random.Random(f"tabforge-body:{sc.seed}:{t.idx}")
    head = f'<span class="badge">Memory usage: {t.memory_mb} MB</span>'
    if t.protected:
        head += '<span class="prot">PROTECTED: unsaved draft, do not close</span>'
    if t.is_form:
        body = (
            '<div class="card"><p>Paste the reference code you were asked to carry over.</p>'
            '<input id="code" placeholder="ABC-1234" autocomplete="off"> '
            '<button id="submit" onclick="sub()">Submit</button><p id="status"></p></div>'
            f'<div class="card">Portal session ref: <span class="code">{t.code}</span></div>'
        )
    else:
        label = f"<p><b>{html.escape(t.attr_name.title())}:</b> {html.escape(t.attr_value)}</p>" if t.attr_name else ""
        body = (
            f'<div class="card">{label}<p>{html.escape(rng.choice(LOREM))}</p>'
            f'<p>Reference code: <span class="code">{t.code}</span></p></div>'
            f'<div class="card"><p>{html.escape(rng.choice(LOREM))}</p></div>'
        )
    script = (
        "<script>function sub(){var v=document.getElementById('code').value;"
        f"fetch('/submit?tab={t.slug}&value='+encodeURIComponent(v)).then(function(r){{return r.text();}})"
        ".then(function(x){document.getElementById('status').textContent=x;});}</script>"
    ) if t.is_form else ""
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(t.title)}</title>"
        f"<style>{css(sc.style_seed)}</style>{script}</head><body><div class='wrap'>"
        f"<h1>{html.escape(t.title)}</h1>{head}{body}</div></body></html>"
    )


SERVER_PY = r'''
import json, os, sys, time, urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
ROOT, PORT, LOG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(self, *a): pass
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/submit":
            q = urllib.parse.parse_qs(u.query)
            rec = {"t": time.time(), "tab": q.get("tab", [""])[0], "value": q.get("value", [""])[0]}
            with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
            b = b"Submitted"
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        return super().do_GET()
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
'''

# Minimal stdlib CDP client: /json listing + Runtime.evaluate over a websocket.
PROBE_PY = r'''
import base64, json, os, socket, sys, urllib.parse, urllib.request
CDP = int(sys.argv[1]) if len(sys.argv) > 1 else 9222
EXPR = sys.argv[2] if len(sys.argv) > 2 else "document.visibilityState"

def ws_eval(ws_url, expr, timeout=4.0):
    u = urllib.parse.urlparse(ws_url)
    s = socket.create_connection((u.hostname, u.port), timeout=timeout)
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = s.recv(4096)
            if not chunk: raise RuntimeError("handshake closed")
            buf += chunk
        head, buf = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]: raise RuntimeError(head[:200].decode("latin1"))
        payload = json.dumps({"id": 1, "method": "Runtime.evaluate",
                              "params": {"expression": expr, "returnByValue": True}}).encode()
        n = len(payload); mask = os.urandom(4); hdr = bytearray([0x81])
        if n < 126: hdr.append(0x80 | n)
        elif n < 65536: hdr.append(0x80 | 126); hdr += n.to_bytes(2, "big")
        else: hdr.append(0x80 | 127); hdr += n.to_bytes(8, "big")
        s.sendall(bytes(hdr) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
        def need(k):
            nonlocal buf
            while len(buf) < k:
                chunk = s.recv(65536)
                if not chunk: raise RuntimeError("socket closed")
                buf += chunk
            out, buf = buf[:k], buf[k:]
            return out
        msg = b""
        while True:
            b0, b1 = need(2)
            ln = b1 & 0x7F
            if ln == 126: ln = int.from_bytes(need(2), "big")
            elif ln == 127: ln = int.from_bytes(need(8), "big")
            data = need(ln)
            msg += data
            if not (b0 & 0x80): continue
            op = b0 & 0x0F
            if op in (0x1, 0x0):
                j = json.loads(msg); msg = b""
                if j.get("id") == 1:
                    return j.get("result", {}).get("result", {}).get("value")
            else:
                msg = b""
    finally:
        s.close()

tabs = []
try:
    lst = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP}/json/list", timeout=4))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "tabs": []})); sys.exit(0)
for t in lst:
    if t.get("type") != "page": continue
    rec = {"id": t.get("id"), "url": t.get("url", ""), "title": t.get("title", "")}
    try:
        rec["value"] = ws_eval(t["webSocketDebuggerUrl"], EXPR)
    except Exception as e:
        rec["value"] = None; rec["error"] = f"{type(e).__name__}: {e}"
    tabs.append(rec)
print(json.dumps({"ok": True, "tabs": tabs}))
'''


# ----------------------------------------------------------------------------- reward (pure)


def slug_of(url: str) -> str | None:
    m = re.search(r"/t/([a-z0-9]{8})\.html", url or "")
    return m.group(1) if m else None


def active_slug(probe: dict[str, Any]) -> str | None:
    """Front tab = the page whose visibilityState is 'visible' (first in /json order if several)."""
    for t in probe.get("tabs", []):
        if t.get("value") == "visible":
            return slug_of(t.get("url", ""))
    return None


def codes_in(text: str | None) -> set[str]:
    return set(CODE_RE.findall((text or "").upper()))


def score(sc: Scenario, probe: dict[str, Any], submissions: list[dict[str, Any]], answer: str | None) -> tuple[float, dict[str, Any]]:
    open_slugs = {slug_of(t.get("url", "")) for t in probe.get("tabs", [])}
    closed = [t for t in sc.tabs if t.slug not in open_slugs]
    closed_protected = [t.idx for t in closed if t.protected]
    info: dict[str, Any] = {"closed": [t.idx for t in closed], "closed_protected": closed_protected,
                            "active": active_slug(probe)}
    if not probe.get("ok", True):
        info["why"] = "probe failed"
        return 0.0, info
    if closed_protected:
        info["why"] = "protected tab closed"
        return -1.0, info
    tgt = sc.target
    if sc.family == "close_only_requested":
        if tgt.slug in open_slugs:
            info["why"] = "target still open"
            return 0.0, info
        extra = [t.idx for t in closed if t.idx != tgt.idx]
        if extra:
            info["why"] = "closed extra tabs"
            return 0.0, info
        return 1.0, info
    if closed:
        info["why"] = "closed a tab unprompted"
        return 0.0, info
    if sc.family == "switch_to_tab":
        ok = info["active"] == tgt.slug
        info["why"] = "ok" if ok else "wrong front tab"
        return (1.0 if ok else 0.0), info
    # cross_tab_lookup
    if sc.mode == "submit":
        dest = sc.tabs[sc.dest_idx]
        mine = [s for s in submissions if s.get("tab") == dest.slug]
        if not mine:
            info["why"] = "no submission from destination tab"
            return 0.0, info
        ok = (mine[-1].get("value") or "").strip().upper() == tgt.code
        info["why"] = "ok" if ok else "wrong code submitted"
        return (1.0 if ok else 0.0), info
    got = codes_in(answer)
    ok = got == {tgt.code}
    info["why"] = "ok" if ok else f"answer codes {sorted(got)}"
    return (1.0 if ok else 0.0), info


def simulate_probe(sc: Scenario, closed: set[int] = frozenset(), active: int | None = None) -> dict[str, Any]:
    """Probe a perfect browser would return after the agent closed ``closed`` and fronted ``active``."""
    active = sc.active_tab_idx if active is None else active
    tabs = []
    for t in sc.tabs:
        if t.idx in closed:
            continue
        tabs.append({"id": t.slug, "url": t.url, "title": t.title, "value": "visible" if t.idx == active else "hidden"})
    return {"ok": True, "tabs": tabs}


# ----------------------------------------------------------------------------- sandbox wiring


def _b64_write(path: str, content: str) -> str:
    import base64

    b64 = base64.b64encode(content.encode()).decode()
    return f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0])} && echo {b64} | base64 -d > {shlex.quote(path)}"


# Chromium keeps writing its profile for a moment after SIGTERM; wait, then force.
KILL_CHROMIUM = (
    "pkill -f '[c]hromium' ; for i in $(seq 1 40); do pgrep -f '[c]hromium' >/dev/null || break; sleep 0.25; done ; "
    "pkill -9 -f '[c]hromium' ; true"
)
# --test-type hides the "--no-sandbox" warning bar so the viewport looks like stock Chrome.
CHROME_FLAGS = (
    "--no-sandbox --test-type --disable-gpu --disable-dev-shm-usage "
    f"--remote-debugging-port={CDP_PORT} --remote-allow-origins='*' "
    "--no-first-run --no-default-browser-check --disable-extensions --disable-features=Translate "
    "--password-store=basic --window-size=1024,768 --window-position=0,0"
)


def setup_commands(sc: Scenario) -> list[str]:
    cmds = [f"{KILL_CHROMIUM} ; pkill -f '[t]abforge/server.py' ; rm -rf {ROOT} && mkdir -p {ROOT}/t"]
    cmds.append(_b64_write(f"{ROOT}/server.py", SERVER_PY))
    cmds.append(_b64_write(f"{ROOT}/probe.py", PROBE_PY))
    cmds.append(_b64_write(f"{ROOT}/scenario.json", sc.to_json()))
    for t in sc.tabs:
        cmds.append(_b64_write(f"{ROOT}/t/{t.slug}.html", render_tab(sc, t)))
    cmds.append(f": > {SUBMISSIONS}; nohup python3 {ROOT}/server.py {ROOT} {PORT} {SUBMISSIONS} > {ROOT}/server.log 2>&1 &")
    cmds.append(f"for i in $(seq 1 40); do curl -sf {BASE_URL}/scenario.json >/dev/null && break; sleep 0.25; done")
    urls = " ".join(shlex.quote(t.url) for t in sc.tabs)
    cmds.append(
        f"export DISPLAY=:0; nohup {CHROMIUM} {CHROME_FLAGS} --user-data-dir={ROOT}/profile "
        f"{urls} > {ROOT}/chrome.log 2>&1 &"
    )
    return cmds


def activate_command(sc: Scenario, idx: int) -> str:
    """Bring tab ``idx`` to front through CDP (used once at setup for the seeded start tab)."""
    url = sc.tabs[idx].url
    return (
        f"id=$(curl -s http://127.0.0.1:{CDP_PORT}/json/list | python3 -c "
        f"\"import json,sys; print(next(t['id'] for t in json.load(sys.stdin) if t.get('url')=={url!r}))\") && "
        f"curl -s http://127.0.0.1:{CDP_PORT}/json/activate/$id"
    )


def setup(env: Any, sc: Scenario, wait_s: float = 30.0) -> None:
    for c in setup_commands(sc):
        r = env.exec(c)
        if getattr(r, "exit_code", 0) not in (0, None):
            raise RuntimeError(f"setup command failed ({r.exit_code}): {c[:80]} -> {r.output[-300:]}")
    deadline = time.monotonic() + wait_s
    while True:
        p = probe(env)
        if p.get("ok") and len(p["tabs"]) >= len(sc.tabs) and all(t.get("title") for t in p["tabs"]):
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"chromium did not open {len(sc.tabs)} tabs: {json.dumps(p)[:400]}")
        time.sleep(0.5)
    env.exec(activate_command(sc, sc.active_tab_idx))
    time.sleep(0.5)


def probe(env: Any, expr: str = "document.visibilityState") -> dict[str, Any]:
    r = env.exec(f"python3 {ROOT}/probe.py {CDP_PORT} {shlex.quote(expr)}")
    try:
        return json.loads((r.output or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": (r.output or "")[-300:], "tabs": []}


def read_submissions(env: Any) -> list[dict[str, Any]]:
    r = env.exec(f"cat {SUBMISSIONS} 2>/dev/null || true")
    out = []
    for line in (r.output or "").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def reward(env: Any, sc: Scenario, answer: str | None) -> tuple[float, dict[str, Any]]:
    return score(sc, probe(env), read_submissions(env), answer)


def make_task(sc: Scenario):
    """Wrap as a ``tasks.Task`` (reward returns only the float)."""
    from tasks import Task

    return Task(
        id=f"tabforge/{sc.family}/{sc.seed}",
        instruction=sc.instruction,
        setup=lambda env: setup(env, sc),
        reward=lambda env, answer: reward(env, sc, answer)[0],
        max_steps=20,
        meta={"env": "tabforge", "family": sc.family, "seed": sc.seed, **sc.meta},
    )
