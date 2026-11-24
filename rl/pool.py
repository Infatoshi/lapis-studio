"""Persistent pool of Daytona ``daytona-small`` sandboxes for fast CUA RL episodes.

Each worker boots ONCE (about 10 s): sandbox create, ``computer_use`` (Xvfb desktop), a stdlib
HTTP control server on port 8765, and one Chromium with the DevTools port 9222. Episodes after
that never rebuild anything: ``reset_worker`` posts a new app spec to ``POST /__reset``, which
swaps the served HTML/JS, the in-memory SQLite db and the event log, then navigates Chromium via
CDP ``Page.navigate`` and waits for the app's ``ready`` event.

Control server routes (all ``/__*`` routes need the per-pool key in ``X-Pool-Key``, so the
policy cannot read ground truth by typing a URL into Chromium):
  * ``GET  /``             the active app page (episode id injected as ``window.__EP``)
  * ``POST /api/event``    app -> server audit log; ``set`` patches the ground-truth state;
                           events from a stale episode are dropped
  * ``POST /api/sql``      query the episode's SQLite db (for apps that keep state there)
  * ``GET  /__health``     liveness + CDP reachability (no key, no data)
  * ``POST /__reset``      swap app + navigate; returns a timing breakdown in ms
  * ``GET  /__state``      ``{ep, name, state, events, dropped}``
  * ``POST /__ax``         ``Accessibility.getFullAXTree`` + per-node layout bounds + viewport origin
  * ``POST /__cdp``        raw CDP passthrough ``{method, params}``

The Mac talks to it through Daytona's preview proxy (``get_preview_link(8765)``), falling back
to ``process.exec`` + curl if the proxy is unreachable. The server binds 0.0.0.0 so the proxy
can reach it; Chromium loads it at 127.0.0.1:8765.

Cost guard: at most 32 workers (32 x $0.07/hr = $2.24/hr). Sandboxes carry a ``lapis-pool``
label, auto-stop after ``idle_stop_min`` idle minutes and auto-delete on stop, and
``close_all`` (also run at exit) deletes every one and checks the label query comes back empty.
"""

from __future__ import annotations

import atexit
import base64
import concurrent.futures as cf
import json
import os
import re
import secrets
import shlex
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from studio_contract import (
    StudioAction,
    format_cdp_ax_lines,
    parse_cua_action,
    parse_studio_actions,
    point_to_pixels,
)

MAX_WORKERS = 32
USD_PER_WORKER_HR = 0.07
CTL_PORT = 8765
CDP_PORT = 9222
SCREEN_W, SCREEN_H = 1024, 768
POOL_DIR = "/tmp/lapis_pool"
LABEL = "lapis-pool"
CHROME_FLAGS = (
    "--no-sandbox --test-type --disable-gpu --disable-dev-shm-usage "
    f"--remote-debugging-port={CDP_PORT} '--remote-allow-origins=*' "  # quoted: the sandbox shell is zsh
    f"--window-size={SCREEN_W},{SCREEN_H} --window-position=0,0 "
    f"--user-data-dir={POOL_DIR}/chrome --no-first-run --no-default-browser-check "
    "--disable-extensions --disable-component-update --disable-features=Translate,MediaRouter "
    "--disable-session-crashed-bubble --password-store=basic"
)

# --------------------------------------------------------------------------- in-sandbox control server

CONTROL_SERVER_SRC = r'''
import base64, json, os, socket, sqlite3, struct, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get("POOL_PORT", "8765"))
CDP_PORT = int(os.environ.get("POOL_CDP_PORT", "9222"))
KEY = os.environ["POOL_KEY"]
IDLE_HTML = "<!doctype html><html><head><title>pool idle</title></head><body><h1>idle</h1></body></html>"


class WS:
    """Minimal RFC 6455 client (text frames, ping/pong, fragmentation) for one CDP target."""

    def __init__(self, url):
        u = urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET %s HTTP/1.1\r\nHost: %s:%s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n") % (u.path, u.hostname, u.port, key)
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("ws handshake closed")
            buf += chunk
        head, rest = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError("ws handshake failed: %r" % head[:200])
        self.buf = bytearray(rest)

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self.buf)))
            if not chunk:
                raise ConnectionError("ws closed")
            self.buf += chunk
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def send(self, data, opcode=1):
        if isinstance(data, str):
            data = data.encode()
        n = len(data)
        hdr = bytearray([0x80 | opcode])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127)
            hdr += struct.pack(">Q", n)
        mask = os.urandom(4)
        hdr += mask
        body = b""
        if n:
            m = (mask * (n // 4 + 1))[:n]
            body = (int.from_bytes(data, "big") ^ int.from_bytes(m, "big")).to_bytes(n, "big")
        self.sock.sendall(bytes(hdr) + body)

    def recv(self, timeout=None):
        self.sock.settimeout(timeout)
        parts = []
        while True:
            b0, b1 = self._read(2)
            fin, op, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if mask and n:
                m = (mask * (n // 4 + 1))[:n]
                payload = (int.from_bytes(payload, "big") ^ int.from_bytes(m, "big")).to_bytes(n, "big")
            if op == 8:
                raise ConnectionError("ws close frame")
            if op == 9:
                self.send(payload, 10)
                continue
            if op == 10:
                continue
            parts.append(payload)
            if fin:
                return b"".join(parts).decode("utf-8", "replace")


class CDPError(Exception):
    pass


class CDP:
    def __init__(self):
        self.ws = None
        self.n = 0
        self.lock = threading.RLock()
        self.events = []

    def _connect(self):
        with urllib.request.urlopen("http://127.0.0.1:%d/json" % CDP_PORT, timeout=2) as r:
            tabs = json.load(r)
        page = next(t for t in tabs if t.get("type") == "page")
        self.ws = WS(page["webSocketDebuggerUrl"])
        self.events = []
        self._call("Page.enable", {}, 5)

    def _call(self, method, params, timeout):
        self.n += 1
        mid = self.n
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            msg = json.loads(self.ws.recv(max(0.01, deadline - time.monotonic())))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise CDPError(json.dumps(msg["error"]))
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)
                del self.events[:-200]

    def call(self, method, params=None, timeout=10.0):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.ws is None:
                        self._connect()
                    return self._call(method, params, timeout)
                except (OSError, ConnectionError, StopIteration, ValueError):
                    self.ws = None
                    if attempt:
                        raise

    def wait_event(self, name, timeout):
        with self.lock:
            deadline = time.monotonic() + timeout
            while True:
                for i, e in enumerate(self.events):
                    if e.get("method") == name:
                        del self.events[: i + 1]
                        return e
                left = deadline - time.monotonic()
                if left <= 0 or self.ws is None:
                    return None
                try:
                    msg = json.loads(self.ws.recv(left))
                except (OSError, ConnectionError, ValueError):
                    return None
                if "method" in msg:
                    self.events.append(msg)


CDPC = CDP()
LOCK = threading.Lock()
READY = threading.Condition(LOCK)
APP = {"ep": 0, "name": "idle", "html": IDLE_HTML, "state": {}, "events": [], "dropped": 0, "ready_ep": -1,
       "db": sqlite3.connect(":memory:", check_same_thread=False)}


def inject(html, ep):
    tag = "<script>window.__EP=%d;</script>" % ep
    i = html.find("<head>")
    return html[: i + 6] + tag + html[i + 6 :] if i >= 0 else tag + html


def ms(a, b):
    return round((b - a) * 1000, 2)


def reset(spec):
    t0 = time.perf_counter()
    db = sqlite3.connect(":memory:", check_same_thread=False)
    if spec.get("sql"):
        db.executescript(spec["sql"])
    with LOCK:
        APP["ep"] += 1
        ep = APP["ep"]
        APP.update(name=spec.get("name", "app"), html=inject(spec.get("html") or IDLE_HTML, ep),
                   state=dict(spec.get("state") or {}), events=[], dropped=0, ready_ep=-1, db=db)
    t1 = time.perf_counter()
    out = {"ep": ep, "swap_ms": ms(t0, t1)}
    if spec.get("navigate", True):
        with CDPC.lock:
            CDPC.events = [e for e in CDPC.events if e.get("method") != "Page.loadEventFired"]
            CDPC.call("Page.navigate", {"url": "http://127.0.0.1:%d/?ep=%d" % (PORT, ep)})
        t2 = time.perf_counter()
        out["navigate_ms"] = ms(t1, t2)
        if spec.get("wait_ready", True):
            with READY:
                ok = READY.wait_for(lambda: APP["ready_ep"] == ep, timeout=float(spec.get("ready_timeout", 3.0)))
            out["ready"] = bool(ok)
        else:
            out["ready"] = CDPC.wait_event("Page.loadEventFired", 3.0) is not None
        out["ready_ms"] = ms(t2, time.perf_counter())
    out["total_ms"] = ms(t0, time.perf_counter())
    return out


def ax_snapshot():
    t0 = time.perf_counter()
    tree = CDPC.call("Accessibility.getFullAXTree", {})
    snap = CDPC.call("DOMSnapshot.captureSnapshot", {"computedStyles": []})
    geo = json.loads(CDPC.call("Runtime.evaluate", {
        "expression": "JSON.stringify([screenX,screenY,outerWidth,outerHeight,innerWidth,innerHeight,devicePixelRatio,scrollX,scrollY])",
        "returnByValue": True})["result"]["value"])
    doc = snap["documents"][0]
    backend = doc["nodes"]["backendNodeId"]
    sx, sy = doc.get("scrollOffsetX", 0), doc.get("scrollOffsetY", 0)
    bounds = {}
    for ni, b in zip(doc["layout"]["nodeIndex"], doc["layout"]["bounds"]):
        bounds[backend[ni]] = [b[0] - sx, b[1] - sy, b[2], b[3]]
    keep = ("nodeId", "parentId", "childIds", "role", "name", "value", "ignored", "backendDOMNodeId")
    nodes = []
    for n in tree["nodes"]:
        s = {k: n[k] for k in keep if k in n}
        b = bounds.get(n.get("backendDOMNodeId"))
        if b is not None:
            s["bounds"] = b
        nodes.append(s)
    scr_x, scr_y, ow, oh, iw, ih, dpr, _, _ = geo
    inset = max(0, (ow - iw) / 2)  # client-side frame shadow, same on left/right/bottom
    origin = [scr_x + inset, scr_y + max(0, oh - ih - inset)]  # estimate; the pool calibrates
    return {"nodes": nodes, "origin": origin, "scale": 1.0, "dpr": dpr, "inner": [iw, ih],
            "scroll": [geo[7], geo[8]], "ms": ms(t0, time.perf_counter())}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        if self.headers.get("X-Pool-Key") == KEY:
            return True
        return parse_qs(urlparse(self.path).query).get("k", [""])[0] == KEY

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with LOCK:
                html = APP["html"]
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        if path == "/__health":
            try:
                CDPC.call("Runtime.evaluate", {"expression": "1", "returnByValue": True}, 2.0)
                cdp = True
            except Exception as e:
                cdp = False
            with LOCK:
                ep = APP["ep"]
            return self._send(200, {"ok": True, "cdp": cdp, "ep": ep})
        if path == "/__state":
            if not self._authed():
                return self._send(403, {"error": "forbidden"})
            with LOCK:
                out = {k: APP[k] for k in ("ep", "name", "state", "events", "dropped")}
                out = json.loads(json.dumps(out))
            return self._send(200, out)
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if path == "/api/event":
            with LOCK:
                if body.get("ep") != APP["ep"]:
                    APP["dropped"] += 1
                else:
                    body["t"] = time.time()
                    APP["events"].append(body)
                    APP["state"].update(body.get("set") or {})
                    if body.get("type") == "ready":
                        APP["ready_ep"] = APP["ep"]
                        READY.notify_all()
            return self._send(200, {"ok": True})
        if path == "/api/sql":
            with LOCK:
                db = APP["db"]
                try:
                    cur = db.execute(body.get("sql", ""), body.get("params") or [])
                    rows = cur.fetchall()
                    db.commit()
                except sqlite3.Error as e:
                    return self._send(400, {"error": str(e)})
            return self._send(200, {"rows": rows})
        if not self._authed():
            return self._send(403, {"error": "forbidden"})
        try:
            if path == "/__reset":
                return self._send(200, reset(body))
            if path == "/__ax":
                return self._send(200, ax_snapshot())
            if path == "/__cdp":
                t0 = time.perf_counter()
                res = CDPC.call(body["method"], body.get("params") or {}, float(body.get("timeout", 10)))
                return self._send(200, {"result": res, "ms": ms(t0, time.perf_counter())})
        except Exception as e:
            return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})
        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    srv.daemon_threads = True
    srv.serve_forever()
'''

# --------------------------------------------------------------------------- AX lines -> click targets

_LINE_RE = re.compile(
    r'^(?P<role>\w+)(?: "(?P<title>(?:[^"\\]|\\.)*)")?(?: = "(?P<value>(?:[^"\\]|\\.)*)")?'
    r" @(?P<x>-?\d+),(?P<y>-?\d+) (?P<w>\d+)x(?P<h>\d+)(?: \{(?P<acts>[^}]*)\})?$"
)


@dataclass(frozen=True)
class AxLine:
    role: str
    title: str
    value: str
    rect: tuple[int, int, int, int]
    pressable: bool

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.rect
        return x + w // 2, y + h // 2


def _unq(s: str | None) -> str:
    return "" if s is None else json.loads(f'"{s}"') if "\\" in s else s


def parse_ax_lines(text: str) -> list[AxLine]:
    """Studio's compact AX lines -> structured nodes (header and ellipsis lines skipped)."""
    out = []
    for raw in text.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            continue
        out.append(AxLine(
            role=m["role"], title=_unq(m["title"]), value=_unq(m["value"]),
            rect=(int(m["x"]), int(m["y"]), int(m["w"]), int(m["h"])),
            pressable="Press" in (m["acts"] or ""),
        ))
    return out


TEXT_ROLES = frozenset({"TextField", "TextArea", "ComboBox", "SearchField"})


def resolve_target(lines: list[AxLine], target: str, screen: tuple[int, int] = (SCREEN_W, SCREEN_H),
                   text_input: bool = False) -> AxLine | None:
    """Pick the element a Studio click target names: exact title, exact value, title prefix,
    title substring, value substring; within a tier prefer on-screen, then pressable, then
    document order. ``"App|element"`` targets drop the app part. ``text_input`` (cua_type)
    searches text fields first."""
    if "|" in target:
        target = target.split("|", 1)[1]
    t = target.strip().lower()
    if not t:
        return None
    if text_input:
        fields = [n for n in lines if n.role in TEXT_ROLES]
        hit = resolve_target(fields, target, screen) if fields else None
        if hit is not None:
            return hit

    def on_screen(n: AxLine) -> bool:
        cx, cy = n.center
        return n.rect[2] > 0 and n.rect[3] > 0 and 0 <= cx < screen[0] and 0 <= cy < screen[1]

    tiers = (
        lambda n: n.title.lower() == t,
        lambda n: n.value.lower() == t,
        lambda n: n.title.lower().startswith(t),
        lambda n: t in n.title.lower(),
        lambda n: t in n.value.lower(),
    )
    for pred in tiers:
        hits = [n for n in lines if pred(n)]
        if hits:
            return min(hits, key=lambda n: (not on_screen(n), not n.pressable, lines.index(n)))
    return None


_KEY_NAMES = {
    "return": "Return", "enter": "Return", "esc": "Escape", "escape": "Escape", "tab": "Tab", "space": "space",
    "delete": "BackSpace", "backspace": "BackSpace", "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "pageup": "Page_Up", "pagedown": "Page_Down", "home": "Home", "end": "End", "forwarddelete": "Delete",
}
_MOD_NAMES = {"cmd": "ctrl", "command": "ctrl", "meta": "ctrl", "ctrl": "ctrl", "control": "ctrl",
              "opt": "alt", "option": "alt", "alt": "alt", "shift": "shift"}


def map_hotkey(combo: str) -> tuple[str, list[str]]:
    """Studio combo ("cmd+l", "return") -> (X11 key, modifiers) for a Linux sandbox (cmd -> ctrl)."""
    parts = [p for p in combo.lower().split("+") if p]
    mods = [_MOD_NAMES[p] for p in parts[:-1] if p in _MOD_NAMES]
    key = parts[-1] if parts else ""
    return _KEY_NAMES.get(key, key), mods


# --------------------------------------------------------------------------- pool


@dataclass
class Worker:
    idx: int
    sandbox: Any
    key: str
    url: str = ""
    token: str = ""
    via: str = "exec"  # "preview" or "exec"
    http: httpx.Client | None = None
    boot: dict[str, Any] = field(default_factory=dict)
    origin: tuple[float, float] | None = None  # viewport (0,0) in X11 screen px, measured at boot
    lock: threading.Lock = field(default_factory=threading.Lock)


class PoolError(RuntimeError):
    pass


class SandboxPool:
    """``pool.start(n)`` then per-episode ``reset_worker`` / ``act`` / ``get_cdp_ax`` /
    ``screenshot`` / ``state``; ``close_all`` (or the context manager / atexit) tears down."""

    def __init__(self, snapshot: str = "daytona-small", api_key: str | None = None, idle_stop_min: int = 30,
                 pool_id: str | None = None):
        key = (api_key or os.environ.get("DAYTONA_API_KEY", "")).strip()
        if not key:
            raise PoolError("DAYTONA_API_KEY is not set")
        from daytona import Daytona, DaytonaConfig

        self._client = Daytona(DaytonaConfig(api_key=key)).v0
        self.snapshot = snapshot
        self.idle_stop_min = idle_stop_min
        self.pool_id = pool_id or f"pool-{secrets.token_hex(4)}"
        self.workers: list[Worker] = []
        self._closed = False
        atexit.register(self.close_all)

    # ----------------------------------------------------------------- lifecycle

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close_all()

    def start(self, n: int, max_parallel: int = 8, retries: int = 1) -> list[dict[str, Any]]:
        if not 1 <= n <= MAX_WORKERS:
            raise ValueError(f"n must be 1..{MAX_WORKERS} (${MAX_WORKERS * USD_PER_WORKER_HR:.2f}/hr cap)")
        if len(self.workers) + n > MAX_WORKERS:
            raise ValueError(f"pool would exceed {MAX_WORKERS} workers")
        base = len(self.workers)
        results: dict[int, Worker] = {}
        errors: list[str] = []

        def boot(i: int) -> Worker:
            last: Exception | None = None
            for _ in range(retries + 1):
                try:
                    return self._boot_one(i)
                except Exception as e:  # noqa: BLE001 - keep booting the rest
                    last = e
            raise PoolError(f"worker {i} failed to boot: {type(last).__name__}: {last}")

        with cf.ThreadPoolExecutor(max_workers=min(max_parallel, n)) as ex:
            futs = {ex.submit(boot, base + j): base + j for j in range(n)}
            for f in cf.as_completed(futs):
                try:
                    results[futs[f]] = f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
        for i in sorted(results):
            self.workers.append(results[i])
        for j, w in enumerate(self.workers):
            w.idx = j
        if errors and not results:
            raise PoolError("; ".join(errors))
        return [w.boot for w in self.workers[base:]] + [{"error": e} for e in errors]

    def _boot_one(self, i: int) -> Worker:
        from daytona.v0 import CreateSandboxFromSnapshotParams

        t0 = time.perf_counter()
        sb = self._client.create(
            CreateSandboxFromSnapshotParams(
                snapshot=self.snapshot, labels={LABEL: self.pool_id},
                auto_stop_interval=self.idle_stop_min, auto_delete_interval=0,
            ),
            timeout=180,
        )
        w = Worker(idx=i, sandbox=sb, key=secrets.token_urlsafe(24))
        try:
            w.boot["create_s"] = round(time.perf_counter() - t0, 3)
            t = time.perf_counter()
            sb.computer_use.start()
            w.boot["computer_use_s"] = round(time.perf_counter() - t, 3)
            t = time.perf_counter()
            sb.process.exec(f"mkdir -p {POOL_DIR}")
            sb.fs.upload_file(CONTROL_SERVER_SRC.encode(), f"{POOL_DIR}/ctl.py")
            sb.process.exec(
                f"cd {POOL_DIR} && POOL_KEY={shlex.quote(w.key)} POOL_PORT={CTL_PORT} POOL_CDP_PORT={CDP_PORT} "
                f"setsid nohup python3 {POOL_DIR}/ctl.py > {POOL_DIR}/ctl.log 2>&1 < /dev/null &"
            )
            sb.process.exec(
                f"export DISPLAY=${{DISPLAY:-:0}}; setsid nohup chromium {CHROME_FLAGS} "
                f"http://127.0.0.1:{CTL_PORT}/ > {POOL_DIR}/chrome.log 2>&1 < /dev/null &"
            )
            deadline = time.monotonic() + 30
            health: dict[str, Any] = {}
            while time.monotonic() < deadline:
                health = self._exec_json(w, "GET", "/__health", timeout=5) or {}
                if health.get("cdp"):
                    break
                time.sleep(0.3)
            if not health.get("cdp"):
                log = sb.process.exec(f"tail -n 20 {POOL_DIR}/ctl.log {POOL_DIR}/chrome.log").result
                raise PoolError(f"control server / CDP not healthy: {health} {log[-800:]}")
            w.boot["server_chrome_s"] = round(time.perf_counter() - t, 3)
            t = time.perf_counter()
            try:
                link = sb.get_preview_link(CTL_PORT)
                w.url, w.token = link.url.rstrip("/"), link.token
                w.http = httpx.Client(timeout=15, headers={"X-Daytona-Preview-Token": w.token, "X-Pool-Key": w.key})
                r = w.http.get(f"{w.url}/__health")
                r.raise_for_status()
                w.via = "preview" if r.json().get("ok") else "exec"
            except Exception as e:  # noqa: BLE001 - fall back to exec transport
                w.via = "exec"
                w.boot["preview_error"] = f"{type(e).__name__}: {e}"[:300]
            w.boot["preview_s"] = round(time.perf_counter() - t, 3)
            t = time.perf_counter()
            w.origin = self._calibrate(w)
            w.boot.update(calibrate_s=round(time.perf_counter() - t, 3), origin=w.origin)
            w.boot.update(via=w.via, sandbox_id=sb.id, total_s=round(time.perf_counter() - t0, 3))
            return w
        except Exception:
            try:
                sb.delete()
            except Exception:  # noqa: BLE001
                pass
            raise

    def close_all(self, verify_timeout_s: float = 90.0) -> dict[str, Any]:
        """Delete every sandbox this pool made (tracked workers plus anything carrying this
        pool's label) and wait until the label query lists none."""
        if self._closed and not self.workers:
            return {"deleted": 0, "active_sandboxes": self._count_labeled()}
        deleted, errors = 0, []
        for w in self.workers:
            if w.http is not None:
                w.http.close()
            try:
                w.sandbox.delete()
                deleted += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{w.sandbox.id}: {type(e).__name__}: {e}")
        self.workers = []
        deadline = time.monotonic() + verify_timeout_s
        remaining = self._count_labeled(delete_stragglers=True)
        while remaining and time.monotonic() < deadline:
            time.sleep(2)
            remaining = self._count_labeled(delete_stragglers=True)
        self._closed = True
        return {"deleted": deleted, "active_sandboxes": remaining, "errors": errors}

    def _count_labeled(self, delete_stragglers: bool = False) -> int:
        from daytona.v0 import ListSandboxesQuery

        gone = {"destroyed", "destroying"}
        live = []
        for sb in self._client.list(ListSandboxesQuery(labels={LABEL: self.pool_id})):
            state = str(getattr(sb.state, "value", sb.state)).lower()
            if state not in gone:
                live.append(sb)
        if delete_stragglers:
            for sb in live:
                try:
                    sb.delete()
                except Exception:  # noqa: BLE001
                    pass
        return len(live)

    def account_active(self) -> int:
        """Non-destroyed sandboxes on the whole account (other pools included)."""
        n = 0
        for sb in self._client.list():
            if str(getattr(sb.state, "value", sb.state)).lower() not in ("destroyed", "destroying"):
                n += 1
        return n

    # ----------------------------------------------------------------- transport

    def _exec_json(self, w: Worker, method: str, path: str, body: Any = None, timeout: int = 15) -> Any:
        hdr = f"-H 'X-Pool-Key: {w.key}'"
        url = f"http://127.0.0.1:{CTL_PORT}{path}"
        if body is None:
            cmd = f"curl -s -m {timeout} -X {method} {hdr} {url}"
        else:
            b64 = base64.b64encode(json.dumps(body).encode()).decode()
            cmd = (f"echo {b64} | base64 -d | curl -s -m {timeout} -X {method} {hdr} "
                   f"-H 'Content-Type: application/json' --data-binary @- {url}")
        r = w.sandbox.process.exec(cmd, timeout=timeout + 5)
        out = (r.result or "").strip()
        try:
            return json.loads(out) if out else None
        except ValueError:
            return None

    def _calibrate(self, w: Worker) -> tuple[float, float]:
        """Viewport origin in screen px, measured: move the real X11 pointer to a known point and
        read the page's clientX/Y. Chromium draws its own frame and shadow without a window
        manager, so outer-minus-inner window geometry is off by the shadow inset."""
        install = ("window.__pool_mm=null;document.addEventListener('mousemove',"
                   "e=>{window.__pool_mm=[e.clientX,e.clientY]},true);1")
        self._ctl(w, "POST", "/__cdp", {"method": "Runtime.evaluate", "params": {"expression": install}})
        cu = w.sandbox.computer_use
        probe = (SCREEN_W // 2, SCREEN_H // 2 + 60)
        for _ in range(3):
            cu.mouse.move(probe[0] - 40, probe[1] - 40)
            cu.mouse.move(*probe)
            time.sleep(0.1)
            r = self._ctl(w, "POST", "/__cdp", {"method": "Runtime.evaluate", "params": {
                "expression": "JSON.stringify(window.__pool_mm)", "returnByValue": True}})
            mm = json.loads(r["result"]["result"].get("value") or "null")
            if mm:
                return (float(probe[0] - mm[0]), float(probe[1] - mm[1]))
        raise PoolError("viewport calibration saw no mousemove")

    def _ctl(self, i: int | Worker, method: str, path: str, body: Any = None, timeout: float = 15) -> Any:
        w = i if isinstance(i, Worker) else self.workers[i]
        if w.via == "preview" and w.http is not None:
            r = w.http.request(method, f"{w.url}{path}", json=body, timeout=timeout)
            if r.status_code >= 400:
                raise PoolError(f"{path} -> {r.status_code}: {r.text[:300]}")
            return r.json()
        out = self._exec_json(w, method, path, body, int(timeout))
        if out is None or "error" in (out if isinstance(out, dict) else {}):
            raise PoolError(f"{path} via exec failed: {out}")
        return out

    # ----------------------------------------------------------------- episode API

    def reset_worker(self, i: int, app_spec: Mapping[str, Any]) -> dict[str, Any]:
        """Swap worker ``i`` to ``app_spec`` ({name, html, state?, sql?}) and reload Chromium.
        Returns the server's timing breakdown plus the client-measured ``rtt_ms``."""
        t0 = time.perf_counter()
        out = self._ctl(i, "POST", "/__reset", dict(app_spec))
        out["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if not out.get("ready"):
            raise PoolError(f"worker {i} app did not become ready: {out}")
        return out

    def state(self, i: int) -> dict[str, Any]:
        return self._ctl(i, "GET", "/__state")

    def cdp(self, i: int, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._ctl(i, "POST", "/__cdp", {"method": method, "params": dict(params or {})})["result"]

    def eval_js(self, i: int, expression: str) -> Any:
        r = self.cdp(i, "Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return r.get("result", {}).get("value")

    def ax_raw(self, i: int) -> dict[str, Any]:
        return self._ctl(i, "POST", "/__ax", {})

    def get_cdp_ax(self, i: int, max_chars: int = 6000, raw: Mapping[str, Any] | None = None) -> str:
        """Studio-format AX lines (``format_cdp_ax_lines``) in screen pixels."""
        snap = raw or self.ax_raw(i)
        origin = self.workers[i].origin or tuple(snap["origin"])
        return format_cdp_ax_lines(
            snap["nodes"], origin=origin, scale=float(snap.get("dpr") or 1.0),
            app_name="Chromium", max_chars=max_chars,
        )

    def screenshot(self, i: int) -> bytes:
        shot = self.workers[i].sandbox.computer_use.screenshot.take_full_screen()
        return base64.b64decode(shot.screenshot)

    def act(self, i: int, action: StudioAction | tuple[str, str] | str) -> dict[str, Any]:
        """Run one Studio action (``[[ACTION:cua_click:Open post p012]]``, a ``StudioAction``, or
        ``(name, arg)``) with real X11 input through ``computer_use``. Click targets resolve
        against the same AX lines the policy sees; ``@x,y`` (0..1000) points are honored."""
        if isinstance(action, str):
            acts = parse_studio_actions(action)
            if not acts:
                return {"ok": False, "detail": "no [[ACTION:...]] tag"}
            action = acts[0]
        name, arg = (action.name, action.arg) if isinstance(action, StudioAction) else action
        cmd = parse_cua_action(name, arg, allow_point=True)
        t0 = time.perf_counter()
        res: dict[str, Any] = {"ok": False, "kind": cmd.kind, "detail": ""}
        if not cmd.valid:
            res["detail"] = f"invalid action {name}:{arg}"
            return res
        cu = self.workers[i].sandbox.computer_use

        def locate(target: str, text_input: bool = False) -> tuple[int, int] | None:
            raw = self.ax_raw(i)
            lines = parse_ax_lines(self.get_cdp_ax(i, max_chars=60000, raw=raw))
            node = resolve_target(lines, target, text_input=text_input)
            if node is None:
                res["detail"] = f"no element matches {target!r}"
                return None
            res["matched"] = f'{node.role} "{node.title or node.value}"'
            ox, oy = self.workers[i].origin or raw["origin"]
            dpr = float(raw.get("dpr") or 1.0)
            cx, cy = node.center
            if not (ox <= cx < ox + raw["inner"][0] * dpr and oy <= cy < oy + raw["inner"][1] * dpr):
                # a real click there would land on whatever covers that point; make the policy scroll
                res["detail"] = f"{res['matched']} is off screen at {cx},{cy}; scroll first"
                return None
            return node.center

        if cmd.kind == "click":
            xy = point_to_pixels(cmd.point_1000, SCREEN_W, SCREEN_H) if cmd.point_1000 else locate(cmd.target)
            if xy is not None:
                cu.mouse.click(*xy)
                res.update(ok=True, x=xy[0], y=xy[1])
        elif cmd.kind == "type":
            xy = locate(cmd.target, text_input=True)
            if xy is not None:
                cu.mouse.click(*xy)
                cu.keyboard.press("a", modifiers=["ctrl"])
                cu.keyboard.type(cmd.text)
                res.update(ok=True, x=xy[0], y=xy[1])
        elif cmd.kind == "scroll":
            xy = (locate(cmd.target) if cmd.target else None) or (SCREEN_W // 2, SCREEN_H // 2 + 40)
            direction = "up" if cmd.delta_y > 0 else "down"
            cu.mouse.scroll(xy[0], xy[1], direction, max(1, min(30, abs(cmd.delta_y))))
            res.update(ok=True, x=xy[0], y=xy[1], direction=direction)
        elif cmd.kind == "hotkey":
            key, mods = map_hotkey(cmd.text)
            cu.keyboard.press(key, modifiers=mods or None)
            res.update(ok=True, key=key, modifiers=mods)
        elif cmd.kind == "open":
            if cmd.text:
                self.cdp(i, "Page.navigate", {"url": cmd.text})
                res.update(ok=True, url=cmd.text)
            else:
                res["detail"] = "only URL opens are supported in the sandbox"
        elif cmd.kind == "drag":
            a, b = locate(cmd.target), locate(cmd.text)
            if a is None or b is None:
                res["detail"] = "drag endpoints not found"
            else:
                cu.mouse.drag(a[0], a[1], b[0], b[1])
                res.update(ok=True)
        elif cmd.kind == "observe":
            res.update(ok=True, observation=self.get_cdp_ax(i))
        res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res


# --------------------------------------------------------------------------- live smoke test


def _lat(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    return {"n": len(s), "median": round(s[len(s) // 2], 1), "p90": round(s[min(len(s) - 1, int(len(s) * 0.9))], 1),
            "min": round(s[0], 1), "max": round(s[-1], 1)}


def smoke(n: int = 2) -> dict[str, Any]:
    """Live check on ``n`` daytona-small workers: boot, /__reset latency, FeedArena rendering
    (DOM order == Python view), CDP AX extraction, search/sort/scroll/open via real input,
    deterministic rewards, and teardown to zero sandboxes. Writes a receipt to rl/receipts/."""
    from pathlib import Path

    from synth.feed_arena import FeedArena, view

    out_dir = Path(__file__).parent / "receipts"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rec: dict[str, Any] = {"snapshot": "daytona-small", "workers": n, "checks": {}, "episodes": []}
    checks = rec["checks"]
    pool = SandboxPool()
    rec["pool_id"] = pool.pool_id
    rec["account_active_before"] = pool.account_active()

    def dom_ids(i: int) -> list[str]:
        return json.loads(pool.eval_js(i, "JSON.stringify([...document.querySelectorAll('#feed .card')].map(c=>c.dataset.id))"))

    def act(i: int, raw: str, log: list) -> dict[str, Any]:
        r = pool.act(i, raw)
        log.append({"action": raw, **{k: v for k, v in r.items() if k != "observation"}})
        if not r.get("ok"):
            raise PoolError(f"action failed: {raw} -> {r}")
        time.sleep(0.15)  # let the page's fetch('/api/event') land
        return r

    def click_open(i: int, pid: str, log: list) -> None:
        """Scroll until post ``pid``'s Open button sits below the sticky header, then click it."""
        top = pool.workers[i].origin[1] + 150
        for _ in range(12):
            lines = parse_ax_lines(pool.get_cdp_ax(i, max_chars=60000))
            btn = next(x for x in lines if x.title == f"Open post {pid}")
            if top <= btn.center[1] < SCREEN_H - 10:
                act(i, f"[[ACTION:cua_click:Open post {pid}]]", log)
                return
            act(i, f"[[ACTION:cua_scroll:{'down' if btn.center[1] >= SCREEN_H - 10 else 'up'}]]", log)
        raise PoolError(f"Open post {pid} never scrolled on screen")

    def open_first_organic(i: int, arena: FeedArena, log: list, skip: set[str] = frozenset()) -> str:
        ads = {a.id for a in arena.feed.ads}
        lines = parse_ax_lines(pool.get_cdp_ax(i, max_chars=60000))
        opens = [x.title.split()[-1] for x in lines if x.role == "Button" and x.title.startswith("Open post ")]
        pid = next(x for x in opens if x not in ads | set(skip))
        click_open(i, pid, log)
        return pid

    try:
        t = time.perf_counter()
        rec["boot"] = pool.start(n)
        rec["boot_wall_s"] = round(time.perf_counter() - t, 2)
        checks["boot"] = len(pool.workers) == n and all(w.boot.get("sandbox_id") for w in pool.workers)
        rec["transport"] = [w.via for w in pool.workers]

        # 1) /__reset latency sweep, train and ood_eval styles, plus DOM order == Python view
        resets, order_ok = [], True
        for r in range(6):
            for i in range(n):
                a = FeedArena.build(5000 + r * n + i, split="train" if r % 2 == 0 else "ood_eval")
                resets.append(pool.reset_worker(i, a.app_spec()))
                order_ok &= dom_ids(i) == [p.id for p in a.cards()]
        floor = []
        for _ in range(6):
            for i in range(n):
                t = time.perf_counter()
                pool._ctl(i, "GET", "/__health")
                floor.append((time.perf_counter() - t) * 1000)
        rec["proxy_floor_rtt_ms"] = _lat(floor)  # no-op round trip: Mac -> Daytona preview proxy -> sandbox
        rec["reset_rtt_ms"] = _lat([x["rtt_ms"] for x in resets])
        rec["reset_server_ms"] = _lat([x["total_ms"] for x in resets])
        rec["reset_breakdown_last"] = resets[-1]
        checks["reset_rtt_median_lt_500ms"] = rec["reset_rtt_ms"]["median"] < 500
        checks["dom_order_matches_python_view"] = bool(order_ok)

        # 2) CDP AX extraction latency + content
        ax_rtt, ax_srv, fmt_ms = [], [], []
        lines_text = ""
        for _ in range(5):
            for i in range(n):
                t = time.perf_counter()
                raw = pool.ax_raw(i)
                ax_rtt.append((time.perf_counter() - t) * 1000)
                ax_srv.append(raw["ms"])
                t = time.perf_counter()
                lines_text = pool.get_cdp_ax(i, raw=raw)
                fmt_ms.append((time.perf_counter() - t) * 1000)
        rec["ax_rtt_ms"], rec["ax_server_ms"], rec["ax_format_ms"] = _lat(ax_rtt), _lat(ax_srv), _lat(fmt_ms)
        rec["ax_sample"] = lines_text[:1500]
        rec["ax_origin"] = raw["origin"]
        parsed = parse_ax_lines(lines_text)
        checks["ax_has_search_tabs_open_buttons"] = (
            any(x.role == "TextField" and x.title == "Search" for x in parsed)
            and all(any(x.title == tab for x in parsed) for tab in ("Latest", "Top", "Most Liked", "Most Reposted"))
            and sum(x.title.startswith("Open post ") for x in parsed) >= 3
        )
        # AX rect of the first card's Open button vs its DOM rect (+ viewport origin)
        dom_rect = json.loads(pool.eval_js(n - 1, "JSON.stringify((r=>[r.x,r.y,r.width,r.height])(document.querySelector('#feed .card .open').getBoundingClientRect()))"))
        first_open = next(x for x in parsed if x.title.startswith("Open post "))
        cal = pool.workers[n - 1].origin
        exp = (round(dom_rect[0] + cal[0]), round(dom_rect[1] + cal[1]))
        rec["ax_vs_dom_rect"] = {"ax": first_open.rect, "dom_plus_calibrated_origin": exp,
                                 "calibrated_origin": cal, "server_estimated_origin": raw["origin"]}
        checks["ax_rect_matches_dom"] = abs(first_open.rect[0] - exp[0]) <= 2 and abs(first_open.rect[1] - exp[1]) <= 2
        last_open = [x for x in parsed if x.title.startswith("Open post ")][-1]
        refused = pool.act(n - 1, f"[[ACTION:cua_click:{last_open.title}]]")
        rec["offscreen_click"] = refused
        checks["offscreen_click_refused"] = (not refused["ok"]) and "off screen" in refused["detail"]

        def episode(i: int, arena: FeedArena, steps: Any) -> dict[str, Any]:
            log: list = []
            t0 = time.perf_counter()
            pool.reset_worker(i, arena.app_spec())
            steps(i, arena, log)
            st = pool.state(i)
            ep = {"worker": i, "seed": arena.seed, "kind": arena.task.kind, "split": arena.split,
                  "instruction": arena.task.instruction, "target": arena.task.target_id,
                  "reward": arena.score(st), "final_state": st["state"],
                  "events": [{k: v for k, v in e.items() if k not in ("t", "ep")} for e in st["events"]],
                  "actions": log, "wall_s": round(time.perf_counter() - t0, 2)}
            rec["episodes"].append(ep)
            return ep

        # 3) search_author_top: type from:@handle, Enter, Most Liked tab, open first organic
        def s_author(i: int, a: FeedArena, log: list) -> None:
            act(i, f"[[ACTION:cua_type:Search|from:@{a.feed.focus_handle}]]", log)
            act(i, "[[ACTION:cua_hotkey:return]]", log)
            act(i, "[[ACTION:cua_click:Most Liked]]", log)
            got = dom_ids(i)
            if got != [p.id for p in view(a.feed, "Most Liked", f"from:@{a.feed.focus_handle}")]:
                raise PoolError(f"search+sort DOM mismatch: {got}")
            open_first_organic(i, a, log)

        e1 = episode(0, FeedArena.build(7001, kind="search_author_top"), s_author)

        # 4) most_viral: Top tab, open first organic
        def s_viral(i: int, a: FeedArena, log: list) -> None:
            act(i, "[[ACTION:cua_click:Top]]", log)
            open_first_organic(i, a, log)

        e2 = episode(n - 1, FeedArena.build(7002, kind="most_viral", split="ood_eval"), s_viral)

        # 5) open_specific_post with a shotgun detour: open the decoy, Escape, then the target (=> 0.5)
        def s_quote(i: int, a: FeedArena, log: list) -> None:
            act(i, f"[[ACTION:cua_type:Search|{a.feed.snippet}]]", log)
            act(i, "[[ACTION:cua_hotkey:return]]", log)
            click_open(i, a.task.meta["decoy"], log)
            act(i, "[[ACTION:cua_hotkey:escape]]", log)
            click_open(i, a.task.target_id, log)

        e3 = episode(0, FeedArena.build(7003, kind="open_specific_post"), s_quote)

        # 6) latest_non_pinned by scrolling + clearing search; scroll must move the page
        def s_latest(i: int, a: FeedArena, log: list) -> None:
            y0 = pool.eval_js(i, "scrollY")
            act(i, "[[ACTION:cua_scroll:down]]", log)
            y1 = pool.eval_js(i, "scrollY")
            log.append({"scrollY_before": y0, "scrollY_after": y1})
            if not y1 > y0:
                raise PoolError(f"scroll did not move the page ({y0} -> {y1})")
            act(i, "[[ACTION:cua_scroll:up]]", log)
            pinned = {p.id for p in a.feed.posts if p.pinned}
            open_first_organic(i, a, log, skip=pinned)

        e4 = episode(n - 1, FeedArena.build(7004, kind="latest_non_pinned"), s_latest)

        # 7) clear button resets the search; a rejected reward when the detail view is closed
        def s_neg(i: int, a: FeedArena, log: list) -> None:
            act(i, "[[ACTION:cua_type:Search|kernel]]", log)
            act(i, "[[ACTION:cua_hotkey:return]]", log)
            act(i, "[[ACTION:cua_click:Clear search]]", log)
            if dom_ids(i) != [p.id for p in a.cards()]:
                raise PoolError("Clear did not restore the feed")
            click_open(i, a.task.target_id, log)
            act(i, "[[ACTION:cua_click:Close]]", log)

        e5 = episode(0, FeedArena.build(7005, kind="negated_instruction"), s_neg)

        checks["reward_search_author_top_eq_1"] = e1["reward"] == 1.0
        checks["reward_most_viral_eq_1"] = e2["reward"] == 1.0
        checks["reward_shotgun_eq_0.5"] = e3["reward"] == 0.5
        checks["reward_latest_non_pinned_eq_1"] = e4["reward"] == 1.0
        checks["reward_closed_view_eq_0"] = e5["reward"] == 0.0
        # replaying the state read is deterministic
        checks["state_read_deterministic"] = pool.state(0)["events"] == pool.state(0)["events"]

        for i in range(n):
            (out_dir / f"feed_arena_smoke_{stamp}_w{i}.png").write_bytes(pool.screenshot(i))
        rec["screenshots"] = [str(out_dir / f"feed_arena_smoke_{stamp}_w{i}.png") for i in range(n)]
    except Exception as e:  # noqa: BLE001 - record and still tear down
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        t = time.perf_counter()
        rec["close"] = pool.close_all()
        rec["close_s"] = round(time.perf_counter() - t, 2)
        rec["account_active_after"] = pool.account_active()
        checks["pool_active_sandboxes_eq_0"] = rec["close"]["active_sandboxes"] == 0
        rec["all_checks_pass"] = all(checks.values()) and "error" not in rec
        path = out_dir / f"feed_arena_pool_smoke_{stamp}.json"
        path.write_text(json.dumps(rec, indent=2, default=str))
        rec["receipt"] = str(path)
    return rec


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        r = smoke(int(sys.argv[sys.argv.index("--smoke") + 1]) if len(sys.argv) > sys.argv.index("--smoke") + 1 else 2)
        print(json.dumps({k: v for k, v in r.items() if k not in ("episodes", "ax_sample")}, indent=2, default=str))
        sys.exit(0 if r["all_checks_pass"] else 1)
    print("usage: python pool.py --smoke [n]")
