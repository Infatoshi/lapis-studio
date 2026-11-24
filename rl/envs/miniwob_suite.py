"""MiniWoB++ task suite on the persistent Daytona pool: frozen task-level split + scripted oracles.

Split (task level, frozen; ``tests/test_miniwob_suite.py`` enforces it):
  * ``HELDOUT_TASKS`` (30): never trained on. The 5 zero-shot probe tasks from
    ``overnight_stage1/miniwob_env.py`` plus their near-duplicate variants (enter-text-2,
    enter-text-dynamic, focus-text-2, login-user-popup) so no held-out family leaks into train,
    plus 21 tasks spanning every other skill category.
  * ``TRAIN_TASKS`` (100): everything else in ``~/cua_datasets/miniwob/miniwob/html/miniwob``.
  Families (``-2``, ``-nodelay``, ``-easy`` ... variants) never straddle the split.

Runtime (one sandbox from ``pool.SandboxPool``, no pool changes):
  * ``install(pool, i)`` uploads the static MiniWoB tree once and serves it with
    ``python3 -m http.server`` on 127.0.0.1:8772 inside the sandbox.
  * ``app_spec(task, seed)`` is the task page itself with ``<base href=http://127.0.0.1:8772/miniwob/>``
    and a bootstrap that, on load: hooks ``core.endEpisode`` (first reward wins, posted to the pool's
    ``/api/event`` as ground truth), lifts the 10 s episode timer to 1 h, seeds
    ``Math.seedrandom("<task>:<seed>")``, starts the episode, zooms 2x, gives unnamed inputs an
    accessible name, then posts ``ready``. ``pool.reset_worker`` swaps it in via ``POST /__reset``
    + CDP ``Page.navigate`` (no Chromium restart). Seed, zoom and text-field naming match the
    old probe env, so held-out probe problems are identical for a given seed.
  * Observations are the pool's Studio CDP AX lines; actions are Studio ``[[ACTION:cua_...]]`` tags
    executed with real X11 input by ``pool.act``.
  * Reward: ``raw`` = the first ``core.endEpisode`` raw reward; ``reward`` = 1.0 if raw > 0 else 0.0.

Oracles: ``ORACLES[task]`` is a generator ``fn(page)`` that reads DOM state through ``page`` and
yields Studio action strings one at a time (closed loop: each is executed before the next is
computed). Clicks use the element's accessible name when ``resolve_target`` on the live AX lines
lands on that element, else the RL ``name @x,y`` point extension.

  uv run python envs/miniwob_suite.py --smoke 2        # live Daytona oracle smoke (owner=w3-miniwob)
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

RL_DIR = Path(__file__).resolve().parent.parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from studio_contract import parse_cua_action, parse_studio_actions  # noqa: E402

MINIWOB_HTML = Path(os.path.expanduser("~/cua_datasets/miniwob/miniwob/html"))
STATIC_PORT = 8772
STATIC_ROOT = "/tmp/miniwob_suite"
SCREEN_W, SCREEN_H = 1024, 768
ZOOM = 2
EPISODE_MAX_MS = 3_600_000
OWNER_LABELS = {"lapis-rl": "1", "owner": "w3-miniwob"}

# --------------------------------------------------------------------------- task audit + split

SKILLS = (
    "click", "form", "checkbox_radio", "dropdown", "datetime_slider", "drag",
    "table_search", "menu_tabs", "multistep", "visual_reasoning",
)

# Every one of the 130 tasks, by primary skill (audited from each task's genProblem + reward code).
TASK_SKILLS: dict[str, str] = {
    # click/buttons
    **dict.fromkeys([
        "button-delay", "click-button", "click-button-sequence", "click-color", "click-dialog", "click-dialog-2",
        "click-link", "click-shades", "click-test", "click-test-2", "click-test-transfer", "click-widget",
        "focus-text", "focus-text-2"], "click"),
    # forms/typing
    **dict.fromkeys([
        "copy-paste", "copy-paste-2", "enter-password", "enter-text", "enter-text-2", "enter-text-dynamic",
        "form-sequence-2", "login-user", "login-user-popup", "multi-layouts", "multi-orderings", "terminal",
        "text-editor", "text-transform", "unicode-test"], "form"),
    # checkboxes/radios
    **dict.fromkeys([
        "click-checkboxes", "click-checkboxes-large", "click-checkboxes-soft", "click-checkboxes-transfer",
        "click-option", "form-sequence", "number-checkboxes", "sign-agreement"], "checkbox_radio"),
    # dropdowns/selects/autocomplete
    **dict.fromkeys([
        "choose-list", "click-scroll-list", "form-sequence-3", "use-autocomplete", "use-autocomplete-nodelay"],
        "dropdown"),
    # date pickers/sliders/spinners/color wheels
    **dict.fromkeys([
        "choose-date", "choose-date-easy", "choose-date-medium", "choose-date-nodelay", "daily-calendar",
        "enter-date", "enter-time", "use-colorwheel", "use-colorwheel-2", "use-slider", "use-slider-2",
        "use-spinner"], "datetime_slider"),
    # drag-and-drop/drawing/text selection
    **dict.fromkeys([
        "drag-box", "drag-circle", "drag-cube", "drag-items", "drag-items-grid", "drag-shapes", "drag-shapes-2",
        "drag-single-shape", "drag-sort-numbers", "draw-circle", "draw-line", "highlight-text",
        "highlight-text-2", "resize-textarea"], "drag"),
    # tables/search/filtering/reading/scrolling text
    **dict.fromkeys([
        "find-greatest", "find-word", "phone-book", "read-table", "read-table-2", "scroll-text", "scroll-text-2",
        "search-engine", "stock-market"], "table_search"),
    # nested menus/tabs/accordions/trees/pie menus
    **dict.fromkeys([
        "click-collapsible", "click-collapsible-2", "click-collapsible-2-nodelay", "click-collapsible-nodelay",
        "click-menu", "click-menu-2", "click-pie", "click-pie-nodelay", "click-tab", "click-tab-2",
        "click-tab-2-easy", "click-tab-2-hard", "click-tab-2-medium", "navigate-tree"], "menu_tabs"),
    # multi-step email/social/booking/ordering
    **dict.fromkeys([
        "book-flight", "book-flight-nodelay", "buy-ticket", "email-inbox", "email-inbox-delete",
        "email-inbox-forward", "email-inbox-forward-nl", "email-inbox-forward-nl-turk", "email-inbox-important",
        "email-inbox-nl-turk", "email-inbox-noscroll", "email-inbox-reply", "email-inbox-star-reply",
        "order-food", "social-media", "social-media-all", "social-media-some"], "multistep"),
    # canvas/shapes/math/games
    **dict.fromkeys([
        "ascending-numbers", "bisect-angle", "chase-circle", "circle-center", "click-shape", "count-shape",
        "count-sides", "find-midpoint", "generate-number", "grid-coordinate", "guess-number", "hot-cold",
        "hover-shape", "identify-shape", "moving-items", "odd-or-even", "right-angle", "simon-says",
        "simple-algebra", "simple-arithmetic", "tic-tac-toe", "visual-addition"], "visual_reasoning"),
}
ALL_TASKS: tuple[str, ...] = tuple(sorted(TASK_SKILLS))

PROBE_TASKS: tuple[str, ...] = ("click-button", "click-link", "focus-text", "enter-text", "login-user")

# Frozen 2026-11-24. Do not edit: held-out results are only comparable while this set is fixed.
HELDOUT_TASKS: frozenset[str] = frozenset({
    # the 5 zero-shot probes + their variants (family-level hold-out)
    "click-button", "click-link", "focus-text", "focus-text-2", "enter-text", "enter-text-2",
    "enter-text-dynamic", "login-user", "login-user-popup",
    "click-option", "sign-agreement",                                  # checkbox_radio
    "choose-list", "use-autocomplete", "use-autocomplete-nodelay",     # dropdown
    "enter-date", "enter-time", "use-spinner",                         # datetime_slider
    "drag-items", "drag-items-grid", "resize-textarea",                # drag
    "search-engine", "phone-book", "read-table", "read-table-2",       # table_search
    "click-menu", "click-menu-2", "navigate-tree",                     # menu_tabs
    "book-flight", "book-flight-nodelay",                              # multistep
    "count-shape",                                                     # visual_reasoning
})
TRAIN_TASKS: tuple[str, ...] = tuple(t for t in ALL_TASKS if t not in HELDOUT_TASKS)

# Variant -> family root, used to prove no family straddles the split.
_FAMILY_SUFFIX = re.compile(r"-(2|3|nodelay|easy|medium|hard|large|soft|transfer|dynamic|popup|nl|turk|"
                            r"noscroll|delete|forward|important|reply|star|all|some)$")


def family(task: str) -> str:
    t = task
    while True:
        s = _FAMILY_SUFFIX.sub("", t)
        if s == t:
            return t
        t = s


def split_counts() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for t, s in TASK_SKILLS.items():
        row = out.setdefault(s, {"train": 0, "heldout": 0})
        row["heldout" if t in HELDOUT_TASKS else "train"] += 1
    return out


# --------------------------------------------------------------------------- page bootstrap

BOOT_JS = r"""
(function(){
var CFG = __CFG__;
function post(o){ o.ep = window.__EP; try { fetch(location.origin + '/api/event', {method:'POST',
  body: JSON.stringify(o), keepalive: true}); } catch(e) {} }
var TEXTLIKE = {text:1, password:1, email:1, number:1, search:1, tel:1, url:1};
var CNT = {};
function prevLabel(el){ var p = el.previousElementSibling;
  return (p && p.tagName === 'LABEL') ? p.innerText.trim() : ''; }
function hasLabelText(el){ return [].slice.call(el.labels || []).some(function(l){ return l.innerText.trim(); }); }
function nameInputs(initial){
  // Text inputs keep the old probe env's naming (adjacent <label> text, else "text field[ N]") so
  // held-out probe observations are unchanged; checkboxes/radios/textareas without a label get
  // "<kind> N". Elements added later (e.g. a reply box) are named by the MutationObserver.
  var els = [].slice.call(document.querySelectorAll('#area input, #area textarea'));
  var groups = {};
  els.forEach(function(el){
    if (el.dataset.lapisNamed) return;
    var kind = el.tagName === 'TEXTAREA' ? 'text area' : (TEXTLIKE[el.type] ? 'text field' :
               (el.type === 'checkbox' || el.type === 'radio') ? el.type : '');
    if (!kind) return;
    (groups[kind] = groups[kind] || []).push(el);
  });
  Object.keys(groups).forEach(function(kind){
    var g = groups[kind];
    g.forEach(function(el, i){
      el.dataset.lapisNamed = '1';
      if (el.getAttribute('aria-label') || el.placeholder) return;
      if ((kind === 'checkbox' || kind === 'radio') && hasLabelText(el)) return;
      var lab = prevLabel(el);
      if (lab) { el.setAttribute('aria-label', lab); return; }
      var n;
      if (initial) { n = g.length > 1 ? i + 1 : 0; CNT[kind] = g.length; }
      else { CNT[kind] = (CNT[kind] || 0) + 1; n = CNT[kind] > 1 ? CNT[kind] : 0; }
      el.setAttribute('aria-label', kind + (n ? ' ' + n : ''));
    });
  });
}
function zf(){ var w = document.getElementById('wrap'); var k = w ? w.getBoundingClientRect().width / 160 : 1;
  return CFG.zoom / (k || 1); }  // CSS rect px -> visual viewport px, whatever zoom semantics apply
window.__lapis = {
  info: function(el){
    if (!el) return null;
    var r = el.getBoundingClientRect(), f = zf();
    var inside = function(x, y){ var h = document.elementFromPoint(x, y); return !!h && (h === el || el.contains(h)); };
    // center first, then a 5x5 grid: an overlapping sibling may cover the center but not the rest
    var px = null, py = null, fr = [0.5, 0.2, 0.8, 0.35, 0.65];
    for (var a = 0; a < fr.length && px === null; a++) for (var b = 0; b < fr.length; b++) {
      var x = r.left + r.width * fr[b], y = r.top + r.height * fr[a];
      if (inside(x, y)) { px = x; py = y; break; } }
    var cs = getComputedStyle(el);
    return {x: r.left * f, y: r.top * f, w: r.width * f, h: r.height * f,
            px: px === null ? null : px * f, py: py === null ? null : py * f,
            hit: px !== null && cs.visibility !== 'hidden' && r.width > 0,
            label: (el.getAttribute('aria-label') || el.innerText || el.value || '').trim().slice(0, 60)};
  },
  hitAt: function(el, vx, vy){
    if (!el) return false; var f = zf(); var h = document.elementFromPoint(vx / f, vy / f);
    return !!h && (h === el || el.contains(h));
  },
  scroller: function(el){
    for (var p = el && el.parentElement; p; p = p.parentElement) {
      var oy = getComputedStyle(p).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && p.scrollHeight > p.clientHeight + 1) {
        var r = p.getBoundingClientRect(), e = el.getBoundingClientRect(), f = zf();
        return {x: r.left * f, y: r.top * f, w: r.width * f, h: r.height * f,
                dir: (e.top + e.height / 2) > (r.top + r.height / 2) ? 'down' : 'up'};
      }
    }
    return null;
  }
};
function boot(){
  if (typeof core === 'undefined' || typeof genProblem !== 'function') {
    post({type: 'ready', set: {error: 'core/genProblem missing'}}); return; }
  window.__first = null;
  var orig = core.endEpisode;
  core.endEpisode = function(r, t, why){
    if (window.__first === null && core.EP_TIMER !== null) {
      window.__first = [r, String(why || '')];
      post({type: 'end', set: {raw_reward: r, reason: String(why || ''), done: true}});
    }
    return orig.apply(this, arguments);
  };
  core.EPISODE_MAX_TIME = CFG.max_ms;
  Math.seedrandom(CFG.task + ':' + CFG.seed);
  core.startEpisodeReal();
  document.body.style.zoom = String(CFG.zoom);
  nameInputs(true);
  new MutationObserver(function(){ nameInputs(false); }).observe(document.body, {childList: true, subtree: true});
  var q = document.getElementById('query');
  post({type: 'ready', set: {task: CFG.task, seed: CFG.seed, query: q ? q.innerText.replace(/\s+/g, ' ').trim() : ''}});
}
window.addEventListener('load', function(){ setTimeout(boot, 0); });
})();
"""


def task_html(task: str) -> str:
    return (MINIWOB_HTML / "miniwob" / f"{task}.html").read_text(encoding="utf-8")


def app_spec(task: str, seed: int, zoom: int = ZOOM, html: str | None = None) -> dict[str, Any]:
    """``pool.reset_worker`` spec for one MiniWoB episode."""
    if task not in TASK_SKILLS:
        raise KeyError(f"unknown MiniWoB task {task!r}")
    src = html if html is not None else task_html(task)
    cfg = json.dumps({"task": task, "seed": int(seed), "zoom": zoom, "max_ms": EPISODE_MAX_MS})
    # Absolute asset URLs, not <base href>: a base makes "#tabs-1" anchors non-local, which turns
    # jQuery UI tabs into remote (ajax) tabs and breaks click-tab*/search-engine.
    root = f"http://127.0.0.1:{STATIC_PORT}/"
    src = src.replace("../core/", root + "core/").replace("../common/", root + "common/")
    boot = "<script>" + BOOT_JS.replace("__CFG__", cfg) + "</script>"
    j = src.rfind("</body>")
    src = src[:j] + boot + src[j:] if j >= 0 else src + boot
    return {"name": f"miniwob:{task}", "html": src, "state": {"task": task, "seed": int(seed)},
            "ready_timeout": 8.0}


def static_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for sub in ("core", "common", "miniwob"):
            tf.add(MINIWOB_HTML / sub, arcname=sub)
    return buf.getvalue()


def install(pool: Any, i: int) -> dict[str, Any]:
    """Upload the static tree to worker ``i`` and serve it on 127.0.0.1:STATIC_PORT (idempotent)."""
    w = pool.workers[i]
    if getattr(w, "miniwob_ready", False):
        return {"cached": True}
    t0 = time.perf_counter()
    sb = w.sandbox
    sb.process.exec(f"rm -rf {STATIC_ROOT} && mkdir -p {STATIC_ROOT}")
    sb.fs.upload_file(static_tarball(), f"{STATIC_ROOT}/s.tgz")
    r = sb.process.exec(
        f"cd {STATIC_ROOT} && tar xzf s.tgz && rm s.tgz && "
        f"(setsid nohup python3 -m http.server {STATIC_PORT} --bind 127.0.0.1 --directory {STATIC_ROOT} "
        f"> {STATIC_ROOT}/http.log 2>&1 < /dev/null &) ; "
        f"for k in $(seq 1 40); do curl -sf http://127.0.0.1:{STATIC_PORT}/core/core.js >/dev/null && echo up && break; "
        f"sleep 0.25; done", timeout=60)
    if "up" not in (r.result or ""):
        raise RuntimeError(f"miniwob static server did not start: {r.result!r}")
    w.miniwob_ready = True
    return {"install_s": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------- oracle page interface


class OracleError(RuntimeError):
    pass


def _clean(name: str) -> str:
    """Make a label safe inside ``[[ACTION:...]]`` (no ']' or '|', no trailing '@x,y')."""
    s = re.sub(r"[\]\[|]", " ", name or "").strip()
    s = re.sub(r"@\s*\d+\s*,\s*\d+\s*$", "", s).strip()
    return " ".join(s.split())[:60] or "target"


def click_action(name: str, point: tuple[int, int] | None = None) -> str:
    n = _clean(name)
    return f"[[ACTION:cua_click:{n} @{point[0]},{point[1]}]]" if point else f"[[ACTION:cua_click:{n}]]"


def type_action(name: str, text: str) -> str:
    if "]" in text:
        raise OracleError("text with ']' cannot be expressed as a Studio action")
    return f"[[ACTION:cua_type:{_clean(name)}|{text}]]"


def hotkey_action(combo: str) -> str:
    return f"[[ACTION:cua_hotkey:{combo}]]"


def scroll_action(target: str, amount: int) -> str:
    return f"[[ACTION:cua_scroll:{_clean(target)}|{amount}]]" if target else f"[[ACTION:cua_scroll:{amount}]]"


def to_1000(x: float, y: float) -> tuple[int, int]:
    return (max(0, min(1000, round(x * 1000 / SCREEN_W))), max(0, min(1000, round(y * 1000 / SCREEN_H))))


class Page:
    """What an oracle may use. ``js`` reads DOM state; the generator helpers yield Studio actions.
    ``LivePage`` backs it with a pool worker; tests back it with canned values."""

    origin: tuple[float, float] = (0.0, 0.0)

    def js(self, expr: str) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def ax(self) -> list[Any]:  # parsed Studio AX lines (pool.AxLine)
        return []

    def wait(self, s: float) -> None:
        pass

    def resolve(self, name: str, text_input: bool = False) -> Any:
        from pool import resolve_target  # local import keeps the module importable without httpx in odd envs
        return resolve_target(self.ax(), name, text_input=text_input)

    # ---- DOM helpers
    @property
    def query(self) -> str:
        return " ".join(str(self.js("document.getElementById('query').innerText") or "").split())

    def info(self, sel: str) -> dict[str, Any] | None:
        inf = self.js(f"__lapis.info({sel})")
        if inf:
            inf["sx"], inf["sy"] = inf["x"] + self.origin[0], inf["y"] + self.origin[1]
        return inf

    def center(self, inf: dict[str, Any]) -> tuple[int, int]:
        """Visible click point: the center unless something overlaps it, then the first uncovered
        grid point ``__lapis.info`` found."""
        if inf.get("px") is not None:
            return to_1000(inf["px"] + self.origin[0], inf["py"] + self.origin[1])
        return to_1000(inf["sx"] + inf["w"] / 2, inf["sy"] + inf["h"] / 2)

    def hits(self, sel: str, screen_xy: tuple[float, float]) -> bool:
        vx, vy = screen_xy[0] - self.origin[0], screen_xy[1] - self.origin[1]
        return bool(self.js(f"__lapis.hitAt({sel}, {vx:.1f}, {vy:.1f})"))

    # ---- action generators
    def reveal(self, sel: str, max_steps: int = 25) -> Iterator[str]:
        """Scroll the element's scroll container (via a visible named element inside it) until the
        element's center is the topmost hit."""
        for _ in range(max_steps):
            inf = self.info(sel)
            if inf is None:
                raise OracleError(f"no element for {sel}")
            if inf["hit"]:
                return
            sc = self.js(f"__lapis.scroller({sel})")
            if not sc:
                raise OracleError(f"{sel} is not clickable and has no scroll container")
            x0, y0 = sc["x"] + self.origin[0], sc["y"] + self.origin[1]
            lines = self.ax()
            anchor = ""
            for ln in lines:
                label = ln.title or ln.value
                cx, cy = ln.center
                if label and x0 <= cx < x0 + sc["w"] and y0 + 4 <= cy < y0 + sc["h"] - 4 and len(label) <= 60:
                    if self.resolve(label) == ln:
                        anchor = label
                        break
            if not anchor:
                raise OracleError(f"no named anchor inside the scroll container of {sel}")
            yield scroll_action(anchor, -2 if sc["dir"] == "down" else 2)
        raise OracleError(f"could not scroll {sel} into view")

    def click(self, sel: str, name: str | None = None, reveal: bool = True) -> Iterator[str]:
        if reveal:
            yield from self.reveal(sel)
        inf = self.info(sel)
        if inf is None:
            raise OracleError(f"no element for {sel}")
        if name:
            node = self.resolve(name)
            if node is not None and self.hits(sel, node.center):
                yield click_action(name)
                return
        yield click_action(name or inf.get("label") or "target", self.center(inf))

    def type_into(self, sel: str, text: str) -> Iterator[str]:
        """``cua_type`` into the element ``sel`` by its accessible name (must resolve to it)."""
        yield from self.reveal(sel)
        name = self.js(f"(function(e){{return e && (e.getAttribute('aria-label') || e.placeholder || "
                       f"(e.labels && e.labels[0] && e.labels[0].innerText) || '')}})({sel})") or ""
        name = " ".join(str(name).split())
        node = self.resolve(name, text_input=True) if name else None
        if node is None or not self.hits(sel, node.center):
            raise OracleError(f"text field {sel} has no unambiguous accessible name ({name!r})")
        yield type_action(name, text)


# --------------------------------------------------------------------------- oracles

Oracle = Callable[[Page], Iterator[str]]
ORACLES: dict[str, Oracle] = {}


def oracle(*tasks: str) -> Callable[[Oracle], Oracle]:
    def deco(fn: Oracle) -> Oracle:
        for t in tasks:
            if t in HELDOUT_TASKS:
                raise ValueError(f"oracle for held-out task {t}")  # never script held-out tasks
            ORACLES[t] = fn
        return fn
    return deco


Q = "document.querySelector"
SUBMIT = f"{Q}('#subbtn')"


def _quote(s: str) -> str:
    return json.dumps(s)


def _between(q: str, left: str, right: str) -> str:
    i = q.index(left) + len(left)
    return q[i: q.index(right, i)]


@oracle("click-test")
def o_click_test(p: Page) -> Iterator[str]:
    yield from p.click(SUBMIT, "Click Me!")


@oracle("click-test-2")
def o_click_test_2(p: Page) -> Iterator[str]:
    yield from p.click(SUBMIT, "ONE")


@oracle("click-test-transfer")
def o_click_test_transfer(p: Page) -> Iterator[str]:
    two = "button TWO" in p.query
    yield from p.click(f"{Q}('#subbtn2')" if two else SUBMIT, "TWO" if two else "ONE")


@oracle("click-button-sequence")
def o_click_button_sequence(p: Page) -> Iterator[str]:
    yield from p.click(SUBMIT, "ONE")
    yield from p.click(f"{Q}('#subbtn2')", "TWO")


@oracle("click-dialog")
def o_click_dialog(p: Page) -> Iterator[str]:
    yield from p.click(f"{Q}('.ui-dialog-titlebar-close')", "Close")


@oracle("click-dialog-2")
def o_click_dialog_2(p: Page) -> Iterator[str]:
    want = _between(p.query, 'labeled "', '"')
    if want == "x":
        yield from p.click(f"{Q}('.ui-dialog-titlebar-close')", "Close")
    else:
        sel = (f"[...document.querySelectorAll('.ui-dialog-buttonpane button')]"
               f".find(b => b.innerText.trim() === {_quote(want)})")
        yield from p.click(sel, want)


@oracle("click-widget")
def o_click_widget(p: Page) -> Iterator[str]:
    kind = _between(p.query, 'a "', '"')
    sel = f"{Q}('#area [data-type={kind}]')"
    name = p.js(f"(e => e.tagName === 'BUTTON' ? e.innerText.trim() : (e.getAttribute('aria-label') || ''))({sel})")
    yield from p.click(sel, name or kind)


@oracle("enter-password")
def o_enter_password(p: Page) -> Iterator[str]:
    pw = p.js(f"{Q}('#query .bold').innerText")
    yield from p.type_into(f"{Q}('#password')", pw)
    yield from p.type_into(f"{Q}('#verify')", pw)
    yield from p.click(SUBMIT, "Submit")


@oracle("text-transform")
def o_text_transform(p: Page) -> Iterator[str]:
    txt = p.js("[...document.querySelectorAll('#captcha span')].map(s => s.textContent).join('')")
    yield from p.type_into(f"{Q}('#tt')", txt)
    yield from p.click(SUBMIT, "Submit")


def _copy_paste(p: Page, src_sel: str) -> Iterator[str]:
    name = p.js(f"{src_sel}.getAttribute('aria-label')") or "text area"
    yield from p.click(src_sel, name)
    yield hotkey_action("cmd+a")
    yield hotkey_action("cmd+c")
    dst = f"{Q}('#answer-input')"
    yield from p.click(dst, p.js(f"{dst}.getAttribute('aria-label')") or "text field")
    yield hotkey_action("cmd+v")
    yield from p.click(SUBMIT, "Submit")


@oracle("copy-paste")
def o_copy_paste(p: Page) -> Iterator[str]:
    yield from _copy_paste(p, f"{Q}('#to-copy')")


@oracle("copy-paste-2")
def o_copy_paste_2(p: Page) -> Iterator[str]:
    n = {"1st": 1, "2nd": 2, "3rd": 3}[p.js(f"{Q}('#query .bold').innerText").strip()]
    yield from _copy_paste(p, f"{Q}('#text-{n}')")


@oracle("multi-layouts", "multi-orderings")
def o_multi_layouts(p: Page) -> Iterator[str]:
    genre, name, year = p.js("[...document.querySelectorAll('#query b')].map(b => b.innerText)")
    for k, row_kw, val in ((0, "genre", genre), (1, "director", name), (2, "year|date", year)):
        sel = ("[...document.querySelectorAll('#area input[type=text]')].find(e => new RegExp("
               f"{_quote(row_kw)}, 'i').test((e.closest('p,div.row,tr,div.field,div.ui-entry') || e.parentElement)"
               ".innerText))")
        yield from p.type_into(sel, val)
    sub = f"{Q}('#area button, #area .final, #area .ui-submit')"
    yield from p.click(sub, p.js(f"{sub}.innerText.trim()"))


@oracle("simple-arithmetic")
def o_simple_arithmetic(p: Page) -> Iterator[str]:
    a, op, b = p.js(f"{Q}('#math-problem').innerText").split()[:3]
    v = {"+": int(a) + int(b), "-": int(a) - int(b), "x": int(a) * int(b)}[op]
    yield from p.type_into(f"{Q}('#math-answer')", str(v))
    yield from p.click(SUBMIT, "Submit")


@oracle("simple-algebra")
def o_simple_algebra(p: Page) -> Iterator[str]:
    t = p.js(f"{Q}('#math-problem').innerText").split()
    lhs, rhs = t[:3], int(t[-1])
    if lhs[0] == "x":
        d = int(lhs[2])
        v = rhs - d if lhs[1] == "+" else rhs + d
    else:
        d = int(lhs[0])
        v = rhs - d if lhs[1] == "+" else d - rhs
    yield from p.type_into(f"{Q}('#math-answer')", str(v))
    yield from p.click(SUBMIT, "Submit")


@oracle("click-checkboxes", "click-checkboxes-large", "click-checkboxes-transfer")
def o_click_checkboxes(p: Page) -> Iterator[str]:
    q = p.query
    body = q[len("Select "): q.rindex(" and click Submit.")]
    names = [] if body == "nothing" else [s.strip() for s in body.split(",") if s.strip()]
    for n in names:
        sel = (f"[...document.querySelectorAll('#area input[type=checkbox]')]"
               f".find(c => c.parentElement.textContent.trim() === {_quote(n)})")
        if not p.js(f"{sel}.checked"):
            yield from p.click(sel, n)
    yield from p.click(SUBMIT, "Submit")


def _slider_to(p: Page, slider: str, target: int) -> Iterator[str]:
    """Click the jQuery UI handle, then closed-loop keyboard (page up/down, arrows) to ``target``."""
    handle = f"{slider}.querySelector('.ui-slider-handle')"
    yield from p.click(handle, "slider handle")
    for _ in range(60):
        v = int(p.js(f"$({slider}).slider('value')"))
        if v == target:
            return
        lo, hi = p.js(f"[$({slider}).slider('option','min'), $({slider}).slider('option','max')]")
        page = max(1, (hi - lo) // 5)
        d = target - v
        yield hotkey_action("pageup" if d >= page else "pagedown" if d <= -page else "up" if d > 0 else "down")
    raise OracleError(f"slider never reached {target}")


@oracle("use-slider")
def o_use_slider(p: Page) -> Iterator[str]:
    n = int(re.search(r"Select (-?\d+)", p.query).group(1))
    yield from _slider_to(p, f"{Q}('#slider')", n)
    yield from p.click(SUBMIT, "Submit")


@oracle("form-sequence")
def o_form_sequence(p: Page) -> Iterator[str]:
    m = re.search(r"Select (-?\d+) with the slider, click the (\d)\w\w checkbox", p.query)
    yield from _slider_to(p, f"{Q}('#slider')", int(m.group(1)))
    cb = f"{Q}('#checkbox-{m.group(2)}')"
    yield from p.click(cb, p.js(f"{cb}.getAttribute('aria-label')"))
    yield from p.click(SUBMIT, "Submit")


@oracle("form-sequence-2")
def o_form_sequence_2(p: Page) -> Iterator[str]:
    m = re.search(r"Check the (\d)\w\w radio button and enter the number \"(-?\d+)\" into the (\d)\w\w textbox",
                  p.query)
    radio = f"{Q}('#area input[type=radio][value=\"{m.group(1)}\"]')"
    yield from p.click(radio, p.js(f"{radio}.getAttribute('aria-label')"))
    yield from p.type_into(f"{Q}('#input-{m.group(3)}')", m.group(2))
    yield from p.click(SUBMIT, "Submit")


@oracle("form-sequence-3")
def o_form_sequence_3(p: Page) -> Iterator[str]:
    q = p.query
    want = _between(q, "Choose ", " from the dropdown")
    btn = _between(q, 'labeled "', '"')
    yield from p.click(f"{Q}('.selectric')", p.js(f"{Q}('.selectric .label').innerText") or "dropdown")
    p.wait(0.2)
    li = f"[...document.querySelectorAll('.selectric-items li')].find(l => l.innerText.trim() === {_quote(want)})"
    yield from p.click(li, want)
    yield from p.click(f"[...document.querySelectorAll('#buttons button')].find(b => b.innerText.trim() === {_quote(btn)})",
                       btn)


@oracle("click-scroll-list")
def o_click_scroll_list(p: Page) -> Iterator[str]:
    want = [s.strip() for s in _between(p.query, "Select ", " from the scroll list").split(",")]
    opts = p.js("[...document.querySelectorAll('#options option')].map(o => o.innerText.trim())")
    idx = sorted(opts.index(w) for w in want)
    # <option> rows are not hit-testable, so: click the list's top row (selects option 0 only),
    # plain Down moves the single selection, ctrl+down moves focus only, ctrl+space toggles.
    inf = p.info(f"{Q}('#options')")
    yield click_action(opts[0], to_1000(inf["sx"] + inf["w"] / 2, inf["sy"] + 8))
    for _ in range(idx[0]):
        yield hotkey_action("down")
    cur = idx[0]
    for k in idx[1:]:
        for _ in range(k - cur):
            yield hotkey_action("ctrl+down")
        yield hotkey_action("ctrl+space")
        cur = k
    got = p.js("[...document.getElementById('options').selectedOptions].map(o => o.innerText.trim())")
    if sorted(got) != sorted(want):
        raise OracleError(f"multi-select ended at {got}, wanted {want}")
    yield from p.click(f"{Q}('#area button')", "Submit")


@oracle("choose-date", "choose-date-easy", "choose-date-medium", "choose-date-nodelay")
def o_choose_date(p: Page) -> Iterator[str]:
    mm, dd, yy = (int(x) for x in re.search(r"(\d\d)/(\d\d)/(\d{4})", p.query).groups())
    inp = f"{Q}('#datepicker')"
    yield from p.click(inp, p.js(f"{inp}.getAttribute('aria-label')"))
    p.wait(0.4)
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    for _ in range(14):
        m, y = p.js("[document.querySelector('.ui-datepicker-month').innerText,"
                    " +document.querySelector('.ui-datepicker-year').innerText]")
        cur = y * 12 + months.index(m.strip())
        tgt = yy * 12 + (mm - 1)
        if cur == tgt:
            break
        sel = f"{Q}('.ui-datepicker-{'prev' if tgt < cur else 'next'}')"
        yield from p.click(sel, "Prev" if tgt < cur else "Next", reveal=False)
        p.wait(0.15)
    day = f"[...document.querySelectorAll('.ui-datepicker-calendar a')].find(a => +a.innerText === {dd})"
    yield from p.click(day, str(dd), reveal=False)
    p.wait(0.3)
    yield from p.click(SUBMIT, "Submit")


@oracle("find-word")
def o_find_word(p: Page) -> Iterator[str]:
    pos = p.js(f"{Q}('#query .bold').innerText").split()[0]
    k = int(re.match(r"\d+", pos).group(0)) - 1
    words = p.js(f"{Q}('#area p').innerText").split()
    yield from p.type_into(f"{Q}('#answer-input')", re.sub(r"[^a-zA-Z]", "", words[k]))
    yield from p.click(SUBMIT, "Submit")


def _scroll_textarea(p: Page, sel: str, to_bottom: bool) -> Iterator[str]:
    name = p.js(f"{sel}.getAttribute('aria-label')") or "text area"
    for _ in range(40):
        top, sh, ch = p.js(f"(e => [e.scrollTop, e.scrollHeight, e.clientHeight])({sel})")
        if (to_bottom and top + ch >= sh - 2) or (not to_bottom and top <= 1):
            return
        yield scroll_action(name, -5 if to_bottom else 5)
    raise OracleError("text area never reached the end")


@oracle("scroll-text")
def o_scroll_text(p: Page) -> Iterator[str]:
    ta = f"{Q}('#text-area')"
    yield from _scroll_textarea(p, ta, True)
    last = p.js(f"{ta}.value.split(/\\s/).filter(Boolean).pop()")
    yield from p.type_into(f"{Q}('#answer-input')", re.sub(r"[^a-zA-Z0-9]", "", last))
    yield from p.click(SUBMIT, "Submit")


@oracle("scroll-text-2")
def o_scroll_text_2(p: Page) -> Iterator[str]:
    yield from _scroll_textarea(p, f"{Q}('#text-area')", "bottom" in p.query)
    yield from p.click(SUBMIT, "Submit")


@oracle("click-tab")
def o_click_tab(p: Page) -> Iterator[str]:
    n = re.search(r"Tab #(\d)", p.query).group(1)
    yield from p.click(f"{Q}('#area ul a[href=\"#tabs-{n}\"]')", f"Tab #{n}")


@oracle("click-tab-2", "click-tab-2-easy", "click-tab-2-medium", "click-tab-2-hard")
def o_click_tab_2(p: Page) -> Iterator[str]:
    word = _between(p.query, 'link "', '"')
    link = f"[...document.querySelectorAll('.alink')].find(e => e.innerHTML === {_quote(word)})"
    if not p.js(f"!!{link}"):
        # click-tab-2-medium empties tab #2 after picking the word; then the right answer is tab #2
        tab2 = "document.getElementsByClassName('ui-tab')[1]"
        yield from p.click(tab2, p.js(f"{tab2}.innerText.trim()"))
        return
    panel = p.js(f"{link}.closest('[id^=tabs-]').id")
    tab = f"{Q}('#area ul a[href=\"#{panel}\"]')"
    if not p.js(f"{link}.offsetParent !== null"):
        yield from p.click(tab, p.js(f"{tab}.innerText.trim()"))
        p.wait(0.2)
    yield from p.click(link, word)


def _expand(p: Page, header: str) -> Iterator[str]:
    yield from p.click(header, p.js(f"{header}.innerText.trim()"))
    for _ in range(20):  # jQuery UI accordion animates ~400 ms
        p.wait(0.1)
        if p.js(f"(h => {{var c = h.nextElementSibling; return c && $(c).is(':visible') && !$(c).is(':animated')}})({header})"):
            return


@oracle("click-collapsible", "click-collapsible-nodelay")
def o_click_collapsible(p: Page) -> Iterator[str]:
    yield from _expand(p, f"{Q}('#area h3')")
    yield from p.click(SUBMIT, "Submit")


@oracle("click-collapsible-2", "click-collapsible-2-nodelay")
def o_click_collapsible_2(p: Page) -> Iterator[str]:
    word = _between(p.query, 'link "', '"')
    link = f"[...document.querySelectorAll('.alink')].find(e => e.innerHTML === {_quote(word)})"
    if not p.js(f"$({link}).is(':visible')"):
        yield from _expand(p, f"{link}.closest('div').previousElementSibling")
    yield from p.click(link, word)


@oracle("email-inbox", "email-inbox-delete", "email-inbox-forward", "email-inbox-important",
        "email-inbox-noscroll", "email-inbox-reply")
def o_email_inbox(p: Page) -> Iterator[str]:
    q = p.query
    who = p.js("document.querySelectorAll('#query .bold')[0].innerText")
    # search narrows the list so the thread is on screen without scrolling
    yield from p.click(f"{Q}('#open-search')", "search")
    yield type_action("Search", who)
    p.wait(0.2)
    thread = (f"[...document.querySelectorAll('#search-results .email-thread')]"
              f".find(t => t.querySelector('.email-sender').innerText === {_quote(who)})")
    yield from p.click(f"{thread}.querySelector('.email-sender')", who)
    p.wait(0.2)
    if "trash icon" in q:
        yield from p.click(f"{Q}('#email .trash')", "trash")
    elif "star icon" in q:
        yield from p.click(f"{Q}('#email .star')", "star")
    elif "reply to them" in q:
        text = p.js("document.querySelectorAll('#query .bold')[1].innerText")
        yield from p.click(f"{Q}('#email .email-reply')", "Reply")
        yield from p.type_into(f"{Q}('#reply-text')", text)
        yield from p.click(f"{Q}('#send-reply')", "send")
    elif "forward that email" in q:
        to = p.js("document.querySelectorAll('#query .bold')[1].innerText")
        yield from p.click(f"{Q}('#email .email-forward')", "Forward")
        yield from p.type_into(f"{Q}('#forward .forward-sender')", to)
        yield from p.click(f"{Q}('#send-forward')", "send")
    else:
        raise OracleError(f"unrecognized email query: {q}")


@oracle("social-media")
def o_social_media(p: Page) -> Iterator[str]:
    user, action = p.js("[...document.querySelectorAll('#query .bold')].map(b => b.innerText)")
    post = (f"[...document.querySelectorAll('#area .media')]"
            f".find(m => m.querySelector('.username').innerText === {_quote(user)})")
    simple = {"Reply": "reply", "Retweet": "retweet", "Like": "like"}
    if action in simple:
        yield from p.click(f"{post}.querySelector('.{simple[action]}')", action)
        return
    yield from p.click(f"{post}.querySelector('.more')", "more")
    p.wait(0.15)
    li = f"[...{post}.querySelectorAll('ul li')].find(l => l.innerText.trim().startsWith({_quote(action)}))"
    yield from p.click(li, p.js(f"{li}.innerText.trim()"))


# --------------------------------------------------------------------------- live runner


def validate_action(raw: str) -> bool:
    acts = parse_studio_actions(raw)
    if len(acts) != 1 or not acts[0].name.startswith("cua_"):
        return False
    return parse_cua_action(acts[0].name, acts[0].arg, allow_point=True).valid


class LivePage(Page):
    def __init__(self, pool: Any, i: int):
        self.pool, self.i = pool, i
        self.origin = tuple(pool.workers[i].origin or (0.0, 0.0))

    def js(self, expr: str) -> Any:
        v = self.pool.eval_js(self.i, f"JSON.stringify((() => ({expr}))())")
        return json.loads(v) if isinstance(v, str) else v

    def ax(self) -> list[Any]:
        from pool import parse_ax_lines
        return parse_ax_lines(self.pool.get_cdp_ax(self.i, max_chars=60000))

    def wait(self, s: float) -> None:
        time.sleep(s)


def reset(pool: Any, i: int, task: str, seed: int) -> dict[str, Any]:
    install(pool, i)
    out = pool.reset_worker(i, app_spec(task, seed))
    st = pool.state(i)
    if st["state"].get("error"):
        raise RuntimeError(f"{task}: {st['state']['error']}")
    out["query"] = st["state"].get("query", "")
    return out


def score(pool: Any, i: int, timeout_s: float = 2.0) -> dict[str, Any]:
    """Ground truth from the pool server's event log (first endEpisode), page global as fallback."""
    deadline = time.monotonic() + timeout_s
    while True:
        st = pool.state(i)["state"]
        if st.get("done") or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    raw = st.get("raw_reward")
    if raw is None:
        first = json.loads(pool.eval_js(i, "JSON.stringify(window.__first)") or "null")
        raw = first[0] if first else None
    return {"raw": raw, "reward": 1.0 if (raw is not None and raw > 0) else 0.0, "reason": st.get("reason", "")}


def run_oracle(pool: Any, i: int, task: str, seed: int, step_sleep: float = 0.15, max_steps: int = 80) -> dict[str, Any]:
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"task": task, "seed": seed, "skill": TASK_SKILLS[task], "worker": i, "actions": []}
    try:
        r = reset(pool, i, task, seed)
        rec["query"], rec["reset_ms"] = r["query"], r.get("total_ms")
        page = LivePage(pool, i)
        for n, raw in enumerate(ORACLES[task](page)):
            if n >= max_steps:
                raise OracleError("too many steps")
            if not validate_action(raw):
                raise OracleError(f"invalid Studio action {raw!r}")
            res = pool.act(i, raw)
            rec["actions"].append({"action": raw, "ok": res.get("ok"), "matched": res.get("matched"),
                                   "detail": res.get("detail", "")})
            if not res.get("ok"):
                raise OracleError(f"action failed: {raw} -> {res.get('detail')}")
            time.sleep(step_sleep)
        rec.update(score(pool, i))
    except Exception as e:  # noqa: BLE001 - record, keep going
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
        try:
            rec.update(score(pool, i, timeout_s=0.3))
        except Exception:  # noqa: BLE001
            rec.update(raw=None, reward=0.0)
    rec["wall_s"] = round(time.perf_counter() - t0, 2)
    return rec


def count_owned(pool: Any) -> int:
    """Non-destroyed sandboxes carrying this worker's owner label (any pool)."""
    from daytona.v0 import ListSandboxesQuery

    n = 0
    for sb in pool._client.list(ListSandboxesQuery(labels=dict(OWNER_LABELS))):
        if str(getattr(sb.state, "value", sb.state)).lower() not in ("destroyed", "destroying"):
            n += 1
    return n


def smoke(n_workers: int = 2, tasks: list[str] | None = None, seeds: tuple[int, ...] = (0,)) -> dict[str, Any]:
    """Live: ``n_workers`` daytona-small sandboxes, every oracle task x seeds, teardown to 0."""
    import concurrent.futures as cf

    from pool import SandboxPool

    if not 1 <= n_workers <= 4:
        raise ValueError("budget gate: 1..4 daytona-small workers")
    tasks = list(tasks or sorted(ORACLES))
    out_dir = RL_DIR / "receipts"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rec: dict[str, Any] = {"snapshot": "daytona-small", "workers": n_workers, "labels": OWNER_LABELS, "episodes": []}
    pool = SandboxPool(labels=OWNER_LABELS, max_workers=4, idle_stop_min=15)
    rec["pool_id"] = pool.pool_id
    try:
        t = time.perf_counter()
        rec["boot"] = pool.start(n_workers)
        rec["boot_wall_s"] = round(time.perf_counter() - t, 2)
        with cf.ThreadPoolExecutor(len(pool.workers)) as ex:
            rec["install"] = list(ex.map(lambda k: install(pool, k), range(len(pool.workers))))
        jobs = [(t, s) for s in seeds for t in tasks]
        lanes = [jobs[k::len(pool.workers)] for k in range(len(pool.workers))]

        def lane(k: int) -> list[dict[str, Any]]:
            res = []
            for task, seed in lanes[k]:
                e = run_oracle(pool, k, task, seed)
                print(f"[w{k}] {task}:{seed} reward={e.get('reward')} raw={e.get('raw')} "
                      f"steps={len(e['actions'])} {e.get('error', '')}", flush=True)
                res.append(e)
            return res

        with cf.ThreadPoolExecutor(len(pool.workers)) as ex:
            for r in ex.map(lane, range(len(pool.workers))):
                rec["episodes"].extend(r)
        rec["screenshot"] = str(out_dir / f"miniwob_suite_{stamp}_w0.png")
        Path(rec["screenshot"]).write_bytes(pool.screenshot(0))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        rec["close"] = pool.close_all()
        rec["owned_active_after"] = count_owned(pool)
        eps = rec["episodes"]
        passed = [e for e in eps if e.get("reward") == 1.0 and e.get("raw") == 1]
        rec["summary"] = {
            "episodes": len(eps), "pass_raw_eq_1": len(passed),
            "pass_rate": round(len(passed) / len(eps), 3) if eps else 0.0,
            "tasks_passed": sorted({e["task"] for e in passed}),
            "tasks_failed": sorted({e["task"] for e in eps} - {e["task"] for e in passed}),
            "skills_passed": sorted({e["skill"] for e in passed}),
            "reset_ms_median": sorted(e["reset_ms"] for e in eps if e.get("reset_ms"))[len(eps) // 2] if eps else None,
        }
        path = out_dir / f"miniwob_suite_smoke_{stamp}.json"
        path.write_text(json.dumps(rec, indent=2, default=str))
        rec["receipt"] = str(path)
    return rec


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke" in args:
        k = args.index("--smoke")
        n = int(args[k + 1]) if len(args) > k + 1 and args[k + 1].isdigit() else 2
        only = args[args.index("--tasks") + 1].split(",") if "--tasks" in args else None
        seeds = tuple(int(s) for s in args[args.index("--seeds") + 1].split(",")) if "--seeds" in args else (0,)
        r = smoke(n, only, seeds)
        print(json.dumps({k: r[k] for k in ("summary", "close", "owned_active_after", "receipt") if k in r}
                         | ({"error": r["error"]} if "error" in r else {}), indent=2, default=str))
        sys.exit(0 if r["owned_active_after"] == 0 else 1)
    print(json.dumps({"train": len(TRAIN_TASKS), "heldout": len(HELDOUT_TASKS), "oracles": len(ORACLES),
                      "by_skill": split_counts()}, indent=2))
