"""Python mirror of live Lapis Studio's model I/O, so RL rollouts see and emit exactly what Studio does.

Mirrored from the C++ source (keep in sync when those change):
  * ``parse_studio_output``  <- ``studio::parseStudioModelOutput``  (src/llm/prefix.cpp)
  * ``parse_cua_action``     <- ``parseCuaAction``                  (src/backend.cpp)
  * ``normalize_open_url``   <- ``studio::cua::normalizeOpenUrl``   (src/mac/cua.mm)
  * ``hotkey_combo_valid``   <- ``studio::cua::parseHotkeyCombo(...).valid`` (src/mac/cua.mm)
  * ``format_ax_nodes``      <- ``studio::cua::formatNodesForModel`` (src/mac/cua.mm)
  * ``ax_structural_noise`` / ``ax_tiny_rect`` <- ``axStructuralNoise`` / ``axTinyRect`` (src/mac/cua.mm)
  * ``observation_prompt`` / ``followup_prompt`` <- ``cuaObservationPrompt`` / ``cuaFollowupPrompt`` (src/backend.cpp)

RL-side extensions that Studio does NOT parse yet (a policy trained on them leaks them as text
in Studio until prefix.cpp / parseCuaAction learn them):
  * ``[[NOTE:...]]`` scratchpad tags (``strip_notes``);
  * an ``@x,y`` suffix on a click target, x and y in 0..1000 of the screenshot (``split_point``).

``format_cdp_ax_lines`` maps a Chromium CDP ``Accessibility.getFullAXTree`` result onto Studio's
AxNode walk so a headless-Chrome sandbox produces the same compact lines a Mac observation does.
Lengths are counted in UTF-16 code units, as QString does.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

POST_ACTION_CHARS = 6000  # kPostActionChars: the follow-up observation budget
OBSERVE_CHARS = 8000  # kObserveChars: cua_observe's budget
TEXT_CAP = 280  # formatNodesForModel's per-string cap
ELLIPSIS = "…"

# --------------------------------------------------------------------------- QString helpers


def _qlen(s: str) -> int:
    """QString::size(): UTF-16 code units."""
    return len(s.encode("utf-16-le")) // 2


def _qleft(s: str, n: int) -> str:
    """QString::left(n) in UTF-16 units (an astral char split in half is dropped)."""
    if n <= 0:
        return ""
    if _qlen(s) <= n:
        return s
    return s.encode("utf-16-le")[: 2 * n].decode("utf-16-le", errors="ignore")


def _qspace(c: str) -> bool:
    """QChar::isSpace."""
    return c in "\t\n\v\f\r\x85\xa0" or unicodedata.category(c) in ("Zs", "Zl", "Zp")


def _qpunct(c: str) -> bool:
    """QChar::isPunct: Unicode P* categories."""
    return unicodedata.category(c).startswith("P")


def _trimmed(s: str) -> str:
    i, j = 0, len(s)
    while i < j and _qspace(s[i]):
        i += 1
    while j > i and _qspace(s[j - 1]):
        j -= 1
    return s[i:j]


def _simplified(s: str) -> str:
    """QString::simplified: trim, and collapse every internal whitespace run to one space."""
    out, word = [], []
    for c in s:
        if _qspace(c):
            if word:
                out.append("".join(word))
                word = []
        else:
            word.append(c)
    if word:
        out.append("".join(word))
    return " ".join(out)


# --------------------------------------------------------------------------- model output


@dataclass(frozen=True)
class StudioAction:
    name: str
    arg: str = ""


@dataclass
class ParsedModelOutput:
    actions: list[StudioAction] = field(default_factory=list)
    clean_text: str = ""
    ignore: bool = False
    notes: list[str] = field(default_factory=list)  # RL extension, always empty from Studio's parser


ACTION_RE = re.compile(r"\[\[ACTION:([^\]:]+)(?::([^\]]*))?\]\]")
NOTE_RE = re.compile(r"\[\[NOTE:(.*?)\]\]", re.DOTALL)
_TRAILING_RE = re.compile(r"[ \t]+\n")
_BLANK_RUNS_RE = re.compile(r"\n{3,}")


def parse_studio_output(raw: str) -> ParsedModelOutput:
    """Exact ``parseStudioModelOutput``: ``[[IGNORE]]`` anywhere wins; every
    ``[[ACTION:name]]`` / ``[[ACTION:name:arg]]`` (arg may hold ':' but not ']') is collected
    in order and cut from the text, dropping blanks before a tag that is followed by whitespace
    or punctuation; then trailing blanks and 3+ newline runs are tidied and the text trimmed."""
    out = ParsedModelOutput()
    if "[[IGNORE]]" in raw:
        out.ignore = True
        return out
    text, last = "", 0
    for m in ACTION_RE.finditer(raw):
        name = _trimmed(m.group(1))
        if name:
            out.actions.append(StudioAction(name, _trimmed(m.group(2) or "")))
        text += raw[last : m.start()]
        last = m.end()
        nxt = raw[last] if last < len(raw) else "\n"
        if _qspace(nxt) or _qpunct(nxt):
            text = text.rstrip(" \t")
    text += raw[last:]
    text = _BLANK_RUNS_RE.sub("\n\n", _TRAILING_RE.sub("\n", text))
    out.clean_text = _trimmed(text)
    return out


def parse_studio_actions(raw: str) -> list[StudioAction]:
    return parse_studio_output(raw).actions


def strip_notes(raw: str) -> tuple[str, list[str]]:
    """RL extension: remove ``[[NOTE:...]]`` scratchpad tags (may span lines; ends at the first
    ``]]``). Returns (text without notes, note bodies trimmed). A space before a removed tag is
    dropped the same way Studio drops it before an ACTION tag."""
    notes: list[str] = []
    text, last = "", 0
    for m in NOTE_RE.finditer(raw):
        notes.append(_trimmed(m.group(1)))
        text += raw[last : m.start()]
        last = m.end()
        nxt = raw[last] if last < len(raw) else "\n"
        if _qspace(nxt) or _qpunct(nxt):
            text = text.rstrip(" \t")
    return text + raw[last:], notes


def parse_policy_output(raw: str) -> ParsedModelOutput:
    """What an RL rollout should run: notes stripped first (so a note never executes or shows),
    then Studio's exact parser."""
    text, notes = strip_notes(raw)
    out = parse_studio_output(text)
    out.notes = notes
    return out


# --------------------------------------------------------------------------- cua_* arguments

POINT_RE = re.compile(r"^(.*?)\s*@\s*(\d{1,4})\s*,\s*(\d{1,4})\s*$", re.DOTALL)


def split_point(target: str) -> tuple[str, tuple[int, int] | None]:
    """RL extension: ``"Chrome|Open p004@512,340"`` -> (``"Chrome|Open p004"``, (512, 340)).
    The point is in 0..1000 screenshot-normalized units; out-of-range or malformed suffixes are
    left in the target untouched (that is also what Studio does with every suffix today)."""
    m = POINT_RE.match(target)
    if not m:
        return target, None
    x, y = int(m.group(2)), int(m.group(3))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return target, None
    return _simplified(m.group(1)), (x, y)


def point_to_pixels(point_1000: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    """0..1000 normalized -> pixel, clamped inside the image."""
    x = min(width - 1, max(0, round(point_1000[0] * width / 1000)))
    y = min(height - 1, max(0, round(point_1000[1] * height / 1000)))
    return x, y


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_PORT_RE = re.compile(r"^[^/:]+:\d")
_HOST_RE = re.compile(
    r"^(localhost|[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}|\d{1,3}(\.\d{1,3}){3})(:\d+)?([/?#].*)?$", re.DOTALL
)


def normalize_open_url(raw: str) -> str:
    """Exact ``normalizeOpenUrl``: a scheme URL passes through, a bare host gets https://,
    anything else (or anything with whitespace) is ""."""
    u = _trimmed(raw)
    if not u or re.search(r"\s", u):
        return ""
    if _SCHEME_RE.match(u) and not _PORT_RE.match(u):
        return u
    if not _HOST_RE.match(u):
        return ""
    return "https://" + u


HOTKEY_KEYS = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789-=[]/.,")
    + ["space", "enter", "return", "tab", "esc", "escape", "backspace", "delete", "forwarddelete", "up", "down",
       "left", "right", "home", "end", "pageup", "pagedown"]
    + [f"f{i}" for i in range(1, 13)]
)
_MODIFIERS = frozenset({"cmd", "command", "meta", "shift", "opt", "option", "alt", "ctrl", "control"})


def hotkey_combo_valid(combo: str) -> bool:
    """``parseHotkeyCombo(...).valid``: '+'-joined modifiers plus exactly one known key."""
    keys = 0
    for raw in _trimmed(combo).lower().split("+"):
        p = _trimmed(raw)
        if not p:
            return False
        if p in _MODIFIERS:
            continue
        if keys or p not in HOTKEY_KEYS:
            return False
        keys += 1
    return keys == 1


@dataclass
class CuaCommand:
    kind: str = ""
    target: str = ""
    text: str = ""
    delta_y: int = 0
    valid: bool = False
    point_1000: tuple[int, int] | None = None  # RL extension (cua_click only)


def _parse_int(s: str) -> int | None:
    """QString::toInt: optional sign and digits, surrounding whitespace allowed."""
    s = _trimmed(s)
    if re.fullmatch(r"[+-]?\d+", s):
        v = int(s)
        return v if -(2**31) <= v < 2**31 else None
    return None


def parse_cua_action(action: str, arg: str, allow_point: bool = False) -> CuaCommand:
    """Exact ``parseCuaAction`` (hotkey validity via ``hotkey_combo_valid``). ``allow_point``
    turns on the RL ``@x,y`` click extension."""
    c = CuaCommand()
    a = _trimmed(action).lower()
    if a == "cua_click":
        c.kind = "click"
        c.target = _simplified(arg)
        if allow_point:
            c.target, c.point_1000 = split_point(c.target)
        c.valid = bool(c.target)
    elif a == "cua_type":
        c.kind = "type"
        bar = arg.find("|")
        if bar < 0:
            return c
        c.target = _simplified(arg[:bar])
        c.text = arg[bar + 1 :]
        c.valid = bool(c.target) and bool(c.text)
    elif a == "cua_scroll":
        c.kind = "scroll"
        bar = arg.rfind("|")
        amount = _trimmed(arg if bar < 0 else arg[bar + 1 :]).lower()
        c.target = "" if bar < 0 else _simplified(arg[:bar])
        n = _parse_int(amount)
        if n is not None:
            c.delta_y = max(-100, min(100, n))
        else:
            c.delta_y = {"up": 5, "down": -5, "pageup": 12, "page up": 12, "pagedown": -12, "page down": -12}.get(amount, 0)
        c.valid = c.delta_y != 0
    elif a == "cua_open":
        c.kind = "open"
        bar = arg.find("|")
        app = _simplified(arg if bar < 0 else arg[:bar])
        url = "" if bar < 0 else _trimmed(arg[bar + 1 :])
        if bar < 0 and not app.lower().endswith(".app") and normalize_open_url(app):
            url, app = app, ""
        if url:
            c.text = normalize_open_url(url)
            if not c.text:
                return c
        c.target = app
        c.valid = bool(c.target) or bool(c.text)
    elif a == "cua_observe":
        c.kind = "observe"
        c.target = _simplified(arg)
        c.valid = True
    elif a == "cua_hotkey":
        c.kind = "hotkey"
        bar = arg.find("|")
        c.target = "" if bar < 0 else _simplified(arg[:bar])
        c.text = _simplified(arg if bar < 0 else arg[bar + 1 :]).replace(" ", "")
        c.valid = hotkey_combo_valid(c.text)
    elif a == "cua_drag":
        c.kind = "drag"
        bar = arg.find("|")
        if bar < 0:
            return c
        c.target = _simplified(arg[:bar])
        c.text = _simplified(arg[bar + 1 :])
        c.valid = bool(c.target) and bool(c.text)
    return c


# --------------------------------------------------------------------------- observations


@dataclass
class AxNode:
    role: str  # Mac AX role, e.g. "AXButton"
    title: str = ""
    value: str = ""
    app_name: str = ""
    pid: int = 0
    rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h in screen points
    actions: list[str] = field(default_factory=list)


STRUCTURAL_WRAPPERS = frozenset({
    "AXGroup", "AXGenericElement", "AXUnknown", "AXSection", "AXList", "AXScrollArea", "AXSplitGroup",
    "AXLayoutArea", "AXLandmarkMain", "AXLandmarkNavigation", "AXLandmarkBanner", "AXLandmarkRegion", "AXWebArea",
})


def ax_structural_noise(n: AxNode, depth: int) -> bool:
    return depth > 0 and not _trimmed(n.title) and not _trimmed(n.value) and n.role in STRUCTURAL_WRAPPERS


def ax_tiny_rect(n: AxNode, depth: int) -> bool:
    return depth > 0 and not (n.rect[2] >= 2 and n.rect[3] >= 2)


def _qround(v: float) -> int:
    """qRound: half away from zero."""
    return int(v + 0.5) if v >= 0 else int(v - 0.5)


def _quoted(s: str) -> str:
    out = ['"']
    for c in s:
        if c in '"\\':
            out.append("\\" + c)
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif ord(c) < 0x20:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


def _bare(ax: str) -> str:
    return ax[2:] if ax.startswith("AX") else ax


def format_ax_nodes(nodes: Iterable[AxNode], max_chars: int = POST_ACTION_CHARS) -> str:
    """Exact ``formatNodesForModel``: a ``[App pid N]`` header when the app changes, then
    ``Role "title" = "value" @x,y wxh {Press}`` per node; whole lines only, a lone "…" marks
    omitted nodes once the budget is hit."""
    if max_chars <= 0:
        return ""
    cap_n = min(max_chars, TEXT_CAP)

    def cap(s: str) -> str:
        return _qleft(s, cap_n - 1) + ELLIPSIS if _qlen(s) > cap_n else s

    out, out_len = [], 0
    last_app, last_pid = None, None
    for n in nodes:
        block = ""
        if n.app_name != last_app or n.pid != last_pid:
            block += f"[{_qleft(n.app_name, max_chars)} pid {n.pid}]\n"
        line = _bare(_qleft(n.role, max_chars))
        if n.title:
            line += " " + _quoted(cap(n.title))
        if n.value:
            line += " = " + _quoted(cap(n.value))
        x, y, w, h = n.rect
        line += f" @{_qround(x)},{_qround(y)} {_qround(w)}x{_qround(h)}"
        acts = [_bare(_qleft(a, max_chars)) for a in n.actions[:32] if a not in ("AXShowMenu", "AXScrollToVisible")]
        if acts:
            line += " {" + ",".join(acts) + "}"
        block += line + "\n"
        blen = _qlen(block)
        if out_len + blen > max_chars:
            if out_len < max_chars:
                out.append(ELLIPSIS)
            break
        out.append(block)
        out_len += blen
        last_app, last_pid = n.app_name, n.pid
    return "".join(out)


def observation_prompt(action: str, tier: str, ok: bool, detail: str, observation: str) -> str:
    """Exact ``cuaObservationPrompt``."""
    obs = _trimmed(observation) or "(no accessibility nodes visible)"
    return f"[CUA observation after {action} ({tier or 'ERR'}, ok={'1' if ok else '0'}, detail={_simplified(detail)})]:\n{obs}"


def followup_prompt(steps: Iterable[Mapping[str, Any]]) -> str:
    """Exact ``cuaFollowupPrompt``; each step has action, target, tier, ok, detail, observation."""
    blocks = [
        observation_prompt(
            s["action"] + (":" + s["target"] if s.get("target") else ""),
            s.get("tier", ""), bool(s.get("ok")), s.get("detail", ""), s.get("observation", ""),
        )
        for s in steps
    ]
    return "\n\n".join(blocks) + "\nContinue only if more actions are needed; otherwise give a concise final answer without cua_* tags."


# --------------------------------------------------------------------------- CDP -> Studio AX

# Chromium AX role -> the Mac AX role Chrome exposes for it (what Studio's walker would read).
CDP_TO_AX_ROLE = {
    "RootWebArea": "AXWebArea", "WebArea": "AXWebArea", "button": "AXButton", "link": "AXLink",
    "StaticText": "AXStaticText", "heading": "AXHeading", "textbox": "AXTextField",
    "searchbox": "AXTextField", "TextField": "AXTextField", "textField": "AXTextField", "checkbox": "AXCheckBox",
    "radio": "AXRadioButton", "combobox": "AXComboBox", "ComboBoxGrouping": "AXComboBox", "listbox": "AXList",
    "list": "AXList", "listitem": "AXGroup", "ListMarker": "AXStaticText", "img": "AXImage", "image": "AXImage",
    "menuitem": "AXMenuItem", "menu": "AXMenu", "menubar": "AXMenuBar", "tab": "AXRadioButton", "tablist": "AXTabGroup",
    "slider": "AXSlider", "switch": "AXCheckBox", "table": "AXTable", "row": "AXRow", "cell": "AXCell",
    "gridcell": "AXCell", "columnheader": "AXCell", "paragraph": "AXGroup", "article": "AXGroup",
    "generic": "AXGroup", "group": "AXGroup", "section": "AXGroup", "region": "AXLandmarkRegion",
    "main": "AXLandmarkMain", "navigation": "AXLandmarkNavigation", "banner": "AXLandmarkBanner",
    "contentinfo": "AXGroup", "time": "AXGroup", "code": "AXGroup", "strong": "AXGroup", "emphasis": "AXGroup",
    "LabelText": "AXGroup", "none": "AXGroup", "presentation": "AXGroup", "Pre": "AXGroup",
}
_PRESSABLE = {"AXButton", "AXLink", "AXCheckBox", "AXRadioButton", "AXMenuItem", "AXComboBox"}
_VALUE_ROLES = {"AXTextField", "AXComboBox", "AXCheckBox", "AXRadioButton", "AXSlider"}


def _ax_value(prop: Any) -> str:
    if isinstance(prop, Mapping):
        v = prop.get("value")
        return "" if v is None else str(v)
    return "" if prop is None else str(prop)


def cdp_node_to_ax(node: Mapping[str, Any], rect: tuple[float, float, float, float], app_name: str, pid: int) -> AxNode:
    role = _ax_value(node.get("role"))
    ax_role = CDP_TO_AX_ROLE.get(role, "AXGroup")
    name = _ax_value(node.get("name"))
    value = _ax_value(node.get("value")) if ax_role in _VALUE_ROLES else ""
    if ax_role == "AXStaticText":  # Chrome on the Mac exposes text in AXValue with an empty AXTitle
        name, value = "", name
    return AxNode(
        role=ax_role, title=name, value=value, app_name=app_name, pid=pid, rect=rect,
        actions=["AXPress"] if ax_role in _PRESSABLE else [],
    )


def format_cdp_ax_lines(
    cdp_nodes: list[Mapping[str, Any]],
    bounds: Mapping[Any, tuple[float, float, float, float]] | None = None,
    *,
    app_name: str = "Google Chrome",
    pid: int = 0,
    origin: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    max_chars: int = POST_ACTION_CHARS,
    max_nodes: int = 500,
) -> str:
    """Chromium ``Accessibility.getFullAXTree`` ``nodes`` -> Studio's compact AX lines.

    ``getFullAXTree`` carries no geometry: pass ``bounds`` keyed by ``backendDOMNodeId`` (or by
    ``nodeId``) as CSS-pixel ``(x, y, w, h)`` from ``DOM.getBoxModel``, or put a ``bounds`` key on
    each node. ``origin`` is the viewport's top-left in screen points and ``scale`` CSS px ->
    points, so lines match what Studio reads off a real Chrome window. Walk order is depth first
    from the root in child order (document order, like Studio's in-page walk); ignored nodes,
    nameless wrappers and sub-2x2 rects are walked through but not shown, as in Studio."""
    bounds = bounds or {}
    by_id = {str(n["nodeId"]): n for n in cdp_nodes}
    child_ids = {c for n in cdp_nodes for c in map(str, n.get("childIds", ()))}
    roots = [n for n in cdp_nodes if str(n["nodeId"]) not in child_ids and "parentId" not in n] or cdp_nodes[:1]

    def rect_of(n: Mapping[str, Any]) -> tuple[float, float, float, float]:
        b = n.get("bounds") or bounds.get(n.get("backendDOMNodeId")) or bounds.get(n["nodeId"]) or (0, 0, 0, 0)
        x, y, w, h = b
        return (origin[0] + x * scale, origin[1] + y * scale, w * scale, h * scale)

    kept: list[AxNode] = []
    stack = [(r, 0) for r in reversed(roots)]
    seen: set[str] = set()
    while stack and len(kept) < max_nodes:
        n, depth = stack.pop()
        nid = str(n["nodeId"])
        if nid in seen:
            continue
        seen.add(nid)
        if _ax_value(n.get("role")) == "InlineTextBox":
            continue  # Blink-internal text runs; the Mac tree never exposes them
        if not n.get("ignored"):
            ax = cdp_node_to_ax(n, rect_of(n), app_name, pid)
            if not (ax_structural_noise(ax, depth) or ax_tiny_rect(ax, depth)):
                kept.append(ax)
        kids = [by_id[c] for c in map(str, n.get("childIds", ())) if c in by_id]
        stack.extend((k, depth + 1) for k in reversed(kids))
    return format_ax_nodes(kept, max_chars)
