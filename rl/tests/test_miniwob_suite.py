"""Offline tests for the MiniWoB++ suite: frozen split, skill coverage, Studio action format, oracles
against a canned page. Live oracle smoke (<= 4 daytona-small, owner=w3-miniwob, always deleted):

  uv run python envs/miniwob_suite.py --smoke 2
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import pytest

from envs import miniwob_suite as M
from pool import AxLine
from studio_contract import parse_cua_action, parse_studio_actions


# ----------------------------------------------------------------------------- split


def test_split_sizes_and_disjoint():
    assert len(M.ALL_TASKS) == 130
    assert len(M.HELDOUT_TASKS) == 30
    assert len(M.TRAIN_TASKS) >= 80
    assert set(M.TRAIN_TASKS).isdisjoint(M.HELDOUT_TASKS)
    assert set(M.TRAIN_TASKS) | set(M.HELDOUT_TASKS) == set(M.ALL_TASKS)


def test_probe_tasks_held_out():
    assert set(M.PROBE_TASKS) == {"click-button", "click-link", "focus-text", "enter-text", "login-user"}
    assert set(M.PROBE_TASKS) <= M.HELDOUT_TASKS


def test_split_is_frozen():
    # changing the held-out set invalidates every held-out number; this pin makes it a deliberate edit
    assert sorted(M.HELDOUT_TASKS) == sorted("""
        book-flight book-flight-nodelay choose-list click-button click-link click-menu click-menu-2 click-option
        count-shape drag-items drag-items-grid enter-date enter-text enter-text-2 enter-text-dynamic enter-time
        focus-text focus-text-2 login-user login-user-popup navigate-tree phone-book read-table read-table-2
        resize-textarea search-engine sign-agreement use-autocomplete use-autocomplete-nodelay use-spinner
    """.split())


def test_no_family_straddles_split():
    fams: dict[str, set[bool]] = defaultdict(set)
    for t in M.ALL_TASKS:
        fams[M.family(t)].add(t in M.HELDOUT_TASKS)
    assert [f for f, sides in fams.items() if len(sides) > 1] == []
    assert M.family("email-inbox-forward-nl-turk") == "email-inbox"
    assert M.family("click-collapsible-2-nodelay") == "click-collapsible"


def test_matches_dataset_on_disk():
    d = M.MINIWOB_HTML / "miniwob"
    if not d.is_dir():
        pytest.skip("dataset not staged on this host")
    assert sorted(p.stem for p in d.glob("*.html")) == list(M.ALL_TASKS)


def test_every_skill_covered_on_both_sides():
    assert set(M.TASK_SKILLS.values()) == set(M.SKILLS)
    counts = M.split_counts()
    for s in M.SKILLS:
        assert counts[s]["train"] >= 2, s
        assert counts[s]["heldout"] >= 1, s
    assert sum(c["train"] for c in counts.values()) == len(M.TRAIN_TASKS)
    assert sum(c["heldout"] for c in counts.values()) == 30


# ----------------------------------------------------------------------------- oracles registry


def test_oracles_train_only_and_diverse():
    assert len(M.ORACLES) >= 25
    assert set(M.ORACLES) <= set(M.TRAIN_TASKS)
    assert not set(M.ORACLES) & M.HELDOUT_TASKS
    skills = {M.TASK_SKILLS[t] for t in M.ORACLES}
    assert {"click", "form", "checkbox_radio", "dropdown", "datetime_slider", "table_search", "menu_tabs",
            "multistep", "visual_reasoning"} <= skills
    with pytest.raises(ValueError):
        M.oracle("click-button")(lambda p: iter(()))


# ----------------------------------------------------------------------------- Studio action format


@pytest.mark.parametrize("raw,kind", [
    (M.click_action("Submit"), "click"),
    (M.click_action("slider handle", (120, 340)), "click"),
    (M.type_action("text field 2", "hello world "), "type"),
    (M.hotkey_action("cmd+c"), "hotkey"),
    (M.hotkey_action("ctrl+space"), "hotkey"),
    (M.hotkey_action("pageup"), "hotkey"),
    (M.scroll_action("text area", -5), "scroll"),
])
def test_action_format(raw, kind):
    assert re.fullmatch(r"\[\[ACTION:cua_\w+:[^\]]*\]\]", raw)
    assert M.validate_action(raw)
    (a,) = parse_studio_actions(raw)
    cmd = parse_cua_action(a.name, a.arg, allow_point=True)
    assert cmd.valid and cmd.kind == kind


def test_point_click_and_label_sanitizing():
    raw = M.click_action("a]b|c @3,4", (512, 1000))
    (a,) = parse_studio_actions(raw)
    cmd = parse_cua_action(a.name, a.arg, allow_point=True)
    assert cmd.point_1000 == (512, 1000) and "]" not in cmd.target and "|" not in cmd.target
    assert M.to_1000(1024, 768) == (1000, 1000)
    assert M.to_1000(0, 0) == (0, 0)
    with pytest.raises(M.OracleError):
        M.type_action("x", "bad ] text")
    assert not M.validate_action("[[ACTION:cua_hotkey:notakey]]")


def test_studio_trims_typed_text():
    # Studio trims the action argument, so a trailing space cannot be typed; copy-paste tasks (whose
    # expected text ends in a space) therefore go through the clipboard, not cua_type.
    (a,) = parse_studio_actions(M.type_action("text field", "Lorem ipsum. "))
    assert parse_cua_action(a.name, a.arg).text == "Lorem ipsum."


# ----------------------------------------------------------------------------- app spec


def test_app_spec_shape():
    if not (M.MINIWOB_HTML / "miniwob").is_dir():
        pytest.skip("dataset not staged on this host")
    spec = M.app_spec("click-checkboxes", 7)
    html = spec["html"]
    assert spec["name"] == "miniwob:click-checkboxes"
    assert f'src="http://127.0.0.1:{M.STATIC_PORT}/core/core.js"' in html
    assert "../core/" not in html and "../common/" not in html and "<base" not in html
    assert '"task": "click-checkboxes", "seed": 7' in html and "Math.seedrandom" in html
    assert html.rindex("__lapis") < html.rindex("</body>")
    assert len(html) < 20000  # assets are served in-sandbox, not inlined into every reset
    with pytest.raises(KeyError):
        M.app_spec("not-a-task", 0)


# ----------------------------------------------------------------------------- oracles on a canned page


class FakePage(M.Page):
    """Answers ``js`` from a substring table; AX lines given explicitly."""

    def __init__(self, table: dict[str, Any], lines: list[AxLine], hits: bool = True):
        self.table, self.lines, self._hits = table, lines, hits
        self.origin = (0.0, 80.0)

    def js(self, expr: str) -> Any:
        if expr.startswith("__lapis.info("):
            return {"x": 10.0, "y": 20.0, "w": 40.0, "h": 20.0, "hit": True, "label": "x"}
        if expr.startswith("__lapis.hitAt("):
            return self._hits
        for k, v in self.table.items():
            if k in expr:
                return v
        raise KeyError(expr)

    def ax(self) -> list[AxLine]:
        return self.lines


def _line(role: str, title: str = "", value: str = "") -> AxLine:
    return AxLine(role=role, title=title, value=value, rect=(10, 100, 40, 20), pressable=role in ("Button", "CheckBox"))


def _run(fn, page) -> list[str]:
    acts = list(fn(page))
    assert acts and all(M.validate_action(a) for a in acts), acts
    return acts


def test_oracle_checkboxes_named_clicks():
    page = FakePage({"innerText": "Select ab, cd and click Submit.", ".checked": False},
                    [_line("CheckBox", "ab"), _line("CheckBox", "cd"), _line("Button", "Submit")])
    acts = _run(M.ORACLES["click-checkboxes"], page)
    assert acts == ["[[ACTION:cua_click:ab]]", "[[ACTION:cua_click:cd]]", "[[ACTION:cua_click:Submit]]"]


def test_oracle_falls_back_to_point_when_name_misses():
    page = FakePage({"innerText": "Click button ONE."}, [_line("Button", "ONE")], hits=False)
    (a,) = _run(M.ORACLES["click-test-2"], page)
    assert a.startswith("[[ACTION:cua_click:ONE @") and a.endswith("]]")


def test_oracle_arithmetic_types_answer():
    page = FakePage({"math-problem": "7 x 8 =", "aria-label": "text field"},
                    [_line("TextField", "text field"), _line("Button", "Submit")])
    acts = _run(M.ORACLES["simple-arithmetic"], page)
    assert acts == ["[[ACTION:cua_type:text field|56]]", "[[ACTION:cua_click:Submit]]"]


def test_oracle_algebra_cases():
    for prob, ans in (("x + 3 = 10", "7"), ("x - 3 = 10", "13"), ("3 + x = 10", "7"), ("3 - x = 10", "-7")):
        page = FakePage({"math-problem": prob, "aria-label": "text field"},
                        [_line("TextField", "text field"), _line("Button", "Submit")])
        assert _run(M.ORACLES["simple-algebra"], page)[0] == f"[[ACTION:cua_type:text field|{ans}]]"


def test_oracle_copy_paste_uses_clipboard_hotkeys():
    page = FakePage({"getAttribute('aria-label')": "text area"},
                    [_line("TextField", "text area"), _line("TextField", "text field"), _line("Button", "Submit")])
    acts = _run(M.ORACLES["copy-paste"], page)
    assert [a for a in acts if "hotkey" in a] == [
        "[[ACTION:cua_hotkey:cmd+a]]", "[[ACTION:cua_hotkey:cmd+c]]", "[[ACTION:cua_hotkey:cmd+v]]"]
