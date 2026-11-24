"""Offline tests for FeedArena (rl/synth/feed_arena.py) and the pool's pure helpers (rl/pool.py)."""

from __future__ import annotations

import collections
import json
import random
import shutil
import subprocess

import pytest

import pool as P
from synth import feed_arena as FA
from synth.feed_arena import FeedArena, TASK_KINDS, TABS, opened_state, score_state, view

AUDIT_SEEDS = 10_000

# --------------------------------------------------------------------------- shortcut policies
# Each looks only at the default (Latest, no search) view, like a policy that never uses the UI.


def pick_top_card(cards):
    return cards[0].id


def pick_3rd_card(cards):
    return cards[2].id


def pick_max_likes(cards):
    return max(cards, key=lambda p: (p.likes, -cards.index(p))).id


def pick_max_reposts(cards):
    return max(cards, key=lambda p: (p.reposts, -cards.index(p))).id


def pick_fixed_repost(cards, fixed=60):
    """The old feed hard-coded tie reposts at 60/100: pick the card closest to a fixed value."""
    return min(cards, key=lambda p: (abs(p.reposts - fixed), cards.index(p))).id


SHORTCUTS = {
    "pick_3rd_card": pick_3rd_card,
    "pick_top_card": pick_top_card,
    "pick_max_likes": pick_max_likes,
    "pick_max_reposts": pick_max_reposts,
    "pick_fixed_repost": pick_fixed_repost,
}


def _audit(n_seeds: int):
    """mean reward per (task kind, policy); 'random' is the exact expectation 1/len(cards)."""
    tot = collections.defaultdict(float)
    for s in range(n_seeds):
        feed = FA.generate_feed(s)
        cards = view(feed)
        for kind in TASK_KINDS:
            task = FA.make_task(feed, kind, random.Random(f"audit-{s}-{kind}"))
            for name, pol in SHORTCUTS.items():
                tot[(kind, name)] += score_state(task.target_id, opened_state([pol(cards)]))
            tot[(kind, "random")] += 1.0 / len(cards)
            tot[(kind, "shotgun_all")] += score_state(task.target_id, opened_state(
                [c.id for c in cards if c.id != task.target_id] + [task.target_id]))
    return {k: v / n_seeds for k, v in tot.items()}


@pytest.fixture(scope="module")
def audit():
    return _audit(AUDIT_SEEDS)


def test_shortcut_audit_10k(audit):
    names = [*SHORTCUTS, "random", "shotgun_all"]
    rows = [f"{'task':22s} " + " ".join(f"{n:>17s}" for n in names)]
    for kind in TASK_KINDS:
        rows.append(f"{kind:22s} " + " ".join(f"{audit[(kind, n)]:17.3f}" for n in names))
    overall = {n: sum(audit[(k, n)] for k in TASK_KINDS) / len(TASK_KINDS) for n in names}
    rows.append(f"{'overall':22s} " + " ".join(f"{overall[n]:17.3f}" for n in names))
    print("\n" + "\n".join(rows))
    for name in SHORTCUTS:
        assert overall[name] < 0.25, (name, overall[name])
        for kind in TASK_KINDS:
            assert audit[(kind, name)] < 0.25, (kind, name, audit[(kind, name)])
    # uniform guessing is ~1/n (n = 10..19 cards), and shotgun clicking never beats it
    assert 0.05 < overall["random"] < 0.10
    for kind in TASK_KINDS:
        assert audit[(kind, "shotgun_all")] == pytest.approx(audit[(kind, "random")], rel=1e-9)


# --------------------------------------------------------------------------- generation


def test_deterministic():
    a, b = FeedArena.build(123), FeedArena.build(123)
    assert a.html() == b.html() and a.task == b.task
    assert FeedArena.build(124).html() != a.html()


def test_knob_ranges_and_traps():
    pinned, ads, layouts, ties, tie_reposts = set(), set(), set(), 0, set()
    likes_bait = reposts_bait = ad_likes_bait = 0
    n = 2000
    for s in range(n):
        f = FA.generate_feed(s)
        k = sum(p.pinned for p in f.posts)
        pinned.add(k)
        ads.add(len(f.ads))
        layouts.add(f.pinned_layout)
        # distinct likes and reposts among organic posts: every truth is unique without tie-breaks
        assert len({p.likes for p in f.posts}) == len(f.posts)
        assert len({p.reposts for p in f.posts}) == len(f.posts)
        win = FA.truth(f, "most_viral")
        if f.tie:
            ties += 1
            top = sorted(f.posts, key=FA.SORT_KEYS["Top"])
            assert top[0].virality == top[1].virality and top[0].likes > top[1].likes
            tie_reposts.add(top[0].reposts)
        likes_bait += max(f.posts, key=lambda p: p.likes).id != win.id
        reposts_bait += max(f.posts, key=lambda p: p.reposts).id != win.id
        if f.ads:
            ad_likes_bait += max(f.ads, key=lambda p: p.likes).likes > max(p.likes for p in f.posts)
    assert pinned == {0, 1, 2, 3, 4}
    assert ads == {0, 1, 2, 3}
    assert layouts == set(FA.PINNED_LAYOUTS)
    assert 0.28 < ties / n < 0.42
    assert len(tie_reposts) > 0.8 * ties  # tie values are random, never a fixed 60/100
    assert likes_bait / n > 0.8 and reposts_bait / n > 0.6 and ad_likes_bait > 0


def test_quote_pair_and_focus_author():
    for s in range(500):
        f = FA.generate_feed(s)
        hits = [p for p in f.posts if f.snippet in p.text]
        assert {p.id for p in hits} == {f.quote_target, f.quote_decoy}
        assert len({p.handle for p in hits}) == 2
        focus = [p for p in f.posts if p.handle == f.focus_handle]
        assert len(focus) == 4
        assert FA.truth(f, "search_author_top").id != max(f.posts, key=lambda p: p.likes).id


def test_targets_are_organic_and_negations_differ():
    for s in range(500):
        f = FA.generate_feed(s)
        organic = {p.id for p in f.posts}
        for kind in TASK_KINDS:
            assert FA.truth(f, kind).id in organic
        assert not FA.truth(f, "latest_non_pinned").pinned
        for neg in FA.NEGATIONS:
            t = FA.truth(f, "negated_instruction", neg)
            assert not t.pinned
        assert FA.truth(f, "negated_instruction", "oldest").id != FA.truth(f, "latest_non_pinned").id


# --------------------------------------------------------------------------- views


def test_views_ordering_search_and_ads():
    for s in range(300):
        f = FA.generate_feed(s)
        latest = [p for p in view(f) if not p.promoted]
        k = sum(p.pinned for p in f.posts)
        assert all(p.pinned for p in latest[:k]) and not any(p.pinned for p in latest[k:])
        assert [p.ts for p in latest[k:]] == sorted((p.ts for p in latest[k:]), reverse=True)
        for tab in TABS:
            v = view(f, tab)
            assert len(v) == len(f.posts) + len(f.ads)
            for ad in f.ads:
                assert v.index(ad) == f.ad_slots[ad.id]  # unfiltered views are longer than any slot
        likes = [p.likes for p in view(f, "Most Liked") if not p.promoted]
        assert likes == sorted(likes, reverse=True)
        from_focus = [p for p in view(f, "Latest", f"from:@{f.focus_handle}") if not p.promoted]
        assert {p.handle for p in from_focus} == {f.focus_handle} and len(from_focus) == 4
        assert len([p for p in view(f, "Latest", f"“{f.snippet}”") if not p.promoted]) == 2
        assert [p for p in view(f, "Top", "zzzz") if not p.promoted] == []


def _smart_pick(arena: FeedArena) -> list[str]:
    """Two-step UI policies (sort tab and/or search, then open) for each task kind."""
    f, t = arena.feed, arena.task

    def first(tab, q=""):
        return next(p for p in view(f, tab, q) if not p.promoted)

    if t.kind == "latest_non_pinned":
        return [next(p for p in view(f) if not p.promoted and not p.pinned).id]
    if t.kind == "most_viral":
        return [first("Top").id]
    if t.kind == "search_author_top":
        return [first("Most Liked", f"from:@{f.focus_handle}").id]
    if t.kind == "open_specific_post":
        handle = f.by_id(t.target_id).handle
        return [first("Latest", f"from:@{handle} {f.snippet}").id]
    neg = t.meta["negation"]
    tab = {"least_liked": "Most Liked", "fewest_reposts": "Most Reposted", "oldest": "Latest"}[neg]
    return [[p for p in view(f, tab) if not p.promoted and not p.pinned][-1].id]


def test_smart_two_step_policies_score_one():
    for s in range(2000):
        for kind in TASK_KINDS:
            a = FeedArena.build(s, kind=kind)
            assert a.score(opened_state(_smart_pick(a))) == 1.0, (s, kind)


def test_scoring_open_close_and_shotgun():
    a = FeedArena.build(7, kind="most_viral")
    tgt = a.task.target_id
    other = next(c.id for c in a.cards() if c.id != tgt)
    assert a.score(opened_state([tgt])) == 1.0
    assert a.score(opened_state([tgt, tgt])) == 1.0  # re-opening the same post is not shotgun
    assert a.score(opened_state([other, tgt])) == 0.5
    assert a.score(opened_state([tgt, other])) == 0.0
    assert a.score(opened_state([tgt], final_open=None)) == 0.0  # opened then closed
    assert a.score({"state": {}, "events": []}) == 0.0
    ad = a.feed.ads[0].id if a.feed.ads else None
    if ad:
        assert a.score(opened_state([ad])) == 0.0


# --------------------------------------------------------------------------- styles / html


def test_style_splits_are_disjoint_and_varied():
    train = [FA.arena_style(s, "train") for s in range(400)]
    ood = [FA.arena_style(s, "ood_eval") for s in range(400)]
    assert {s["theme"] for s in train} == set(FA.SPLIT_THEMES["train"])
    assert {s["theme"] for s in ood} == set(FA.SPLIT_THEMES["ood_eval"])
    assert not {s["theme"] for s in train} & {s["theme"] for s in ood}
    assert not {s["font_px"] for s in train} & {s["font_px"] for s in ood}
    assert {s["sidebar"] for s in train} == {"none", "left"}
    assert {s["sidebar"] for s in ood} == {"right", "both"}
    combos = {tuple(sorted(s.items())) for s in train}
    assert len(combos) > 300
    with pytest.raises(ValueError):
        FA.arena_style(0, "test")


def test_html_contents_and_app_spec():
    for s in range(50):
        a = FeedArena.build(s, split="ood_eval" if s % 2 else "train")
        h = a.html()
        assert h.count('role="tab"') == 4 and 'aria-label="Clear search"' in h and 'role="dialog"' in h
        assert "</script>" in h and h.count("</script>") == 1  # data blob cannot close the script
        for p in a.feed.posts:
            assert p.id in h
        spec = a.app_spec()
        assert spec["state"] == {"open_post": None, "tab": "Latest", "query": ""}
        assert a.task.target_id not in json.dumps({k: v for k, v in spec.items() if k != "html"})


@pytest.mark.skipif(shutil.which("bun") is None and shutil.which("node") is None, reason="no JS runtime")
def test_page_js_matches_python_views(tmp_path):
    """Run the page's own sort/search code (stubbed DOM) and compare with ``view`` for many seeds."""
    cases, expect = [], []
    for s in range(150):
        a = FeedArena.build(s)
        f = a.feed
        for tab in TABS:
            for q in ("", f"from:@{f.focus_handle}", f"“{f.snippet}”", "kernel cache", "FROM:@ADA"):
                cases.append({"seed": s, "tab": tab, "q": q})
                expect.append([p.id for p in view(f, tab, q)])
    blobs = {}
    for s in range(150):
        h = FeedArena.build(s).html()
        start = h.index("<script>") + len("<script>")
        blobs[s] = h[start:h.index("</script>")]
    harness = ["const out = [];", "const document = {addEventListener(){}}; const window = {};",
               "function fetch(){ return {catch(){}}; }"]
    for s, js in blobs.items():
        mine = [c for c in cases if c["seed"] == s]
        harness.append("(function(){" + js + ";\n" + "".join(
            f"tab={json.dumps(c['tab'])};query={json.dumps(c['q'])};out.push(currentView().map(p=>p.id));"
            for c in mine) + "})();")
    harness.append("console.log(JSON.stringify(out));")
    script = tmp_path / "views.js"
    script.write_text("\n".join(harness))
    exe = shutil.which("bun") or shutil.which("node")
    got = json.loads(subprocess.run([exe, str(script)], capture_output=True, text=True, check=True).stdout)
    assert got == expect


# --------------------------------------------------------------------------- pool helpers (no network)


AX_SAMPLE = """[Chromium pid 0]
WebArea "Relay" @0,85 1024x683
TextField "Search" @300,120 380x30 {Press}
Button "Search" @690,120 70x30 {Press}
RadioButton "Most Liked" @520,160 120x28 {Press}
StaticText = "kernel \\"cache\\" latency" @300,220 400x18
Button "Open post p123" @700,260 60x24 {Press}
Button "Open post p456" @700,2600 60x24 {Press}
…"""


def test_parse_ax_lines_and_resolve():
    lines = P.parse_ax_lines(AX_SAMPLE)
    assert [n.role for n in lines] == ["WebArea", "TextField", "Button", "RadioButton", "StaticText", "Button", "Button"]
    assert lines[4].value == 'kernel "cache" latency'
    assert P.resolve_target(lines, "Search").role == "TextField"  # exact title, first on screen + pressable
    assert P.resolve_target(lines, "Chromium|Most Liked").title == "Most Liked"
    assert P.resolve_target(lines, "Open post p123").center == (730, 272)
    assert P.resolve_target(lines, "Open post").title == "Open post p123"  # prefix, on-screen first
    assert P.resolve_target(lines, "cache").role == "StaticText"
    assert P.resolve_target(lines, "nothing here") is None


def test_ax_lines_roundtrip_from_cdp_shape():
    nodes = [
        {"nodeId": "1", "role": {"value": "RootWebArea"}, "name": {"value": "Relay"}, "childIds": ["2", "3"],
         "bounds": [0, 0, 1024, 683]},
        {"nodeId": "2", "parentId": "1", "role": {"value": "button"}, "name": {"value": "Open post p001"},
         "bounds": [600, 100, 50, 20]},
        {"nodeId": "3", "parentId": "1", "role": {"value": "searchbox"}, "name": {"value": "Search"},
         "value": {"value": "from:@ada"}, "bounds": [10, 10, 300, 30]},
    ]
    from studio_contract import format_cdp_ax_lines

    text = format_cdp_ax_lines(nodes, origin=(0, 85))
    lines = P.parse_ax_lines(text)
    btn = P.resolve_target(lines, "Open post p001")
    assert btn.rect == (600, 185, 50, 20) and btn.pressable
    assert P.resolve_target(lines, "Search").value == "from:@ada"


def test_map_hotkey_and_guards():
    assert P.map_hotkey("return") == ("Return", [])
    assert P.map_hotkey("cmd+l") == ("l", ["ctrl"])
    assert P.map_hotkey("shift+cmd+t") == ("t", ["shift", "ctrl"])
    assert P.MAX_WORKERS * P.USD_PER_WORKER_HR == pytest.approx(2.24)
    assert "--test-type" in P.CHROME_FLAGS and "--remote-debugging-port=9222" in P.CHROME_FLAGS
    compile(P.CONTROL_SERVER_SRC, "ctl.py", "exec")
