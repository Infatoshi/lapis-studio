from __future__ import annotations

import io
import json

import pytest
from PIL import Image

import smoke_test_daytona as smoke
import tasks as T
from daytona_env import (
    DaytonaCUAEnv,
    DaytonaSDKClient,
    ExecResult,
    MacReservationRequired,
    MissingDaytonaKey,
    SandboxSpec,
    key_fingerprint,
    load_api_key,
    summarize,
)


def _png(w: int = 64, h: int = 48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


class FakeSandbox:
    def __init__(self, sid: str, log: list, exec_output: str = ""):
        self.id = sid
        self.log = log
        self.exec_output = exec_output
        self.deleted = False
        self.fail_screenshots = 0

    def start_computer_use(self):
        self.log.append(("start_cu",))

    def screenshot_png(self):
        if self.fail_screenshots:
            self.fail_screenshots -= 1
            raise RuntimeError("desktop not up")
        self.log.append(("screenshot",))
        return _png()

    def click(self, x, y, button="left", double=False):
        self.log.append(("click", x, y, button, double))

    def move(self, x, y):
        self.log.append(("move", x, y))

    def scroll(self, x, y, direction, amount):
        self.log.append(("scroll", x, y, direction, amount))

    def type_text(self, text):
        self.log.append(("type", text))

    def hotkey(self, keys):
        self.log.append(("hotkey", keys))

    def key(self, key):
        self.log.append(("key", key))

    def exec(self, command, timeout_s=60):
        self.log.append(("exec", command))
        return ExecResult(0, self.exec_output)

    def info(self):
        return {"id": self.id}

    def delete(self):
        self.deleted = True
        self.log.append(("delete",))


class FakeClient:
    def __init__(self, fail_create: bool = False):
        self.log: list = []
        self.created: list[FakeSandbox] = []
        self.specs: list[SandboxSpec] = []
        self.fail_create = fail_create
        self.closed = False

    def create_sandbox(self, spec):
        if self.fail_create:
            raise RuntimeError("quota")
        self.specs.append(spec)
        sb = FakeSandbox(f"sb{len(self.created)}", self.log)
        self.created.append(sb)
        return sb

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- key handling


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    with pytest.raises(MissingDaytonaKey):
        load_api_key()
    with pytest.raises(MissingDaytonaKey):
        DaytonaCUAEnv("linux")


def test_key_fingerprint_hides_value():
    fp = key_fingerprint("dtn_abcdefghijklmnop123456")
    assert fp == "len=26 last6=123456"
    assert "abcdef" not in fp
    assert key_fingerprint("") is None


def test_bad_target():
    with pytest.raises(ValueError):
        DaytonaCUAEnv("windows", client=FakeClient())


# --------------------------------------------------------------------------- lifecycle


def test_lazy_create_and_reset_obs():
    c = FakeClient()
    env = DaytonaCUAEnv("linux", client=c)
    assert c.created == []  # lazy
    obs = env.reset()
    assert len(c.created) == 1
    assert obs["screenshot"][:8] == b"\x89PNG\r\n\x1a\n"
    assert (obs["width"], obs["height"]) == (64, 48)
    assert obs["latency_s"] >= 0
    for k in ("create", "computer_use_start", "first_screenshot", "boot_total", "screenshot"):
        assert len(env.timings[k]) == 1, k
    env.reset()  # reuses sandbox
    assert len(c.created) == 1
    env.close()


def test_close_always_deletes_even_on_error():
    c = FakeClient()
    with pytest.raises(ZeroDivisionError):
        with DaytonaCUAEnv("linux", client=c) as env:
            env.reset()
            1 / 0
    assert c.created[0].deleted
    assert env.timings["teardown"]
    env.close()  # idempotent
    assert [e for e in c.log if e == ("delete",)] == [("delete",)]


def test_close_without_sandbox_is_noop():
    env = DaytonaCUAEnv("linux", client=FakeClient())
    env.close()
    with pytest.raises(RuntimeError):
        env.reset()


def test_wait_ready_retries(monkeypatch):
    monkeypatch.setattr("daytona_env.time.sleep", lambda s: None)
    c = FakeClient()
    orig = c.create_sandbox

    def create(spec):
        sb = orig(spec)
        sb.fail_screenshots = 2
        return sb

    c.create_sandbox = create
    env = DaytonaCUAEnv("linux", client=c)
    env.reset()
    assert c.log.count(("screenshot",)) >= 2
    env.close()


def test_spec_from_env(monkeypatch):
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.setenv("DAYTONA_MAC_RESERVATION", "res_1")
    c = FakeClient()
    env = DaytonaCUAEnv("macos", client=c, resources={"cpu": 4, "memory": 8, "disk": 100})
    env.reset()
    s = c.specs[0]
    assert (s.target, s.region, s.mac_reservation) == ("macos", "us", "res_1")
    env.close()


def test_sdk_client_refuses_mac_purchase_without_reservation():
    client = DaytonaSDKClient.__new__(DaytonaSDKClient)  # skip network/config
    client._v2 = None
    with pytest.raises(MacReservationRequired):
        client.create_sandbox(SandboxSpec(target="macos"))


# --------------------------------------------------------------------------- step dispatch


@pytest.mark.parametrize(
    "action,expected",
    [
        ({"type": "click", "x": 1, "y": 2}, ("click", 1, 2, "left", False)),
        ({"type": "click", "x": 1, "y": 2, "button": "right"}, ("click", 1, 2, "right", False)),
        ({"type": "double_click", "x": 3, "y": 4}, ("click", 3, 4, "left", True)),
        ({"type": "type", "text": "hi"}, ("type", "hi")),
        ({"type": "scroll", "x": 5, "y": 6, "direction": "up", "amount": 2}, ("scroll", 5, 6, "up", 2)),
        ({"type": "hotkey", "keys": ["ctrl", "c"]}, ("hotkey", "ctrl+c")),
        ({"type": "hotkey", "keys": "cmd+space"}, ("hotkey", "cmd+space")),
        ({"type": "key", "key": "enter"}, ("key", "enter")),
        ({"type": "move", "x": 7, "y": 8}, ("move", 7, 8)),
    ],
)
def test_step_dispatch(action, expected):
    c = FakeClient()
    env = DaytonaCUAEnv("linux", client=c)
    env.reset()
    obs, reward, done, info = env.step(action)
    assert expected in c.log
    assert not done and reward == 0.0 and "error" not in info
    assert obs["width"] == 64
    name = "double_click" if action["type"] == "double_click" else action["type"]
    assert len(env.timings[name]) == 1
    env.close()


def test_step_wait_done_and_errors(monkeypatch):
    monkeypatch.setattr("daytona_env.time.sleep", lambda s: None)
    env = DaytonaCUAEnv("linux", client=FakeClient())
    env.reset()
    _, _, done, info = env.step({"type": "wait", "seconds": 0.5})
    assert not done and env.timings["wait"]
    _, _, done, info = env.step({"type": "bogus"})
    assert "unknown action" in info["error"] and not done
    _, _, done, info = env.step({"type": "click", "x": 1})  # missing y
    assert "KeyError" in info["error"]
    _, _, done, info = env.step({"type": "scroll", "x": 1, "y": 1, "direction": "left"})
    assert "ValueError" in info["error"]
    _, reward, done, _ = env.step({"type": "done", "answer": "x"})
    assert done and reward == 0.0 and env.last_answer == "x"
    env.close()


def test_step_reward_and_truncation():
    calls = []
    task = T.Task(
        id="t", instruction="", setup=lambda e: calls.append("setup"),
        reward=lambda e, a: 1.0 if a == "42" else 0.0, max_steps=3,
    )
    env = DaytonaCUAEnv("linux", client=FakeClient())
    env.reset(task)
    assert calls == ["setup"] and env.timings["task_setup"]
    _, r, done, _ = env.step({"type": "done", "answer": "42"})
    assert done and r == 1.0
    env.reset(task)
    env.step({"type": "move", "x": 0, "y": 0})
    env.step({"type": "move", "x": 0, "y": 0})
    _, r, done, info = env.step({"type": "move", "x": 0, "y": 0})
    assert done and info.get("truncated") and r == 0.0
    env.close()


def test_summarize_percentiles():
    s = summarize([float(i) for i in range(1, 101)])
    assert (s["n"], s["p50"], s["p95"], s["min"], s["max"]) == (100, 50.0, 95.0, 1.0, 100.0)
    assert summarize([]) == {"n": 0}
    assert summarize([2.0])["p95"] == 2.0


# --------------------------------------------------------------------------- smoke CLI


def test_smoke_blocked_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    out = tmp_path / "r.json"
    rc = smoke.main(["--target", "both", "--json-out", str(out)])
    assert rc == 2
    assert "BLOCKED: DAYTONA_API_KEY not set" in capsys.readouterr().out
    assert json.loads(out.read_text())["status"] == "blocked"


def test_smoke_vlm_only_without_key_reports_connection_error(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    out = tmp_path / "r.json"
    rc = smoke.main(["--vlm-only", "--vlm-url", "http://127.0.0.1:9/v1/chat/completions", "--json-out", str(out)])
    assert rc == 1
    rec = json.loads(out.read_text())["vlm"]
    assert rec["ok"] is False and "connection error" in rec["error"]


def test_smoke_full_run_with_fake(monkeypatch, tmp_path):
    monkeypatch.setenv("DAYTONA_API_KEY", "fake_key_000000")
    clients: list[FakeClient] = []

    def factory(**kw):
        c = FakeClient()
        clients.append(c)
        kw.pop("allow_mac_purchase")
        return DaytonaCUAEnv(client=c, **kw)

    out = tmp_path / "r.json"
    rc = smoke.main(["--target", "both", "--iters", "3", "--no-vlm", "--json-out", str(out)], env_factory=factory)
    assert rc == 0
    r = json.loads(out.read_text())
    assert r["api_key"] == "len=15 last6=000000"
    assert set(r["targets"]) == {"linux", "macos"}
    for t in r["targets"].values():
        assert t["ok"] and t["screenshot"]["n"] == 3 and t["click"]["n"] == 3 and "teardown_s" in t
    assert all(c.created[0].deleted for c in clients)
    assert ("hotkey", "cmd+a") in clients[1].log


def test_smoke_tears_down_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("DAYTONA_API_KEY", "fake_key_000000")
    c = FakeClient()

    def factory(**kw):
        kw.pop("allow_mac_purchase")
        env = DaytonaCUAEnv(client=c, **kw)
        env.info = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        env.exec = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
        return env

    with pytest.raises(KeyboardInterrupt):
        smoke.main(["--target", "linux", "--iters", "1", "--no-vlm", "--json-out", str(tmp_path / "r.json")],
                   env_factory=factory)
    assert c.created[0].deleted


def test_models_url():
    assert smoke.models_url("http://h:1/v1/chat/completions") == "http://h:1/v1/models"


# --------------------------------------------------------------------------- tasks


def test_feed_deterministic_by_seed():
    assert T.generate_feed(7) == T.generate_feed(7)
    assert T.generate_feed(7) != T.generate_feed(8)
    assert T.render_feed_html(T.generate_feed(3)) == T.render_feed_html(T.generate_feed(3))


@pytest.mark.parametrize("seed", range(200))
def test_feed_invariants(seed):
    posts = T.generate_feed(seed)
    params = T.feed_params(seed)
    k, layout = params["n_pinned"], params["pinned_layout"]
    assert len({p.id for p in posts}) == len(posts)
    assert sum(p.pinned for p in posts) == k and 0 <= k <= 4
    by_time = sorted(posts, key=lambda p: -p.timestamp)
    if k and layout == "newest":
        assert all(p.pinned for p in by_time[:k]) and not by_time[k].pinned
    if k and layout == "mixed":
        assert by_time[0].pinned
    latest = T.latest_non_pinned(posts)
    assert not latest.pinned
    assert latest.timestamp == max(p.timestamp for p in posts if not p.pinned)
    d = T.display_order(posts)
    assert all(p.pinned for p in d[:k]) and d[k] is latest  # first un-badged card
    top = max(p.virality for p in posts)
    tied = [p for p in posts if p.virality == top]
    assert len(tied) == (2 if params["tie"] else 1)
    winner = T.most_viral(posts)
    assert winner.likes == max(p.likes for p in tied)
    if params["tie"]:
        assert tied[0].reposts != tied[1].reposts


def test_feed_explicit_params_and_bounds():
    for k in range(5):
        for layout in T.PINNED_LAYOUTS:
            posts = T.generate_feed(11, n_pinned=k, pinned_layout=layout)
            assert sum(p.pinned for p in posts) == k
            assert not T.latest_non_pinned(posts).pinned
    with pytest.raises(ValueError):
        T.generate_feed(0, n=6, n_pinned=4)
    with pytest.raises(ValueError):
        T.generate_feed(0, pinned_layout="bogus")


AUDIT_SEEDS = 10_000


def shortcut_audit(n_seeds: int = AUDIT_SEEDS) -> dict[str, float]:
    """Score of each surface heuristic over ``n_seeds`` feeds (fraction solved)."""
    import collections
    import random

    hits: collections.Counter = collections.Counter()
    winner_reposts: collections.Counter = collections.Counter()
    pinned_counts: collections.Counter = collections.Counter()
    layouts: collections.Counter = collections.Counter()
    rnd = random.Random(1234)
    ties = 0
    for s in range(n_seeds):
        posts = T.generate_feed(s)
        d = T.display_order(posts)
        rank = {p.id: i for i, p in enumerate(d)}  # heuristics break ties by on-screen position
        viral, latest = T.most_viral(posts), T.latest_non_pinned(posts)
        hits["pick_3rd_card@latest_non_pinned"] += d[2] is latest
        hits["pick_top_card@latest_non_pinned"] += d[0] is latest
        hits["pick_newest_timestamp@latest_non_pinned"] += max(posts, key=lambda p: p.timestamp) is latest
        hits["random@latest_non_pinned"] += rnd.choice(posts) is latest
        hits["pick_max_likes@most_viral"] += max(posts, key=lambda p: (p.likes, -rank[p.id])) is viral
        hits["pick_max_reposts@most_viral"] += max(posts, key=lambda p: (p.reposts, -rank[p.id])) is viral
        hits["pick_3rd_card@most_viral"] += d[2] is viral
        hits["random@most_viral"] += rnd.choice(posts) is viral
        top = max(p.virality for p in posts)
        ties += sum(p.virality == top for p in posts) == 2
        hits["winner_has_fewer_likes_than_some_post"] += any(p.likes > viral.likes for p in posts)
        winner_reposts[viral.reposts] += 1
        pinned_counts[sum(p.pinned for p in posts)] += 1
        layouts[T.feed_params(s)["pinned_layout"]] += 1
    out = {k: v / n_seeds for k, v in hits.items()}
    # best possible "always pick the post whose reposts == K" over every K (60, 100, ...)
    out["pick_any_fixed_repost@most_viral"] = max(winner_reposts.values()) / n_seeds
    out["pick_reposts==60@most_viral"] = winner_reposts[60] / n_seeds
    out["pick_reposts==100@most_viral"] = winner_reposts[100] / n_seeds
    out["tie_rate"] = ties / n_seeds
    out["min_share_per_n_pinned"] = min(pinned_counts[k] for k in range(5)) / n_seeds
    out["min_share_per_layout"] = min(layouts[l] for l in T.PINNED_LAYOUTS) / n_seeds
    return out


def test_shortcut_audit_10k_seeds():
    a = shortcut_audit()
    print("\n10,000-seed shortcut audit:")
    for k in sorted(a):
        print(f"  {k:45s} {a[k]:.4f}")
    n = 16
    assert a["pick_3rd_card@latest_non_pinned"] < 0.30
    assert a["pick_top_card@latest_non_pinned"] < 0.30
    assert a["pick_newest_timestamp@latest_non_pinned"] < 0.40
    assert a["pick_max_likes@most_viral"] < 0.45
    assert a["pick_max_reposts@most_viral"] < 0.45
    assert a["pick_any_fixed_repost@most_viral"] < 0.05
    assert a["pick_reposts==60@most_viral"] < 0.01 and a["pick_reposts==100@most_viral"] < 0.01
    for k in ("random@most_viral", "random@latest_non_pinned", "pick_3rd_card@most_viral"):
        assert abs(a[k] - 1 / n) < 0.015, k
    assert 0.30 <= a["tie_rate"] <= 0.40
    assert a["winner_has_fewer_likes_than_some_post"] > 0.8
    assert a["min_share_per_n_pinned"] > 0.17 and a["min_share_per_layout"] > 0.15


def test_feed_style_knobs():
    legacy = T.render_feed_html(T.generate_feed(1))
    assert "font:16px sans-serif" in legacy and "max-width:760px" in legacy
    styles = [T.feed_style(s) for s in range(200)]
    for knob in ("theme", "density", "font_px", "border", "radius", "width"):
        assert len({s[knob] for s in styles}) >= 2, knob
    assert {s["theme"] for s in styles} == {"light", "dark"}
    assert {s["density"] for s in styles} >= {"compact", "spacious"}
    assert T.feed_style(5) == T.feed_style(5)
    posts = T.generate_feed(2)
    html_a = T.render_feed_html(posts, style_seed=3)
    assert html_a == T.render_feed_html(posts, style_seed=3)
    assert len({T.render_feed_html(posts, style_seed=s) for s in range(10)}) > 1
    for p in posts:  # click hooks and badges survive every theme
        assert f"pick('{p.id}')" in html_a
    assert html_a.count('class="pin"') == sum(p.pinned for p in posts)


def test_most_viral_tie_breaking_handmade():
    P = T.Post
    posts = [
        P("p001", "a", 100, 50, 50, False, ""),
        P("p002", "a", 200, 60, 40, False, ""),  # same virality, more likes -> wins
        P("p003", "a", 300, 60, 40, False, ""),  # same likes too, newer -> wins
        P("p004", "a", 300, 60, 40, False, ""),  # same everything, larger id -> loses
    ]
    assert T.most_viral(posts).id == "p003"
    assert T.most_viral(posts[:2]).id == "p002"


def test_latest_non_pinned_handmade():
    P = T.Post
    posts = [P("p009", "a", 999, 0, 0, True, ""), P("p001", "a", 500, 0, 0, False, ""),
             P("p002", "a", 500, 0, 0, False, ""), P("p003", "a", 100, 0, 0, False, "")]
    assert T.latest_non_pinned(posts).id == "p001"
    with pytest.raises(ValueError):
        T.latest_non_pinned([posts[0]])


def test_sort_and_nth():
    P = T.Post
    posts = [P("p001", "a", 1, 5, 0, False, ""), P("p002", "a", 2, 9, 0, False, ""),
             P("p003", "a", 3, 5, 0, False, "")]
    assert [p.id for p in T.sort_posts(posts, "likes")] == ["p002", "p003", "p001"]
    assert T.nth_by(posts, "likes", 2).id == "p003"
    assert T.nth_by(posts, "likes", 1, descending=False).id == "p003"  # tie -> newer first
    with pytest.raises(ValueError):
        T.nth_by(posts, "likes", 4)
    with pytest.raises(ValueError):
        T.sort_posts(posts, "author")


def test_parse_clicks_and_extract():
    log = (
        '127.0.0.1 - - [23/Nov/2026 10:00:00] "GET / HTTP/1.1" 200 -\n'
        '127.0.0.1 - - [23/Nov/2026 10:00:01] code 404, message File not found\n'
        '127.0.0.1 - - [23/Nov/2026 10:00:01] "GET /click?id=p004 HTTP/1.1" 404 -\n'
        '127.0.0.1 - - [23/Nov/2026 10:00:02] "GET /click?id=p011 HTTP/1.1" 404 -\n'
    )
    assert T.parse_clicks(log) == ["p004", "p011"]
    assert T.extract_post_id("The answer is P012.") == "p012"
    assert T.extract_post_id("nothing") is None
    assert T.extract_post_id(None) is None


class FakeTaskEnv:
    def __init__(self, target="linux", log=""):
        self.target = target
        self.log_text = log
        self.cmds: list[str] = []

    def exec(self, cmd, timeout_s=60):
        self.cmds.append(cmd)
        return ExecResult(0, self.log_text if cmd.startswith("cat ") else "")


def test_feed_task_rewards():
    seed = 4
    posts = T.generate_feed(seed)
    tasks = {t.id.rsplit("_s", 1)[0]: t for t in T.feed_tasks(seed)}
    viral = tasks["feed_most_viral"]
    truth = T.most_viral(posts).id
    assert viral.meta["truth"] == truth
    env = FakeTaskEnv()
    assert viral.reward(env, truth) == 1.0
    assert viral.reward(env, f"post {truth}") == 1.0
    wrong = next(p.id for p in posts if p.id != truth)
    assert viral.reward(env, wrong) == 0.0
    # click path: last click counts
    env.log_text = f'"GET /click?id={wrong} HTTP/1.1" 404 -\n"GET /click?id={truth} HTTP/1.1" 404 -\n'
    assert viral.reward(env, None) == 1.0
    env.log_text = f'"GET /click?id={truth} HTTP/1.1" 404 -\n"GET /click?id={wrong} HTTP/1.1" 404 -\n'
    assert viral.reward(env, None) == 0.0
    # a wrong explicit answer is not rescued by a right click
    env.log_text = f'"GET /click?id={truth} HTTP/1.1" 404 -\n'
    assert viral.reward(env, wrong) == 0.0
    latest = tasks["feed_latest_non_pinned"]
    pinned_top = T.display_order(posts)[0]
    assert latest.reward(FakeTaskEnv(), pinned_top.id) == 0.0
    assert latest.reward(FakeTaskEnv(), T.latest_non_pinned(posts).id) == 1.0
    sort_t = next(t for k, t in tasks.items() if k.startswith("feed_sort"))
    assert sort_t.meta["truth"] == T.nth_by(posts, sort_t.meta["key"], sort_t.meta["n"]).id


def test_web_task_rewards_and_setup():
    tasks = {t.id: t for t in T.web_tasks(1)}
    code = T.secret_code(1)
    assert T.secret_code(1) == code and T.secret_code(2) != code
    rc = tasks["web_read_code_s1"]
    assert rc.reward(FakeTaskEnv(), f" {code}. ") == 1.0
    assert rc.reward(FakeTaskEnv(), "ZZZZZZ") == 0.0
    assert rc.reward(FakeTaskEnv(), None) == 0.0
    nav = tasks["web_navigate_settings_s1"]
    assert nav.reward(FakeTaskEnv(log='"GET /settings.html HTTP/1.1" 200 -'), None) == 1.0
    assert nav.reward(FakeTaskEnv(log='"GET /docs.html HTTP/1.1" 200 -'), None) == 0.0
    assert tasks["web_example_heading"].reward(FakeTaskEnv(), "example domain") == 1.0
    env = FakeTaskEnv(target="macos")
    rc.setup(env)
    assert any("base64 -d" in c and "info.html" in c for c in env.cmds)
    assert any("http.server 8765" in c for c in env.cmds)
    assert env.cmds[-1].startswith("open ")


def test_written_html_roundtrips():
    import base64
    import re

    html = T.render_feed_html(T.generate_feed(0))
    cmd = T.write_file_command("/tmp/x/index.html", html)
    b64 = re.search(r"echo (\S+) \|", cmd).group(1)
    assert base64.b64decode(b64).decode() == html
    for p in T.generate_feed(0):
        assert f"pick('{p.id}')" in html


def test_build_suite():
    suite = T.build_suite(seeds=(0, 1))
    ids = [t.id for t in suite]
    assert len(ids) == len(set(ids))
    assert sum(1 for i in ids if i == "web_example_heading") == 1
    assert all(0 < t.max_steps <= 25 for t in suite)
    assert not any(t.meta.get("needs_internet") for t in T.build_suite((0,), include_internet=False))


# --------------------------------------------------------------------------- studio contract

import studio_contract as SC  # noqa: E402


def test_studio_action_parser_mirrors_prefix_cpp():
    out = SC.parse_studio_output("Opening it [[ACTION:cua_open:Chrome|https://x.com:8080/a]] now.")
    assert out.actions == [SC.StudioAction("cua_open", "Chrome|https://x.com:8080/a")]
    assert out.clean_text == "Opening it now."
    # no arg, padded name, arg with ':' kept, ']' ends the arg
    out = SC.parse_studio_output("[[ACTION: go_home ]][[ACTION:cua_click:Chrome|Open p004]]\nDone")
    assert out.actions == [SC.StudioAction("go_home", ""), SC.StudioAction("cua_click", "Chrome|Open p004")]
    assert out.clean_text == "Done"
    # a space before a tag is dropped only when whitespace/punctuation follows the tag
    assert SC.parse_studio_output("a [[ACTION:x]] b").clean_text == "a b"
    assert SC.parse_studio_output("a [[ACTION:x]], b").clean_text == "a, b"
    assert SC.parse_studio_output("a [[ACTION:x]]b").clean_text == "a b"
    # empty name is not an action (regex needs 1+ chars; whitespace-only name dropped)
    out = SC.parse_studio_output("x [[ACTION: ]] y")
    assert out.actions == [] and out.clean_text == "x y"
    # tidy: trailing blanks before newline, 3+ newlines -> 2
    assert SC.parse_studio_output("a  \n\n\n\n[[ACTION:x]]\nb").clean_text == "a\n\nb"
    assert SC.parse_studio_output("hi [[IGNORE]] [[ACTION:x]]").ignore
    assert SC.parse_studio_output("hi [[IGNORE]] [[ACTION:x]]").actions == []
    assert SC.parse_studio_actions("[[ACTION:cua_scroll:Chrome|down]]") == [SC.StudioAction("cua_scroll", "Chrome|down")]
    assert SC.parse_studio_actions("[ACTION:x] [[ACTION:y:z]") == []


def test_studio_cpp_test_vectors():
    """Ported verbatim from tests/test_llm.cpp, tests/test_cua.cpp and src/tests.cpp."""
    P = SC.parse_studio_output
    assert P("[[IGNORE]]").ignore and P("[[IGNORE]]").clean_text == "" and P("[[IGNORE]]").actions == []
    assert P("  just chatting [[IGNORE]] \n").ignore
    c = P("[[ACTION:focus_project:lapis]] [[ACTION:play_track:Nights (slowed)]]\n"
          "Opening lapis and starting music. [[ACTION:spawn_agent:profile the KV cache: c1 vs c8]]")
    assert [(a.name, a.arg) for a in c.actions] == [
        ("focus_project", "lapis"), ("play_track", "Nights (slowed)"), ("spawn_agent", "profile the KV cache: c1 vs c8")]
    assert c.clean_text == "Opening lapis and starting music."
    na = P("Going home [[ACTION:go_home]] now.")
    assert na.actions == [SC.StudioAction("go_home", "")] and na.clean_text == "Going home now."
    assert P("  It is 126 tok/s.\n    indented  code\n").clean_text == "It is 126 tok/s.\n    indented  code"
    p = P("Sure. [[ACTION:cua_type:To: field|a|b]]")
    assert p.actions == [SC.StudioAction("cua_type", "To: field|a|b")] and p.clean_text == "Sure."

    A = SC.parse_cua_action
    click = A("cua_click", "  Send   button ")
    assert click.valid and click.kind == "click" and click.target == "Send button"
    assert not A("cua_click", "  ").valid
    t = A("cua_type", "To: field|a|b ")
    assert t.valid and (t.target, t.text) == ("To: field", "a|b ")
    assert not A("cua_type", "no separator").valid and not A("cua_type", "field|").valid
    sc = A("cua_scroll", "a|b list|-7")
    assert sc.valid and (sc.target, sc.delta_y) == ("a|b list", -7)
    assert A("cua_scroll", "page|down").delta_y == -5 and A("cua_scroll", "up").delta_y == 5
    assert A("cua_scroll", "up").target == ""
    assert not A("cua_scroll", "list|0").valid and not A("cua_scroll", "list|lots").valid
    assert A("cua_scroll", "list|9999").delta_y == 100
    k = A("cua_hotkey", "cmd+l")
    assert k.valid and k.target == "" and k.text == "cmd+l"
    kt = A("cua_hotkey", " Chrome | cmd + t ")
    assert kt.valid and (kt.target, kt.text) == ("Chrome", "cmd+t")
    for bad in ("", "cmd+banana", "Chrome|", "cmd+shift"):
        assert not A("cua_hotkey", bad).valid, bad
    d = A("cua_drag", " slider   handle | volume max ")
    assert d.valid and (d.target, d.text) == ("slider handle", "volume max")
    for bad in ("slider handle", "|volume max", "slider handle| "):
        assert not A("cua_drag", bad).valid, bad
    app = A("cua_open", " Chrome ")
    assert app.valid and (app.target, app.text) == ("Chrome", "")
    url = A("cua_open", "https://x.com/elonmusk")
    assert url.valid and (url.target, url.text) == ("", "https://x.com/elonmusk")
    both = A("cua_open", "Chrome|x.com/elonmusk")
    assert both.valid and (both.target, both.text) == ("Chrome", "https://x.com/elonmusk")
    bare = A("cua_open", "x.com/pmarca")
    assert bare.valid and (bare.target, bare.text) == ("", "https://x.com/pmarca")
    assert not A("cua_open", "").valid and not A("cua_open", "Chrome|not a url").valid
    assert A("cua_open", "Notion.app").target == "Notion.app"
    assert A("cua_scroll", "Chrome|pagedown").delta_y == -12 and A("cua_scroll", "Chrome|pageup").delta_y == 12
    assert A("cua_scroll", "Chrome|down").target == "Chrome"
    obs = A("cua_observe", " Discord ")
    assert obs.valid and obs.kind == "observe" and obs.target == "Discord"
    assert A("cua_observe", "").valid and A("cua_observe", "").target == ""

    U = SC.normalize_open_url
    assert U("x.com/elonmusk") == "https://x.com/elonmusk" and U("https://x.com/pmarca") == "https://x.com/pmarca"
    assert U(" github.com ") == "https://github.com" and U("localhost:8080/a") == "https://localhost:8080/a"
    assert U("file:///tmp/a.html") == "file:///tmp/a.html"
    assert U("Chrome") == "" and U("") == "" and U("elon musk") == ""

    F = SC.format_ax_nodes
    n = SC.AxNode("AXButton", 'Save\n"document"', "a\\b", "Example", 42, (-100, -50, 80, 24), ["AXPress"])
    text = F([n])
    assert text == '[Example pid 42]\nButton "Save\\n\\"document\\"" = "a\\\\b" @-100,-50 80x24 {Press}\n'
    assert text.count("\n") == 2 and F([n, n]).count("\n") == 3
    assert F([]) == "" and F([n], 0) == "" and F([n], -1) == ""
    for cap in (1, 2, 10, 100, 200, 6000):
        assert len(F([n, n], cap)) <= cap
    assert F([n], 1) == "…"
    assert F([n], len(text)) == text
    assert F([n, n], len(text) + 1) == text + "…"
    big = SC.AxNode("AXButton", n.title, "x" * 100000, "Example", 42, n.rect, ["AXPress"])
    assert len(F([big], 500)) <= 500
    para = SC.AxNode("AXStaticText", "p" * 2000, "", "Google Chrome", 600, (126.33333333333333, 40.06, 80.25, 19.999))
    line = F([para], 14000)
    assert "@126,40 80x20" in line and "126.3" not in line
    assert "p" * 279 + "…" + '"' in line and len(line) < 600
    page = [SC.AxNode("AXLink", f"Post {i} by someone","", "Google Chrome", 600, (40, 200.0 + i * 30, 600, 20), ["AXPress"])
            for i in range(120)]
    assert F(page, SC.POST_ACTION_CHARS).count("\n") >= 60


def test_note_stripping():
    raw = "Plan [[NOTE:latest = first card\nwithout PINNED]] then [[ACTION:cua_click:Chrome|Open p004]]."
    out = SC.parse_policy_output(raw)
    assert out.notes == ["latest = first card\nwithout PINNED"]
    assert out.actions == [SC.StudioAction("cua_click", "Chrome|Open p004")]
    assert out.clean_text == "Plan then."
    assert "NOTE" not in out.clean_text
    text, notes = SC.strip_notes("[[NOTE: a ]][[NOTE:b]]c")
    assert (text, notes) == ("c", ["a", "b"])
    assert SC.strip_notes("no notes") == ("no notes", [])
    # Studio itself does not strip notes: its parser leaves them visible
    assert "[[NOTE:x]]" in SC.parse_studio_output("[[NOTE:x]] hi").clean_text


def test_point_suffix_and_cua_args():
    assert SC.split_point("Chrome|Open p004@512,340") == ("Chrome|Open p004", (512, 340))
    assert SC.split_point("Chrome|Open p004 @ 0 , 1000") == ("Chrome|Open p004", (0, 1000))
    assert SC.split_point("Chrome|Open p004@1001,5") == ("Chrome|Open p004@1001,5", None)
    assert SC.split_point("Chrome|mail me@x.com") == ("Chrome|mail me@x.com", None)
    assert SC.split_point("Chrome|Open p004") == ("Chrome|Open p004", None)
    assert SC.point_to_pixels((512, 340), 1000, 500) == (512, 170)
    assert SC.point_to_pixels((1000, 1000), 1280, 800) == (1279, 799)
    c = SC.parse_cua_action("cua_click", "  Chrome|Open   p004@512,340 ", allow_point=True)
    assert (c.kind, c.target, c.point_1000, c.valid) == ("click", "Chrome|Open p004", (512, 340), True)
    c = SC.parse_cua_action("CUA_CLICK", "Chrome|Open p004@512,340")
    assert c.target == "Chrome|Open p004@512,340" and c.point_1000 is None  # Studio today
    c = SC.parse_cua_action("cua_type", "Discord search| John\\n")
    assert (c.target, c.text, c.valid) == ("Discord search", " John\\n", True)
    assert not SC.parse_cua_action("cua_type", "no bar").valid
    for arg, dy in [("Chrome|down", -5), ("up", 5), ("Chrome|page down", -12), ("x|-300", -100), ("x|7", 7)]:
        c = SC.parse_cua_action("cua_scroll", arg)
        assert c.delta_y == dy and c.valid, arg
    assert not SC.parse_cua_action("cua_scroll", "Chrome|sideways").valid
    c = SC.parse_cua_action("cua_open", "Chrome|example.com/a")
    assert (c.target, c.text, c.valid) == ("Chrome", "https://example.com/a", True)
    c = SC.parse_cua_action("cua_open", "localhost:8765/index.html")
    assert (c.target, c.text) == ("", "https://localhost:8765/index.html")
    c = SC.parse_cua_action("cua_open", "Chrome")
    assert (c.target, c.text, c.valid) == ("Chrome", "", True)
    assert not SC.parse_cua_action("cua_open", "Chrome|not a url").valid
    assert SC.normalize_open_url("http://127.0.0.1:8765/") == "http://127.0.0.1:8765/"
    assert SC.normalize_open_url("127.0.0.1:8765/x") == "https://127.0.0.1:8765/x"
    c = SC.parse_cua_action("cua_hotkey", "Chrome|cmd + r")
    assert (c.target, c.text, c.valid) == ("Chrome", "cmd+r", True)
    assert SC.parse_cua_action("cua_drag", "a|b").valid and not SC.parse_cua_action("cua_drag", "ab").valid
    assert SC.parse_cua_action("cua_observe", "").valid
    assert not SC.parse_cua_action("cua_bogus", "x").valid


def test_format_ax_nodes_mirrors_cua_mm():
    nodes = [
        SC.AxNode("AXButton", "Open p004", "", "Google Chrome", 42, (412.4, 339.6, 88, 28), ["AXPress", "AXShowMenu"]),
        SC.AxNode("AXStaticText", "", 'say "hi"\n\tx\\', "Google Chrome", 42, (10, 20, 100.5, 14)),
        SC.AxNode("AXTextField", "Search", "abc", "Finder", 7, (0, 0, 50, 20), ["AXConfirm", "AXScrollToVisible"]),
    ]
    assert SC.format_ax_nodes(nodes) == (
        "[Google Chrome pid 42]\n"
        'Button "Open p004" @412,340 88x28 {Press}\n'
        'StaticText = "say \\"hi\\"\\n\\tx\\\\" @10,20 101x14\n'
        "[Finder pid 7]\n"
        'TextField "Search" = "abc" @0,0 50x20 {Confirm}\n'
    )
    # 280-char text cap with an ellipsis
    long = SC.format_ax_nodes([SC.AxNode("AXStaticText", "", "x" * 500, "A", 1, (0, 0, 9, 9))])
    assert '"' + "x" * 279 + "…" + '"' in long
    # budget: whole lines only, then a lone ellipsis
    many = [SC.AxNode("AXButton", f"b{i}", "", "A", 1, (i, i, 20, 20), ["AXPress"]) for i in range(1000)]
    s = SC.format_ax_nodes(many, SC.POST_ACTION_CHARS)
    assert len(s) <= SC.POST_ACTION_CHARS + 1 and s.endswith("\n…")
    assert all(line.startswith(("[A pid", "Button")) for line in s.split("\n")[:-1])
    assert SC.format_ax_nodes(many, 0) == ""
    assert SC.ax_structural_noise(SC.AxNode("AXGroup"), 1) and not SC.ax_structural_noise(SC.AxNode("AXGroup"), 0)
    assert not SC.ax_structural_noise(SC.AxNode("AXGroup", "named"), 3)
    assert SC.ax_tiny_rect(SC.AxNode("AXButton", rect=(0, 0, 1, 40)), 1)


def test_observation_prompts():
    assert SC.observation_prompt("cua_click:Chrome|Open p004", "L1_AX", True, "  pressed\n ok ", "") == (
        "[CUA observation after cua_click:Chrome|Open p004 (L1_AX, ok=1, detail=pressed ok)]:\n"
        "(no accessibility nodes visible)"
    )
    f = SC.followup_prompt([{"action": "cua_open", "target": "Chrome", "tier": "", "ok": False, "detail": "x",
                             "observation": " Button @0,0 1x1\n"}])
    assert f.startswith("[CUA observation after cua_open:Chrome (ERR, ok=0, detail=x)]:\nButton @0,0 1x1\n")
    assert f.endswith("without cua_* tags.")


def _cdp_tree():
    def node(nid, role, name="", children=(), backend=None, ignored=False, value=None, parent=None):
        n = {"nodeId": nid, "ignored": ignored, "role": {"type": "role", "value": role},
             "name": {"type": "computedString", "value": name}, "childIds": list(children)}
        if backend is not None:
            n["backendDOMNodeId"] = backend
        if value is not None:
            n["value"] = {"type": "string", "value": value}
        if parent is not None:
            n["parentId"] = parent
        return n

    nodes = [
        node("1", "RootWebArea", "Feed 3", ["2"], backend=1),
        node("2", "generic", "", ["3", "6", "7", "9"], backend=2, parent="1"),
        node("3", "article", "", ["4", "5"], backend=3, parent="2"),
        node("4", "StaticText", "@ada kernel cache", ["4b"], backend=4, parent="3"),
        node("4b", "InlineTextBox", "@ada kernel cache", [], parent="4"),
        node("5", "button", "Open p004", [], backend=5, parent="3"),
        node("6", "none", "", [], ignored=True, parent="2"),
        node("7", "textbox", "Search", [], backend=7, value="viral", parent="2"),
        node("9", "link", "hidden", [], backend=9, parent="2"),
    ]
    bounds = {1: (0, 0, 800, 600), 2: (0, 0, 800, 600), 3: (10, 50, 700, 90), 4: (20, 60, 300, 18),
              5: (400, 100, 88, 28), 7: (10, 10, 200, 24), 9: (0, 0, 0, 0)}
    return nodes, bounds


def test_format_cdp_ax_lines():
    nodes, bounds = _cdp_tree()
    s = SC.format_cdp_ax_lines(nodes, bounds, pid=99, origin=(0, 87), scale=1.0)
    assert s == (
        "[Google Chrome pid 99]\n"
        'WebArea "Feed 3" @0,87 800x600\n'
        'StaticText = "@ada kernel cache" @20,147 300x18\n'
        'Button "Open p004" @400,187 88x28 {Press}\n'
        'TextField "Search" = "viral" @10,97 200x24\n'
    )
    # retina-style scale and a node-level bounds key
    nodes[5]["bounds"] = (1, 2, 3, 4)
    s2 = SC.format_cdp_ax_lines(nodes, bounds, pid=99, scale=2.0)
    assert 'Button "Open p004" @2,4 6x8 {Press}' in s2
    # order is document order even if the CDP list is shuffled
    shuffled = list(reversed(nodes))
    assert SC.format_cdp_ax_lines(shuffled, bounds, pid=99, scale=2.0) == s2
    # bounded to Studio's 6,000-char follow-up budget
    big = [{"nodeId": "r", "role": {"value": "RootWebArea"}, "name": {"value": "p"},
            "childIds": [f"b{i}" for i in range(2000)], "bounds": (0, 0, 800, 600)}]
    big += [{"nodeId": f"b{i}", "parentId": "r", "role": {"value": "button"}, "name": {"value": f"Open p{i:03d}"},
             "childIds": [], "bounds": (0, i, 80, 20)} for i in range(2000)]
    out = SC.format_cdp_ax_lines(big)
    assert len(out) <= SC.POST_ACTION_CHARS + 1 and out.endswith("…")
    assert out.count("\n") < 500


def test_studio_contract_roundtrip_on_feed():
    """A rendered feed's 'Open' button text is exactly what a policy targets in cua_click."""
    posts = T.generate_feed(5)
    truth = T.latest_non_pinned(posts).id
    reply = f"[[NOTE:skip PINNED cards]]Opening the latest post [[ACTION:cua_click:Chrome|Open {truth}@500,420]]"
    out = SC.parse_policy_output(reply)
    cmd = SC.parse_cua_action(out.actions[0].name, out.actions[0].arg, allow_point=True)
    assert cmd.target == f"Chrome|Open {truth}" and cmd.point_1000 == (500, 420)
    assert T.extract_post_id(cmd.target) == truth
    assert f">Open {truth}</button>" in T.render_feed_html(posts, style_seed=5)


# --------------------------------------------------------------------------- merge.py pwrite


def _write_safetensors(path, tensors):
    """Minimal safetensors writer: {name: (dtype, shape, bytes)} in insertion order."""
    import struct

    header, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _fake_ckpt(tmp_path):
    import os

    src = tmp_path / "src"
    src.mkdir()
    rng = __import__("random").Random(0)
    tensors = {
        "model.llm.layers.0.attn.wq_du.weight": ("BF16", (4, 8), rng.randbytes(64)),
        "model.llm.layers.0.mlp.experts.w1.weight": ("U8", (16, 8), rng.randbytes(128)),  # NVFP4 packed
        "model.llm.layers.0.mlp.experts.w1.weight_scale": ("F8_E4M3", (16, 2), rng.randbytes(32)),
        "model.llm.layers.0.mlp.gate.weight": ("BF16", (2, 8), rng.randbytes(32)),
        "model.visual.proj_in.weight": ("BF16", (3, 4), rng.randbytes(24)),
    }
    _write_safetensors(src / "model-00001-of-00001.safetensors", tensors)
    (src / "config.json").write_text('{"a": 1}')
    os.makedirs(src / "sub")
    (src / "sub" / "x.txt").write_text("x")
    return src, tensors


def test_pwrite_patches_only_target_tensors(tmp_path):
    import hashlib
    import os

    from inkling2_train_opt import merge as M

    src, tensors = _fake_ckpt(tmp_path)
    dst = tmp_path / "shm_ckpt"
    assert M.ensure_base_copy(str(src), str(dst)) is True
    st = dst / M.ST_NAME
    before = st.read_bytes()
    assert before == (src / M.ST_NAME).read_bytes() and (dst / "sub" / "x.txt").exists()
    header, base, raw_header = M.read_header(str(st))
    targets = ["model.llm.layers.0.attn.wq_du.weight", "model.visual.proj_in.weight"]
    new = {t: bytes((b ^ 0x5A) for b in tensors[t][2]) for t in targets}
    ino = os.stat(st).st_ino
    assert M.pwrite_tensors(str(st), new) == sum(len(v) for v in new.values())
    after = st.read_bytes()
    assert os.stat(st).st_ino == ino and len(after) == len(before)  # same file, patched in place
    assert after[:base] == before[:base] == raw_header  # header byte-identical
    assert M.read_tensor_bytes(str(st), targets) == new
    changed = bytearray(len(after))
    for t in targets:
        s, e = M.tensor_span(header, base, t)
        changed[s:e] = b"\x01" * (e - s)
    diff = [i for i in range(len(after)) if after[i] != before[i]]
    assert diff and all(changed[i] for i in diff)  # nothing outside the target spans moved
    for t in tensors:  # untouched tensors (NVFP4 experts, router) are byte-identical
        if t not in targets:
            assert M.read_tensor_bytes(str(st), [t])[t] == tensors[t][2]
    # the result still parses as safetensors with the same header and size accounting
    h2, b2, _ = M.read_header(str(st))
    assert h2 == header and b2 + max(v["data_offsets"][1] for k, v in h2.items() if k != "__metadata__") == len(after)
    # second sync: base already present and header matches -> no recopy, patch survives
    assert M.ensure_base_copy(str(src), str(dst)) is False
    assert hashlib.sha256(st.read_bytes()).digest() == hashlib.sha256(after).digest()
    # restore-from-src helper puts the base bytes back
    M.pwrite_tensors(str(st), M.read_tensor_bytes(str(src / M.ST_NAME), targets))
    assert st.read_bytes() == before
    # a different src header forces a fresh copy
    tensors2 = dict(tensors)
    tensors2["model.visual.proj_in.weight"] = ("BF16", (4, 3), tensors["model.visual.proj_in.weight"][2])
    _write_safetensors(src / M.ST_NAME, tensors2)
    assert M.ensure_base_copy(str(src), str(dst)) is True
    assert st.read_bytes() == (src / M.ST_NAME).read_bytes()


def test_pwrite_rejects_bad_patches(tmp_path):
    from inkling2_train_opt import merge as M

    src, tensors = _fake_ckpt(tmp_path)
    st = str(src / M.ST_NAME)
    before = (src / M.ST_NAME).read_bytes()
    with pytest.raises(ValueError):  # wrong length
        M.pwrite_tensors(st, {"model.visual.proj_in.weight": b"\x00" * 23})
    with pytest.raises(ValueError):  # NVFP4 experts are never patched
        M.pwrite_tensors(st, {"model.llm.layers.0.mlp.experts.w1.weight": b"\x00" * 128})
    with pytest.raises(KeyError):
        M.pwrite_tensors(st, {"nope": b""})
    # validation happens before any write
    with pytest.raises(ValueError):
        M.pwrite_tensors(st, {"model.visual.proj_in.weight": b"\x01" * 24, "model.llm.layers.0.mlp.gate.weight": b"x"})
    assert (src / M.ST_NAME).read_bytes() == before
    assert M.target_name("layers.3.attn.wk") == "model.llm.layers.3.attn.wk_dv.weight"
    assert M.target_name("layers.3.router.lin") == "model.llm.layers.3.mlp.gate.weight"
    modules = [f"layers.{i}.attn.{w}" for i in range(24) for w in ("wq", "wk", "wv", "wo")]
    modules += [f"layers.{i}.router.lin" for i in range(24)] + ["visual.proj_in", "visual.proj_out"]
    assert len({M.target_name(m) for m in modules}) == 122


def test_patched_sidecar_roundtrip(tmp_path):
    from inkling2_train_opt import merge as M

    assert M.load_patched(str(tmp_path)) == set()
    M.save_patched(str(tmp_path), {"b": b"", "a": b""})
    assert M.load_patched(str(tmp_path)) == {"a", "b"}
