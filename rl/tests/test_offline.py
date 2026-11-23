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


@pytest.mark.parametrize("seed", range(25))
def test_feed_invariants(seed):
    posts = T.generate_feed(seed)
    assert len({p.id for p in posts}) == len(posts)
    newest = max(posts, key=lambda p: p.timestamp)
    assert newest.pinned  # trap: newest overall is pinned
    latest = T.latest_non_pinned(posts)
    assert not latest.pinned and latest.id != newest.id
    assert latest.timestamp == max(p.timestamp for p in posts if not p.pinned)
    # the page shows pinned first, so naive "top post" is wrong
    assert T.display_order(posts)[0].pinned
    # forced tie for most viral, broken by more likes
    top = max(p.virality for p in posts)
    tied = [p for p in posts if p.virality == top]
    assert len(tied) >= 2
    assert T.most_viral(posts).likes == max(p.likes for p in tied)


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
