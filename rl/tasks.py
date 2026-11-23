"""Verifiable CUA tasks for DaytonaCUAEnv.

Two families:
  (a) login-free web tasks with deterministic checks (local static pages served
      inside the sandbox, plus one public page whose content is fixed).
  (b) mock social-feed search/sort tasks over a seeded HTML feed served by
      ``python3 -m http.server`` inside the sandbox.

Rewards read either the agent's ``done`` answer or the http.server access log
(each post's "Open" button fires ``GET /click?id=<post id>``; clicking a link
fires ``GET /<page>``). Feed generation, ground truth and log parsing are pure
functions so they are unit-testable offline.

Every reward function takes ``(env, answer)`` where ``env`` only needs
``.exec(cmd) -> ExecResult`` and ``.target``.
"""

from __future__ import annotations

import base64
import html
import random
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable

PORT = 8765
ROOT = "/tmp/lapis_rl_site"
LOG = f"{ROOT}/server.log"
BASE_URL = f"http://127.0.0.1:{PORT}"

POST_ID_RE = re.compile(r"\bp\d{3}\b")
CLICK_RE = re.compile(r'"GET /click\?id=(p\d{3})[ &"]')


@dataclass
class Task:
    id: str
    instruction: str
    setup: Callable[[Any], None] | None
    reward: Callable[[Any, str | None], float]
    max_steps: int = 15
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- feed (pure)


@dataclass(frozen=True)
class Post:
    id: str
    author: str
    timestamp: int  # unix seconds
    likes: int
    reposts: int
    pinned: bool
    text: str

    @property
    def virality(self) -> int:
        return self.likes + self.reposts


AUTHORS = ["ada", "bjarne", "grace", "linus", "margaret", "ken", "dennis", "barbara", "edsger", "alan"]
WORDS = ["kernel", "latency", "cache", "tensor", "shader", "compiler", "vector", "socket", "thread", "matrix"]
SORT_KEYS = ("likes", "reposts", "timestamp", "virality")


def generate_feed(seed: int, n: int = 16, n_pinned: int = 2, base_ts: int = 1_790_000_000) -> list[Post]:
    """Deterministic feed. Guarantees, for any seed:
    * the newest post overall is pinned (so "latest" must skip pinned posts);
    * the two most viral posts are non-pinned and tie on likes+reposts, so
      "most viral" must apply the tie-break (more likes wins).
    """
    if n < n_pinned + 3:
        raise ValueError("n too small")
    rng = random.Random(seed)
    ts = sorted(rng.sample(range(base_ts, base_ts + 30 * 86400), n), reverse=True)
    posts: list[dict[str, Any]] = []
    for i in range(n):
        posts.append(
            dict(
                id=f"p{i:03d}",
                author=rng.choice(AUTHORS),
                timestamp=ts[i],
                likes=rng.randint(0, 500),
                reposts=rng.randint(0, 200),
                pinned=False,
                text=" ".join(rng.choice(WORDS) for _ in range(6)),
            )
        )
    # newest post (index 0) is pinned; other pinned posts are older ones
    pinned_idx = [0] + rng.sample(range(n // 2, n), n_pinned - 1) if n_pinned else []
    for i in pinned_idx:
        posts[i]["pinned"] = True
    # force a virality tie between two non-pinned posts at the top of the non-pinned range
    free = [i for i in range(n) if i not in pinned_idx]
    a, b = rng.sample(free, 2)
    top = max(200, max(p["likes"] + p["reposts"] for p in posts)) + rng.randint(1, 50)
    posts[a]["likes"], posts[a]["reposts"] = top - 100, 100
    posts[b]["likes"], posts[b]["reposts"] = top - 60, 60
    ids = [f"p{i:03d}" for i in range(n)]
    rng.shuffle(ids)  # ids do not encode order
    for p, pid in zip(posts, ids):
        p["id"] = pid
    return [Post(**p) for p in posts]


def display_order(posts: list[Post]) -> list[Post]:
    """How the page shows the feed: pinned first (newest first), then the rest newest first."""
    pinned = sorted((p for p in posts if p.pinned), key=lambda p: -p.timestamp)
    rest = sorted((p for p in posts if not p.pinned), key=lambda p: -p.timestamp)
    return pinned + rest


def most_viral(posts: list[Post]) -> Post:
    """Max likes+reposts over all posts. Ties: more likes, then newer, then smaller id."""
    return min(posts, key=lambda p: (-p.virality, -p.likes, -p.timestamp, p.id))


def latest_non_pinned(posts: list[Post]) -> Post:
    """Newest non-pinned post. Ties: smaller id."""
    cands = [p for p in posts if not p.pinned]
    if not cands:
        raise ValueError("no non-pinned posts")
    return min(cands, key=lambda p: (-p.timestamp, p.id))


def sort_posts(posts: list[Post], key: str, descending: bool = True) -> list[Post]:
    """Sort by ``key``; ties broken by newer timestamp, then smaller id (stable, total)."""
    if key not in SORT_KEYS:
        raise ValueError(f"key must be one of {SORT_KEYS}")
    sign = -1 if descending else 1
    return sorted(posts, key=lambda p: (sign * getattr(p, key), -p.timestamp, p.id))


def nth_by(posts: list[Post], key: str, n: int, descending: bool = True) -> Post:
    """1-indexed Nth post after ``sort_posts``."""
    if n < 1 or n > len(posts):
        raise ValueError("n out of range")
    return sort_posts(posts, key, descending)[n - 1]


def extract_post_id(answer: str | None) -> str | None:
    if not answer:
        return None
    m = POST_ID_RE.findall(answer.lower())
    return m[-1] if m else None


def parse_clicks(log_text: str) -> list[str]:
    """Post ids clicked, in order, from python http.server access log text."""
    return CLICK_RE.findall(log_text)


def fmt_ts(ts: int) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def render_feed_html(posts: list[Post], title: str = "Feed") -> str:
    rows = []
    for p in display_order(posts):
        badge = '<span class="pin">PINNED</span> ' if p.pinned else ""
        rows.append(
            f'<article class="post" id="{p.id}" data-id="{p.id}">'
            f"<header>{badge}<b>@{html.escape(p.author)}</b> "
            f'<time datetime="{p.timestamp}">{fmt_ts(p.timestamp)}</time> '
            f'<code>{p.id}</code></header>'
            f"<p>{html.escape(p.text)}</p>"
            f'<footer>likes <span class="likes">{p.likes}</span> | reposts '
            f'<span class="reposts">{p.reposts}</span> '
            f'<button onclick="pick(\'{p.id}\')">Open {p.id}</button></footer></article>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px sans-serif;max-width:760px;margin:20px auto}}
.post{{border:1px solid #888;border-radius:6px;padding:8px 12px;margin:8px 0}}
.pin{{background:#fc0;padding:1px 4px}} button{{margin-left:12px}}</style>
<script>function pick(id){{fetch('/click?id='+id).catch(()=>{{}});
location.hash='clicked='+id;document.title='clicked '+id;}}</script></head>
<body><h1>{html.escape(title)}</h1>{"".join(rows)}</body></html>"""


# --------------------------------------------------------------------------- sandbox setup


def open_url_command(target: str, url: str) -> str:
    q = shlex.quote(url)
    if target == "macos":
        return f"open {q}"
    # Linux: which browser the default Daytona image ships is unverified; try in order.
    return (
        "export DISPLAY=${DISPLAY:-:0}; "
        f"(command -v xdg-open >/dev/null && nohup xdg-open {q} || "
        f"command -v firefox >/dev/null && nohup firefox {q} || "
        f"command -v chromium >/dev/null && nohup chromium --no-sandbox {q} || "
        f"command -v google-chrome >/dev/null && nohup google-chrome --no-sandbox {q}) "
        ">/dev/null 2>&1 &"
    )


def write_file_command(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0])} && echo {b64} | base64 -d > {shlex.quote(path)}"


def start_server_command() -> str:
    return (
        f"cd {ROOT} && : > {LOG} && "
        # bracket trick: the pattern must not match this shell's own command line
        f"(pkill -f '[h]ttp.server {PORT}' || true) && "
        f"nohup python3 -m http.server {PORT} --bind 127.0.0.1 >> {LOG} 2>&1 &"
    )


def _serve(files: dict[str, str], open_path: str) -> Callable[[Any], None]:
    def setup(env: Any) -> None:
        env.exec(f"rm -rf {ROOT} && mkdir -p {ROOT}")
        for name, content in files.items():
            r = env.exec(write_file_command(f"{ROOT}/{name}", content))
            if r.exit_code not in (0, None):
                raise RuntimeError(f"writing {name} failed: {r.output[-300:]}")
        env.exec(start_server_command())
        env.exec(f"for i in $(seq 1 20); do curl -sf {BASE_URL}/ >/dev/null && break; sleep 0.25; done")
        env.exec(open_url_command(env.target, f"{BASE_URL}/{open_path}"))

    return setup


def _read_log(env: Any) -> str:
    r = env.exec(f"cat {LOG} 2>/dev/null || true")
    return r.output or ""


def _answer_or_click_reward(truth_id: str) -> Callable[[Any, str | None], float]:
    def reward(env: Any, answer: str | None) -> float:
        got = extract_post_id(answer)
        if got is not None:
            return 1.0 if got == truth_id else 0.0
        clicks = parse_clicks(_read_log(env))
        return 1.0 if clicks and clicks[-1] == truth_id else 0.0

    return reward


def _exact_answer_reward(expected: str) -> Callable[[Any, str | None], float]:
    def reward(env: Any, answer: str | None) -> float:
        if answer is None:
            return 0.0
        return 1.0 if answer.strip().strip(".").lower() == expected.lower() else 0.0

    return reward


def _visited_reward(path: str) -> Callable[[Any, str | None], float]:
    pat = re.compile(rf'"GET /{re.escape(path)}[ ?#]')

    def reward(env: Any, answer: str | None) -> float:
        return 1.0 if pat.search(_read_log(env)) else 0.0

    return reward


# --------------------------------------------------------------------------- suite


def secret_code(seed: int) -> str:
    rng = random.Random(f"code-{seed}")
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))


def web_tasks(seed: int) -> list[Task]:
    code = secret_code(seed)
    info_html = (
        f"<!doctype html><html><head><title>Info {seed}</title></head><body>"
        f"<h1>Status page</h1><p>Build: 42</p><p>The access code is <b id='code'>{code}</b>.</p>"
        f"</body></html>"
    )
    index_html = (
        "<!doctype html><html><head><title>Start</title></head><body><h1>Start</h1>"
        "<ul><li><a href='/docs.html'>Docs</a></li><li><a href='/settings.html'>Settings</a></li>"
        "<li><a href='/info.html'>Info</a></li></ul></body></html>"
    )
    settings_html = "<!doctype html><html><head><title>Settings</title></head><body><h1>Settings</h1></body></html>"
    docs_html = "<!doctype html><html><head><title>Docs</title></head><body><h1>Docs</h1></body></html>"
    site = {"index.html": index_html, "info.html": info_html, "settings.html": settings_html, "docs.html": docs_html}
    return [
        Task(
            id=f"web_read_code_s{seed}",
            instruction="The browser shows a status page. Read the access code and finish with done(answer=<code>).",
            setup=_serve(site, "info.html"),
            reward=_exact_answer_reward(code),
            max_steps=8,
            meta={"expected": code},
        ),
        Task(
            id=f"web_navigate_settings_s{seed}",
            instruction="The browser shows a Start page. Click the 'Settings' link, then finish with done().",
            setup=_serve(site, "index.html"),
            reward=_visited_reward("settings.html"),
            max_steps=8,
        ),
        Task(
            # Needs sandbox egress to example.com; content is fixed by IANA.
            id="web_example_heading",
            instruction="Read the main heading of the open page and finish with done(answer=<heading>).",
            setup=lambda env: env.exec(open_url_command(env.target, "https://example.com")),
            reward=_exact_answer_reward("Example Domain"),
            max_steps=6,
            meta={"needs_internet": True},
        ),
    ]


def feed_tasks(seed: int) -> list[Task]:
    posts = generate_feed(seed)
    site = {"index.html": render_feed_html(posts, title=f"Feed {seed}")}
    setup = _serve(site, "index.html")
    viral = most_viral(posts)
    latest = latest_non_pinned(posts)
    rng = random.Random(f"sort-{seed}")
    key = rng.choice(SORT_KEYS)
    n = rng.randint(2, 5)
    nth = nth_by(posts, key, n)
    tail = "Report the post id with done(answer=<id>) or press that post's Open button and then done()."
    return [
        Task(
            id=f"feed_most_viral_s{seed}",
            instruction=f"In this feed, find the most viral post (highest likes + reposts; tie: more likes, then newer). {tail}",
            setup=setup,
            reward=_answer_or_click_reward(viral.id),
            max_steps=20,
            meta={"truth": viral.id},
        ),
        Task(
            id=f"feed_latest_non_pinned_s{seed}",
            instruction=f"Find the most recent post that is NOT pinned (pinned posts sit at the top). {tail}",
            setup=setup,
            reward=_answer_or_click_reward(latest.id),
            max_steps=15,
            meta={"truth": latest.id},
        ),
        Task(
            id=f"feed_sort_{key}_{n}_s{seed}",
            instruction=(
                f"Order all posts by {key} (likes+reposts for virality), highest first, ties by newer "
                f"timestamp. Open post number {n} in that order by pressing its Open button, then done()."
            ),
            setup=setup,
            reward=_answer_or_click_reward(nth.id),
            max_steps=25,
            meta={"truth": nth.id, "key": key, "n": n},
        ),
    ]


def build_suite(seeds: list[int] | tuple[int, ...] = (0,), include_internet: bool = True) -> list[Task]:
    out: list[Task] = []
    for s in seeds:
        out += [t for t in web_tasks(s) if include_internet or not t.meta.get("needs_internet")]
        out += feed_tasks(s)
    # web_example_heading has no seed; dedupe by id
    seen: set[str] = set()
    return [t for t in out if not (t.id in seen or seen.add(t.id))]
