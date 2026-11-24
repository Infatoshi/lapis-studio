"""FeedArena: a knob-driven social-feed micro-app with verifiable, shortcut-proof tasks.

One seed gives one feed (organic posts plus PROMOTED ads), one visual style, and one task. The
app is a single self-contained HTML page (search bar with ``from:@handle`` and keyword terms, a
Clear button, sort tabs ``Latest | Top | Most Liked | Most Reposted``, a post detail modal and an
optional sidebar). It reports every interaction to the pool control server (``POST /api/event``)
so rewards come from server-side state, never from what the policy says.

Every task is answered the same way: open the target post's detail view and leave it open.
``score_state`` gives 1/len(distinct posts opened) when the open view is the target, else 0, so
opening many cards ("shotgun clicking") never beats a uniform guess.

Traps built into every feed (see ``generate_feed``):
  * 0..4 pinned posts placed as the newest, newest-plus-random or scattered, so "Nth card" fails;
  * 0..3 PROMOTED cards at random slots, often with the largest like/repost counts on the page;
  * the most viral post usually has fewer likes than a likes-bait post and fewer reposts than a
    reposts-bait post, and ~35% of seeds tie the top two on likes+reposts at random repost values
    (tie-break: more likes);
  * a quoted snippet appears in two posts by different authors (only the author separates them).

Smart policies finish in two steps (sort tab or search, then open); scrolling also works.

Visual knobs come from ``style_seed`` and a split: ``train`` themes are light/dark/dim with
mid-range fonts and no right sidebar; ``ood_eval`` themes are high-contrast/sepia with extreme
fonts, padding and widths, a right sidebar, pill tabs and a different count format.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

TASK_KINDS = ("latest_non_pinned", "most_viral", "open_specific_post", "search_author_top", "negated_instruction")
NEGATIONS = ("least_liked", "fewest_reposts", "oldest")
TABS = ("Latest", "Top", "Most Liked", "Most Reposted")
SPLITS = ("train", "ood_eval")

HANDLES = [
    ("ada", "Ada Lovelace"), ("bjarne", "Bjarne S."), ("grace", "Grace Hopper"), ("linus", "Linus T."),
    ("margaret", "Margaret H."), ("ken", "Ken Thompson"), ("dennis", "Dennis R."), ("barbara", "Barbara Liskov"),
    ("edsger", "Edsger D."), ("alan", "Alan Turing"), ("katherine", "Katherine J."), ("john", "John Backus"),
    ("frances", "Frances Allen"), ("donald", "Donald Knuth"), ("radia", "Radia Perlman"), ("tim", "Tim B-L"),
    ("hedy", "Hedy Lamarr"), ("claudes", "Claude Shannon"), ("sophie", "Sophie Wilson"), ("guido", "Guido R."),
]
ADVERTISERS = [
    ("cloudforge", "CloudForge"), ("gpuhub", "GPU Hub"), ("swiftvpn", "SwiftVPN"),
    ("devcourse", "DevCourse"), ("byteboost", "ByteBoost"), ("stackmart", "StackMart"),
]
AD_TEXTS = [
    "Train models ten times faster. Start free today.",
    "Ship code with zero downtime. Book a demo now.",
    "Private browsing on every device. Get 80 percent off.",
    "Learn systems programming in six weeks. Enroll today.",
    "Boost your build cache hit rate. Try it free.",
    "Every developer tool you need in one bundle.",
]
WORDS = [
    "kernel", "latency", "cache", "tensor", "shader", "compiler", "vector", "socket", "thread", "matrix",
    "pipeline", "scheduler", "allocator", "parser", "runtime", "bytecode", "register", "mutex", "buffer", "packet",
    "gradient", "router", "checksum", "lexer", "heap", "stack", "branch", "commit", "release", "benchmark",
]
# The quoted-snippet words never occur in ordinary post text, so a snippet matches exactly two posts.
SNIPPET_ADJ = ["quiet", "amber", "hollow", "frozen", "silver", "crooked", "velvet", "rusty", "lucid", "brisk"]
SNIPPET_NOUN = ["lattice", "harbor", "lantern", "meadow", "compass", "orchard", "glacier", "beacon", "canyon", "ember"]

PINNED_LAYOUTS = ("newest", "mixed", "scattered")
PINNED_LAYOUT_WEIGHTS = (4, 4, 2)
TIE_RATE = 0.35
LIKES_BAIT_RATE = 0.85
REPOSTS_BAIT_RATE = 0.75
AD_LIKES_BAIT_RATE = 0.6
AD_REPOSTS_BAIT_RATE = 0.5
ENGAGEMENT_SCALES = (1, 2, 3, 5, 8, 13, 20, 40)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class FPost:
    id: str
    handle: str  # without "@"
    name: str
    ts: int
    likes: int
    reposts: int
    replies: int
    pinned: bool
    promoted: bool
    text: str

    @property
    def virality(self) -> int:
        return self.likes + self.reposts

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "handle": self.handle, "name": self.name, "ts": self.ts, "time": fmt_time(self.ts),
            "likes": self.likes, "reposts": self.reposts, "replies": self.replies, "pinned": self.pinned,
            "promoted": self.promoted, "text": self.text,
        }


@dataclass(frozen=True)
class Feed:
    seed: int
    posts: tuple[FPost, ...]  # organic posts, generation order (index 0 newest)
    ads: tuple[FPost, ...]
    ad_slots: Mapping[str, int]  # ad id -> display slot, applied in every view
    quote_target: str  # id of the post carrying the quoted snippet by ``quote_handle``
    quote_decoy: str  # same snippet, different author
    snippet: str
    focus_handle: str  # author with 4 posts (search_author_top)
    pinned_layout: str
    tie: bool

    def by_id(self, pid: str) -> FPost:
        for p in (*self.posts, *self.ads):
            if p.id == pid:
                return p
        raise KeyError(pid)


def fmt_time(ts: int) -> str:
    d = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
    return f"{d.strftime('%b')} {d.day}, {d.strftime('%H:%M')}"


def _engagement(rng: random.Random, n: int) -> tuple[list[int], list[int], int, bool]:
    """Distinct-ish likes/reposts with an engineered winner, baits and an optional virality tie."""
    scale = rng.choice(ENGAGEMENT_SCALES)
    lmax = rng.randint(max(4 * n, 60), 900)
    rmax = rng.randint(max(4 * n, 40), 400)
    likes = [x * scale + rng.randint(0, scale - 1) for x in rng.sample(range(lmax), n)]
    reposts = [x * scale + rng.randint(0, scale - 1) for x in rng.sample(range(rmax), n)]
    tie = rng.random() < TIE_RATE
    top = max(a + b for a, b in zip(likes, reposts))
    v = top + rng.randint(max(2, top // 25), max(3, top // 4))
    w, s, b = rng.sample(range(n), 3)
    if tie:
        ra = rng.randint(max(1, v // 10), v * 55 // 100)
        rb = rng.randint(ra + 1, min(v - 1, ra + max(1, v // 2)))
        likes[w], reposts[w] = v - ra, ra  # winner: same sum, more likes
        likes[s], reposts[s] = v - rb, rb
    else:
        rw = rng.randint(max(1, v // 4), v * 8 // 10)
        likes[w], reposts[w] = v - rw, rw
        if rng.random() < REPOSTS_BAIT_RATE and rw + 1 <= v - 1:
            r_e = rng.randint(rw + 1, v - 1)
            reposts[s], likes[s] = r_e, rng.randint(0, (v - 1 - r_e) // 2)
    lw = likes[w]
    if rng.random() < LIKES_BAIT_RATE and lw + 1 <= v - 1:
        l_d = rng.randint(lw + 1, v - 1)
        likes[b], reposts[b] = l_d, rng.randint(0, (v - 1 - l_d) // 2)
    return likes, reposts, w, tie


def generate_feed(seed: int, n_posts: int | None = None, n_pinned: int | None = None,
                  n_promoted: int | None = None) -> Feed:
    """Deterministic feed for ``seed``; see the module docstring for the traps."""
    rng = random.Random(f"feed-arena-{seed}")
    n = rng.randint(10, 16) if n_posts is None else n_posts
    k = rng.randint(0, 4) if n_pinned is None else n_pinned
    m = rng.choices((0, 1, 2, 3), weights=(2, 3, 3, 2))[0] if n_promoted is None else n_promoted
    if n < 8 or not 0 <= k <= n - 4 or not 0 <= m <= 4:
        raise ValueError("bad feed size knobs")
    layout = rng.choices(PINNED_LAYOUTS, weights=PINNED_LAYOUT_WEIGHTS)[0]

    for _ in range(200):  # rejection: likes and reposts must each be distinct among organic posts
        likes, reposts, _w, tie = _engagement(rng, n)
        if len(set(likes)) == n and len(set(reposts)) == n:
            break
    else:  # pragma: no cover - astronomically unlikely
        raise RuntimeError("engagement rejection failed")

    base = 1_780_000_000 + rng.randint(0, 10_000_000)
    ts = sorted(rng.sample(range(base, base + 21 * 86400), n), reverse=True)

    authors = rng.sample(HANDLES, rng.randint(4, 7))
    focus, others = authors[0], authors[1:]
    focus_idx = set(rng.sample(range(n), 4))
    who = [focus if i in focus_idx else rng.choice(others) for i in range(n)]
    # the focus author's most-liked post must not be the page's most-liked organic post
    top_focus = max(focus_idx, key=lambda i: likes[i])
    if top_focus == max(range(n), key=lambda i: likes[i]):
        swap = rng.choice([i for i in range(n) if i not in focus_idx])
        who[top_focus], who[swap] = rng.choice(others), focus

    texts: list[str] = []
    seen: set[str] = set()
    for _ in range(n):
        while True:
            t = " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 9)))
            if t not in seen:
                seen.add(t)
                texts.append(t)
                break
    snippet = f"{rng.choice(SNIPPET_ADJ)} {rng.choice(SNIPPET_NOUN)}"
    q_t = rng.randrange(n)
    q_d = rng.choice([i for i in range(n) if who[i][0] != who[q_t][0]])
    for i in (q_t, q_d):
        words = texts[i].split()
        at = rng.randint(0, len(words))
        texts[i] = " ".join(words[:at] + snippet.split() + words[at:])

    if k == 0:
        pinned_idx: list[int] = []
    elif layout == "newest":
        pinned_idx = list(range(k))
    elif layout == "mixed":
        pinned_idx = [0] + rng.sample(range(1, n), k - 1)
    else:
        pinned_idx = rng.sample(range(n), k)

    ids = [f"p{i:03d}" for i in rng.sample(range(1000), n + m)]  # ids encode nothing
    posts = tuple(
        FPost(id=ids[i], handle=who[i][0], name=who[i][1], ts=ts[i], likes=likes[i], reposts=reposts[i],
              replies=rng.randint(0, 40), pinned=i in pinned_idx, promoted=False, text=texts[i])
        for i in range(n)
    )
    maxl, maxr = max(likes), max(reposts)
    ads = []
    for j, (h, nm) in enumerate(rng.sample(ADVERTISERS, m)):
        al = rng.randint(maxl + 1, maxl * 2 + 5) if rng.random() < AD_LIKES_BAIT_RATE else rng.randint(0, maxl)
        ar = rng.randint(maxr + 1, maxr * 2 + 5) if rng.random() < AD_REPOSTS_BAIT_RATE else rng.randint(0, maxr)
        ads.append(FPost(id=ids[n + j], handle=h, name=nm, ts=rng.randint(base, base + 21 * 86400), likes=al,
                         reposts=ar, replies=rng.randint(0, 9), pinned=False, promoted=True,
                         text=rng.choice(AD_TEXTS)))
    slots = rng.sample(range(0, min(6, n)), m)
    return Feed(
        seed=seed, posts=posts, ads=tuple(ads), ad_slots={a.id: s for a, s in zip(ads, slots)},
        quote_target=posts[q_t].id, quote_decoy=posts[q_d].id, snippet=snippet, focus_handle=focus[0],
        pinned_layout=layout, tie=tie,
    )


# --------------------------------------------------------------------------- views (mirrored in the page JS)

_QUOTES_RE = re.compile("[\"“”'‘’]")
_FROM_RE = re.compile(r"^from:@?(.+)$", re.IGNORECASE)


def parse_query(q: str) -> tuple[str | None, list[str]]:
    frm, terms = None, []
    for tok in q.strip().split():
        tok = _QUOTES_RE.sub("", tok)
        if not tok:
            continue
        m = _FROM_RE.match(tok)
        if m:
            frm = m.group(1).lower()
        else:
            terms.append(tok.lower())
    return frm, terms


def matches(p: FPost, q: str) -> bool:
    frm, terms = parse_query(q)
    if frm is not None and p.handle.lower() != frm:
        return False
    hay = f"{p.text} @{p.handle} {p.name}".lower()
    return all(t in hay for t in terms)


SORT_KEYS = {
    "Latest": lambda p: (not p.pinned, -p.ts, p.id),
    "Top": lambda p: (-p.virality, -p.likes, -p.ts, p.id),
    "Most Liked": lambda p: (-p.likes, -p.ts, p.id),
    "Most Reposted": lambda p: (-p.reposts, -p.ts, p.id),
}


def view(feed: Feed, tab: str = "Latest", query: str = "") -> list[FPost]:
    """Cards in on-screen order for a tab and search query. Pinned posts lead only in Latest;
    ads keep their slots in every view (clamped to the list end) and ignore the search."""
    if tab not in SORT_KEYS:
        raise ValueError(f"tab must be one of {TABS}")
    out = sorted((p for p in feed.posts if matches(p, query)), key=SORT_KEYS[tab])
    for ad in sorted(feed.ads, key=lambda a: feed.ad_slots[a.id]):
        out.insert(min(feed.ad_slots[ad.id], len(out)), ad)
    return out


# --------------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class FeedTask:
    kind: str
    instruction: str
    target_id: str
    meta: Mapping[str, Any] = field(default_factory=dict)


_OPEN_TAIL = (
    " Open that post's detail view and leave it open. Promoted posts are ads and never count.",
    " Click it to open its detail view, then stop. Ignore PROMOTED ads.",
    " Open it (the detail view must stay open when you finish). Ads marked PROMOTED do not count.",
)


def _organic_non_pinned(feed: Feed) -> list[FPost]:
    return [p for p in feed.posts if not p.pinned]


def truth(feed: Feed, kind: str, negation: str = "least_liked") -> FPost:
    if kind == "latest_non_pinned":
        return max(_organic_non_pinned(feed), key=lambda p: p.ts)
    if kind == "most_viral":
        return min(feed.posts, key=SORT_KEYS["Top"])
    if kind == "open_specific_post":
        return feed.by_id(feed.quote_target)
    if kind == "search_author_top":
        return max((p for p in feed.posts if p.handle == feed.focus_handle), key=lambda p: p.likes)
    if kind == "negated_instruction":
        cands = _organic_non_pinned(feed)
        key = {"least_liked": lambda p: p.likes, "fewest_reposts": lambda p: p.reposts, "oldest": lambda p: p.ts}
        return min(cands, key=key[negation])
    raise ValueError(f"kind must be one of {TASK_KINDS}")


def make_task(feed: Feed, kind: str, rng: random.Random) -> FeedTask:
    tail = rng.choice(_OPEN_TAIL)
    meta: dict[str, Any] = {}
    if kind == "latest_non_pinned":
        ins = rng.choice((
            "Find the most recent post that is NOT pinned.",
            "Which post is the newest one, ignoring pinned posts?",
            "Locate the latest non-pinned post in this feed.",
        ))
    elif kind == "most_viral":
        ins = rng.choice((
            "Find the most viral post: the highest likes + reposts combined (tie: more likes wins).",
            "Which post has the largest total of likes plus reposts? Break ties by more likes.",
        ))
    elif kind == "open_specific_post":
        tgt = feed.by_id(feed.quote_target)
        ins = rng.choice((
            f"Open the post by @{tgt.handle} that mentions “{feed.snippet}”.",
            f"Find @{tgt.handle}'s post containing the words “{feed.snippet}”.",
        ))
        meta["decoy"] = feed.quote_decoy
    elif kind == "search_author_top":
        ins = rng.choice((
            f"Find the most-liked post written by @{feed.focus_handle}.",
            f"Among posts from @{feed.focus_handle}, which one has the most likes?",
        ))
    elif kind == "negated_instruction":
        neg = rng.choice(NEGATIONS)
        meta["negation"] = neg
        ins = {
            "least_liked": "Find the LEAST liked post among the posts that are NOT pinned.",
            "fewest_reposts": "Find the non-pinned post with the FEWEST reposts (not the most).",
            "oldest": "Find the OLDEST post that is NOT pinned (not the newest).",
        }[neg]
    else:
        raise ValueError(f"kind must be one of {TASK_KINDS}")
    target = truth(feed, kind, meta.get("negation", "least_liked"))
    return FeedTask(kind=kind, instruction=ins + tail, target_id=target.id, meta=meta)


def score_state(target_id: str, server_state: Mapping[str, Any]) -> float:
    """Reward from the control server's ``/__state``: 1/len(distinct opened) if the open detail
    view is the target, else 0."""
    st = server_state.get("state") or {}
    if st.get("open_post") != target_id:
        return 0.0
    opened = [e.get("id") for e in server_state.get("events", ()) if e.get("type") == "open"]
    distinct = set(opened) or {target_id}
    return 1.0 / len(distinct)


# --------------------------------------------------------------------------- style

THEMES = {
    # train
    "light": {"bg": "#f5f7fa", "card": "#ffffff", "fg": "#0f1419", "muted": "#536471", "border": "#d6dde3",
              "accent": "#1d7bd8", "pin": "#ffd400", "pin_fg": "#111", "ad": "#e6e9ed", "ad_fg": "#536471"},
    "dark": {"bg": "#000000", "card": "#16181c", "fg": "#e7e9ea", "muted": "#8b98a5", "border": "#2f3336",
             "accent": "#1d9bf0", "pin": "#8a6d00", "pin_fg": "#fff", "ad": "#2f3336", "ad_fg": "#c4cbd1"},
    "dim": {"bg": "#15202b", "card": "#192734", "fg": "#f7f9f9", "muted": "#8b98a5", "border": "#38444d",
            "accent": "#4aa8ff", "pin": "#b38f00", "pin_fg": "#fff", "ad": "#38444d", "ad_fg": "#d0d6db"},
    # ood_eval
    "high_contrast": {"bg": "#000000", "card": "#000000", "fg": "#ffffff", "muted": "#ffff00", "border": "#ffffff",
                      "accent": "#00ffff", "pin": "#ffff00", "pin_fg": "#000", "ad": "#ff00ff", "ad_fg": "#000"},
    "sepia": {"bg": "#f4ecd8", "card": "#fbf5e6", "fg": "#433422", "muted": "#7a6a53", "border": "#c8b88f",
              "accent": "#a0522d", "pin": "#d2a24c", "pin_fg": "#2b1d0e", "ad": "#e3d5b0", "ad_fg": "#5b4a32"},
}
SPLIT_THEMES = {"train": ("dark", "dim", "light"), "ood_eval": ("high_contrast", "sepia")}
APP_NAMES = {"train": ("Chirp", "Relay", "Murmur"), "ood_eval": ("Kite", "Buzzline")}


def arena_style(style_seed: int, split: str = "train") -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rng = random.Random(f"feed-arena-style-{split}-{style_seed}")
    train = split == "train"
    return {
        "split": split,
        "theme": rng.choice(SPLIT_THEMES[split]),
        "font_px": rng.randint(13, 17) if train else rng.choice((11, 12, 18, 19, 20)),
        "pad_px": rng.choice((6, 10, 14)) if train else rng.choice((3, 22)),
        "gap_px": rng.choice((4, 8, 12)) if train else rng.choice((0, 20)),
        "radius": rng.choice((0, 8, 14)),
        "width": rng.choice((560, 620, 680)) if train else rng.choice((480, 760)),
        "sidebar": rng.choice(("none", "left")) if train else rng.choice(("right", "both")),
        "tab_style": "underline" if train else "pill",
        "count_fmt": "words" if train else "labels",
        "app_name": rng.choice(APP_NAMES[split]),
    }


def _css(st: Mapping[str, Any]) -> str:
    t = THEMES[st["theme"]]
    tab_on = (f"border-bottom:3px solid {t['accent']};font-weight:700" if st["tab_style"] == "underline"
              else f"background:{t['accent']};color:{t['bg']};border-radius:999px;font-weight:700")
    return f"""
*{{box-sizing:border-box}}
body{{margin:0;background:{t['bg']};color:{t['fg']};font:{st['font_px']}px/1.35 -apple-system,'Segoe UI',Roboto,'DejaVu Sans',sans-serif}}
.shell{{display:flex;justify-content:center;gap:16px;padding:8px}}
.col{{width:{st['width']}px;max-width:100%}}
.side{{width:170px;flex:none;color:{t['muted']}}} .side a{{display:block;color:{t['fg']};padding:6px 4px;text-decoration:none}}
.side h3{{margin:6px 4px;font-size:1em;color:{t['fg']}}}
.top{{position:sticky;top:0;z-index:5;background:{t['bg']};padding:6px 0;border-bottom:1px solid {t['border']}}}
.brand{{font-weight:800;font-size:1.15em;margin:0 0 6px 2px}}
.search{{display:flex;gap:6px}} .search input{{flex:1;padding:6px 10px;font-size:1em;border-radius:999px;border:1px solid {t['border']};background:{t['card']};color:{t['fg']}}}
button{{font-size:.95em;cursor:pointer;border:1px solid {t['border']};background:{t['card']};color:{t['fg']};border-radius:6px;padding:4px 10px}}
.tabs{{display:flex;gap:4px;margin-top:6px}} .tabs button{{flex:1;border:none;background:transparent;padding:6px 4px;color:{t['muted']}}}
.tabs button[aria-selected=true]{{color:{t['fg']};{tab_on}}}
.card{{background:{t['card']};border:1px solid {t['border']};border-radius:{st['radius']}px;padding:{st['pad_px']}px;margin:{st['gap_px']}px 0;cursor:pointer}}
.hd{{color:{t['muted']}}} .hd b{{color:{t['fg']}}}
.badge{{font-size:.75em;font-weight:700;padding:1px 5px;border-radius:4px;margin-right:6px}}
.pin{{background:{t['pin']};color:{t['pin_fg']}}} .ad{{background:{t['ad']};color:{t['ad_fg']}}}
.txt{{margin:4px 0}} .ft{{display:flex;gap:14px;align-items:center;color:{t['muted']}}} .ft .open{{margin-left:auto}}
.pid{{font-family:monospace;font-size:.85em}}
.empty{{padding:20px;color:{t['muted']}}}
#modal{{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-start;justify-content:center;z-index:20;padding-top:60px}}
#modal.on{{display:flex}} .dlg{{background:{t['card']};border:1px solid {t['border']};border-radius:12px;padding:18px;width:min(560px,92vw)}}
.dlg .txt{{font-size:1.2em}}
"""


_JS = r"""
const D = __DATA__;
const EP = window.__EP || 0;
let tab = 'Latest', query = '', openId = null;
function send(ev) {
  ev.ep = EP;
  fetch('/api/event', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(ev), keepalive: true}).catch(() => {});
}
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function parseQ(q) {
  let from = null; const terms = [];
  for (let t of q.trim().split(/\s+/)) {
    t = t.replace(/["“”'‘’]/g, '');
    if (!t) continue;
    const m = t.match(/^from:@?(.+)$/i);
    if (m) from = m[1].toLowerCase(); else terms.push(t.toLowerCase());
  }
  return {from, terms};
}
function match(p, q) {
  const {from, terms} = parseQ(q);
  if (from !== null && p.handle.toLowerCase() !== from) return false;
  const hay = (p.text + ' @' + p.handle + ' ' + p.name).toLowerCase();
  return terms.every(t => hay.includes(t));
}
function cmpTuple(a, b) { for (let i = 0; i < a.length; i++) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; } return 0; }
const KEYS = {
  'Latest': p => [p.pinned ? 0 : 1, -p.ts, p.id],
  'Top': p => [-(p.likes + p.reposts), -p.likes, -p.ts, p.id],
  'Most Liked': p => [-p.likes, -p.ts, p.id],
  'Most Reposted': p => [-p.reposts, -p.ts, p.id],
};
function currentView() {
  const k = KEYS[tab];
  const out = D.posts.filter(p => match(p, query)).sort((a, b) => cmpTuple(k(a), k(b)));
  const ads = D.ads.slice().sort((a, b) => D.slots[a.id] - D.slots[b.id]);
  for (const ad of ads) out.splice(Math.min(D.slots[ad.id], out.length), 0, ad);
  return out;
}
function counts(p) {
  return D.style.count_fmt === 'words'
    ? `<span>${p.likes} likes</span><span>${p.reposts} reposts</span><span>${p.replies} replies</span>`
    : `<span>Likes: ${p.likes}</span><span>Reposts: ${p.reposts}</span><span>Replies: ${p.replies}</span>`;
}
function badge(p) {
  if (p.promoted) return '<span class="badge ad">PROMOTED</span>';
  if (p.pinned && tab === 'Latest') return '<span class="badge pin">PINNED</span>';
  return '';
}
function cardHTML(p) {
  return `<article class="card" data-id="${p.id}" aria-label="Post by @${esc(p.handle)}">
<div class="hd">${badge(p)}<b>${esc(p.name)}</b> @${esc(p.handle)} · <time>${esc(p.time)}</time></div>
<p class="txt">${esc(p.text)}</p>
<div class="ft">${counts(p)}<span class="pid">${p.id}</span>
<button class="open" aria-label="Open post ${p.id}">Open</button></div></article>`;
}
function render() {
  const v = currentView();
  const list = document.getElementById('feed');
  list.innerHTML = v.length ? v.map(cardHTML).join('') : '<div class="empty">No results</div>';
  for (const b of document.querySelectorAll('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === tab));
  document.getElementById('status').textContent = query ? `Results for "${query}"` : '';
}
function openPost(id) {
  const p = D.posts.concat(D.ads).find(x => x.id === id);
  if (!p) return;
  openId = id;
  const dlg = document.getElementById('dlg');
  dlg.setAttribute('aria-label', 'Post detail ' + id);
  dlg.innerHTML = `<h2>Post ${p.id}</h2><div class="hd">${p.promoted ? '<span class="badge ad">PROMOTED</span>' : (p.pinned ? '<span class="badge pin">PINNED</span>' : '')}<b>${esc(p.name)}</b> @${esc(p.handle)} · <time>${esc(p.time)}</time></div>
<p class="txt">${esc(p.text)}</p><div class="ft">${counts(p)}</div><p><button id="close">Close</button></p>`;
  document.getElementById('modal').classList.add('on');
  document.getElementById('close').onclick = closePost;
  send({type: 'open', id: id, set: {open_post: id}});
}
function closePost() {
  if (openId === null) return;
  const id = openId; openId = null;
  document.getElementById('modal').classList.remove('on');
  send({type: 'close', id: id, set: {open_post: null}});
}
function doSearch(q) { query = q; send({type: 'search', q: q, set: {query: q}}); render(); }
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('feed').addEventListener('click', e => {
    const card = e.target.closest('.card'); if (card) openPost(card.dataset.id);
  });
  for (const b of document.querySelectorAll('.tabs button')) b.onclick = () => {
    tab = b.dataset.tab; send({type: 'tab', tab: tab, set: {tab: tab}}); render(); window.scrollTo(0, 0);
  };
  const inp = document.getElementById('q');
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(inp.value); });
  document.getElementById('go').onclick = () => doSearch(inp.value);
  document.getElementById('clear').onclick = () => { inp.value = ''; doSearch(''); };
  document.getElementById('modal').addEventListener('click', e => { if (e.target.id === 'modal') closePost(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closePost(); });
  for (const a of document.querySelectorAll('.side a')) a.onclick = e => { e.preventDefault(); send({type: 'nav', label: a.textContent}); };
  render();
  send({type: 'ready'});
});
"""

_LEFT_NAV = ("Home", "Explore", "Notifications", "Messages", "Bookmarks", "Profile")


def render_html(feed: Feed, style: Mapping[str, Any], title: str | None = None) -> str:
    data = {
        "posts": [p.to_json() for p in feed.posts],
        "ads": [a.to_json() for a in feed.ads],
        "slots": dict(feed.ad_slots),
        "style": {"count_fmt": style["count_fmt"]},
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    name = _html.escape(style["app_name"])
    left = right = ""
    if style["sidebar"] in ("left", "both"):
        left = '<nav class="side" aria-label="Main">' + "".join(f'<a href="#">{x}</a>' for x in _LEFT_NAV) + "</nav>"
    if style["sidebar"] in ("right", "both"):
        rng = random.Random(f"trends-{feed.seed}")
        trends = rng.sample(WORDS, 5)
        right = ('<aside class="side" aria-label="Trending"><h3>Trending</h3>'
                 + "".join(f'<a href="#">#{w}</a>' for w in trends) + "</aside>")
    tabs = "".join(f'<button role="tab" data-tab="{t}" aria-selected="false">{t}</button>' for t in TABS)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{_html.escape(title or name)}</title>
<style>{_css(style)}</style><script>{_JS.replace('__DATA__', blob)}</script></head>
<body><div class="shell">{left}<main class="col">
<div class="top"><div class="brand">{name}</div>
<div class="search" role="search"><input id="q" type="search" placeholder="Search posts or from:@handle" aria-label="Search">
<button id="go">Search</button><button id="clear" aria-label="Clear search">Clear</button></div>
<div class="tabs" role="tablist">{tabs}</div></div>
<div id="status" class="hd" aria-live="polite"></div>
<section id="feed" aria-label="Feed"></section></main>{right}</div>
<div id="modal"><div class="dlg" id="dlg" role="dialog" aria-modal="true"></div></div>
</body></html>"""


# --------------------------------------------------------------------------- arena


@dataclass(frozen=True)
class FeedArena:
    seed: int
    split: str
    feed: Feed
    task: FeedTask
    style: Mapping[str, Any]

    @classmethod
    def build(cls, seed: int, kind: str | None = None, split: str = "train", style_seed: int | None = None,
              n_pinned: int | None = None, n_promoted: int | None = None) -> "FeedArena":
        feed = generate_feed(seed, n_pinned=n_pinned, n_promoted=n_promoted)
        rng = random.Random(f"feed-arena-task-{seed}")
        kind = kind or rng.choice(TASK_KINDS)
        task = make_task(feed, kind, rng)
        style = arena_style(seed if style_seed is None else style_seed, split)
        return cls(seed=seed, split=split, feed=feed, task=task, style=style)

    def cards(self, tab: str = "Latest", query: str = "") -> list[FPost]:
        return view(self.feed, tab, query)

    def html(self) -> str:
        return render_html(self.feed, self.style)

    def app_spec(self) -> dict[str, Any]:
        """What ``pool.reset_worker`` sends to ``POST /__reset``."""
        return {
            "name": f"feed_arena/{self.task.kind}/s{self.seed}",
            "html": self.html(),
            "state": {"open_post": None, "tab": "Latest", "query": ""},
        }

    def score(self, server_state: Mapping[str, Any]) -> float:
        return score_state(self.task.target_id, server_state)


def opened_state(opened: Sequence[str], final_open: str | None = "__last__") -> dict[str, Any]:
    """A ``/__state``-shaped dict for offline scoring of a sequence of opens."""
    last = opened[-1] if opened else None
    return {
        "state": {"open_post": last if final_open == "__last__" else final_open},
        "events": [{"type": "open", "id": pid} for pid in opened],
    }
