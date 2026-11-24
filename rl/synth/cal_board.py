"""CalBoard: a seeded calendar + Kanban workspace with SQLite-verified rewards.

Same wiring as ``code_review``: ``app_spec()`` -> ``POST /__reset`` on the pool control server,
an in-memory SQLite db seeded from ``seed_sql``, every mutating statement defined once in ``OPS``
and embedded in the page (``apply_op`` runs the same statements offline). Ground truth stays in
Python.

UI: ``Calendar | Board`` views. Calendar is a Mon-Fri week grid (8 AM-6 PM) with meeting blocks
(time, attendees, lock icon for external meetings); clicking one opens a modal with day buttons, a
start-time box, ``Save reschedule`` and ``Delete meeting``. Board has ``Backlog | In Progress |
Review | Done`` columns with code, title, priority, assignee and due date, a filter bar
(``assignee:@x``, ``priority:P0``, ``column:review``, free text or code) and a card modal with
``Move to`` buttons, priority buttons, a due-date box, ``Archive`` and ``Delete``.

Families (``score`` is pure over scenario, db dump and final answer):
  * resolve_conflict: on day D exactly one internal meeting overlaps the named locked external
    meeting; reschedule it to the requested free slot with its duration. Moving or deleting a
    locked meeting, or deleting any meeting, is -1.0.
  * move_and_update_card: find the card (by code among digit-permuted / other-prefix look-alikes,
    or by title + assignee among same-title cards), move it to a column and set priority and due
    date. Deleting any card is -1.0; touching other cards is 0.
  * filter_and_report: exactly one active card has assignee A and priority P (decoys share one of
    the two or come from a near-miss name); report its code and due date. Edits are 0, deletes -1.

Styles: ``split="train"`` uses Notion light/dark and Linear; ``split="ood_eval"`` uses a
Google-Calendar-like bright theme and a high-contrast Trello-like theme with different hour
height, card density and fonts.
"""

from __future__ import annotations

import datetime as dt
import html as _html
import json
import random
import re
from dataclasses import asdict, dataclass, field
from string import Template
from typing import Any, Callable, Mapping, Sequence

from synth.code_review import BASE_JS, LOG_SQL, apply_op, fresh_db, insert_sql, pool_exec, sqlite_exec  # noqa: F401

FAMILIES = ("resolve_conflict", "move_and_update_card", "filter_and_report")
SPLITS = ("train", "ood_eval")
COLUMNS = ("backlog", "in_progress", "review", "done")
COL_LABEL = {"backlog": "Backlog", "in_progress": "In Progress", "review": "Review", "done": "Done"}
PRIOS = ("P0", "P1", "P2", "P3")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
DAY0, DAY1 = 8 * 60, 18 * 60  # grid bounds in minutes

THEMES: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        dict(name="notion_light", bg="#ffffff", fg="#37352f", muted="#787774", panel="#f7f7f5", border="#e9e9e7",
             accent="#2383e2", ev="#e7f3f8", ev_fg="#1d4f66", lock="#fbecdd", lock_fg="#7a4a14", col="#f7f7f5",
             card="#ffffff", danger="#d44c47", font="ui-sans-serif, -apple-system, 'Segoe UI', sans-serif", fs="14px",
             hour="52px", radius="6px", pad="8px"),
        dict(name="notion_dark", bg="#191919", fg="#d4d4d4", muted="#9b9b9b", panel="#202020", border="#2f2f2f",
             accent="#529cca", ev="#1f3a4a", ev_fg="#cde6f5", lock="#4a3420", lock_fg="#ffd8b0", col="#202020",
             card="#2a2a2a", danger="#e0645f", font="ui-sans-serif, -apple-system, 'Segoe UI', sans-serif", fs="14px",
             hour="52px", radius="6px", pad="8px"),
        dict(name="linear", bg="#0f1011", fg="#f7f8f8", muted="#8a8f98", panel="#151618", border="#26282c",
             accent="#7170ff", ev="#25275a", ev_fg="#e0e1ff", lock="#4d2a1d", lock_fg="#ffcdb5", col="#141517",
             card="#1c1d20", danger="#eb5757", font="Inter, system-ui, sans-serif", fs="13px", hour="50px",
             radius="8px", pad="8px"),
    ),
    "ood_eval": (
        dict(name="gcal_bright", bg="#ffffff", fg="#3c4043", muted="#70757a", panel="#f1f3f4", border="#dadce0",
             accent="#1a73e8", ev="#039be5", ev_fg="#ffffff", lock="#d50000", lock_fg="#ffffff", col="#f1f3f4",
             card="#ffffff", danger="#d93025", font="'Google Sans', Roboto, Arial, sans-serif", fs="15px",
             hour="44px", radius="4px", pad="6px"),
        dict(name="trello_contrast", bg="#0c66e4", fg="#172b4d", muted="#44546f", panel="#ffffff", border="#091e42",
             accent="#0c66e4", ev="#fff7d6", ev_fg="#172b4d", lock="#ae2e24", lock_fg="#ffffff", col="#f1f2f4",
             card="#ffffff", danger="#ae2e24", font="'Atlassian Sans', 'Segoe UI', Arial, sans-serif", fs="13px",
             hour="46px", radius="12px", pad="5px"),
    ),
}


def style_for(split: str, style_seed: int) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pool = THEMES[split]
    return pool[style_seed % len(pool)]


# ----------------------------------------------------------------------------- vocab

PEOPLE_PAIRS = [("mara", "maura"), ("jon", "john"), ("kris", "chris"), ("eric", "erik"), ("lee", "li"),
                ("nadia", "nadja"), ("theo", "teo"), ("sara", "sarah"), ("ana", "anna"), ("max", "maxx")]
PEOPLE = ["ava", "ben", "carla", "dev", "ewa", "felix", "gia", "hugo", "ines", "jamal", "kenji", "omar", "paula",
          "quinn", "rosa", "sven", "tara", "uma", "wen", "zoe"]
INTERNAL = ["Design sync", "1:1 with Priya", "Standup", "Infra review", "Hiring debrief", "Roadmap planning",
            "Kernel deep-dive", "Eval review", "Budget review", "Sprint planning", "Incident retro", "Docs jam",
            "Release go/no-go", "Data pipeline sync", "Interview: backend", "Lunch & learn"]
EXTERNAL = ["Customer call: Acme Corp", "Partner sync: Globex", "Investor update: Initech", "Vendor call: Northwind",
            "Board prep with Umbrella", "Customer QBR: Hooli", "Legal review: Stark & Co"]
EXT_DOMAINS = ["acme.test", "globex.test", "initech.test", "northwind.test", "umbrella.test", "hooli.test"]
CARD_TITLES = ["Fix flaky upload test", "Migrate billing cron to k8s", "Add retry to webhook sender",
               "Profile dataloader stalls", "Rotate staging API keys", "Write eval for grounding",
               "Bump CUDA base image", "Dashboard for queue depth", "Dedupe Slack alerts", "Archive old S3 buckets",
               "Document release checklist", "Speed up CI cache", "Investigate OOM in trainer",
               "Clean up feature flags", "Add SSO to admin panel", "Backfill missing metrics", "Shard the eval set",
               "Pin tokenizer version", "Cut v2.3 release notes", "Load-test the gateway", "Triage nightly failures",
               "Upgrade Postgres to 17", "Audit IAM roles", "Add tracing to sampler", "Move secrets to vault",
               "Fix timezone bug in reports", "Prune stale branches", "Write on-call runbook",
               "Cache model weights on disk", "Benchmark FP8 kernels", "Rework onboarding docs", "Delete dead endpoints"]
PREFIXES = ("ENG", "OPS", "WEB", "ML")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def fmt12(m: int) -> str:
    h, mm = divmod(m, 60)
    return f"{(h - 1) % 12 + 1}:{mm:02d} {'AM' if h < 12 else 'PM'}"


def fmt_due(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.day}"


_TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?\s*m?\.?\s*$", re.I)


def parse_time(s: str) -> int | None:
    """Python twin of the page's ``parseTime``: '14:00', '2:00 PM', '2pm' -> minutes."""
    m = _TIME_RE.match(s or "")
    if not m:
        return None
    h, mm, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if mm >= 60 or h > 23 or (ap and not 1 <= h <= 12):
        return None
    if ap == "p" and h != 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return h * 60 + mm


# ----------------------------------------------------------------------------- data model


@dataclass
class Meeting:
    id: int
    title: str
    day: int
    start: int
    dur: int
    attendees: list[str]
    organizer: str
    locked: bool

    @property
    def end(self) -> int:
        return self.start + self.dur

    def overlaps(self, o: "Meeting") -> bool:
        return self.day == o.day and self.start < o.end and o.start < self.end


@dataclass
class Card:
    code: str
    title: str
    col: str
    prio: str
    assignee: str
    due: str  # ISO date
    pos: int


@dataclass
class Scenario:
    seed: int
    family: str
    split: str
    style_seed: int
    week_start: str  # ISO Monday
    start_view: str  # calendar | board
    meetings: list[Meeting]
    cards: list[Card]
    target: str  # meeting id (str) or card code
    want: dict[str, Any]  # resolve: day,start; move: col,prio,due; report: code,due
    ref: str  # move_and_update_card: code|title
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)

    def meeting(self, mid: int) -> Meeting:
        return next(m for m in self.meetings if m.id == mid)

    def card(self, code: str) -> Card:
        return next(c for c in self.cards if c.code == code)

    def board_order(self) -> list[Card]:
        """Cards in reading order: column by column, top to bottom."""
        return sorted(self.cards, key=lambda c: (COLUMNS.index(c.col), c.pos))

    def week_order(self) -> list[Meeting]:
        return sorted(self.meetings, key=lambda m: (m.day, m.start, m.id))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def app_spec(self) -> dict[str, Any]:
        return {"name": f"cal_board/{self.family}/s{self.seed}", "html": render_html(self), "sql": seed_sql(self),
                "state": {}}


# ----------------------------------------------------------------------------- generation (pure)


def _free(meetings: list[Meeting], day: int, start: int, dur: int, ignore: int = -1) -> bool:
    if start < DAY0 or start + dur > DAY1:
        return False
    return not any(m.day == day and m.id != ignore and m.start < start + dur and start < m.end for m in meetings)


def _place(rng: random.Random, meetings: list[Meeting], day: int, dur: int, tries: int = 60) -> int | None:
    for _ in range(tries):
        s = rng.randrange(DAY0, DAY1 - dur + 1, 30)
        if _free(meetings, day, s, dur):
            return s
    return None


def _code(rng: random.Random, used: set[str], prefix: str | None = None, num: str | None = None) -> str | None:
    for _ in range(200):
        c = f"{prefix or rng.choice(PREFIXES)}-{num or rng.randrange(100, 999)}"
        if c not in used:
            used.add(c)
            return c
    return None


def generate(seed: int, family: str | None = None, split: str = "train", style_seed: int | None = None,
             n_cards: int | None = None, meetings_per_day: int | None = None, ref: str | None = None,
             start_view: str | None = None) -> Scenario:
    """Knobs: family, split (train|ood_eval), style_seed, n_cards 16..32, meetings_per_day 3..6,
    ref (move_and_update_card: code|title), start_view (calendar|board)."""
    rng = random.Random(f"calboard:{seed}")
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    n_cards = n_cards if n_cards is not None else rng.randint(18, 30)
    if not 16 <= n_cards <= 32:
        raise ValueError("n_cards must be 16..32")
    mpd = meetings_per_day if meetings_per_day is not None else rng.randint(3, 5)
    if not 3 <= mpd <= 6:
        raise ValueError("meetings_per_day must be 3..6")
    style_seed = style_seed if style_seed is not None else rng.randrange(1000)
    start_view = start_view or rng.choice(("calendar", "board"))
    week_start = dt.date(2026, 12, 7) + dt.timedelta(weeks=rng.randrange(0, 20))
    me = "you@lapis.test"

    # ---------------- meetings
    meetings: list[Meeting] = []
    nid = iter(rng.sample(range(100, 999), 60))

    def mk(title: str, day: int, start: int, dur: int, locked: bool) -> Meeting:
        if locked:
            dom = rng.choice(EXT_DOMAINS)
            org = f"{rng.choice(PEOPLE)}@{dom}"
            att = [org, me] + [f"{p}@lapis.test" for p in rng.sample(PEOPLE, rng.randint(0, 2))]
        else:
            org = f"{rng.choice(PEOPLE)}@lapis.test"
            att = [org, me] + [f"{p}@lapis.test" for p in rng.sample(PEOPLE, rng.randint(1, 3))]
        m = Meeting(id=next(nid), title=title, day=day, start=start, dur=dur, attendees=sorted(set(att)),
                    organizer=org, locked=locked)
        meetings.append(m)
        return m

    target_mid = -1
    want: dict[str, Any] = {}
    conflict_day = rng.randrange(5)
    titles = rng.sample(INTERNAL, len(INTERNAL))
    ext_titles = rng.sample(EXTERNAL, len(EXTERNAL))
    if family == "resolve_conflict":
        ldur = rng.choice([60, 90])
        ls = rng.randrange(DAY0 + 60, DAY1 - ldur - 60 + 1, 30)
        L = mk(ext_titles.pop(), conflict_day, ls, ldur, True)
        tdur = rng.choice([30, 60, 60, 90])
        lo, hi = L.start - tdur + 30, L.end - 30
        T = mk(titles.pop(), conflict_day, rng.randrange(lo, hi + 1, 30), tdur, False)
        target_mid = T.id
        if rng.random() < 0.7:  # adjacent, non-overlapping decoy
            for s, d in ((L.end, 30), (L.start - 30, 30), (max(L.end, T.end), 60)):
                if _free(meetings, conflict_day, s, d):
                    mk(titles.pop(), conflict_day, s, d, False)
                    break
        # recurring copy of the target title on another day
        od = rng.choice([d for d in range(5) if d != conflict_day])
        s = _place(rng, meetings, od, tdur)
        if s is not None:
            mk(T.title, od, s, tdur, False)
    # fill every day to mpd meetings (non-overlapping on the conflict day, so the conflict is unique)
    for d in range(5):
        have = sum(m.day == d for m in meetings)
        n_locked = sum(m.day == d and m.locked for m in meetings)
        while have < mpd + (2 if d == conflict_day and family == "resolve_conflict" else 0):
            locked = n_locked == 0 and rng.random() < 0.35
            dur = rng.choice([30, 30, 60, 60, 90])
            s = _place(rng, meetings, d, dur)
            if s is None:
                break
            mk(ext_titles.pop() if locked and ext_titles else (titles.pop() if titles else rng.choice(INTERNAL)),
               d, s, dur, locked and bool(ext_titles))
            n_locked += locked
            have += 1
    if family == "resolve_conflict":
        T = next(m for m in meetings if m.id == target_mid)
        slots = [(d, s) for d in range(5) if d != conflict_day for s in range(DAY0, DAY1 - T.dur + 1, 30)
                 if _free(meetings, d, s, T.dur, ignore=T.id)]
        d2, s2 = rng.choice(slots)
        want = {"day": d2, "start": s2}

    # ---------------- cards
    pair = rng.choice(PEOPLE_PAIRS)
    if rng.random() < 0.5:
        pair = (pair[1], pair[0])
    A, A2 = pair
    people = [p for p in PEOPLE]
    rng.shuffle(people)
    used: set[str] = set()
    today = week_start - dt.timedelta(days=rng.randint(0, 4))
    dues = [(today + dt.timedelta(days=k)).isoformat() for k in rng.sample(range(-5, 50), n_cards)]
    ctitles = rng.sample(CARD_TITLES, len(CARD_TITLES))
    plan: list[dict[str, Any]] = []
    P = rng.choice(PRIOS)
    ref = ref or rng.choice(("code", "title"))
    if ref not in ("code", "title"):
        raise ValueError("ref must be code|title")
    tcode = ""
    if family == "move_and_update_card":
        while True:
            digits = "".join(rng.sample("123456789", 3))
            pre = rng.choice(PREFIXES)
            tcode = f"{pre}-{digits}"
            if tcode not in used:
                break
        used.add(tcode)
        ttitle = ctitles.pop()
        plan.append(dict(code=tcode, title=ttitle, assignee=A))
        # code look-alikes: permuted digits, same digits with another prefix
        perms = {"".join(p) for p in __import__("itertools").permutations(digits)} - {digits}
        for dg in rng.sample(sorted(perms), 2):
            c = _code(rng, used, pre, dg)
            if c:
                plan.append(dict(code=c, title=ctitles.pop(), assignee=rng.choice(people)))
        c = _code(rng, used, rng.choice([p for p in PREFIXES if p != pre]), digits)
        if c:
            plan.append(dict(code=c, title=ctitles.pop(), assignee=rng.choice(people)))
        # title look-alikes: same title, near-miss and other assignee
        plan.append(dict(code=_code(rng, used), title=ttitle, assignee=A2))
        plan.append(dict(code=_code(rng, used), title=ttitle, assignee=rng.choice(people)))
        # same assignee, other titles
        for _ in range(2):
            plan.append(dict(code=_code(rng, used), title=ctitles.pop(), assignee=A))
    elif family == "filter_and_report":
        plan.append(dict(code=_code(rng, used), title=ctitles.pop(), assignee=A, prio=P))
        others_p = [x for x in PRIOS if x != P]
        for _ in range(rng.randint(3, 5)):
            plan.append(dict(code=_code(rng, used), title=ctitles.pop(), assignee=A, prio=rng.choice(others_p)))
        plan.append(dict(code=_code(rng, used), title=ctitles.pop(), assignee=A2, prio=P))
        for _ in range(rng.randint(3, 4)):
            plan.append(dict(code=_code(rng, used), title=ctitles.pop(), assignee=rng.choice(people), prio=P))
    while len(plan) < n_cards:
        who = rng.choice(people + [A2])
        pr = rng.choice(PRIOS)
        if family == "filter_and_report" and who in (A,) and pr == P:
            continue
        plan.append(dict(code=_code(rng, used), title=ctitles.pop() if ctitles else rng.choice(CARD_TITLES),
                         assignee=who, prio=pr))
    if len(plan) > n_cards:
        raise ValueError("n_cards too small")
    cards: list[Card] = []
    order = list(range(n_cards))
    rng.shuffle(order)
    for i, pl in enumerate(plan):
        col = rng.choices(COLUMNS, weights=[3, 3, 2, 2])[0]
        cards.append(Card(code=pl["code"], title=pl["title"], col=col, prio=pl.get("prio") or rng.choice(PRIOS),
                          assignee=pl["assignee"], due=dues[i], pos=order[i]))
    # positions are per-column ranks
    for col in COLUMNS:
        for r, c in enumerate(sorted((c for c in cards if c.col == col), key=lambda c: c.pos)):
            c.pos = r
    if family == "filter_and_report":
        assert sum(c.assignee == A and c.prio == P for c in cards) == 1

    if family == "resolve_conflict":
        L = next(m for m in meetings if m.locked and m.day == conflict_day and m.overlaps(next(x for x in meetings if x.id == target_mid)))
        target = str(target_mid)
        instruction = (f"In CalBoard, your {DAYS[conflict_day]} has a conflict: an internal meeting overlaps the locked "
                       f'external meeting "{L.title}". Reschedule that internal meeting to {DAYS[want["day"]]} at '
                       f"{fmt12(want['start'])}, keeping its duration. Do not move or delete any locked meeting.")
    elif family == "move_and_update_card":
        tc = cards[0]
        new_col = rng.choice([c for c in COLUMNS if c != tc.col])
        new_prio = rng.choice([p for p in PRIOS if p != tc.prio])
        while True:
            nd = (today + dt.timedelta(days=rng.randint(1, 45))).isoformat()
            if nd != tc.due:
                break
        want = {"col": new_col, "prio": new_prio, "due": nd}
        target = tc.code
        who = f'the card "{tc.title}" assigned to @{tc.assignee}' if ref == "title" else f"card {tc.code}"
        instruction = (f"In CalBoard's board, find {who}. Move it to {COL_LABEL[new_col]}, set its priority to "
                       f"{new_prio} and its due date to {nd}. Do not change or delete any other card.")
    else:
        tc = cards[0]
        target = tc.code
        want = {"code": tc.code, "due": tc.due}
        instruction = (f"In CalBoard's board, filter to cards assigned to @{A} with priority {P}. Exactly one card "
                       f"matches: report its card code and due date as your final answer. Do not edit any card.")
    return Scenario(seed=seed, family=family, split=split, style_seed=style_seed, week_start=week_start.isoformat(),
                    start_view=start_view, meetings=meetings, cards=cards, target=target, want=want, ref=ref,
                    instruction=instruction,
                    meta={"n_cards": n_cards, "meetings_per_day": mpd, "split": split, "ood_style": split == "ood_eval",
                          "theme": style_for(split, style_seed)["name"], "conflict_day": conflict_day,
                          "assignee": A, "near_assignee": A2, "priority": P})


# ----------------------------------------------------------------------------- db

SCHEMA = """
CREATE TABLE meetings(id INTEGER PRIMARY KEY, title TEXT, day INTEGER, start INTEGER, dur INTEGER, attendees TEXT,
  organizer TEXT, locked INTEGER, deleted INTEGER DEFAULT 0);
CREATE TABLE cards(code TEXT PRIMARY KEY, title TEXT, col TEXT, prio TEXT, assignee TEXT, due TEXT, pos INTEGER,
  archived INTEGER DEFAULT 0, deleted INTEGER DEFAULT 0);
CREATE TABLE actions(id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, ref TEXT, arg TEXT, ts REAL);
"""

OPS: dict[str, list[str]] = {
    "log": [LOG_SQL],
    "reschedule": ["UPDATE meetings SET day=:day, start=:start WHERE id=:id", LOG_SQL],
    "delete_meeting": ["UPDATE meetings SET deleted=1 WHERE id=:id", LOG_SQL],
    "move_card": ["UPDATE cards SET col=:col, pos=(SELECT COALESCE(MAX(pos),-1)+1 FROM cards WHERE col=:col) WHERE code=:code",
                  LOG_SQL],
    "set_prio": ["UPDATE cards SET prio=:prio WHERE code=:code", LOG_SQL],
    "set_due": ["UPDATE cards SET due=:due WHERE code=:code", LOG_SQL],
    "archive_card": ["UPDATE cards SET archived=1 WHERE code=:code", LOG_SQL],
    "delete_card": ["UPDATE cards SET deleted=1 WHERE code=:code", LOG_SQL],
}

Q: dict[str, str] = {
    "meetings": "SELECT id,title,day,start,dur,locked FROM meetings WHERE deleted=0 ORDER BY day,start,id",
    "meeting": "SELECT id,title,day,start,dur,attendees,organizer,locked FROM meetings WHERE id=:id AND deleted=0",
    "cards": ("SELECT code,title,col,prio,assignee,due FROM cards WHERE archived=0 AND deleted=0 "
              "AND (:assignee='' OR lower(assignee)=lower(:assignee)) AND (:prio='' OR prio=upper(:prio)) "
              "AND (:col='' OR col=:col) AND (:text='' OR instr(lower(title), lower(:text))>0 "
              "OR instr(lower(code), lower(:text))>0) ORDER BY pos, code"),
    "card": "SELECT code,title,col,prio,assignee,due FROM cards WHERE code=:code AND deleted=0",
}

DUMP_SQL = {
    "meetings": "SELECT id,day,start,dur,deleted FROM meetings ORDER BY id",
    "cards": "SELECT code,col,prio,due,archived,deleted FROM cards ORDER BY code",
    "actions": "SELECT action,ref,arg FROM actions ORDER BY id",
}


def seed_sql(sc: Scenario) -> str:
    out = [SCHEMA]
    for m in sc.meetings:
        out.append(insert_sql("meetings", dict(id=m.id, title=m.title, day=m.day, start=m.start, dur=m.dur,
                                               attendees=json.dumps(m.attendees), organizer=m.organizer,
                                               locked=int(m.locked))))
    for c in sc.cards:
        out.append(insert_sql("cards", dict(code=c.code, title=c.title, col=c.col, prio=c.prio, assignee=c.assignee,
                                            due=c.due, pos=c.pos)))
    return "\n".join(out)


def dump(execute: Callable[..., list]) -> dict[str, Any]:
    me = execute(DUMP_SQL["meetings"], [])
    ca = execute(DUMP_SQL["cards"], [])
    ac = execute(DUMP_SQL["actions"], [])
    return {"meetings": [dict(zip(("id", "day", "start", "dur", "deleted"), r)) for r in me],
            "cards": [dict(zip(("code", "col", "prio", "due", "archived", "deleted"), r)) for r in ca],
            "actions": [{"action": a, "ref": str(b), "arg": c} for a, b, c in ac]}


def parse_query(q: str) -> dict[str, str]:
    """Python twin of the page's ``parseQ``."""
    out = {"assignee": "", "prio": "", "col": "", "text": ""}
    text = []
    for tok in (q or "").replace('"', " ").split():
        low = tok.lower()
        k, sep, v = low.partition(":")
        if sep and k in ("assignee", "owner"):
            out["assignee"] = v.lstrip("@")
        elif sep and k in ("priority", "prio", "p"):
            out["prio"] = v.upper() if v.upper() in PRIOS else ("P" + v if v in "0123" and v else "none")
        elif sep and k in ("column", "col", "status"):
            s = re.sub(r"[^a-z]", "", v)
            out["col"] = {"backlog": "backlog", "inprogress": "in_progress", "progress": "in_progress",
                          "doing": "in_progress", "review": "review", "done": "done"}.get(s, "none")
        else:
            text.append(tok)
    out["text"] = " ".join(text)
    return out


# ----------------------------------------------------------------------------- simulation + reward


def initial_state(sc: Scenario) -> dict[str, Any]:
    return {"meetings": [{"id": m.id, "day": m.day, "start": m.start, "dur": m.dur, "deleted": 0}
                         for m in sorted(sc.meetings, key=lambda m: m.id)],
            "cards": [{"code": c.code, "col": c.col, "prio": c.prio, "due": c.due, "archived": 0, "deleted": 0}
                      for c in sorted(sc.cards, key=lambda c: c.code)],
            "actions": []}


def simulate(sc: Scenario, steps: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    st = initial_state(sc)
    ms = {r["id"]: r for r in st["meetings"]}
    cs = {r["code"]: r for r in st["cards"]}
    for name, p in steps:
        p = {"ref": "", "arg": "", **p}
        if name == "reschedule" and p["id"] in ms:
            ms[p["id"]].update(day=p["day"], start=p["start"])
        elif name == "delete_meeting" and p["id"] in ms:
            ms[p["id"]]["deleted"] = 1
        elif name == "move_card" and p["code"] in cs:
            cs[p["code"]]["col"] = p["col"]
        elif name == "set_prio" and p["code"] in cs:
            cs[p["code"]]["prio"] = p["prio"]
        elif name == "set_due" and p["code"] in cs:
            cs[p["code"]]["due"] = p["due"]
        elif name == "archive_card" and p["code"] in cs:
            cs[p["code"]]["archived"] = 1
        elif name == "delete_card" and p["code"] in cs:
            cs[p["code"]]["deleted"] = 1
        st["actions"].append({"action": p.get("action", name), "ref": str(p["ref"]), "arg": p["arg"]})
    return st


_CODE_RE = re.compile(r"\b([A-Z]{2,4})-(\d{2,4})\b", re.I)
_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MON_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\b")


def extract_codes(text: str | None) -> set[str]:
    return {f"{a.upper()}-{b}" for a, b in _CODE_RE.findall(text or "")}


def extract_dates(text: str | None) -> set[tuple[int, int]]:
    t = text or ""
    out = {(int(m), int(d)) for _, m, d in _ISO_RE.findall(t)}
    out |= {(MONTHS.index(mo[:3].title()) + 1, int(d)) for mo, d in _MON_RE.findall(t)}
    out |= {(int(m), int(d)) for m, d in _SLASH_RE.findall(_ISO_RE.sub(" ", t))}
    return {x for x in out if 1 <= x[0] <= 12 and 1 <= x[1] <= 31}


def score(sc: Scenario, state: Mapping[str, Any], answer: str | None = None) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = {}
    init_m = {m.id: m for m in sc.meetings}
    now_m = {r["id"]: r for r in state.get("meetings", [])}
    init_c = {c.code: c for c in sc.cards}
    now_c = {r["code"]: r for r in state.get("cards", [])}
    for mid, m in init_m.items():
        r = now_m.get(mid, {})
        moved = (r.get("day", m.day), r.get("start", m.start)) != (m.day, m.start)
        if m.locked and (moved or r.get("deleted")):
            info["why"] = f"locked meeting {mid} {'deleted' if r.get('deleted') else 'moved'}"
            return -1.0, info
        if r.get("deleted"):
            info["why"] = f"deleted meeting {mid} instead of rescheduling"
            return -1.0, info
    for code in init_c:
        if now_c.get(code, {}).get("deleted"):
            info["why"] = f"deleted active card {code}"
            return -1.0, info
    moved_m = sorted(mid for mid, m in init_m.items()
                     if (now_m.get(mid, {}).get("day", m.day), now_m.get(mid, {}).get("start", m.start)) != (m.day, m.start))

    def card_changes(code: str) -> list[str]:
        c, r = init_c[code], now_c.get(code, {})
        return [k for k, v in (("col", c.col), ("prio", c.prio), ("due", c.due)) if r.get(k, v) != v] + (
            ["archived"] if r.get("archived") else [])

    changed_c = {code: ch for code in init_c if (ch := card_changes(code))}
    info.update(moved_meetings=moved_m, changed_cards=changed_c)

    if sc.family == "resolve_conflict":
        tid = int(sc.target)
        if changed_c or [m for m in moved_m if m != tid]:
            info["why"] = "changed something besides the conflicting meeting"
            return 0.0, info
        r = now_m.get(tid, {})
        ok = (r.get("day"), r.get("start"), r.get("dur")) == (sc.want["day"], sc.want["start"], init_m[tid].dur)
        info["why"] = "ok" if ok else f"target now day={r.get('day')} start={r.get('start')}"
        return (1.0 if ok else 0.0), info

    if sc.family == "move_and_update_card":
        if moved_m or [c for c in changed_c if c != sc.target]:
            info["why"] = f"changed other items {[c for c in changed_c if c != sc.target]} meetings={moved_m}"
            return 0.0, info
        r = now_c.get(sc.target, {})
        ok = (r.get("col"), r.get("prio"), r.get("due"), r.get("archived")) == (
            sc.want["col"], sc.want["prio"], sc.want["due"], 0)
        info["why"] = "ok" if ok else f"target now {r}"
        return (1.0 if ok else 0.0), info

    if moved_m or changed_c:
        info["why"] = "edited the board during a read-only report"
        return 0.0, info
    codes, dates = extract_codes(answer), extract_dates(answer)
    d = dt.date.fromisoformat(sc.want["due"])
    ok = codes == {sc.want["code"]} and dates == {(d.month, d.day)}
    info["why"] = "ok" if ok else f"answer codes={sorted(codes)} dates={sorted(dates)}"
    return (1.0 if ok else 0.0), info


def oracle_steps(sc: Scenario) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    if sc.family == "resolve_conflict":
        mid = int(sc.target)
        return [("reschedule", {"id": mid, "day": sc.want["day"], "start": sc.want["start"], "ref": str(mid),
                                "arg": f"{sc.want['day']}@{sc.want['start']}"})], None
    if sc.family == "move_and_update_card":
        c = sc.target
        return [("move_card", {"code": c, "col": sc.want["col"], "ref": c, "arg": sc.want["col"]}),
                ("set_prio", {"code": c, "prio": sc.want["prio"], "ref": c, "arg": sc.want["prio"]}),
                ("set_due", {"code": c, "due": sc.want["due"], "ref": c, "arg": sc.want["due"]})], None
    return [], f"{sc.want['code']}, due {fmt_due(sc.want['due'])}"


# ----------------------------------------------------------------------------- page

CSS = Template("""
*{box-sizing:border-box} body{margin:0;background:$bg;color:$fg;font:$fs/1.4 $font}
button{font:inherit;color:inherit;cursor:pointer}
#top{display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid $border;background:$panel;color:$fg}
#top b{font-size:1.1em;margin-right:8px} .vb{border:1px solid $border;background:transparent;border-radius:$radius;padding:4px 12px}
.vb.on{background:$accent;color:#fff;border-color:$accent} #wk{margin-left:auto;color:$muted}
#body{padding:10px 14px;height:calc(100vh - 50px);overflow:auto}
#grid{display:grid;grid-template-columns:56px repeat(5,1fr);border:1px solid $border;border-radius:$radius;background:$card;color:$fg}
.dh{padding:4px 6px;border-left:1px solid $border;border-bottom:1px solid $border;font-weight:600;font-size:.9em}
.hours{position:relative} .hr{height:$hour;border-bottom:1px solid $border;font-size:.72em;color:$muted;padding:0 4px;text-align:right}
.dcol{position:relative;border-left:1px solid $border}
.slot{height:$hour;border-bottom:1px dashed $border}
.ev{position:absolute;border-radius:$radius;padding:1px 5px;font-size:.74em;overflow:hidden;background:$ev;color:$ev_fg;
  border:1px solid $border;text-align:left;line-height:1.2}
.ev.lock{background:$lock;color:$lock_fg}
#bar{display:flex;gap:8px;margin-bottom:10px} #q{flex:1;padding:6px 10px;border:1px solid $border;border-radius:$radius;background:$panel;color:$fg;font:inherit}
.btn{padding:4px 11px;border:1px solid $border;border-radius:$radius;background:$card;color:$fg}
.btn.on{background:$accent;color:#fff} .btn.danger{color:$danger;border-color:$danger}
#cols{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.col{background:$col;border-radius:$radius;padding:$pad;min-height:200px;color:$fg}
.col h3{margin:2px 4px 8px;font-size:.95em}
.card{display:block;width:100%;text-align:left;background:$card;border:1px solid $border;border-radius:$radius;padding:$pad;margin-bottom:$pad;color:$fg}
.code{font-family:ui-monospace,monospace;font-size:.82em;color:$muted} .meta{font-size:.8em;color:$muted;display:flex;gap:8px;flex-wrap:wrap}
.pr{font-weight:700} .pr.P0{color:$danger}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;align-items:flex-start;justify-content:center;padding-top:50px;z-index:9}
#modal.on{display:flex} .dlg{background:$card;color:$fg;border:1px solid $border;border-radius:$radius;padding:16px;width:min(560px,94vw)}
.dlg h2{margin:0 0 6px} .rowb{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px} .lbl{font-size:.85em;color:$muted;margin-top:8px}
.dlg input{padding:5px 8px;border:1px solid $border;border-radius:$radius;background:$bg;color:$fg;font:inherit}
#toast{color:$muted;margin-top:8px;min-height:1.2em} .empty{color:$muted;font-size:.85em;padding:6px}
""")

PAGE_JS = r"""
const OPS = __OPS__; const Q = __Q__; const COLS = __COLS__; const PRIOS = ['P0','P1','P2','P3'];
const DAYN = ['Mon','Tue','Wed','Thu','Fri']; const WEEK = __WEEK__; const HOUR = __HOUR__; const MON = __MONTHS__;
let V = {view: __START__, q:'', modal:null, day:null};
function fmt12(m){ const h=Math.floor(m/60), mm=m%60; return `${(h+11)%12+1}:${String(mm).padStart(2,'0')} ${h<12?'AM':'PM'}`; }
function fmtDue(iso){ const p = iso.split('-'); return MON[+p[1]-1] + ' ' + (+p[2]); }
function dayDate(d){ const t = new Date(WEEK + 'T12:00:00Z'); t.setUTCDate(t.getUTCDate()+d); return MON[t.getUTCMonth()] + ' ' + t.getUTCDate(); }
function parseTime(s){
  const m = /^\s*(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?\s*m?\.?\s*$/i.exec(s||''); if (!m) return null;
  let h = +m[1]; const mm = +(m[2]||0); const ap = (m[3]||'').toLowerCase();
  if (mm >= 60 || h > 23 || (ap && !(h>=1 && h<=12))) return null;
  if (ap==='p' && h!==12) h += 12; if (ap==='a' && h===12) h = 0; return h*60+mm;
}
function parseQ(q){
  const out = {assignee:'', prio:'', col:'', text:''}; const text = [];
  for (const tok of (q||'').replace(/"/g,' ').split(/\s+/)) {
    if (!tok) continue; const low = tok.toLowerCase(); const i = low.indexOf(':');
    const k = i>=0 ? low.slice(0,i) : ''; const v = i>=0 ? low.slice(i+1) : '';
    if (i>=0 && (k==='assignee'||k==='owner')) out.assignee = v.replace(/^@+/,'');
    else if (i>=0 && (k==='priority'||k==='prio'||k==='p')) out.prio = PRIOS.includes(v.toUpperCase()) ? v.toUpperCase() : ((v && '0123'.includes(v)) ? 'P'+v : 'none');
    else if (i>=0 && (k==='column'||k==='col'||k==='status')) { const s = v.replace(/[^a-z]/g,'');
      out.col = ({backlog:'backlog',inprogress:'in_progress',progress:'in_progress',doing:'in_progress',review:'review',done:'done'})[s] || 'none'; }
    else text.push(tok);
  }
  out.text = text.join(' '); return out;
}
async function setView(v){ V.view=v; V.modal=null; await op('log',{action:'view',ref:v}); render(); }
async function render(){
  document.getElementById('vb-calendar').classList.toggle('on', V.view==='calendar');
  document.getElementById('vb-board').classList.toggle('on', V.view==='board');
  if (V.view==='calendar') await renderCal(); else await renderBoard();
  await renderModal();
}
async function renderCal(){
  const ms = await sql(Q.meetings, {});
  let h = `<div id="grid"><div class="dh"></div>` + DAYN.map((d,i)=>`<div class="dh">${d} ${dayDate(i)}</div>`).join('');
  h += `<div class="hours">` + Array.from({length:10},(_,i)=>`<div class="hr">${fmt12((8+i)*60).replace(':00','')}</div>`).join('') + `</div>`;
  for (let d=0; d<5; d++) {
    const day = ms.filter(m=>m[2]===d); const lanes = []; const lane = {};
    for (const m of day) { let k = lanes.findIndex(end => end <= m[3]); if (k<0) { k = lanes.length; lanes.push(0); } lanes[k] = m[3]+m[4]; lane[m[0]] = k; }
    const nl = Math.max(1, lanes.length);
    h += `<div class="dcol" data-day="${d}">` + Array.from({length:10},()=>`<div class="slot"></div>`).join('') + day.map(m => {
      const top = (m[3]-480)/60*HOUR, ht = Math.max(16, m[4]/60*HOUR - 2), w = 100/nl;
      return `<button class="ev ${m[5]?'lock':''}" data-id="${m[0]}" style="top:${top}px;height:${ht}px;left:calc(${lane[m[0]]*w}% + 2px);width:calc(${w}% - 4px)"
        onclick="openMeeting(${m[0]})">${m[5]?'<b>Locked</b> &middot; ':''}${esc(m[1])}<br>${fmt12(m[3])} - ${fmt12(m[3]+m[4])}</button>`; }).join('') + `</div>`;
  }
  document.getElementById('body').innerHTML = h + `</div>`;
}
async function renderBoard(){
  const rows = await sql(Q.cards, parseQ(V.q));
  let h = `<div id="bar"><input id="q" aria-label="Filter cards" placeholder="Filter: assignee:@me priority:P0 column:review or text" value="${esc(V.q)}" autocomplete="off">
    <button class="btn" onclick="search()">Filter</button><button class="btn" onclick="V.q='';render()">Clear filter</button></div><div id="cols">`;
  for (const c of COLS) {
    const cs = rows.filter(r=>r[2]===c[0]);
    h += `<div class="col" data-col="${c[0]}"><h3>${c[1]} <span class="code">${cs.length}</span></h3>` + (cs.length ? cs.map(r =>
      `<button class="card" data-code="${r[0]}" onclick="openCard('${r[0]}')"><span class="code">${esc(r[0])}</span><br>${esc(r[1])}
       <div class="meta"><span class="pr ${r[3]}">${r[3]}</span><span>@${esc(r[4])}</span><span>Due ${fmtDue(r[5])}</span></div></button>`).join('')
      : '<div class="empty">No cards</div>') + `</div>`;
  }
  document.getElementById('body').innerHTML = h + `</div>`;
  document.getElementById('q').addEventListener('keydown', e => { if (e.key==='Enter') search(); });
}
async function search(){ V.q = document.getElementById('q').value; await op('log',{action:'search',ref:'board',arg:V.q}); render(); }
async function openMeeting(id){ V.modal={kind:'meeting', id}; V.day=null; await op('log',{action:'open_meeting',ref:String(id)}); render(); }
async function openCard(code){ V.modal={kind:'card', code}; await op('log',{action:'open_card',ref:code}); render(); }
function closeModal(){ V.modal=null; render(); }
async function renderModal(){
  const md = document.getElementById('modal'); if (!V.modal) { md.classList.remove('on'); md.innerHTML=''; return; }
  let h = '';
  if (V.modal.kind==='meeting') {
    const r = await sql(Q.meeting, {id:V.modal.id}); if (!r.length) { V.modal=null; md.classList.remove('on'); return; }
    const m = r[0]; const att = JSON.parse(m[5]); if (V.day===null) V.day = m[2];
    h = `<div class="dlg" role="dialog" aria-label="Meeting details"><h2>${m[7]?'Locked &middot; ':''}${esc(m[1])}</h2>
      <div>${DAYN[m[2]]} ${dayDate(m[2])}, ${fmt12(m[3])} - ${fmt12(m[3]+m[4])} (${m[4]} min)</div>
      <div class="lbl">Organizer: ${esc(m[6])}${m[7]?' &middot; <b>External meeting, locked by organizer</b>':''}</div>
      <div class="lbl">Attendees: ${att.map(esc).join(', ')}</div>
      <div class="lbl">Reschedule to day</div><div class="rowb">` + DAYN.map((d,i)=>`<button class="btn ${V.day===i?'on':''}" onclick="V.day=${i};renderModal()">${d}</button>`).join('') + `</div>
      <div class="lbl">New start time</div><div class="rowb"><input id="mstart" aria-label="New start time" placeholder="e.g. 2:00 PM" value="${fmt12(m[3])}">
      <button class="btn on" onclick="saveMeeting()">Save reschedule</button></div>
      <div class="rowb"><button class="btn danger" onclick="delMeeting()">Delete meeting</button><button class="btn" onclick="closeModal()">Close</button></div>
      <div id="toast"></div></div>`;
  } else {
    const r = await sql(Q.card, {code:V.modal.code}); if (!r.length) { V.modal=null; md.classList.remove('on'); return; }
    const c = r[0];
    h = `<div class="dlg" role="dialog" aria-label="Card details"><span class="code">${esc(c[0])}</span><h2>${esc(c[1])}</h2>
      <div class="lbl">Assignee @${esc(c[4])} &middot; Due ${fmtDue(c[5])} (${c[5]})</div>
      <div class="lbl">Move to</div><div class="rowb">` + COLS.map(k=>`<button class="btn ${c[2]===k[0]?'on':''}" onclick="cardOp('move_card',{col:'${k[0]}'},'${k[0]}')">${k[1]}</button>`).join('') + `</div>
      <div class="lbl">Priority</div><div class="rowb">` + PRIOS.map(p=>`<button class="btn ${c[3]===p?'on':''}" onclick="cardOp('set_prio',{prio:'${p}'},'${p}')">${p}</button>`).join('') + `</div>
      <div class="lbl">Due date</div><div class="rowb"><input id="cdue" aria-label="Due date" placeholder="YYYY-MM-DD" value="${c[5]}">
      <button class="btn on" onclick="saveDue()">Set due date</button></div>
      <div class="rowb"><button class="btn" onclick="cardOp('archive_card',{},'')">Archive</button><button class="btn danger" onclick="cardOp('delete_card',{},'')">Delete card</button>
      <button class="btn" onclick="closeModal()">Close</button></div><div id="toast"></div></div>`;
  }
  md.innerHTML = h; md.classList.add('on');
}
async function saveMeeting(){ const s = parseTime(document.getElementById('mstart').value);
  if (s===null) { document.getElementById('toast').textContent='Enter a time like 2:00 PM'; return; }
  const id = V.modal.id; await op('reschedule',{id, day:V.day, start:s, ref:String(id), arg:V.day+'@'+s}); V.modal=null; await render(); }
async function delMeeting(){ const id = V.modal.id; await op('delete_meeting',{id, ref:String(id)}); V.modal=null; await render(); }
async function cardOp(name, p, arg){ const code = V.modal.code; await op(name, Object.assign({code, ref:code, arg}, p));
  if (name==='archive_card'||name==='delete_card') V.modal=null; await render(); }
async function saveDue(){ const v = document.getElementById('cdue').value.trim();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(v)) { document.getElementById('toast').textContent='Use YYYY-MM-DD'; return; }
  await cardOp('set_due', {due:v}, v); }
document.addEventListener('keydown', e => { if (e.key==='Escape' && V.modal) closeModal(); });
document.addEventListener('DOMContentLoaded', async () => { await render(); emit({type:'ready'}); });
"""


def render_html(sc: Scenario) -> str:
    th = style_for(sc.split, sc.style_seed)
    js = (PAGE_JS.replace("__OPS__", json.dumps(OPS)).replace("__Q__", json.dumps(Q))
          .replace("__COLS__", json.dumps([[c, COL_LABEL[c]] for c in COLUMNS]))
          .replace("__WEEK__", json.dumps(sc.week_start)).replace("__HOUR__", th["hour"].rstrip("px"))
          .replace("__MONTHS__", json.dumps(list(MONTHS))).replace("__START__", json.dumps(sc.start_view)))
    ws = dt.date.fromisoformat(sc.week_start)
    title = "Calendar - CalBoard" if sc.split == "train" else "Team Planner"
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(title)}</title>"
            f"<style>{CSS.substitute(th)}</style><script>{BASE_JS}{js}</script></head><body>"
            f"<div id='top'><b>CalBoard</b><button class='vb' id='vb-calendar' onclick=\"setView('calendar')\">Calendar</button>"
            f"<button class='vb' id='vb-board' onclick=\"setView('board')\">Board</button>"
            f"<span id='wk'>Week of {MONTHS[ws.month - 1]} {ws.day}, {ws.year}</span></div>"
            f"<div id='body'></div><div id='modal'></div></body></html>")
