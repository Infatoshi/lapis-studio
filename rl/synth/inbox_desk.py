"""InboxDesk: seeded email/chat triage micro-app with a SQLite-verified reward.

Pure stdlib inside the sandbox: ``app.py`` (``APP_PY`` below) is one
``http.server`` process plus one SQLite file (well under 30 MiB RSS). The page
is a dual-pane inbox: thread list with unread badges, sender, timestamp and a
search bar (``from:``, ``subject:``, ``is:unread``, ``is:starred``, free
text), and a reading pane with Reply / Reply All / Archive / Delete / Star /
Mark Unread and a compose box.

Every UI action goes through the HTTP API into SQLite (threads, sent, actions
tables). ``score`` is a pure function of (scenario, db dump, answer); the
offline tests exec ``APP_PY`` locally so they exercise the exact DB code that
runs in the sandbox.

Families:
  * find_and_open_thread: open the right thread (logged ``open``) and report
    its fact; the answer must contain exactly the true fact of that type.
  * safe_triage: archive or star ONLY the target. Delete anywhere -> -1.0,
    any other thread modified -> -1.0.
  * reply_vs_reply_all: exactly one send, on the target thread, in the right
    mode, containing the required phrase. Delete -> -1.0.

A decoy thread (look-alike sender name, same topic, different fact) and fact
decoys on every thread keep "first thread", "first name match", "always reply
all" and random guessing near chance.
"""

from __future__ import annotations

import html
import json
import random
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PORT = 8772
CDP_PORT = 9222
ROOT = "/tmp/inboxdesk"
DB = f"{ROOT}/inbox.db"
BASE_URL = f"http://127.0.0.1:{PORT}"
CHROMIUM = "/usr/bin/chromium"
ME = "me@lapis.test"

FAMILIES = ("find_and_open_thread", "safe_triage", "reply_vs_reply_all")
FACT_TYPES = ("meeting_time", "tracking_code", "pr_number")
FACT_LABEL = {"meeting_time": "meeting time", "tracking_code": "tracking code", "pr_number": "PR number"}

# style_seed % len(THEMES); 0-2 train, 3-4 held-out (OOD).
THEMES: tuple[dict[str, str], ...] = (
    {"name": "light", "bg": "#ffffff", "fg": "#1f1f1f", "side": "#f2f4f7", "sel": "#d3e3fd", "accent": "#0b57d0", "font": "system-ui, sans-serif", "row": "10px", "fs": "14px"},
    {"name": "dark", "bg": "#1e1f22", "fg": "#e3e3e3", "side": "#2b2d31", "sel": "#404249", "accent": "#8ab4f8", "font": "system-ui, sans-serif", "row": "10px", "fs": "14px"},
    {"name": "compact", "bg": "#fbfbfb", "fg": "#202124", "side": "#eceff1", "sel": "#cfe0fc", "accent": "#1a73e8", "font": "Arial, sans-serif", "row": "4px", "fs": "12px"},
    {"name": "ood_slack", "bg": "#ffffff", "fg": "#1d1c1d", "side": "#3f0e40", "sel": "#1164a3", "accent": "#007a5a", "font": "Lato, Helvetica, sans-serif", "row": "8px", "fs": "15px"},
    {"name": "ood_gmail", "bg": "#f6f8fc", "fg": "#202124", "side": "#ffffff", "sel": "#c2e7ff", "accent": "#d93025", "font": "Roboto, Arial, sans-serif", "row": "12px", "fs": "14px"},
)
N_TRAIN_THEMES = 3
ACTIONS = ("reply", "reply_all", "archive", "delete", "star", "mark_unread")
ACTION_LABEL = {"reply": "Reply", "reply_all": "Reply All", "archive": "Archive", "delete": "Delete",
                "star": "Star", "mark_unread": "Mark Unread"}


def theme_for(style_seed: int) -> dict[str, str]:
    return THEMES[style_seed % len(THEMES)]


def is_ood_style(style_seed: int) -> bool:
    return style_seed % len(THEMES) >= N_TRAIN_THEMES


# ----------------------------------------------------------------------------- generation (pure)

DECOY_PAIRS = [
    ("Dana Whitfield", "Dana Whitford"), ("Jon Park", "John Park"), ("Mara Lin", "Maura Lin"),
    ("Chris Adler", "Kris Adler"), ("Sam Ortiz", "Sam Ortega"), ("Lee Chen", "Li Chen"),
    ("Erik Strand", "Eric Strand"), ("Nadia Karim", "Nadia Kareem"), ("Theo Grant", "Teo Grant"),
]
OTHER_NAMES = [
    "Ava Novak", "Ben Murphy", "Carla Diaz", "Dev Patel", "Ewa Kowalski", "Felix Braun", "Gia Russo",
    "Hugo Silva", "Ines Moreau", "Jamal Wright", "Kenji Mori", "Lucia Romero", "Mo Farah", "Nina Petrova",
    "Omar Aziz", "Paula Costa", "Quinn Hayes", "Rosa Vidal", "Sven Berg", "Tara Walsh", "Uma Rao",
    "Victor Hale", "Wen Zhou", "Yara Haddad", "Zoe Clarke",
]
TOPICS = [
    "Q4 vendor review", "launch readiness", "GPU cluster renewal", "offsite planning", "security audit",
    "design crit", "hiring loop", "billing migration", "incident retro", "roadmap sync", "API deprecation",
    "contract renewal", "on-call handoff", "data retention policy", "customer escalation",
]
SUBJ_FORMS = ["{t}", "Re: {t}", "{t}: next steps", "Quick question on {t}", "Fwd: {t} notes", "{t} follow-up"]
PHRASES = ["Confirmed for Thursday", "Approved on my side", "Looping in finance today", "Shipping it tomorrow",
           "Works for me, thanks", "Let's push to next week"]


def _email(name: str) -> str:
    return name.lower().replace(" ", ".") + "@" + random.Random(name).choice(["acme.test", "corp.test", "mail.test"])


def make_fact(kind: str, rng: random.Random) -> str:
    if kind == "meeting_time":
        return f"{rng.randint(1, 12)}:{rng.choice(['00', '15', '30', '45'])} {rng.choice(['AM', 'PM'])}"
    if kind == "tracking_code":
        return f"TRK-{rng.randrange(10000, 99999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
    return f"#{rng.randrange(1000, 9999)}"


FACT_SENTENCE = {
    "meeting_time": "Can we meet at {f} to go over it?",
    "tracking_code": "The shipment tracking code is {f}.",
    "pr_number": "The fix is in PR {f}, please take a look.",
}

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?")
_TRK_RE = re.compile(r"\bTRK-\d{5}-[A-Z]{2}\b")
_PR_RE = re.compile(r"(?:#|\bPR\s*#?\s*)(\d{4})\b", re.I)


def extract_facts(kind: str, text: str | None) -> set[str]:
    text = text or ""
    if kind == "meeting_time":
        return {f"{int(h)}:{m} {ap.upper()}M" for h, m, ap in _TIME_RE.findall(text)}
    if kind == "tracking_code":
        return set(_TRK_RE.findall(text.upper()))
    return {f"#{n}" for n in _PR_RE.findall(text)}


@dataclass
class Thread:
    id: int
    subject: str
    sender_name: str
    sender_email: str
    to: list[str]
    cc: list[str]
    ts: int
    unread: bool
    starred: bool
    fact_type: str
    fact: str
    body: str
    topic: str


@dataclass
class Scenario:
    seed: int
    family: str
    style_seed: int
    threads: list[Thread]
    target_id: int
    decoy_id: int  # -1 when has_decoy_sender is False
    fact_type: str
    action: str  # safe_triage: archive|star; reply_vs_reply_all: reply|reply_all; find: ""
    phrase: str
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> Thread:
        return next(t for t in self.threads if t.id == self.target_id)

    def ordered(self) -> list[Thread]:
        """Inbox order as the UI shows it: newest first."""
        return sorted(self.threads, key=lambda t: (-t.ts, t.id))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def generate(
    seed: int,
    family: str | None = None,
    n_threads: int | None = None,
    n_unread: int | None = None,
    has_decoy_sender: bool | None = None,
    style_seed: int | None = None,
    action: str | None = None,
) -> Scenario:
    rng = random.Random(f"inboxdesk:{seed}")
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    n_threads = n_threads if n_threads is not None else rng.randint(12, 30)
    if not 12 <= n_threads <= 30:
        raise ValueError("n_threads must be in 12..30")
    n_unread = n_unread if n_unread is not None else rng.randint(2, n_threads // 2)
    n_unread = max(0, min(n_unread, n_threads))
    has_decoy_sender = has_decoy_sender if has_decoy_sender is not None else rng.random() < 0.8
    style_seed = style_seed if style_seed is not None else rng.randrange(N_TRAIN_THEMES)

    pair = rng.choice(DECOY_PAIRS)
    if rng.random() < 0.5:
        pair = (pair[1], pair[0])
    target_name, decoy_name = pair
    topics = rng.sample(TOPICS, len(TOPICS))
    target_topic = topics.pop()
    fact_type = rng.choice(FACT_TYPES)
    others = rng.sample(OTHER_NAMES, len(OTHER_NAMES))

    # (sender, topic, forced fact type) per thread
    plan: list[tuple[str, str, str | None]] = [(target_name, target_topic, fact_type)]
    if has_decoy_sender:
        plan.append((decoy_name, target_topic, fact_type))
    if rng.random() < 0.6:  # same sender, different topic and same fact type
        plan.append((target_name, topics.pop(), fact_type))
    if rng.random() < 0.6:  # different sender, same topic
        plan.append((others.pop(), target_topic, fact_type))
    plan.append((others.pop(), rng.choice(topics), fact_type))  # always >= 2 facts of the asked type
    while len(plan) < n_threads:
        plan.append((others.pop() if others else rng.choice(OTHER_NAMES), rng.choice(topics), None))

    now = 1_795_000_000 + rng.randrange(0, 86400 * 30)
    stamps = sorted(rng.sample(range(now - 86400 * 14, now), n_threads), reverse=True)
    rng.shuffle(stamps)  # target position in the newest-first list is uniform
    facts_used: set[str] = set()
    pool = [n for n in OTHER_NAMES]
    threads: list[Thread] = []
    for i, (sender, topic, ft) in enumerate(plan):
        ft = ft or rng.choice(FACT_TYPES)
        while True:
            f = make_fact(ft, rng)
            if f not in facts_used:
                facts_used.add(f)
                break
        to = [ME] + rng.sample(pool, rng.randint(0, 2))
        cc = rng.sample(pool, rng.randint(1, 3)) if (i == 0 or rng.random() < 0.6) else []
        to = [x if "@" in x else _email(x) for x in to]
        cc = [_email(x) for x in cc if x != sender]
        if i == 0 and not cc:
            cc = [_email(rng.choice([p for p in pool if p != sender]))]
        subj = rng.choice(SUBJ_FORMS).format(t=topic)
        filler = rng.choice(["Hi,", "Hey team,", "Hello,", "Quick update:"])
        body = (f"{filler} {FACT_SENTENCE[ft].format(f=f)} "
                f"Context is in the {topic} doc. Thanks, {sender.split()[0]}")
        threads.append(Thread(id=0, subject=subj, sender_name=sender, sender_email=_email(sender), to=to, cc=cc,
                              ts=stamps[i], unread=False, starred=rng.random() < 0.15, fact_type=ft, fact=f,
                              body=body, topic=topic))
    # ids follow a seeded shuffle so id order leaks nothing
    ids = list(range(1, n_threads + 1))
    rng.shuffle(ids)
    for t, i in zip(threads, ids):
        t.id = i
    for t in rng.sample(threads, n_unread):
        t.unread = True
    target = threads[0]
    decoy_id = threads[1].id if has_decoy_sender else -1
    # target subject differs from the decoy's so subject disambiguates when the name is a near miss
    if has_decoy_sender:
        forms = rng.sample(SUBJ_FORMS, 2)
        target.subject = forms[0].format(t=target_topic)
        threads[1].subject = forms[1].format(t=target_topic)

    if family == "safe_triage":
        action = action or rng.choice(("archive", "star"))
        if action == "star":
            target.starred = False
    elif family == "reply_vs_reply_all":
        action = action or rng.choice(("reply", "reply_all"))
    else:
        action = ""
    phrase = rng.choice(PHRASES) if family == "reply_vs_reply_all" else ""
    instruction = _instruction(family, target, fact_type, action, phrase, rng)
    return Scenario(
        seed=seed, family=family, style_seed=style_seed, threads=threads, target_id=target.id, decoy_id=decoy_id,
        fact_type=fact_type, action=action, phrase=phrase, instruction=instruction,
        meta={"n_threads": n_threads, "n_unread": n_unread, "has_decoy_sender": has_decoy_sender,
              "ood_style": is_ood_style(style_seed)},
    )


def _instruction(family: str, t: Thread, fact_type: str, action: str, phrase: str, rng: random.Random) -> str:
    who = f"{t.sender_name} ({t.sender_email})" if rng.random() < 0.5 else t.sender_name
    ref = f'the thread from {who} with the subject "{t.subject}"'
    if family == "find_and_open_thread":
        return f"In InboxDesk, open {ref} and report the {FACT_LABEL[fact_type]} it mentions as your final answer."
    if family == "safe_triage":
        verb = "Archive" if action == "archive" else "Star"
        extra = " Archive it, do not delete it." if action == "archive" else ""
        return f"In InboxDesk, {verb.lower()} {ref}. Do not change any other thread.{extra}"
    if action == "reply":
        how = rng.choice([f"Reply privately so only {t.sender_name.split()[0]} receives it",
                          "Reply to the sender only, not the whole thread"])
    else:
        how = rng.choice(["Reply to everyone on the thread, including Cc",
                          "Make sure every recipient of the thread gets your answer (Reply All)"])
    return f'In InboxDesk, open {ref}. {how}, with a message containing "{phrase}", and send it.'


# ----------------------------------------------------------------------------- in-sandbox app

APP_PY = r'''
import json, os, shlex, sqlite3, sys, time, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ME = "me@lapis.test"
SCHEMA = """
CREATE TABLE threads(id INTEGER PRIMARY KEY, subject TEXT, sender_name TEXT, sender_email TEXT,
  to_json TEXT, cc_json TEXT, ts INTEGER, unread INTEGER, starred INTEGER, archived INTEGER DEFAULT 0,
  deleted INTEGER DEFAULT 0, body TEXT);
CREATE TABLE sent(id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id INTEGER, mode TEXT,
  recipients_json TEXT, body TEXT, ts REAL);
CREATE TABLE actions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, action TEXT, thread_id INTEGER, arg TEXT);
"""

def init_db(path, scenario):
    if os.path.exists(path): os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for t in scenario["threads"]:
        con.execute("INSERT INTO threads(id,subject,sender_name,sender_email,to_json,cc_json,ts,unread,starred,body)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (t["id"], t["subject"], t["sender_name"], t["sender_email"], json.dumps(t["to"]),
                     json.dumps(t["cc"]), t["ts"], int(t["unread"]), int(t["starred"]), t["body"]))
    con.commit()
    return con

def log(con, action, tid, arg=""):
    con.execute("INSERT INTO actions(ts,action,thread_id,arg) VALUES(?,?,?,?)", (time.time(), action, tid, arg))

def parse_query(q):
    try: toks = shlex.split(q or "")
    except ValueError: toks = (q or "").split()
    where, args, box = [], [], "inbox"
    for tok in toks:
        low = tok.lower()
        if low.startswith("from:"):
            v = tok[5:]; where.append("(sender_name LIKE ? OR sender_email LIKE ?)"); args += [f"%{v}%"] * 2
        elif low.startswith("subject:"):
            where.append("subject LIKE ?"); args.append(f"%{tok[8:]}%")
        elif low == "is:unread": where.append("unread=1")
        elif low == "is:starred": where.append("starred=1")
        elif low == "in:archive": box = "archive"
        elif tok:
            where.append("(subject LIKE ? OR body LIKE ? OR sender_name LIKE ?)"); args += [f"%{tok}%"] * 3
    where.append("deleted=0")
    where.append("archived=1" if box == "archive" else "archived=0")
    return " AND ".join(where), args

def list_threads(con, q=""):
    w, a = parse_query(q)
    rows = con.execute(f"SELECT id,subject,sender_name,sender_email,ts,unread,starred,substr(body,1,18)||'...'"
                       f" FROM threads WHERE {w} ORDER BY ts DESC, id ASC", a).fetchall()
    unread = con.execute("SELECT COUNT(*) FROM threads WHERE unread=1 AND archived=0 AND deleted=0").fetchone()[0]
    keys = ("id", "subject", "sender_name", "sender_email", "ts", "unread", "starred", "snippet")
    return {"threads": [dict(zip(keys, r)) for r in rows], "unread_total": unread}

def open_thread(con, tid):
    r = con.execute("SELECT id,subject,sender_name,sender_email,to_json,cc_json,ts,unread,starred,body"
                    " FROM threads WHERE id=? AND deleted=0", (tid,)).fetchone()
    if not r: return None
    con.execute("UPDATE threads SET unread=0 WHERE id=?", (tid,)); log(con, "open", tid); con.commit()
    keys = ("id", "subject", "sender_name", "sender_email", "to", "cc", "ts", "unread", "starred", "body")
    d = dict(zip(keys, r)); d["to"] = json.loads(d["to"]); d["cc"] = json.loads(d["cc"]); d["unread"] = 0
    return d

SETS = {"archive": "archived=1", "unarchive": "archived=0", "delete": "deleted=1", "star": "starred=1",
        "unstar": "starred=0", "mark_unread": "unread=1", "mark_read": "unread=0"}

def act(con, tid, action):
    if action not in SETS: raise ValueError(action)
    con.execute(f"UPDATE threads SET {SETS[action]} WHERE id=?", (tid,)); log(con, action, tid); con.commit()

def recipients(con, tid, mode):
    sender, to, cc = con.execute("SELECT sender_email,to_json,cc_json FROM threads WHERE id=?", (tid,)).fetchone()
    if mode == "reply": return [sender]
    out = [sender] + [x for x in json.loads(to) + json.loads(cc) if x != ME and x != sender]
    return list(dict.fromkeys(out))

def send(con, tid, mode, body):
    if mode not in ("reply", "reply_all"): raise ValueError(mode)
    rc = recipients(con, tid, mode)
    con.execute("INSERT INTO sent(thread_id,mode,recipients_json,body,ts) VALUES(?,?,?,?,?)",
                (tid, mode, json.dumps(rc), body, time.time()))
    log(con, "send_" + mode, tid); con.commit()
    return rc

def dump(con):
    th = con.execute("SELECT id,unread,starred,archived,deleted FROM threads ORDER BY id").fetchall()
    se = con.execute("SELECT thread_id,mode,recipients_json,body FROM sent ORDER BY id").fetchall()
    ac = con.execute("SELECT action,thread_id,arg FROM actions ORDER BY id").fetchall()
    return {"threads": [dict(zip(("id", "unread", "starred", "archived", "deleted"), r)) for r in th],
            "sent": [{"thread_id": a, "mode": b, "recipients": json.loads(c), "body": d} for a, b, c, d in se],
            "actions": [{"action": a, "thread_id": b, "arg": c} for a, b, c in ac]}

def serve(root, port, db):
    con = sqlite3.connect(db, check_same_thread=False)
    import threading
    lock = threading.Lock()
    index = open(os.path.join(root, "index.html"), "rb").read()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, obj, ctype="application/json"):
            b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
            with lock:
                if u.path in ("/", "/index.html"): return self._send(200, index, "text/html; charset=utf-8")
                if u.path == "/api/threads":
                    q0 = q.get("q", [""])[0]
                    if q0: log(con, "search", None, q0); con.commit()
                    return self._send(200, list_threads(con, q0))
                if u.path == "/api/thread":
                    d = open_thread(con, int(q.get("id", ["0"])[0]))
                    return self._send(200 if d else 404, d or {"error": "not found"})
            return self._send(404, {"error": "no route"})
        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try: body = json.loads(self.rfile.read(n) or b"{}")
            except Exception: return self._send(400, {"error": "bad json"})
            try:
                with lock:
                    if u.path == "/api/action":
                        act(con, int(body["id"]), body["action"]); return self._send(200, {"ok": True})
                    if u.path == "/api/send":
                        rc = send(con, int(body["id"]), body["mode"], body.get("body", ""))
                        return self._send(200, {"ok": True, "recipients": rc})
            except (KeyError, ValueError) as e:
                return self._send(400, {"error": str(e)})
            return self._send(404, {"error": "no route"})
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

if __name__ == "__main__":
    cmd, root, db = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd == "init":
        init_db(db, json.load(open(os.path.join(root, "scenario.json")))).close()
    elif cmd == "serve":
        serve(root, int(sys.argv[4]), db)
    elif cmd == "dump":
        print(json.dumps(dump(sqlite3.connect(db))))
'''


def load_app() -> dict[str, Any]:
    """Exec ``APP_PY`` locally (offline tests use the exact in-sandbox DB code)."""
    ns: dict[str, Any] = {"__name__": "inboxdesk_app"}
    exec(compile(APP_PY, "inboxdesk_app.py", "exec"), ns)
    return ns


# ----------------------------------------------------------------------------- page


def render_index(sc: Scenario) -> str:
    th = theme_for(sc.style_seed)
    order = list(ACTIONS)
    random.Random(f"inboxdesk-buttons:{sc.style_seed}").shuffle(order)
    buttons = "".join(
        f'<button id="btn-{a}" data-action="{a}" onclick="doAction(\'{a}\')">{ACTION_LABEL[a]}</button>' for a in order
    )
    slack = th["name"] == "ood_slack"
    side_fg = "#ffffff" if slack else th["fg"]
    css = f"""
*{{box-sizing:border-box}} body{{margin:0;font-family:{th['font']};font-size:{th['fs']};background:{th['bg']};color:{th['fg']}}}
#top{{display:flex;align-items:center;gap:12px;padding:8px 12px;border-bottom:1px solid #8884}}
#top b{{font-size:18px}} #q{{flex:1;font-size:15px;padding:7px 10px;border:1px solid #8886;border-radius:18px;background:{th['bg']};color:{th['fg']}}}
#badge{{background:{th['accent']};color:#fff;border-radius:10px;padding:1px 8px;font-weight:700}}
#main{{display:flex;height:calc(100vh - 52px)}}
#list{{width:42%;overflow-y:auto;background:{th['side']};color:{side_fg};border-right:1px solid #8884}}
.row{{padding:{th['row']} 12px;border-bottom:1px solid #8883;cursor:pointer}} .row.sel{{background:{th['sel']}}}
.row.unread .who,.row.unread .subj{{font-weight:800}} .dot{{color:{th['accent']}}}
.who{{display:flex;justify-content:space-between}} .ts{{opacity:.7;font-size:12px}} .snip{{opacity:.7;font-size:12px}}
#pane{{flex:1;padding:16px;overflow-y:auto}} #acts{{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}}
#acts button,#send{{padding:6px 12px;border:1px solid #8886;border-radius:6px;background:{th['bg']};color:{th['fg']};cursor:pointer}}
#send{{background:{th['accent']};color:#fff;border:0}} .meta{{opacity:.8;font-size:13px}}
#compose{{display:none;border:1px solid #8886;border-radius:8px;padding:10px;margin-top:12px}}
#body{{width:100%;height:90px;font-size:14px;background:{th['bg']};color:{th['fg']}}} #toast{{margin-top:8px;opacity:.85}}
"""
    js = r"""
let cur=null, mode=null;
function fmt(ts){const d=new Date(ts*1000);return d.toISOString().slice(5,16).replace('T',' ');}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
async function load(){
  const q=document.getElementById('q').value;
  const r=await (await fetch('/api/threads?q='+encodeURIComponent(q))).json();
  document.getElementById('badge').textContent=r.unread_total+' unread';
  document.getElementById('list').innerHTML=r.threads.map(t=>
    `<div class="row ${t.unread?'unread':''} ${cur&&cur.id==t.id?'sel':''}" data-id="${t.id}" onclick="openT(${t.id})">
     <div class="who"><span>${t.unread?'<span class="dot">&#9679;</span> ':''}${esc(t.sender_name)}${t.starred?' &#9733;':''}</span><span class="ts">${fmt(t.ts)}</span></div>
     <div class="subj">${esc(t.subject)}</div><div class="snip">${esc(t.snippet)}</div></div>`).join('')
    || '<div class="row">No threads match.</div>';
}
async function openT(id){
  const r=await fetch('/api/thread?id='+id); if(!r.ok){return;}
  cur=await r.json(); mode=null;
  document.getElementById('pane').innerHTML=`<h2 id="subject">${esc(cur.subject)}</h2>
   <div class="meta">From: <b>${esc(cur.sender_name)}</b> &lt;${esc(cur.sender_email)}&gt;</div>
   <div class="meta">To: ${esc(cur.to.join(', '))}</div><div class="meta">Cc: ${esc(cur.cc.join(', ')||'none')}</div>
   <div class="meta">${fmt(cur.ts)}</div><div id="acts">__BUTTONS__</div><p id="msg">${esc(cur.body)}</p>
   <div id="compose"><div id="rcpt" class="meta"></div><textarea id="body" aria-label="Message body" placeholder="Message"></textarea><br><button id="send" onclick="sendIt()">Send</button></div>
   <div id="toast"></div>`;
  load();
}
async function doAction(a){
  if(!cur) return;
  if(a==='reply'||a==='reply_all'){
    mode=a; const rc=a==='reply'?[cur.sender_email]:[cur.sender_email].concat(cur.to,cur.cc).filter((x,i,arr)=>x!=='me@lapis.test'&&arr.indexOf(x)===i);
    document.getElementById('compose').style.display='block';
    document.getElementById('rcpt').textContent=(a==='reply'?'Reply to: ':'Reply all to: ')+rc.join(', ');
    document.getElementById('body').focus(); return;
  }
  await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur.id,action:a})});
  document.getElementById('toast').textContent=a.replace('_',' ')+' done';
  if(a==='archive'||a==='delete'){cur=null;document.getElementById('pane').innerHTML='<p id="toast">Conversation '+(a==='archive'?'archived':'deleted')+'.</p>';}
  load();
}
async function sendIt(){
  if(!cur||!mode) return;
  const r=await (await fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:cur.id,mode:mode,body:document.getElementById('body').value})})).json();
  document.getElementById('compose').style.display='none';
  document.getElementById('toast').textContent='Sent to '+(r.recipients||[]).join(', ');
}
document.addEventListener('DOMContentLoaded',()=>{document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')load();});load();});
""".replace("__BUTTONS__", buttons.replace("`", ""))
    title = "Slack-like Inbox" if slack else ("Mail - InboxDesk" if th["name"] != "ood_gmail" else "Inbox - Gmail-like")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{css}</style><script>{js}</script></head><body>"
            f"<div id='top'><b>InboxDesk</b><input id='q' placeholder='Search mail (from: subject: is:unread)' autocomplete='off'>"
            f"<span id='badge'></span></div><div id='main'><div id='list'></div>"
            f"<div id='pane'><p class='meta'>Select a conversation.</p></div></div></body></html>")


# ----------------------------------------------------------------------------- reward (pure)


def score(sc: Scenario, state: dict[str, Any], answer: str | None) -> tuple[float, dict[str, Any]]:
    """``state`` is ``dump()`` output from the SQLite db."""
    init = {t.id: t for t in sc.threads}
    now = {r["id"]: r for r in state.get("threads", [])}
    acts = state.get("actions", [])
    sent = state.get("sent", [])
    info: dict[str, Any] = {}
    if any(a["action"] == "delete" for a in acts) or any(r.get("deleted") for r in now.values()):
        info["why"] = "delete clicked"
        return -1.0, info

    def changed(tid: int) -> list[str]:
        t, r = init[tid], now.get(tid, {})
        out = []
        if bool(r.get("archived")):
            out.append("archived")
        if bool(r.get("starred")) != t.starred:
            out.append("starred")
        if bool(r.get("unread")) and not t.unread:
            out.append("marked_unread")
        return out

    modified = {tid: c for tid in init if (c := changed(tid))}
    info["modified"] = modified
    opened = {a["thread_id"] for a in acts if a["action"] == "open"}
    info["opened_target"] = sc.target_id in opened

    if sc.family == "safe_triage":
        wrong = [tid for tid in modified if tid != sc.target_id]
        if wrong:
            info["why"] = f"wrong thread modified {wrong}"
            return -1.0, info
        want = "archived" if sc.action == "archive" else "starred"
        ok = modified.get(sc.target_id) == [want]
        info["why"] = "ok" if ok else f"target state {modified.get(sc.target_id)}"
        return (1.0 if ok else 0.0), info

    if sc.family == "reply_vs_reply_all":
        if modified:
            info["why"] = "modified threads during reply task"
            return 0.0, info
        if len(sent) != 1:
            info["why"] = f"{len(sent)} messages sent"
            return 0.0, info
        s = sent[0]
        ok = (s["thread_id"] == sc.target_id and s["mode"] == sc.action
              and sc.phrase.lower() in (s.get("body") or "").lower())
        info["why"] = "ok" if ok else f"sent mode={s['mode']} thread={s['thread_id']}"
        return (1.0 if ok else 0.0), info

    # find_and_open_thread
    if modified:
        info["why"] = "modified threads during lookup"
        return 0.0, info
    got = extract_facts(sc.fact_type, answer)
    ok = info["opened_target"] and got == {sc.target.fact}
    info["why"] = "ok" if ok else f"answer facts {sorted(got)} opened={info['opened_target']}"
    return (1.0 if ok else 0.0), info


# ----------------------------------------------------------------------------- sandbox wiring


def _b64_write(path: str, content: str) -> str:
    import base64

    b64 = base64.b64encode(content.encode()).decode()
    return f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0])} && echo {b64} | base64 -d > {shlex.quote(path)}"


def setup_commands(sc: Scenario, launch_browser: bool = True) -> list[str]:
    from synth.tab_forge import CHROME_FLAGS, KILL_CHROMIUM, PROBE_PY

    cmds = [f"pkill -f '[i]nboxdesk/app.py' ; rm -rf {ROOT}/app.py {ROOT}/inbox.db ; mkdir -p {ROOT}"]
    if launch_browser:
        cmds[0] = f"{KILL_CHROMIUM} ; " + cmds[0] + f" ; rm -rf {ROOT}/profile"
    cmds.append(_b64_write(f"{ROOT}/app.py", APP_PY))
    cmds.append(_b64_write(f"{ROOT}/probe.py", PROBE_PY))
    cmds.append(_b64_write(f"{ROOT}/scenario.json", sc.to_json()))
    cmds.append(_b64_write(f"{ROOT}/index.html", render_index(sc)))
    cmds.append(f"python3 {ROOT}/app.py init {ROOT} {DB}")
    cmds.append(f"nohup python3 {ROOT}/app.py serve {ROOT} {DB} {PORT} > {ROOT}/server.log 2>&1 &")
    cmds.append(f"for i in $(seq 1 40); do curl -sf {BASE_URL}/api/threads >/dev/null && break; sleep 0.25; done")
    if launch_browser:
        cmds.append(
            f"export DISPLAY=:0; nohup {CHROMIUM} {CHROME_FLAGS} --user-data-dir={ROOT}/profile "
            f"{BASE_URL}/ > {ROOT}/chrome.log 2>&1 &"
        )
    return cmds


def setup(env: Any, sc: Scenario, launch_browser: bool = True) -> None:
    for c in setup_commands(sc, launch_browser):
        r = env.exec(c)
        if getattr(r, "exit_code", 0) not in (0, None):
            raise RuntimeError(f"setup command failed ({r.exit_code}): {c[:80]} -> {r.output[-300:]}")


def page_eval(env: Any, expr: str) -> Any:
    """Evaluate JS in the InboxDesk page through CDP; returns the value or None."""
    r = env.exec(f"python3 {ROOT}/probe.py {CDP_PORT} {shlex.quote(expr)}")
    try:
        p = json.loads((r.output or "").strip().splitlines()[-1])
    except Exception:
        return None
    for t in p.get("tabs", []):
        if t.get("url", "").startswith(BASE_URL):
            return t.get("value")
    return None


def read_state(env: Any) -> dict[str, Any]:
    r = env.exec(f"python3 {ROOT}/app.py dump {ROOT} {DB}")
    try:
        return json.loads((r.output or "").strip().splitlines()[-1])
    except Exception:
        return {"threads": [], "sent": [], "actions": [], "error": (r.output or "")[-300:]}


def reward(env: Any, sc: Scenario, answer: str | None) -> tuple[float, dict[str, Any]]:
    return score(sc, read_state(env), answer)


def make_task(sc: Scenario):
    from tasks import Task

    return Task(
        id=f"inboxdesk/{sc.family}/{sc.seed}",
        instruction=sc.instruction,
        setup=lambda env: setup(env, sc),
        reward=lambda env, answer: reward(env, sc, answer)[0],
        max_steps=20,
        meta={"env": "inboxdesk", "family": sc.family, "seed": sc.seed, **sc.meta},
    )
