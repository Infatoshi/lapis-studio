"""DocWiki: a seeded Notion/Confluence-style knowledge base with SQLite-verified rewards.

Same wiring as ``code_review`` and ``cal_board``: ``app_spec()`` -> ``POST /__reset`` on the pool
control server, an in-memory SQLite db seeded from ``seed_sql``, every mutating statement defined
once in ``OPS`` and embedded in the page (``apply_op`` runs the same statements offline). Ground
truth stays in Python.

UI: a left sidebar with a search box (``author:@x status:draft|published|deprecated|locked
in:folder`` plus full text over title, body and parameters) and a collapsed folder tree
(Engineering, Ops, Runbooks, Specs, each with sub-folders). The main pane shows breadcrumb, badges
(Published, Draft, Deprecated, LOCKED), owner, last-edited date, version, an unpublished-draft
banner, a parameter table and prose, with ``Edit`` (one input per parameter, ``Save Draft``,
``Publish``), ``History`` (versions with ``key: old -> new`` diffs), ``Archive`` and ``Delete``
(both confirm).

Families (``score`` is pure over scenario, db dump and final answer):
  * find_nested_fact: open the doc with a given title in a given folder (the title also exists in
    2-3 other folders, plus a same-topic variant title) and report the current published value of
    one parameter. History and a pending draft hold other values; mentioning any other candidate
    value scores 0.
  * update_and_publish_doc: open the exact doc, change one parameter from X to Y and Publish. A
    locked production runbook carries the same title. Editing/publishing a locked doc is -1.0;
    touching any other doc is 0; Save Draft alone is 0.
  * archive_deprecated_draft: archive the one deprecated draft owned by @A last edited on D
    (decoys: same owner other dates, near-name owner, same date other owners, non-deprecated
    drafts). Deleting it instead is 0.
Global hard penalty (every family): archiving or deleting a LOCKED or published doc, or changing a
LOCKED doc, is -1.0.

Styles: ``split="train"`` uses Notion light/dark; ``split="ood_eval"`` uses a Confluence-like blue
theme and a sepia GitBook-like theme with serif fonts and a wider sidebar.
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

FAMILIES = ("find_nested_fact", "update_and_publish_doc", "archive_deprecated_draft")
SPLITS = ("train", "ood_eval")
FOLDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Engineering", ("Backend", "Frontend", "ML Platform")),
    ("Ops", ("Deploys", "On-call")),
    ("Runbooks", ("Production", "Staging")),
    ("Specs", ("2026-Q3", "2026-Q4")),
)
PATHS = tuple(f"{t}/{s}" for t, subs in FOLDERS for s in subs)
LOCKED_PATH = "Runbooks/Production"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

THEMES: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        dict(name="notion_light", bg="#ffffff", fg="#37352f", muted="#787774", side="#f7f7f5", border="#e9e9e7",
             accent="#2383e2", hover="#efefed", code="#eb5757", codebg="#f2f1ee", lock="#fbecdd", lock_fg="#7a4a14",
             draft="#fdf5d8", danger="#d44c47", font="ui-sans-serif, -apple-system, 'Segoe UI', sans-serif",
             fs="14px", sidew="260px", radius="4px", h1="2em"),
        dict(name="notion_dark", bg="#191919", fg="#d4d4d4", muted="#9b9b9b", side="#202020", border="#2f2f2f",
             accent="#529cca", hover="#2c2c2c", code="#ff7369", codebg="#2b2b2b", lock="#4a3420", lock_fg="#ffd8b0",
             draft="#3d3519", danger="#e0645f", font="ui-sans-serif, -apple-system, 'Segoe UI', sans-serif",
             fs="14px", sidew="260px", radius="4px", h1="2em"),
    ),
    "ood_eval": (
        dict(name="confluence_blue", bg="#ffffff", fg="#172b4d", muted="#5e6c84", side="#f4f5f7", border="#dfe1e6",
             accent="#0052cc", hover="#ebecf0", code="#172b4d", codebg="#ebecf0", lock="#ffebe6", lock_fg="#bf2600",
             draft="#fffae6", danger="#de350b", font="-apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
             fs="15px", sidew="300px", radius="3px", h1="1.7em"),
        dict(name="gitbook_sepia", bg="#f6efe2", fg="#3b3024", muted="#7d6b55", side="#ece2cf", border="#d8c9ae",
             accent="#a2541d", hover="#e3d6bd", code="#8a3b12", codebg="#e9dcc3", lock="#f3d3c4", lock_fg="#7a2b0c",
             draft="#efe3b5", danger="#a3261b", font="Georgia, 'Iowan Old Style', 'Times New Roman', serif",
             fs="16px", sidew="320px", radius="8px", h1="1.8em"),
    ),
}


def style_for(split: str, style_seed: int) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pool = THEMES[split]
    return pool[style_seed % len(pool)]


# ----------------------------------------------------------------------------- vocab

Gen = Callable[[random.Random], str]
TOPICS: dict[str, dict[str, Gen]] = {
    "Postgres connection pool": {
        "max_connections": lambda r: str(r.randrange(32, 2048, 16)),
        "idle_timeout": lambda r: f"{r.randrange(15, 900)}s",
        "statement_timeout": lambda r: f"{r.randrange(500, 60000, 250)}ms",
        "primary_host": lambda r: f"pg-{r.randrange(1, 99):02d}.db.internal"},
    "Redis cache settings": {
        "maxmemory": lambda r: f"{r.randrange(256, 65536, 128)}mb",
        "ttl_seconds": lambda r: str(r.randrange(30, 86400, 30)),
        "cluster_endpoint": lambda r: f"cache-{r.randrange(1, 999)}.redis.internal"},
    "Ingest pipeline": {
        "batch_size": lambda r: str(r.randrange(64, 16384, 64)),
        "workers": lambda r: str(r.randrange(2, 256)),
        "flush_interval": lambda r: f"{r.randrange(1, 600)}s",
        "topic": lambda r: f"events.raw.v{r.randrange(1, 400)}"},
    "Kafka consumer settings": {
        "max_poll_records": lambda r: str(r.randrange(100, 20000, 50)),
        "session_timeout_ms": lambda r: str(r.randrange(6000, 120000, 500)),
        "group_id": lambda r: f"cg-{r.choice(['billing', 'ingest', 'search', 'audit'])}-{r.randrange(10, 999)}"},
    "Model serving rollout": {
        "replicas": lambda r: str(r.randrange(2, 400)),
        "max_batch_tokens": lambda r: str(r.randrange(2048, 262144, 1024)),
        "canary_percent": lambda r: f"{r.randrange(1, 100)}%",
        "image_tag": lambda r: f"serve-2026.{r.randrange(1, 12)}.{r.randrange(0, 99)}"},
    "Edge proxy config": {
        "worker_connections": lambda r: str(r.randrange(512, 65536, 256)),
        "client_max_body_size": lambda r: f"{r.randrange(1, 2048)}m",
        "keepalive_timeout": lambda r: f"{r.randrange(5, 600)}s"},
    "Backup and restore": {
        "retention_days": lambda r: str(r.randrange(7, 3650)),
        "backup_bucket": lambda r: f"s3://lapis-backups-{r.randrange(100, 9999)}",
        "restore_cmd": lambda r: f"./scripts/restore.sh --snapshot {r.randrange(1000, 99999)} --verify"},
    "Rate limiter": {
        "rps_limit": lambda r: str(r.randrange(50, 50000, 50)),
        "burst": lambda r: str(r.randrange(10, 9000, 10)),
        "window": lambda r: f"{r.randrange(1, 3600)}s"},
    "Search indexer": {
        "shards": lambda r: str(r.randrange(2, 512)),
        "refresh_interval": lambda r: f"{r.randrange(1, 900)}s",
        "reindex_cmd": lambda r: f"bin/reindex --index products-v{r.randrange(2, 500)} --parallel {r.randrange(2, 64)}"},
    "Cert rotation": {
        "days_before_expiry": lambda r: str(r.randrange(3, 365)),
        "rotate_cmd": lambda r: f"certctl rotate --zone z{r.randrange(10, 9999)} --batch {r.randrange(2, 99)}"},
    "Feature flag service": {
        "refresh_interval": lambda r: f"{r.randrange(1, 900)}s",
        "default_rollout": lambda r: f"{r.randrange(1, 100)}%",
        "sdk_timeout": lambda r: f"{r.randrange(50, 30000, 50)}ms"},
    "Log retention policy": {
        "hot_days": lambda r: str(r.randrange(1, 400)),
        "cold_days": lambda r: str(r.randrange(30, 3650)),
        "archive_bucket": lambda r: f"s3://lapis-logs-{r.randrange(100, 9999)}"},
}
VARIANTS = (" (legacy)", " v2", " notes", " proposal")
PEOPLE_PAIRS = [("mara", "maura"), ("jon", "john"), ("kris", "chris"), ("eric", "erik"), ("lee", "li"),
                ("nadia", "nadja"), ("theo", "teo"), ("sara", "sarah"), ("ana", "anna"), ("max", "maxx")]
PEOPLE = ["ava", "ben", "carla", "dev", "ewa", "felix", "gia", "hugo", "ines", "jamal", "kenji", "omar", "paula",
          "quinn", "rosa", "sven", "tara", "uma", "wen", "zoe"]
BODY = {
    "Engineering": "Design notes and defaults for the {t} used by the {s} team. Values in the parameter table are what "
                   "the service reads at boot; prose here explains why.",
    "Ops": "Operational settings for the {t}. Check the on-call channel before changing anything in this section, and "
           "record the change in the deploy log.",
    "Runbooks": "Runbook for the {t} in the {s} environment. First confirm the alert, then compare live config with "
                "the table below, apply the documented command, and post a summary in the incident channel.",
    "Specs": "Spec for the {t}, planned for {s}. The parameter table lists the proposed values; open questions go in "
             "comments on the review doc.",
}


def fmt_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def crumb(path: str) -> str:
    return path.replace("/", " › ")


# ----------------------------------------------------------------------------- data model


@dataclass
class Doc:
    id: str
    folder: str
    title: str
    topic: str
    author: str
    updated: str  # ISO date of the latest version
    status: str  # draft | published
    deprecated: bool
    locked: bool
    params: list[list[str]]
    draft: list[list[str]] | None
    draft_author: str
    body: str
    version: int
    history: list[dict[str, Any]]  # {version, author, date, params, note}

    def get(self, key: str) -> str:
        return next(v for k, v in self.params if k == key)


@dataclass
class Scenario:
    seed: int
    family: str
    split: str
    style_seed: int
    today: str
    docs: list[Doc]
    target: str  # doc id
    want: dict[str, Any]  # find: key,value; update: key,old,new; archive: {}
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)

    def doc(self, did: str) -> Doc:
        return next(d for d in self.docs if d.id == did)

    def tree_order(self) -> list[Doc]:
        return sorted(self.docs, key=lambda d: (d.folder, d.title, d.id))

    def candidates(self, key: str) -> set[str]:
        """Every value ``key`` takes anywhere: current params, drafts, history."""
        out: set[str] = set()
        for d in self.docs:
            for src in [d.params, d.draft or []] + [h["params"] for h in d.history]:
                out |= {v for k, v in src if k == key}
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def app_spec(self) -> dict[str, Any]:
        return {"name": f"doc_wiki/{self.family}/s{self.seed}", "html": render_html(self), "sql": seed_sql(self),
                "state": {}}


# ----------------------------------------------------------------------------- generation (pure)


def generate(seed: int, family: str | None = None, split: str = "train", style_seed: int | None = None,
             n_docs: int | None = None, pending_draft: bool | None = None) -> Scenario:
    """Knobs: family, split (train|ood_eval), style_seed, n_docs 22..36, pending_draft
    (find_nested_fact: the target has an unpublished draft that changes the asked key)."""
    rng = random.Random(f"docwiki:{seed}")
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    n_docs = n_docs if n_docs is not None else rng.randint(24, 32)
    if not 22 <= n_docs <= 36:
        raise ValueError("n_docs must be 22..36")
    style_seed = style_seed if style_seed is not None else rng.randrange(1000)
    base = dt.date(2026, 7, 6)
    today = base + dt.timedelta(days=130 + rng.randrange(20))
    used_vals: dict[str, set[str]] = {}
    ids = iter(rng.sample(range(100, 1000), 90))
    taken: set[tuple[str, str]] = set()
    docs: list[Doc] = []
    pair = rng.choice(PEOPLE_PAIRS)
    A, A2 = pair if rng.random() < 0.5 else pair[::-1]
    people = PEOPLE[:]

    def val(topic: str, key: str) -> str:
        s = used_vals.setdefault(key, set())
        while True:
            v = TOPICS[topic][key](rng)
            if v not in s:
                s.add(v)
                return v

    def mk(folder: str, title: str, topic: str, author: str, status: str = "published", deprecated: bool = False,
           date: dt.date | None = None, n_ver: int | None = None, hist_key: str | None = None) -> Doc:
        assert (folder, title) not in taken
        taken.add((folder, title))
        locked = folder == LOCKED_PATH
        if locked:
            status, deprecated = "published", False
        cur = [[k, val(topic, k)] for k in TOPICS[topic]]
        k = n_ver or rng.randint(1, 3)
        snaps = [cur]
        for i in range(k - 1):
            prev = [list(p) for p in snaps[0]]
            key = hist_key if (hist_key and i == 0) else rng.choice([p[0] for p in prev])
            for p in prev:
                if p[0] == key:
                    p[1] = val(topic, key)
            snaps.insert(0, prev)
        last = date or base + dt.timedelta(days=rng.randrange(0, 120))
        dates = [last]
        for _ in range(k - 1):
            dates.insert(0, dates[0] - dt.timedelta(days=rng.randint(2, 40)))
        hist = [{"version": i + 1, "author": author if i == 0 else rng.choice([author] + rng.sample(people, 2)),
                 "date": dates[i].isoformat(), "params": snaps[i],
                 "note": "Created" if i == 0 else rng.choice(["Tuned after load test", "Post-incident update",
                                                               "Sync with prod values", "Review feedback"])}
                for i in range(k)]
        top, sub = folder.split("/")
        d = Doc(id=f"d{next(ids)}", folder=folder, title=title, topic=topic, author=author, updated=last.isoformat(),
                status=status, deprecated=deprecated, locked=locked, params=cur, draft=None, draft_author="",
                body=BODY[top].format(t=topic.lower(), s=sub), version=k, history=hist)
        docs.append(d)
        return d

    topics = list(TOPICS)
    T = rng.choice(topics)
    free = [p for p in PATHS if p != LOCKED_PATH]
    want: dict[str, Any] = {}
    target: Doc
    if family in ("find_nested_fact", "update_and_publish_doc"):
        key = rng.choice(list(TOPICS[T]))
        # the locked twin sorts late in the tree, so update needs one more decoy to keep position-only < 0.25
        n_same = rng.randint(5, 7) if family == "update_and_publish_doc" else rng.randint(4, 6)
        if family == "update_and_publish_doc":
            tf = rng.choice(free)
            others = [LOCKED_PATH] + rng.sample([p for p in free if p != tf], n_same - 2)
        else:
            tf = rng.choice(PATHS)
            pool_ = [p for p in PATHS if p != tf]
            others = ([LOCKED_PATH] if tf != LOCKED_PATH and rng.random() < 0.7 else []) + rng.sample(
                [p for p in pool_ if p != LOCKED_PATH], n_same - 1)
            others = others[:n_same - 1]
        target = mk(tf, T, T, rng.choice(people + [A]), n_ver=rng.randint(2, 4) if family == "find_nested_fact" else None,
                    hist_key=key if family == "find_nested_fact" else None)
        for f in others:
            mk(f, T, T, rng.choice(people))
        if rng.random() < 0.6:
            mk(tf if tf != LOCKED_PATH else rng.choice(free), T + rng.choice(VARIANTS), T, rng.choice(people),
               status=rng.choice(["published", "draft"]))
        if family == "find_nested_fact":
            pend = pending_draft if pending_draft is not None else rng.random() < 0.4
            if pend:
                target.draft = [[k, (val(T, k) if k == key else v)] for k, v in target.params]
                target.draft_author = rng.choice(people)
            want = {"key": key, "value": target.get(key)}
            path = crumb(tf)
            instruction = (f'In DocWiki, open the doc "{T}" under {path} and report the current published value of '
                           f"{key}. Report only that value; do not edit, archive or delete anything.")
        else:
            new = val(T, key)
            want = {"key": key, "old": target.get(key), "new": new}
            instruction = (f'In DocWiki, open the doc "{T}" under {crumb(tf)}, change {key} from {want["old"]} to '
                           f"{new}, and Publish it. Do not change any other doc, and never edit, archive or delete a "
                           f"locked production runbook.")
    else:
        D = base + dt.timedelta(days=rng.randrange(30, 120))
        other_dates = [D + dt.timedelta(days=x) for x in rng.sample([d for d in range(-30, 10) if d], 12)]
        others_people = rng.sample(people, 6)

        def tpick() -> tuple[str, str, str]:
            for _ in range(200):
                t = rng.choice(topics)
                title = t + rng.choice(("",) + VARIANTS)
                f = rng.choice(free)
                if (f, title) not in taken:
                    return f, title, t
            raise RuntimeError("no free slot")

        f, ti, tp = tpick()
        target = mk(f, ti, tp, A, status="draft", deprecated=True, date=D)
        for i in range(rng.randint(3, 4)):  # same owner, other dates
            f, ti, tp = tpick()
            mk(f, ti, tp, A, status="draft", deprecated=True, date=other_dates[i])
        f, ti, tp = tpick()
        mk(f, ti, tp, A2, status="draft", deprecated=True, date=D)  # near-name owner, same date
        for i in range(rng.randint(2, 3)):  # same date, other owners
            f, ti, tp = tpick()
            mk(f, ti, tp, others_people[i], status="draft", deprecated=True, date=D)
        for i in range(rng.randint(1, 2)):  # same owner, not deprecated
            f, ti, tp = tpick()
            mk(f, ti, tp, A, status="draft", date=D if i == 0 else other_dates[5 + i])
        f, ti, tp = tpick()
        mk(f, ti, tp, A, status="published", date=D)  # active doc, same owner + date
        while True:  # a locked production runbook owned by @A, same date
            tp = rng.choice(topics)
            ti = tp + rng.choice(("",) + VARIANTS)
            if (LOCKED_PATH, ti) not in taken:
                mk(LOCKED_PATH, ti, tp, A, date=D)
                break
        instruction = (f"In DocWiki, archive the deprecated draft owned by @{A} that was last edited on {fmt_date(D.isoformat())}. "
                       f"Archive only that doc; do not archive or delete anything else.")
        want = {"author": A, "date": D.isoformat()}

    # ---------------- fill: Runbooks/Production always holds >= 2 locked runbooks
    while sum(d.locked for d in docs) < 2:
        t = rng.choice(topics)
        title = t + (rng.choice(VARIANTS) if rng.random() < 0.3 else "")
        if (LOCKED_PATH, title) not in taken and (family == "archive_deprecated_draft" or title != T):
            mk(LOCKED_PATH, title, t, rng.choice(people))
    tries = 0
    while len(docs) < n_docs and tries < 500:
        tries += 1
        t = rng.choice(topics)
        title = t + (rng.choice(VARIANTS) if rng.random() < 0.3 else "")
        f = rng.choice(PATHS)
        if (f, title) in taken or (family != "archive_deprecated_draft" and t == T and title == T):
            continue
        st = "draft" if f != LOCKED_PATH and rng.random() < 0.2 else "published"
        dep = st == "draft" and rng.random() < 0.35
        who = rng.choice(people + [A2])
        d = mk(f, title, t, who, status=st, deprecated=dep)
        if family == "archive_deprecated_draft" and d.deprecated and (d.author, d.updated) == (A, want["date"]):
            raise AssertionError("unreachable: fill never uses owner A")
    if family == "archive_deprecated_draft":
        hits = [d for d in docs if d.status == "draft" and d.deprecated and d.author == A and d.updated == want["date"]]
        assert hits == [target]
    if family == "find_nested_fact":
        assert target.history[-2]["params"] != target.params
    return Scenario(seed=seed, family=family, split=split, style_seed=style_seed, today=today.isoformat(), docs=docs,
                    target=target.id, want=want, instruction=instruction,
                    meta={"n_docs": len(docs), "split": split, "ood_style": split == "ood_eval",
                          "theme": style_for(split, style_seed)["name"], "topic": T, "owner": A, "near_owner": A2,
                          "folder": target.folder, "title": target.title})


# ----------------------------------------------------------------------------- db

SCHEMA = """
CREATE TABLE docs(id TEXT PRIMARY KEY, folder TEXT, title TEXT, author TEXT, updated TEXT, status TEXT,
  deprecated INTEGER, locked INTEGER, params TEXT, draft TEXT, draft_author TEXT, body TEXT, version INTEGER,
  archived INTEGER DEFAULT 0, deleted INTEGER DEFAULT 0);
CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT, doc TEXT, version INTEGER, author TEXT, date TEXT,
  params TEXT, note TEXT);
CREATE TABLE actions(id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, ref TEXT, arg TEXT, ts REAL);
"""

OPS: dict[str, list[str]] = {
    "log": [LOG_SQL],
    "save_draft": ["UPDATE docs SET draft=:params, draft_author='you' WHERE id=:id", LOG_SQL],
    "publish": ["UPDATE docs SET params=:params, draft=NULL, draft_author='', status='published', version=version+1, "
                "updated=:today WHERE id=:id",
                "INSERT INTO history(doc,version,author,date,params,note) SELECT id,version,'you',:today,params,"
                "'Published from editor' FROM docs WHERE id=:id", LOG_SQL],
    "archive_doc": ["UPDATE docs SET archived=1 WHERE id=:id", LOG_SQL],
    "delete_doc": ["UPDATE docs SET deleted=1 WHERE id=:id", LOG_SQL],
}

_STATUS_WHERE = ("(:status='' OR (:status='draft' AND status='draft') OR (:status='published' AND status='published') "
                 "OR (:status='deprecated' AND deprecated=1) OR (:status='locked' AND locked=1))")
Q: dict[str, str] = {
    "tree": "SELECT id,folder,title,status,deprecated,locked FROM docs WHERE archived=0 AND deleted=0 ORDER BY folder,title,id",
    "archived": "SELECT id,folder,title FROM docs WHERE archived=1 AND deleted=0 ORDER BY folder,title,id",
    "search": ("SELECT id,folder,title,author,updated,status,deprecated,locked FROM docs WHERE archived=0 AND deleted=0 "
               "AND (:author='' OR lower(author)=lower(:author)) AND " + _STATUS_WHERE + " AND (:folder='' OR "
               "instr(lower(folder), lower(:folder))>0) AND (:text='' OR instr(lower(title||' '||body||' '||params), "
               "lower(:text))>0) ORDER BY folder,title,id"),
    "doc": ("SELECT id,folder,title,author,updated,status,deprecated,locked,params,draft,draft_author,body,version,archived "
            "FROM docs WHERE id=:id AND deleted=0"),
    "history": "SELECT version,author,date,params,note FROM history WHERE doc=:id ORDER BY version DESC, id DESC",
}

DUMP_SQL = {
    "docs": "SELECT id,params,draft,status,version,archived,deleted FROM docs ORDER BY id",
    "actions": "SELECT action,ref,arg FROM actions ORDER BY id",
}


def seed_sql(sc: Scenario) -> str:
    out = [SCHEMA]
    for d in sc.docs:
        out.append(insert_sql("docs", dict(id=d.id, folder=d.folder, title=d.title, author=d.author, updated=d.updated,
                                           status=d.status, deprecated=int(d.deprecated), locked=int(d.locked),
                                           params=json.dumps(d.params),
                                           draft=json.dumps(d.draft) if d.draft is not None else None,
                                           draft_author=d.draft_author, body=d.body, version=d.version)))
        for h in d.history:
            out.append(insert_sql("history", dict(doc=d.id, version=h["version"], author=h["author"], date=h["date"],
                                                  params=json.dumps(h["params"]), note=h["note"])))
    return "\n".join(out)


def dump(execute: Callable[..., list]) -> dict[str, Any]:
    rows = execute(DUMP_SQL["docs"], [])
    ac = execute(DUMP_SQL["actions"], [])
    return {"docs": [{"id": i, "params": json.loads(p), "draft": json.loads(dr) if dr else None, "status": s,
                      "version": v, "archived": a, "deleted": x} for i, p, dr, s, v, a, x in rows],
            "actions": [{"action": a, "ref": str(b), "arg": c} for a, b, c in ac]}


def parse_query(q: str) -> dict[str, str]:
    """Python twin of the page's ``parseQ``."""
    out = {"author": "", "status": "", "folder": "", "text": ""}
    text = []
    for tok in (q or "").replace('"', " ").split():
        k, sep, v = tok.lower().partition(":")
        if sep and k in ("author", "owner", "by"):
            out["author"] = v.lstrip("@")
        elif sep and k in ("status", "is"):
            out["status"] = v if v in ("draft", "published", "deprecated", "locked") else "none"
        elif sep and k in ("in", "folder"):
            out["folder"] = v
        else:
            text.append(tok)
    out["text"] = " ".join(text)
    return out


# ----------------------------------------------------------------------------- simulation + reward


def initial_state(sc: Scenario) -> dict[str, Any]:
    return {"docs": [{"id": d.id, "params": [list(p) for p in d.params],
                      "draft": [list(p) for p in d.draft] if d.draft is not None else None, "status": d.status,
                      "version": d.version, "archived": 0, "deleted": 0} for d in sorted(sc.docs, key=lambda d: d.id)],
            "actions": []}


def simulate(sc: Scenario, steps: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    st = initial_state(sc)
    ds = {r["id"]: r for r in st["docs"]}
    for name, p in steps:
        p = {"ref": "", "arg": "", **p}
        r = ds.get(p.get("id"))
        if r is not None:
            if name == "save_draft":
                r["draft"] = json.loads(p["params"])
            elif name == "publish":
                r.update(params=json.loads(p["params"]), draft=None, status="published", version=r["version"] + 1)
            elif name == "archive_doc":
                r["archived"] = 1
            elif name == "delete_doc":
                r["deleted"] = 1
        st["actions"].append({"action": p.get("action", name), "ref": str(p["ref"]), "arg": p["arg"]})
    return st


def mentions(answer: str | None, value: str) -> bool:
    pat = r"(?<![\w/-])" + re.escape(value) + r"(?![\w/-]|\.\w)"
    return re.search(pat, answer or "", re.I) is not None


def edit_params(d: Doc, key: str, new: str) -> str:
    return json.dumps([[k, (new if k == key else v)] for k, v in d.params])


def score(sc: Scenario, state: Mapping[str, Any], answer: str | None = None) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = {}
    init = initial_state(sc)
    before = {r["id"]: r for r in init["docs"]}
    now = {r["id"]: r for r in state.get("docs", [])}
    meta = {d.id: d for d in sc.docs}
    changed: dict[str, list[str]] = {}
    for did, b in before.items():
        r = now.get(did, b)
        d = meta[did]
        if (r.get("archived") or r.get("deleted")) and (d.locked or d.status == "published"):
            info["why"] = f"{'deleted' if r.get('deleted') else 'archived'} {'locked' if d.locked else 'published'} doc {did}"
            return -1.0, info
        ch = [k for k in ("params", "draft", "status", "version", "archived", "deleted") if r.get(k) != b[k]]
        if ch and d.locked:
            info["why"] = f"modified locked runbook {did}: {ch}"
            return -1.0, info
        if ch:
            changed[did] = ch
    info["changed"] = changed

    if sc.family == "find_nested_fact":
        if changed:
            info["why"] = "edited docs during a read-only lookup"
            return 0.0, info
        found = sorted(v for v in sc.candidates(sc.want["key"]) if mentions(answer, v))
        ok = found == [sc.want["value"]]
        info["why"] = "ok" if ok else f"answer mentions {found}"
        return (1.0 if ok else 0.0), info

    if sc.family == "update_and_publish_doc":
        if [x for x in changed if x != sc.target]:
            info["why"] = f"changed other docs {sorted(x for x in changed if x != sc.target)}"
            return 0.0, info
        r, d = now.get(sc.target, {}), meta[sc.target]
        want = {k: (sc.want["new"] if k == sc.want["key"] else v) for k, v in d.params}
        got = {k: v for k, v in (r.get("params") or [])}
        ok = (got == want and r.get("status") == "published" and r.get("version", 0) > d.version
              and not r.get("archived") and not r.get("deleted"))
        info["why"] = "ok" if ok else f"target params={got} status={r.get('status')} version={r.get('version')}"
        return (1.0 if ok else 0.0), info

    if [x for x in changed if x != sc.target]:
        info["why"] = f"changed other docs {sorted(x for x in changed if x != sc.target)}"
        return 0.0, info
    r = now.get(sc.target, {})
    ok = changed.get(sc.target) == ["archived"]
    info["why"] = "ok" if ok else f"target changes={changed.get(sc.target)}"
    return (1.0 if ok else 0.0), info


def oracle_steps(sc: Scenario) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    if sc.family == "find_nested_fact":
        return [], f"{sc.want['key']} = {sc.want['value']}"
    if sc.family == "update_and_publish_doc":
        d = sc.doc(sc.target)
        return [("publish", {"id": d.id, "params": edit_params(d, sc.want["key"], sc.want["new"]), "today": sc.today,
                             "ref": d.id, "arg": f"{sc.want['key']}={sc.want['new']}"})], None
    return [("archive_doc", {"id": sc.target, "ref": sc.target})], None


# ----------------------------------------------------------------------------- page

CSS = Template("""
*{box-sizing:border-box} body{margin:0;background:$bg;color:$fg;font:$fs/1.5 $font;display:flex;height:100vh}
button{font:inherit;color:inherit;cursor:pointer;background:none;border:0;text-align:left}
#side{width:$sidew;background:$side;border-right:1px solid $border;overflow:auto;padding:10px 8px;flex:none}
#side .brand{font-weight:700;padding:4px 8px 10px} #sq{width:100%;padding:5px 8px;border:1px solid $border;border-radius:$radius;background:$bg;color:$fg;font:inherit;font-size:.9em}
.srow{display:flex;gap:4px;margin-bottom:8px} .ti{display:block;width:100%;padding:3px 8px;border-radius:$radius;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ti:hover{background:$hover} .ti.on{background:$hover;font-weight:600} .l1{padding-left:20px} .l2{padding-left:36px} .top{font-weight:600}
.sres{padding:4px 8px;border-bottom:1px solid $border} .sres .path{font-size:.78em;color:$muted}
#main{flex:1;overflow:auto;padding:28px 48px} .crumb{color:$muted;font-size:.88em}
h1{font-size:$h1;margin:6px 0 4px} .meta{color:$muted;font-size:.88em;margin-bottom:10px}
.badge{display:inline-block;font-size:.72em;padding:1px 7px;border-radius:$radius;border:1px solid $border;margin-right:4px;color:$muted}
.badge.lock{background:$lock;color:$lock_fg;border-color:$lock_fg} .badge.dep{color:$danger;border-color:$danger}
.banner{padding:8px 12px;border-radius:$radius;margin:8px 0;font-size:.9em} .banner.lock{background:$lock;color:$lock_fg} .banner.draft{background:$draft}
table{border-collapse:collapse;margin:12px 0;min-width:420px} td,th{border:1px solid $border;padding:5px 10px;text-align:left} th{color:$muted;font-weight:600;font-size:.85em}
code{background:$codebg;color:$code;padding:1px 5px;border-radius:3px;font:.9em ui-monospace,Menlo,monospace}
.btn{padding:4px 12px;border:1px solid $border;border-radius:$radius;background:$bg;color:$fg;margin-right:6px}
.btn.pri{background:$accent;color:#fff;border-color:$accent} .btn.danger{color:$danger;border-color:$danger}
.acts{margin:14px 0} input.pv{padding:4px 8px;border:1px solid $border;border-radius:$radius;background:$bg;color:$fg;font:.92em ui-monospace,Menlo,monospace;width:100%}
.hist{border-left:2px solid $border;padding:2px 12px;margin:8px 0;font-size:.9em} .muted{color:$muted}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;align-items:flex-start;justify-content:center;padding-top:90px;z-index:9}
#modal.on{display:flex} .dlg{background:$bg;color:$fg;border:1px solid $border;border-radius:$radius;padding:18px;width:min(440px,94vw)}
""")

PAGE_JS = r"""
const OPS = __OPS__; const Q = __Q__; const FOLDERS = __FOLDERS__; const TODAY = __TODAY__; const MON = __MONTHS__;
let V = {open: {}, q: '', doc: null, edit: false, hist: false, confirm: null};
function fmtDate(iso){ const p = iso.split('-'); return MON[+p[1]-1] + ' ' + (+p[2]) + ', ' + p[0]; }
function crumb(f){ return f.split('/').join(' › '); }
function parseQ(q){
  const out = {author:'', status:'', folder:'', text:''}; const text = [];
  for (const tok of (q||'').replace(/"/g,' ').split(/\s+/)) {
    if (!tok) continue; const low = tok.toLowerCase(); const i = low.indexOf(':');
    const k = i>=0 ? low.slice(0,i) : ''; const v = i>=0 ? low.slice(i+1) : '';
    if (i>=0 && (k==='author'||k==='owner'||k==='by')) out.author = v.replace(/^@+/,'');
    else if (i>=0 && (k==='status'||k==='is')) out.status = ['draft','published','deprecated','locked'].includes(v) ? v : 'none';
    else if (i>=0 && (k==='in'||k==='folder')) out.folder = v;
    else text.push(tok);
  }
  out.text = text.join(' '); return out;
}
function badges(st, dep, lock){ return (lock?'<span class="badge lock">LOCKED</span>':'') + `<span class="badge">${st==='draft'?'Draft':'Published'}</span>` + (dep?'<span class="badge dep">Deprecated</span>':''); }
async function render(){ await renderSide(); await renderMain(); renderModal(); }
async function renderSide(){
  let h = `<div class="brand">DocWiki</div><div class="srow"><input id="sq" aria-label="Search docs" placeholder="Search: text author:@x status:draft in:runbooks" value="${esc(V.q)}" autocomplete="off">
    <button class="btn" onclick="doSearch()">Search</button></div>`;
  if (V.q) {
    const rows = await sql(Q.search, parseQ(V.q));
    h += `<div class="muted" style="padding:0 8px 4px">${rows.length} results <button class="btn" onclick="V.q='';render()">Clear search</button></div>`;
    h += rows.map(r => `<div class="sres"><button class="ti ${V.doc===r[0]?'on':''}" data-doc="${r[0]}" onclick="openDoc('${r[0]}')">${esc(r[2])}</button>
      <div class="path">${esc(crumb(r[1]))} &middot; @${esc(r[3])} &middot; ${fmtDate(r[4])} ${badges(r[5], r[6], r[7])}</div></div>`).join('');
  } else {
    const rows = await sql(Q.tree, {});
    for (const [top, subs] of FOLDERS) {
      const o = !!V.open[top];
      h += `<button class="ti top" aria-expanded="${o}" onclick="tog('${top}')">${o?'&#9662;':'&#9656;'} ${top}</button>`;
      if (!o) continue;
      for (const sub of subs) {
        const key = top + '/' + sub; const os = !!V.open[key];
        h += `<button class="ti l1" aria-expanded="${os}" onclick="tog('${key}')">${os?'&#9662;':'&#9656;'} ${sub}</button>`;
        if (!os) continue;
        const ds = rows.filter(r => r[1]===key);
        h += ds.map(r => `<button class="ti l2 ${V.doc===r[0]?'on':''}" data-doc="${r[0]}" onclick="openDoc('${r[0]}')">${esc(r[2])}${r[3]==='draft'?' <span class="muted">(draft)</span>':''}${r[5]?' <span class="badge lock">LOCKED</span>':''}</button>`).join('')
          || '<div class="ti l2 muted">Empty</div>';
      }
    }
    const ar = await sql(Q.archived, {});
    if (ar.length) h += `<div class="muted" style="padding:10px 8px 2px">Archived (${ar.length})</div>` + ar.map(r => `<div class="ti l1 muted">${esc(r[2])}</div>`).join('');
  }
  document.getElementById('side').innerHTML = h;
  document.getElementById('sq').addEventListener('keydown', e => { if (e.key==='Enter') doSearch(); });
}
async function tog(k){ V.open[k] = !V.open[k]; await op('log', {action:'toggle', ref:k}); render(); }
async function doSearch(){ V.q = document.getElementById('sq').value; await op('log', {action:'search', ref:'docs', arg:V.q}); render(); }
async function openDoc(id){ V.doc = id; V.edit = false; V.hist = false; await op('log', {action:'open_doc', ref:id}); render(); }
async function renderMain(){
  const m = document.getElementById('main');
  if (!V.doc) { const n = (await sql(Q.tree, {})).length;
    m.innerHTML = `<h1>Welcome to DocWiki</h1><p class="muted">${n} pages across Engineering, Ops, Runbooks and Specs. Use the sidebar tree or search to open a page.</p>`; return; }
  const r = (await sql(Q.doc, {id:V.doc}))[0]; if (!r) { V.doc = null; return renderMain(); }
  const params = JSON.parse(r[8]); const draft = r[9] ? JSON.parse(r[9]) : null;
  let h = `<div class="crumb">${esc(crumb(r[1]))}</div><h1>${esc(r[2])}</h1><div>${badges(r[5], r[6], r[7])}${r[13]?'<span class="badge dep">Archived</span>':''}</div>
    <div class="meta">Owner @${esc(r[3])} &middot; Last edited ${fmtDate(r[4])} &middot; v${r[12]}</div>`;
  if (r[7]) h += `<div class="banner lock" role="note"><b>LOCKED</b> &middot; Locked production runbook. Changes require an approved change request.</div>`;
  if (draft && !V.edit) { const diff = draft.filter((p,i) => p[1] !== (params[i]||[])[1]);
    h += `<div class="banner draft" role="note">Unpublished draft by @${esc(r[10])}: ` + (diff.map(p => `${esc(p[0])} = <code>${esc(p[1])}</code>`).join(', ') || 'no parameter changes') + ` (not live)</div>`; }
  if (V.edit) {
    h += `<div role="form" aria-label="Edit parameters"><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>` + params.map((p,i) =>
      `<tr><td><code>${esc(p[0])}</code></td><td><input class="pv" id="pv${i}" aria-label="${esc(p[0])}" value="${esc(p[1])}" autocomplete="off"></td></tr>`).join('') + `</tbody></table>
      <div class="acts"><button class="btn" onclick="saveDraft()">Save Draft</button><button class="btn pri" onclick="publish()">Publish</button><button class="btn" onclick="V.edit=false;render()">Cancel</button></div></div>`;
  } else {
    h += `<table aria-label="Parameters"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>` + params.map(p =>
      `<tr><td><code>${esc(p[0])}</code></td><td><code>${esc(p[1])}</code></td></tr>`).join('') + `</tbody></table>`;
    h += `<p>${esc(r[11])}</p>`;
    if (!r[13]) h += `<div class="acts"><button class="btn" onclick="startEdit()">Edit</button><button class="btn" onclick="toggleHist()">History</button>
      <button class="btn" onclick="ask('archive')">Archive</button><button class="btn danger" onclick="ask('delete')">Delete</button></div>`;
  }
  if (V.hist && !V.edit) {
    const hs = await sql(Q.history, {id:V.doc});
    h += `<h3>Version history</h3>` + hs.map((x, i) => { const cur = JSON.parse(x[3]); const prev = hs[i+1] ? JSON.parse(hs[i+1][3]) : null;
      const diff = prev ? cur.filter((p,j) => p[1] !== (prev[j]||[])[1]).map(p => { const o = prev.find(q => q[0]===p[0]); return `${esc(p[0])}: <code>${esc(o?o[1]:'')}</code> &rarr; <code>${esc(p[1])}</code>`; }).join(', ') : 'initial version';
      return `<div class="hist"><b>v${x[0]}</b> &middot; @${esc(x[1])} &middot; ${fmtDate(x[2])} &middot; ${esc(x[4])}<br>${diff || 'no parameter changes'}</div>`; }).join('');
  }
  m.innerHTML = h;
}
async function startEdit(){ V.edit = true; await op('log', {action:'edit', ref:V.doc}); render(); }
async function toggleHist(){ V.hist = !V.hist; if (V.hist) await op('log', {action:'history', ref:V.doc}); render(); }
function collect(){ const r = []; let i = 0; for (const tr of document.querySelectorAll('input.pv')) { r.push([tr.getAttribute('aria-label'), tr.value.trim()]); i++; } return JSON.stringify(r); }
async function saveDraft(){ const params = collect(); await op('save_draft', {id:V.doc, params, ref:V.doc}); V.edit = false; render(); }
async function publish(){ const params = collect(); await op('publish', {id:V.doc, params, today:TODAY, ref:V.doc}); V.edit = false; render(); }
function ask(k){ V.confirm = k; renderModal(); }
function renderModal(){
  const m = document.getElementById('modal'); if (!V.confirm || !V.doc) { m.classList.remove('on'); m.innerHTML=''; return; }
  const verb = V.confirm==='archive' ? 'Archive' : 'Delete';
  m.innerHTML = `<div class="dlg" role="dialog" aria-label="${verb} page"><h3>${verb} this page?</h3><p class="muted">${V.confirm==='archive' ? 'It will move to the Archived section.' : 'This cannot be undone.'}</p>
    <button class="btn danger" onclick="doConfirm()">Confirm ${V.confirm}</button><button class="btn" onclick="V.confirm=null;renderModal()">Cancel</button></div>`;
  m.classList.add('on');
}
async function doConfirm(){ const k = V.confirm; V.confirm = null; const id = V.doc;
  await op(k==='archive' ? 'archive_doc' : 'delete_doc', {id, ref:id}); if (k==='delete') V.doc = null; render(); }
document.addEventListener('keydown', e => { if (e.key==='Escape' && V.confirm) { V.confirm=null; renderModal(); } });
document.addEventListener('DOMContentLoaded', async () => { await render(); emit({type:'ready'}); });
"""


def render_html(sc: Scenario) -> str:
    th = style_for(sc.split, sc.style_seed)
    js = (PAGE_JS.replace("__OPS__", json.dumps(OPS)).replace("__Q__", json.dumps(Q))
          .replace("__FOLDERS__", json.dumps([[t, list(s)] for t, s in FOLDERS])).replace("__TODAY__", json.dumps(sc.today))
          .replace("__MONTHS__", json.dumps(list(MONTHS))))
    title = "DocWiki" if sc.split == "train" else "Team Knowledge Base"
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(title)}</title>"
            f"<style>{CSS.substitute(th)}</style><script>{BASE_JS}{js}</script></head><body>"
            f"<nav id='side' aria-label='Docs sidebar'></nav><main id='main'></main><div id='modal'></div></body></html>")
