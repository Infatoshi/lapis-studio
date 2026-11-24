"""REAL / WebClones (agi-inc/agisdk) as a Studio-contract CUA environment with a frozen app split.

Upstream: agisdk ``src/agisdk/REAL/browsergym/webclones`` (Apache-2.0), staged by
``stage_data()`` under ``~/cua_datasets/webclones/agisdk`` (shallow clone, ~51 MB, pinned to
``AGISDK_COMMIT``). Each task is a JSON config ``{id, goal, website.url, evals[]}``; the clone is a
hosted Next.js SPA that keeps its state in the browser, and ``<url>/finish`` renders the run's
state diff as JSON inside a ``<pre>``. Upstream task flow (base.py ``setup``/``get_finish_json``):
``<url>/config?run_id=0&task_id=<v.id>&latency=0`` -> ``<url>/finish`` -> ``<url>`` (the agent's
page) ... then ``<url>/finish`` again and grade that JSON.

Grader kinds upstream:
  * ``jmespath`` (v1): a JMESPath query over the finish JSON must equal ``expected_value``;
  * ``script``   (v2): ``python eval_<app>_<n>.py <finish.json>`` prints SUCCESS / FAILURE;
  * ``llm_boolean`` (v1): an LLM grades the agent's free-text answer against a rubric.

What this module keeps (``classify``): tasks whose every grader is a pure function of the finish
state. Dropped, with the reason recorded: any ``llm_boolean`` grader; v2 scripts that read their
input file's path / sibling files, the wall clock, randomness or the ``run_id`` (static audit,
``audit_script``: several upstream scripts were fitted to labeled runs and fall back to folder
timestamps or run_id parity); and anything in ``BANNED_TASKS``. Reward is exactly 0 or 1.

Anti-benchmaxing: ``TRAIN_APPS`` and ``HELDOUT_APPS`` are frozen at the app level (a held-out app
is never reset in a train episode, and ``episode_samples`` refuses to emit training samples from a
held-out episode). ``SPLIT_SHA256`` pins the split; tests fail if it is edited.

Live path: ``WebClonesEnv`` subclasses the stage-1 ``StudioEnv`` (rl/overnight_stage1/envs.py), so
observations (1024x768 JPEG + Studio CDP AX lines), actions (``[[ACTION:cua_*]]`` with real X11
input) and the episode loop (``run_episode`` -> ``studio_format.turn_messages``) are byte-for-byte
the ones the trainer and evals already use. The pool's own Chromium navigates to the hosted clone
via CDP; nothing extra runs in the sandbox.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

RL = Path(__file__).resolve().parents[1]
if str(RL) not in sys.path:
    sys.path.insert(0, str(RL))

# --------------------------------------------------------------------------- data location

DATA_ROOT = Path(os.environ.get("WEBCLONES_ROOT", str(Path.home() / "cua_datasets" / "webclones")))
AGISDK_REPO = "https://github.com/agi-inc/agisdk.git"
AGISDK_COMMIT = "97a245dcbccd3e1cf4d5ee00d5b1707942258518"  # main @ 2025-12-05, staged 2026-11-24
JMESPATH_VERSION = "1.0.1"


def webclones_src(root: Path | None = None) -> Path:
    return (root or DATA_ROOT) / "agisdk" / "src" / "agisdk" / "REAL" / "browsergym" / "webclones"


def stage_data(root: Path | None = None) -> dict[str, Any]:
    """Idempotent: shallow-clone agisdk at ``AGISDK_COMMIT`` and install pure-Python jmespath into
    ``<root>/pydeps`` (the rl venv does not carry it). Returns what is on disk."""
    root = root or DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "agisdk"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", AGISDK_REPO, str(repo)], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pydeps = root / "pydeps"
    if not (pydeps / "jmespath").exists():
        subprocess.run(["uv", "pip", "install", "--target", str(pydeps), f"jmespath=={JMESPATH_VERSION}"], check=True)
    return {"root": str(root), "agisdk_head": head, "pinned": head == AGISDK_COMMIT, "pydeps": str(pydeps)}


def _jmespath():
    """jmespath from the venv if present, else the staged ``pydeps`` copy."""
    try:
        import jmespath  # type: ignore

        return jmespath
    except ImportError:
        pass
    pydeps = DATA_ROOT / "pydeps"
    spec = importlib.util.spec_from_file_location("jmespath", pydeps / "jmespath" / "__init__.py",
                                                  submodule_search_locations=[str(pydeps / "jmespath")])
    if spec is None or spec.loader is None:
        raise ImportError(f"jmespath not installed; run stage_data() (looked in {pydeps})")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jmespath"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- the frozen app split

# Every app id that appears in agisdk v1 or v2 task configs.
ALL_APPS = ("dashdish", "fly-unified", "gocalendar", "gomail", "marrisuite", "networkin", "omnizon",
            "opendining", "staynb", "topwork", "udriver", "zilloft")

# 8 train apps. Every family Studio users hit daily keeps a train sibling of each held-out app:
# mail (gomail) for calendar; food/rides/commerce checkout (dashdish, udriver, omnizon) for
# restaurant booking; flights and real-estate listing search (fly-unified, zilloft) for stays search.
TRAIN_APPS = ("dashdish", "fly-unified", "gomail", "networkin", "omnizon", "topwork", "udriver", "zilloft")
# 3 apps never trained on: scheduling (GoCalendar ~ Google Calendar), reservations
# (OpenDining ~ OpenTable), listing search + booking (StayNB ~ Airbnb). Frozen 2026-11-24 after the
# live sweep (Zilloft was first held out, but its v2 deployment is gone, leaving it 2 live tasks).
HELDOUT_APPS = ("gocalendar", "opendining", "staynb")
# Not part of the 11-app REAL release: one v2 task whose grader reads its file path. Never used.
EXCLUDED_APPS = ("marrisuite",)

assert set(TRAIN_APPS).isdisjoint(HELDOUT_APPS), "train / held-out apps overlap"
assert set(EXCLUDED_APPS).isdisjoint(set(TRAIN_APPS) | set(HELDOUT_APPS))
assert sorted(TRAIN_APPS + HELDOUT_APPS + EXCLUDED_APPS) == sorted(ALL_APPS), "every app sits in exactly one bucket"
assert len(TRAIN_APPS) == 8 and len(HELDOUT_APPS) == 3


def split_fingerprint(train: Iterable[str] = TRAIN_APPS, heldout: Iterable[str] = HELDOUT_APPS) -> str:
    blob = "train=" + ",".join(sorted(train)) + ";heldout=" + ",".join(sorted(heldout))
    return hashlib.sha256(blob.encode()).hexdigest()


SPLIT_SHA256 = "333a9ea6c5da39a033115779254a183d5d97fd1db604c261bc405faccd6e979f"
assert split_fingerprint() == SPLIT_SHA256, "the WebClones app split is frozen; do not edit it"


def app_split(app: str) -> str:
    if app in TRAIN_APPS:
        return "train"
    if app in HELDOUT_APPS:
        return "heldout"
    return "excluded"


SPLIT_ALIASES = {"train": "train", "val": "train", "heldout": "heldout", "held_out": "heldout",
                 "eval": "heldout", "ood_eval": "heldout", "test": "heldout"}

# --------------------------------------------------------------------------- tasks

DETERMINISTIC_EVALS = frozenset({"jmespath", "script"})

# Per-task bans the static audit cannot see. Keep the reason next to each id.
BANNED_TASKS: dict[str, str] = {
    # null-state probe: the script prints SUCCESS when the state has no initialfinaldiff (do-nothing = 1)
    "v2.gomail-21": "passes_on_empty_state",
}

# Hosted clones that no longer serve the app (seen from a Daytona Chromium, 2026-11-24 live sweep,
# receipts/webclones_smoke_20261124T103408.json): every task on them can never score.
DEAD_DEPLOYMENTS: dict[str, str] = {
    "https://real-zilloft-46k595h3n-real-sites.vercel.app": "410 GONE: removed by Vercel retention policy",
    "https://evals-omnizon.vercel.app": "Vercel DMCA takedown notice instead of the app",
}

# Static audit of v2 grader scripts: patterns that make the verdict depend on something other than
# the finish state. Comments are stripped before matching.
SCRIPT_AUDIT = (
    ("reads_path_or_fs", re.compile(r"\bos\.(path|listdir|walk|scandir|getcwd|environ)\b|\bglob\b|\bpathlib\b|_from_path\b")),
    ("wall_clock", re.compile(r"\b(datetime|date)\.(now|today|utcnow)\s*\(|\btime\.time\s*\(")),
    ("random", re.compile(r"^\s*(import|from)\s+random\b|\brandom\.\w+\s*\(", re.M)),
    ("run_id", re.compile(r"\brun_id\b")),
    ("network", re.compile(r"^\s*(import|from)\s+(requests|urllib|http|socket|openai|anthropic)\b", re.M)),
)


def _strip_comments(src: str) -> str:
    """Drop ``#`` comments (tokenizer-exact, so a '#' inside a string survives)."""
    import io
    import tokenize

    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return src


def audit_script(source: str) -> list[str]:
    """Reasons a grader script is not a pure function of its JSON input ([] = clean)."""
    code = _strip_comments(source)
    return [name for name, rx in SCRIPT_AUDIT if rx.search(code)]


@dataclass(frozen=True)
class Eval:
    type: str
    query: str = ""
    expected: Any = None
    script: str = ""
    rubric: str = ""


@dataclass(frozen=True)
class WebCloneTask:
    id: str  # canonical "<version>.<name>", e.g. "v2.gomail-1"
    version: str
    name: str
    app: str
    goal: str
    start_url: str
    difficulty: str
    challenge_type: str
    evals: tuple[Eval, ...]
    possible: bool = True
    points: float = 1.0
    source_path: str = ""

    @property
    def split(self) -> str:
        return app_split(self.app)

    @property
    def base_url(self) -> str:
        return self.start_url.rstrip("/")

    @property
    def origin(self) -> str:
        u = urlsplit(self.start_url)
        return f"{u.scheme}://{u.netloc}"

    @property
    def config_url(self) -> str:
        return f"{self.base_url}/config?run_id=0&task_id={self.id}&latency=0"

    @property
    def finish_url(self) -> str:
        return f"{self.base_url}/finish"

    @property
    def eval_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({e.type for e in self.evals}))


def parse_task(cfg: Mapping[str, Any], version: str, source_path: str = "") -> WebCloneTask:
    name = str(cfg["id"])
    evals = []
    for e in cfg.get("evals") or ():
        kind = e.get("type") or ("script" if e.get("script") else "")
        evals.append(Eval(type=kind, query=e.get("query", "") or "", expected=e.get("expected_value"),
                          script=e.get("script", "") or "", rubric=e.get("rubric", "") or ""))
    return WebCloneTask(
        id=f"{version}.{name}", version=version, name=name, app=str(cfg["website"]["id"]), goal=str(cfg["goal"]),
        start_url=str(cfg["website"]["url"]), difficulty=str(cfg.get("difficulty", "")),
        challenge_type=str(cfg.get("challengeType", "")), evals=tuple(evals),
        possible=bool(cfg.get("possible", True)), points=float(cfg.get("points", 1) or 1), source_path=source_path,
    )


def script_path(task: WebCloneTask, ev: Eval, root: Path | None = None) -> Path:
    return webclones_src(root) / task.version / "eval_scripts" / ev.script


def classify(task: WebCloneTask, root: Path | None = None) -> tuple[bool, str]:
    """(keep, reason). keep=True only when every grader is a deterministic function of state."""
    if task.id in BANNED_TASKS:
        return False, "banned:" + BANNED_TASKS[task.id]
    if task.app not in ALL_APPS:
        return False, "unknown_app"
    if task.split == "excluded":
        return False, "excluded_app"
    if task.origin in DEAD_DEPLOYMENTS:
        return False, "dead_deployment"
    if not task.possible:
        return False, "impossible_task"
    if not task.evals:
        return False, "no_evals"
    kinds = set(task.eval_kinds)
    if "llm_boolean" in kinds:
        return False, "llm_judge"
    if not kinds <= DETERMINISTIC_EVALS:
        return False, "unknown_eval:" + ",".join(sorted(kinds - DETERMINISTIC_EVALS))
    for ev in task.evals:
        if ev.type == "jmespath" and not ev.query:
            return False, "empty_query"
        if ev.type == "script":
            p = script_path(task, ev, root)
            if not p.exists():
                return False, "missing_script"
            bad = audit_script(p.read_text())
            if bad:
                return False, "script_" + "+".join(bad)
    return True, "deterministic"


def load_all_tasks(root: Path | None = None, versions: Iterable[str] = ("v1", "v2")) -> list[WebCloneTask]:
    out = []
    for v in versions:
        d = webclones_src(root) / v / "tasks"
        for p in sorted(d.glob("*.json")):
            out.append(parse_task(json.loads(p.read_text()), v, str(p)))
    return out


@dataclass
class TaskIndex:
    kept: list[WebCloneTask] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # task id -> reason

    def by_split(self, split: str) -> list[WebCloneTask]:
        s = SPLIT_ALIASES.get(split, split)
        return [t for t in self.kept if t.split == s]

    def get(self, task_id: str) -> WebCloneTask:
        for t in self.kept:
            if t.id == task_id:
                return t
        raise KeyError(f"{task_id} is not a deterministic WebClones task ({self.dropped.get(task_id, 'unknown')})")

    def counts(self) -> dict[str, Any]:
        per_app: dict[str, dict[str, int]] = {}
        for t in self.kept:
            per_app.setdefault(t.app, {"v1": 0, "v2": 0, "total": 0})
            per_app[t.app][t.version] += 1
            per_app[t.app]["total"] += 1
        reasons: dict[str, int] = {}
        for r in self.dropped.values():
            reasons[r] = reasons.get(r, 0) + 1
        return {
            "kept": len(self.kept), "dropped": len(self.dropped),
            "train": len(self.by_split("train")), "heldout": len(self.by_split("heldout")),
            "per_app": dict(sorted(per_app.items())), "drop_reasons": dict(sorted(reasons.items())),
        }


def load_tasks(root: Path | None = None) -> TaskIndex:
    idx = TaskIndex()
    for t in load_all_tasks(root):
        keep, why = classify(t, root)
        if keep:
            idx.kept.append(t)
        else:
            idx.dropped[t.id] = why
    return idx


def pick_task(index: TaskIndex, seed: int, split: str, family: str | None = None) -> WebCloneTask:
    """Deterministic draw. ``family`` may be an app id or a canonical task id; either must belong to
    ``split`` (asking for a held-out app in a train episode raises)."""
    s = SPLIT_ALIASES.get(split)
    if s is None:
        raise ValueError(f"unknown split {split!r}")
    pool = index.by_split(s)
    if family:
        if family in ALL_APPS:
            if app_split(family) != s:
                raise ValueError(f"app {family!r} is {app_split(family)}, not {s}")
            pool = [t for t in pool if t.app == family]
        else:
            t = index.get(family)
            if t.split != s:
                raise ValueError(f"task {family!r} is {t.split}, not {s}")
            return t
    if not pool:
        raise ValueError(f"no deterministic tasks for split={s} family={family}")
    return pool[seed % len(pool)]


# --------------------------------------------------------------------------- verifier


class VerifierError(RuntimeError):
    pass


def parse_finish_text(text: str) -> dict[str, Any]:
    """The ``<pre>`` body of ``/finish`` -> state dict (raises VerifierError when it is not JSON)."""
    s = (text or "").strip()
    if not s:
        raise VerifierError("empty /finish payload")
    try:
        state = json.loads(s)
    except json.JSONDecodeError as e:
        raise VerifierError(f"/finish is not JSON: {e}") from e
    if not isinstance(state, dict):
        raise VerifierError(f"/finish JSON is {type(state).__name__}, not an object")
    return state


def run_script(path: Path, state: Mapping[str, Any], timeout_s: float = 30.0) -> tuple[bool, dict[str, Any]]:
    """Upstream ``execute_eval_script_subprocess``: write the state to a fresh temp dir (so
    nothing else is beside it), run the script with this interpreter in isolated mode, SUCCESS
    in stdout (case-insensitive) passes."""
    with tempfile.TemporaryDirectory(prefix="webclones_eval_") as d:
        fp = Path(d) / "state.json"
        fp.write_text(json.dumps(state))
        try:
            r = subprocess.run([sys.executable, "-I", str(path), str(fp)], capture_output=True, text=True,
                               timeout=timeout_s, cwd=d)
        except subprocess.TimeoutExpired:
            return False, {"script": path.name, "error": f"timeout {timeout_s}s"}
    out = r.stdout.strip()
    return "SUCCESS" in out.upper(), {"script": path.name, "output": out[-300:], "rc": r.returncode,
                                      "stderr": r.stderr.strip()[-300:] or None}


def verify(task: WebCloneTask, state: Mapping[str, Any], root: Path | None = None) -> tuple[float, dict[str, Any]]:
    """Deterministic 0/1 reward: every grader must pass. Never consults an LLM or the answer."""
    keep, why = classify(task, root)
    if not keep:
        raise VerifierError(f"{task.id} is not deterministic ({why}); refusing to grade it")
    results = []
    for ev in task.evals:
        if ev.type == "jmespath":
            try:
                actual = _jmespath().search(ev.query, state)
                ok, info = actual == ev.expected, {"query": ev.query, "actual": actual, "expected": ev.expected}
            except Exception as e:  # noqa: BLE001 - a malformed state fails the check, as upstream
                ok, info = False, {"query": ev.query, "error": f"{type(e).__name__}: {e}"}
        else:
            ok, info = run_script(script_path(task, ev, root), state)
        results.append({"type": ev.type, "ok": bool(ok), **info})
    passed = all(r["ok"] for r in results)
    return (1.0 if passed else 0.0), {"task": task.id, "passed": passed, "results": results}


# --------------------------------------------------------------------------- live env on the pool

# textContent, not innerText: StayNB's /finish dump is multi-MB and innerText forces layout (40 s+).
FINISH_JS = "(()=>{const p=document.querySelector('pre');return p?p.textContent:null})()"
STABLE_JS = ("(()=>{const b=document.body;const t=b?b.innerText:'';"
             "return document.getElementsByTagName('*').length+'|'+t.length+'|'+"
             "performance.getEntriesByType('resource').length+"
             "((/loading application|please wait/i.test(t)||document.querySelector("
             "'[aria-busy=\"true\"],[aria-label*=\"Loading application\" i],[aria-label*=\"please wait\" i]'))"
             "?'|loading':'')})()")
OWNER_LABELS = {"lapis-rl": "1", "owner": "w1-webclones"}


def _stage1_envs():
    """rl/overnight_stage1/envs.py, reusing whatever copy is already imported."""
    path = (RL / "overnight_stage1" / "envs.py").resolve()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve() == path:
            return m
    spec = importlib.util.spec_from_file_location("lapis_stage1_envs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lapis_stage1_envs"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_env(pool: Any, i: int, index: TaskIndex | None = None) -> Any:
    """A ``WebClonesEnv`` on pool worker ``i`` (class built on first use so offline imports stay light)."""
    return _env_class()(pool, i, index)


_ENV_CLASS: Any = None


def _env_class() -> Any:
    global _ENV_CLASS
    if _ENV_CLASS is not None:
        return _ENV_CLASS
    E = _stage1_envs()

    class WebClonesEnv(E.StudioEnv):
        """``reset(seed, split, family)`` -> Studio goal; ``observe``/``execute`` from StudioEnv;
        ``score(answer)`` -> deterministic 0/1 from the clone's /finish state."""

        name = "webclones"

        def __init__(self, pool: Any, i: int, index: TaskIndex | None = None):
            super().__init__(pool, i)
            self.index = index or load_tasks()
            self.task: WebCloneTask | None = None
            self.final_state: dict[str, Any] | None = None

        # -- CDP navigation on the front tab
        def _cdp(self, method: str, params: Mapping[str, Any], attempts: int = 3, timeout_s: float = 30.0) -> Any:
            """CDP on the front tab with a longer timeout than pool.cdp's 10 s, retried."""
            for k in range(attempts):
                try:
                    return self.pool._ctl(self.i, "POST", "/__cdp", {"method": method, "params": dict(params),
                                          "front": True, "timeout": timeout_s}, timeout=timeout_s + 10)["result"]
                except Exception:  # noqa: BLE001
                    if k == attempts - 1:
                        raise
                    time.sleep(1.0)

        def _navigate(self, url: str, settle_s: float = 0.6, timeout_s: float = 25.0, attempts: int = 2,
                      stable: bool = True) -> float:
            """Page.navigate the front tab and wait for load + a still DOM. One retry: the live
            sweep saw ~3% transient CDP navigate timeouts on heavy clones (StayNB, DashDish)."""
            t0 = time.perf_counter()
            for attempt in range(attempts):
                try:
                    self._cdp("Page.navigate", {"url": url}, attempts=1)
                except Exception:  # noqa: BLE001 - CDP timeout while a heavy page commits
                    if attempt == attempts - 1:
                        raise
                    time.sleep(1.0)
                    continue
                deadline = time.monotonic() + timeout_s
                time.sleep(0.2)
                while time.monotonic() < deadline:
                    try:
                        st = self.pool.eval_js(self.i, "document.readyState+'|'+location.href", front=True) or ""
                    except Exception:  # noqa: BLE001 - mid-navigation context swap
                        st = ""
                    if st.startswith("complete|") and not st.endswith("about:blank"):
                        if stable:
                            self._wait_stable(min_s=settle_s)
                        return round(time.perf_counter() - t0, 3)
                    time.sleep(0.15)
            raise E.PoolError(f"worker {self.i}: {url} did not load in {attempts}x{timeout_s}s")

        def _wait_stable(self, min_s: float = 0.3, timeout_s: float = 12.0, quiet_polls: int = 3) -> bool:
            """Upstream waits for Playwright ``networkidle``; here: the page's DOM size, text size and
            resource count must hold still for ``quiet_polls`` polls (0.25 s apart) and no
            "Loading application" splash may be showing (the v1 clones hydrate behind one)."""
            t0 = time.monotonic()
            last, same = None, 0
            while time.monotonic() - t0 < timeout_s:
                try:
                    sig = self.pool.eval_js(self.i, STABLE_JS, front=True)
                except Exception:  # noqa: BLE001
                    sig = None
                if sig and sig == last and not str(sig).endswith("|loading"):
                    same += 1
                    if same >= quiet_polls and time.monotonic() - t0 >= min_s:
                        return True
                else:
                    same = 0
                last = sig
                time.sleep(0.25)
            return False

        def _close_extra_tabs(self) -> int:
            tabs = self.pool.tabs(self.i).get("tabs", [])
            if len(tabs) <= 1:
                return 0
            keep = next((t for t in tabs if t.get("value") == "visible"), tabs[0])
            n = 0
            for t in tabs:
                if t.get("id") != keep.get("id"):
                    try:
                        self.pool.cdp(self.i, "Target.closeTarget", {"targetId": t["id"]}, front=True)
                        n += 1
                    except Exception:  # noqa: BLE001
                        pass
            return n

        def read_finish(self, timeout_s: float = 25.0) -> dict[str, Any]:
            assert self.task is not None
            # no DOM-stability wait here: innerText over a multi-MB <pre> (StayNB) is itself slow
            self._navigate(self.task.finish_url, settle_s=0.0, stable=False)
            deadline, last = time.monotonic() + timeout_s, None
            while time.monotonic() < deadline:
                try:
                    r = self.pool._ctl(self.i, "POST", "/__cdp", {
                        "method": "Runtime.evaluate", "front": True, "timeout": 40,
                        "params": {"expression": FINISH_JS, "returnByValue": True}}, timeout=50)
                    last = (r.get("result") or {}).get("result", {}).get("value")
                except Exception as e:  # noqa: BLE001 - keep polling through a slow render
                    last = f"<{type(e).__name__}: {str(e)[:120]}>"
                    time.sleep(0.5)
                    continue
                if last:
                    try:
                        return parse_finish_text(last)
                    except VerifierError:
                        pass
                time.sleep(0.25)
            raise VerifierError(f"/finish never rendered JSON (last={str(last)[:200]!r})")

        def reset(self, seed: int, split: str = "train", family: str | None = None) -> dict[str, Any]:
            t0 = time.perf_counter()
            task = pick_task(self.index, seed, split, family)
            if self._mode() != "pool":
                self.pool.relaunch_chrome(self.i)
                self._set_mode("pool")
            timing: dict[str, float] = {"closed_tabs": float(self._close_extra_tabs())}
            # fresh browser state for this origin, like upstream's fresh Playwright context
            # leave the previous episode's page first: a multi-MB /finish dump keeps the renderer busy
            self._cdp("Page.navigate", {"url": "about:blank"})
            self._cdp("Storage.clearDataForOrigin", {"origin": task.origin, "storageTypes": "all"})
            self.task, self.final_state = task, None
            timing["config_s"] = self._navigate(task.config_url, settle_s=0.5)
            timing["finish0_s"] = self._navigate(task.finish_url, settle_s=0.3)
            timing["start_s"] = self._navigate(task.start_url, settle_s=0.5)
            self.calibrate()
            self.seed, self.split, self.family, self.goal = seed, task.split, task.app, task.goal
            timing["reset_s"] = round(time.perf_counter() - t0, 3)
            self.reset_timing = timing
            return {"goal": task.goal, "family": task.app, "task_id": task.id, "split": task.split,
                    "start_url": task.start_url}

        def score(self, answer: str | None) -> tuple[float, dict[str, Any]]:
            assert self.task is not None, "reset() first"
            try:
                self.final_state = self.read_finish()
            except Exception as e:  # noqa: BLE001 - no readable state means the task was not done
                return 0.0, {"task": self.task.id, "passed": False, "error": f"{type(e).__name__}: {e}"}
            return verify(self.task, self.final_state)

        def meta(self) -> dict[str, Any]:
            t = self.task
            return {} if t is None else {"task_id": t.id, "app": t.app, "split": t.split, "evals": list(t.eval_kinds),
                                         "difficulty": t.difficulty, "start_url": t.start_url}

    _ENV_CLASS = WebClonesEnv
    return _ENV_CLASS


# --------------------------------------------------------------------------- trainer adapter


def episode_samples(ep: Mapping[str, Any], *, include_failed: bool = False) -> list[dict[str, Any]]:
    """A ``run_episode`` result -> per-turn Studio samples in the trainer's shape: ``messages``
    from ``studio_format.turn_messages`` (exactly what Studio sends at turn k) and
    ``completion_text`` (the reply executed at turn k). Held-out episodes are refused."""
    if ep.get("env") != "webclones":
        raise ValueError(f"not a webclones episode: {ep.get('env')!r}")
    app = ep.get("family") or ""
    if app_split(app) != "train" or SPLIT_ALIASES.get(str(ep.get("split")), "") != "train":
        raise ValueError(f"refusing to emit training samples from {app!r} (split={ep.get('split')!r}): held-out")
    if float(ep.get("reward", 0.0)) < 1.0 and not include_failed:
        return []
    SF = _studio_format()
    turns = ep["turns"]
    task_id = (ep.get("reset_info") or {}).get("task_id", "")
    out = []
    for k, t in enumerate(turns):
        msgs = SF.turn_messages(ep["goal"], prior=turns[:k], steps=turns[k - 1]["steps"] if k else None,
                                time_line=ep["time_line"], image_jpeg=t.get("image"))
        out.append({"src": "webclones", "id": f"webclones_{task_id}_{ep['seed']}_k{k}", "app": app, "task_id": task_id,
                    "split": "train", "turn": k, "messages": msgs, "completion_text": t["raw"],
                    "reward": float(ep["reward"])})
    return out


def _studio_format():
    path = (RL / "overnight_stage1" / "studio_format.py").resolve()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve() == path:
            return m
    spec = importlib.util.spec_from_file_location("studio_format", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["studio_format"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_episode(env: Any, policy: Any, seed: int, split: str, family: str | None = None, **kw: Any) -> dict[str, Any]:
    """Stage-1 ``run_episode`` (the Studio loop) on a WebClonesEnv."""
    return _stage1_envs().run_episode(env, policy, seed, split, family, **kw)


# --------------------------------------------------------------------------- live smoke


# Hand-written replies that solve a task through the real UI (live check that reward 1 is reachable).
ORACLES: dict[str, list[str]] = {
    "v1.gomail-2": [
        "Opening the first inbox email marks it read. "
        "[[ACTION:cua_click:Chromium|Email from Support, Your Account Statement is Ready]]",
        "Done: the first email in the Inbox is now marked as read.",
    ],
}


def scripted_policy(replies: list[str]) -> Any:
    """A policy that plays ``replies`` in order (then answers "Done.")."""
    def policy(msgs: list[dict], ctx: Mapping[str, Any]) -> str:
        k = int(ctx.get("turn", 0))
        return replies[k] if k < len(replies) else "Done."
    return policy


def noop_sweep(pool: Any, index: TaskIndex, task_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Reset every task on the live clone, take no action, read /finish and grade it. A task that
    scores 1 here rewards doing nothing and belongs in BANNED_TASKS. No-op states are saved under
    ``DATA_ROOT/noop_states`` (they are the fixtures the offline tests replay)."""
    import concurrent.futures as cf

    ids = list(task_ids) if task_ids is not None else [t.id for t in index.kept]
    out_dir = DATA_ROOT / "noop_states"
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = [make_env(pool, i, index) for i in range(len(pool.workers))]
    queues: list[list[str]] = [ids[k::len(envs)] for k in range(len(envs))]

    def run(k: int) -> list[dict[str, Any]]:
        env, res = envs[k], []
        for tid in queues[k]:
            t = index.get(tid)
            row: dict[str, Any] = {"task": tid, "app": t.app}
            try:
                env.reset(seed=0, split=t.split, family=tid)
                row["reset_s"] = env.reset_timing.get("reset_s")
                st = env.read_finish()
                (out_dir / f"{tid}.json").write_text(json.dumps(st))
                r, info = verify(t, st)
                row.update(reward=r, keys=sorted(st)[:12])
                if r:
                    row["verdict"] = info
            except Exception as e:  # noqa: BLE001
                import traceback

                frames = [f"{fr.name}:{fr.lineno}" for fr in traceback.extract_tb(e.__traceback__)][-4:]
                row["error"] = f"{type(e).__name__}: {e}"[:300] + " @ " + " < ".join(reversed(frames))
            res.append(row)
        return res

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(len(envs)) as ex:
        rows = [r for part in ex.map(run, range(len(envs))) for r in part]
    return {"n": len(rows), "wall_s": round(time.perf_counter() - t0, 1),
            "noop_pass": sorted(r["task"] for r in rows if r.get("reward") == 1.0),
            "errors": {r["task"]: r["error"] for r in rows if "error" in r}, "rows": rows}


def smoke(n: int = 1, snapshot: str = "daytona-small", tasks: Iterable[str] = (), sweep: bool = False) -> dict[str, Any]:
    """Live check on ``n`` (<= 2) labeled sandboxes: hosted clone loads in the pool's Chromium, CDP
    AX lines list its controls, /finish parses, a no-op episode scores 0, and every sandbox with
    owner=w1-webclones is gone afterwards. Writes rl/receipts/webclones_smoke_<stamp>.json."""
    assert 1 <= n <= 2, "budget: at most 2 sandboxes"
    from pool import SandboxPool

    E = _stage1_envs()
    idx = load_tasks()
    want = list(tasks) or ["v2.gomail-1", "v2.gocalendar-1"]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rec: dict[str, Any] = {"snapshot": snapshot, "workers": n, "labels": OWNER_LABELS, "counts": idx.counts(),
                           "split_sha256": SPLIT_SHA256, "episodes": []}
    pool = SandboxPool(snapshot=snapshot, labels=OWNER_LABELS, max_workers=n, idle_stop_min=15)
    rec["pool_id"] = pool.pool_id
    try:
        t = time.perf_counter()
        rec["boot"] = pool.start(n, max_parallel=n)
        rec["boot_wall_s"] = round(time.perf_counter() - t, 2)
        for w in pool.workers:
            w.env_mode = "pool"
        for k, tid in enumerate(want):
            task = idx.get(tid)
            env = make_env(pool, k % len(pool.workers), idx)
            ep: dict[str, Any] = {"task": tid, "app": task.app, "split": task.split}
            try:
                ep["reset"] = env.reset(seed=0, split=task.split, family=tid)
                ep["reset_timing"] = env.reset_timing
                jpeg, ax = env.observe()
                ep["jpeg_bytes"] = len(jpeg)
                shot = RL / "receipts" / f"webclones_smoke_{stamp}_{task.app}.jpg"
                shot.write_bytes(jpeg)
                ep["screenshot"] = str(shot)
                lines = E.parse_ax_lines(ax)
                ep["ax_chars"] = len(ax)
                ep["ax_lines"] = len(lines)
                ep["ax_controls"] = sum(1 for x in lines if x.role in ("Button", "Link", "TextField", "CheckBox",
                                                                         "RadioButton", "ComboBox"))
                ep["ax"] = ax
                ep["page"] = pool.eval_js(env.i, "document.title+' | '+location.href", front=True)
                obs = env.execute(E.StudioAction("cua_observe", ""))
                ep["observe_step"] = {k2: obs[k2] for k2 in ("tier", "ok", "detail")}
                r, info = env.score("")
                ep["noop_reward"] = r
                ep["noop_verdict"] = info
                ep["finish_keys"] = sorted((env.final_state or {}).keys())
            except Exception as e:  # noqa: BLE001 - record and keep going to teardown
                ep["error"] = f"{type(e).__name__}: {e}"[:600]
            rec["episodes"].append(ep)
        for tid, replies in ORACLES.items():  # full Studio loop with a scripted policy: reward-1 path
            task = idx.get(tid)
            env = make_env(pool, 0, idx)
            orc: dict[str, Any] = {"task": tid, "replies": replies}
            try:
                ep = run_episode(env, scripted_policy(replies), seed=0, split=task.split, family=tid)
                orc.update(reward=ep["reward"], finished=ep["finished"], info=ep["info"], timing=ep["timing"],
                           steps=[{k2: s[k2] for k2 in ("action", "target", "tier", "ok", "detail")}
                                  for t in ep["turns"] for s in t["steps"]])
                samples = episode_samples(ep)
                orc["samples"] = len(samples)
                orc["sample0_roles"] = [m["role"] for m in samples[0]["messages"]] if samples else []
                orc["sample1_user_has_image"] = bool(samples[1:] and isinstance(samples[1]["messages"][-1]["content"], list))
            except Exception as e:  # noqa: BLE001
                orc["error"] = f"{type(e).__name__}: {e}"[:600]
            rec.setdefault("oracles", []).append(orc)
        if sweep:
            rec["noop_sweep"] = noop_sweep(pool, idx)
    finally:
        rec["close"] = pool.close_all()
        rec["owner_live_after"] = _owner_live_count()
        out = RL / "receipts" / f"webclones_smoke_{stamp}.json"
        out.write_text(json.dumps(rec, indent=2, default=str))
        rec["receipt"] = str(out)
    return rec


def _owner_live_count() -> int:
    from daytona import Daytona, DaytonaConfig
    from daytona.v0 import ListSandboxesQuery

    c = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"].strip())).v0
    n = 0
    for sb in c.list(ListSandboxesQuery(labels=dict(OWNER_LABELS))):
        if (getattr(sb, "labels", None) or {}).get("owner") != OWNER_LABELS["owner"]:
            continue
        if str(getattr(sb.state, "value", sb.state)).lower() not in ("destroyed", "destroying"):
            n += 1
    return n


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        args = sys.argv[2:]
        n = int(args.pop(0)) if args and args[0].isdigit() else 1
        sweep = "--sweep" in args
        res = smoke(n, tasks=[a for a in args if a != "--sweep"], sweep=sweep)
        print(json.dumps({k: res.get(k) for k in ("receipt", "close", "owner_live_after")}, default=str))
    else:
        print(json.dumps(load_tasks().counts(), indent=2))
