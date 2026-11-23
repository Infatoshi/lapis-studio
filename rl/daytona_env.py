"""Daytona-backed computer-use (CUA) environment for RL rollouts.

SDK: ``daytona`` 2.x, v2 namespace (``Daytona().v2``). The 0.x-style API used by
harbor (``CreateSandboxFromImageParams`` / ``CreateSandboxFromSnapshotParams``)
now lives under ``Daytona().v0`` and needs the ``daytona[v0-sdk]`` extra; it has
no macOS path, so this module targets v2 only.

Targeting (verified against the installed SDK source, daytona 2.1.0):
  * linux: ``v2.create(image_or_None, resources=Resources(...))``, or
    ``v2.create_from_snapshot(name)`` (daytona/v2/_sync/daytona.py:101, :285).
    ``image=None`` boots the platform default Linux image. The computer-use
    docs say the desktop stack ships with the runtime on any Linux image; which
    browser (if any) the default image carries is NOT verified here.
  * macos: ``v2.create_macos(image, resources=MacosResources(...),
    reservation=...)`` (daytona/v2/_sync/daytona.py:331). Runs on a reserved,
    dedicated Mac. ``reservation=None`` lets the org's auto-reserve BUY a Mac,
    and ``"new"`` always buys one, so this module refuses to create a macOS
    sandbox unless a reservation id is given or ``allow_mac_purchase=True``.
    Docs name ``base-macos`` as the image; no other image names are verified.

Env vars: ``DAYTONA_API_KEY`` (required), ``DAYTONA_TARGET`` (mapped to the v2
``region=`` create kwarg; v2 has no client-level target), ``DAYTONA_V2_API_URL``
(read natively by the SDK). ``DAYTONA_API_URL`` is the 0.x variable; SDK 2.x
ignores it, so we only warn when it is set.

The SDK sits behind ``SandboxClient`` / ``SandboxHandle`` so tests inject fakes.
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol

Target = Literal["macos", "linux"]

DEFAULT_MACOS_IMAGE = "base-macos"
ACTION_TYPES = ("click", "double_click", "type", "scroll", "hotkey", "key", "move", "wait", "done")


class MissingDaytonaKey(RuntimeError):
    """Raised when DAYTONA_API_KEY is not set."""


class MacReservationRequired(RuntimeError):
    """Raised before a macOS create that could buy a Mac reservation."""


class SandboxNotReady(RuntimeError):
    pass


def key_fingerprint(key: str | None) -> str | None:
    """Loggable identity of a secret: length and last 6 chars, never the value."""
    if not key:
        return None
    return f"len={len(key)} last6={key[-6:]}"


def load_api_key(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    key = (env.get("DAYTONA_API_KEY") or "").strip()
    if not key:
        raise MissingDaytonaKey(
            "DAYTONA_API_KEY is not set. Export it (it lives in ~/.env_vars) and rerun."
        )
    return key


@dataclass
class ExecResult:
    exit_code: int | None
    output: str


@dataclass
class SandboxSpec:
    target: Target
    image: str | None = None
    snapshot: str | None = None
    resources: dict[str, float] | None = None  # cpu, memory (GiB), disk (GiB)
    region: str | None = None
    timeout_s: int = 600
    mac_reservation: str | None = None
    allow_mac_purchase: bool = False
    labels: dict[str, str] = field(default_factory=lambda: {"lapis-rl": "1"})


class SandboxHandle(Protocol):
    id: str

    def start_computer_use(self) -> None: ...
    def screenshot_png(self) -> bytes: ...
    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None: ...
    def move(self, x: int, y: int) -> None: ...
    def scroll(self, x: int, y: int, direction: str, amount: int) -> None: ...
    def type_text(self, text: str) -> None: ...
    def hotkey(self, keys: str) -> None: ...
    def key(self, key: str) -> None: ...
    def exec(self, command: str, timeout_s: float = 60) -> ExecResult: ...
    def info(self) -> dict[str, Any]: ...
    def delete(self) -> None: ...


class SandboxClient(Protocol):
    def create_sandbox(self, spec: SandboxSpec) -> SandboxHandle: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- SDK adapter


class _SDKSandbox:
    """Wraps a ``daytona.v2`` sync ``Sandbox``."""

    def __init__(self, sandbox: Any):
        self._sb = sandbox
        self.id = sandbox.id

    def start_computer_use(self) -> None:
        self._sb.computer_use.start()

    def screenshot_png(self) -> bytes:
        shot = self._sb.computer_use.screen.screenshot()
        return base64.b64decode(shot.screenshot)

    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> None:
        self._sb.computer_use.mouse.click(x=int(x), y=int(y), button=button, double=double or None)

    def move(self, x: int, y: int) -> None:
        self._sb.computer_use.mouse.move(x=int(x), y=int(y))

    def scroll(self, x: int, y: int, direction: str, amount: int) -> None:
        self._sb.computer_use.mouse.scroll(x=int(x), y=int(y), direction=direction, amount=int(amount))

    def type_text(self, text: str) -> None:
        self._sb.computer_use.keyboard.type(text=text)

    def hotkey(self, keys: str) -> None:
        self._sb.computer_use.keyboard.hotkey(keys=keys)

    def key(self, key: str) -> None:
        self._sb.computer_use.keyboard.key(key=key)

    def exec(self, command: str, timeout_s: float = 60) -> ExecResult:
        res = self._sb.exec(command, timeout=timeout_s)
        return ExecResult(exit_code=res.exit_code, output=res.output or (res.stdout + res.stderr))

    def info(self) -> dict[str, Any]:
        sb = self._sb
        out: dict[str, Any] = {"id": sb.id}
        for attr in ("sandbox_class", "state", "region", "cpu", "cpu_max", "memory", "memory_max", "disk"):
            try:
                out[attr] = getattr(sb, attr)
            except Exception as exc:  # attribute reads are local, but stay defensive
                out[attr] = f"<error {exc}>"
        try:
            displays = sb.computer_use.display.info().displays
            out["displays"] = [
                {"w": d.width, "h": d.height, "active": d.is_active} for d in displays
            ]
        except Exception as exc:
            out["displays"] = f"<error {type(exc).__name__}: {exc}>"
        return out

    def delete(self) -> None:
        self._sb.delete(timeout=120)


class DaytonaSDKClient:
    """Thin adapter over ``daytona.Daytona().v2``."""

    def __init__(self, api_key: str):
        if os.environ.get("DAYTONA_API_URL"):
            warnings.warn(
                "DAYTONA_API_URL is a daytona 0.x variable and is ignored by SDK 2.x; "
                "set DAYTONA_V2_API_URL for a non-default v2 endpoint.",
                stacklevel=2,
            )
        from daytona import Daytona, DaytonaConfig

        self._root = Daytona(DaytonaConfig(api_key=api_key))
        self._v2 = self._root.v2

    def create_sandbox(self, spec: SandboxSpec) -> SandboxHandle:
        from daytona.v2 import MacosResources, Resources

        common = dict(
            labels=spec.labels,
            region=spec.region,
            wait_seconds=min(spec.timeout_s, 600),
            create_timeout_seconds=spec.timeout_s,
        )
        if spec.target == "macos":
            if spec.snapshot:
                raise ValueError("macOS sandboxes are created from an image name, not a snapshot")
            reservation = spec.mac_reservation
            if reservation is None and not spec.allow_mac_purchase:
                raise MacReservationRequired(
                    "macOS sandboxes run on a reserved Mac; with no reservation id the "
                    "org's auto-reserve may buy one. Pass mac_reservation=<id> "
                    "(or DAYTONA_MAC_RESERVATION) or allow_mac_purchase=True."
                )
            res = MacosResources(**spec.resources) if spec.resources else None
            sb = self._v2.create_macos(
                spec.image or DEFAULT_MACOS_IMAGE,
                resources=res,
                reservation=reservation,
                **common,
            )
        elif spec.snapshot:
            sb = self._v2.create_from_snapshot(spec.snapshot, **common)
        else:
            res = Resources(**spec.resources) if spec.resources else None
            sb = self._v2.create(spec.image, resources=res, **common)

        if sb.state != "started":
            err = getattr(sb, "error", None)
            try:
                sb.delete(timeout=120)
            finally:
                raise SandboxNotReady(f"sandbox {sb.id} ended create in state {sb.state!r}: {err}")
        return _SDKSandbox(sb)

    def close(self) -> None:
        self._root.close()


# --------------------------------------------------------------------------- env


def png_size(png: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        return im.size


class DaytonaCUAEnv:
    """Gym-style CUA env over one Daytona sandbox (created lazily on reset)."""

    def __init__(
        self,
        target: Target = "linux",
        resources: dict[str, float] | None = None,
        image: str | None = None,
        snapshot: str | None = None,
        timeout: int = 600,
        region: str | None = None,
        mac_reservation: str | None = None,
        allow_mac_purchase: bool = False,
        client: SandboxClient | None = None,
        api_key: str | None = None,
        ready_timeout: float = 120.0,
        settle_s: float = 0.0,
    ):
        if target not in ("macos", "linux"):
            raise ValueError(f"target must be 'macos' or 'linux', got {target!r}")
        self.target: Target = target
        self.spec = SandboxSpec(
            target=target,
            image=image,
            snapshot=snapshot,
            resources=resources,
            region=region or os.environ.get("DAYTONA_TARGET") or None,
            timeout_s=timeout,
            mac_reservation=mac_reservation or os.environ.get("DAYTONA_MAC_RESERVATION") or None,
            allow_mac_purchase=allow_mac_purchase,
        )
        self._owns_client = client is None
        if client is None:
            key = api_key or load_api_key()
            self.key_fingerprint = key_fingerprint(key)
            self._client_factory = lambda: DaytonaSDKClient(key)
            self.client: SandboxClient | None = None
        else:
            self.key_fingerprint = None
            self._client_factory = None
            self.client = client
        self.ready_timeout = ready_timeout
        self.settle_s = settle_s
        self.sandbox: SandboxHandle | None = None
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.task: Any = None
        self.steps = 0
        self.last_answer: str | None = None
        self.closed = False

    # ---- timing

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name].append(time.perf_counter() - t0)

    def timing_summary(self) -> dict[str, dict[str, float]]:
        return {k: summarize(v) for k, v in self.timings.items()}

    # ---- lifecycle

    def _ensure_sandbox(self) -> SandboxHandle:
        if self.closed:
            raise RuntimeError("env is closed")
        if self.sandbox is not None:
            return self.sandbox
        if self.client is None:
            self.client = self._client_factory()
        t_boot = time.perf_counter()
        with self._timed("create"):
            self.sandbox = self.client.create_sandbox(self.spec)
        with self._timed("computer_use_start"):
            self.sandbox.start_computer_use()
        with self._timed("first_screenshot"):
            self._wait_ready()
        self.timings["boot_total"].append(time.perf_counter() - t_boot)
        return self.sandbox

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                png = self.sandbox.screenshot_png()
                if png[:8] == b"\x89PNG\r\n\x1a\n":
                    return
                last = SandboxNotReady("screenshot was not a PNG")
            except Exception as exc:  # desktop still coming up
                last = exc
            time.sleep(1.0)
        raise SandboxNotReady(f"computer use not ready after {self.ready_timeout}s: {last}")

    def reset(self, task: Any = None) -> dict[str, Any]:
        self._ensure_sandbox()
        self.task = task
        self.steps = 0
        self.last_answer = None
        if task is not None and getattr(task, "setup", None) is not None:
            with self._timed("task_setup"):
                task.setup(self)
        return self.screenshot()

    def close(self) -> None:
        """Delete the sandbox (always attempted) and release the client. Idempotent."""
        if self.closed:
            return
        self.closed = True
        sb, self.sandbox = self.sandbox, None
        try:
            if sb is not None:
                with self._timed("teardown"):
                    sb.delete()
        finally:
            if self._owns_client and self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass

    def __enter__(self) -> "DaytonaCUAEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- primitives

    def screenshot(self) -> dict[str, Any]:
        sb = self._ensure_sandbox()
        t0 = time.perf_counter()
        with self._timed("screenshot"):
            png = sb.screenshot_png()
        latency = time.perf_counter() - t0
        w, h = png_size(png)
        return {"screenshot": png, "width": w, "height": h, "t": time.time(), "latency_s": latency}

    def click(self, x: int, y: int, button: str = "left") -> None:
        sb = self._ensure_sandbox()
        with self._timed("click"):
            sb.click(x, y, button=button)

    def double_click(self, x: int, y: int) -> None:
        sb = self._ensure_sandbox()
        with self._timed("double_click"):
            sb.click(x, y, button="left", double=True)

    def move(self, x: int, y: int) -> None:
        sb = self._ensure_sandbox()
        with self._timed("move"):
            sb.move(x, y)

    def type(self, text: str) -> None:
        sb = self._ensure_sandbox()
        with self._timed("type"):
            sb.type_text(text)

    def scroll(self, x: int, y: int, direction: str = "down", amount: int = 3) -> None:
        if direction not in ("up", "down"):
            raise ValueError(f"scroll direction must be up/down, got {direction!r}")
        sb = self._ensure_sandbox()
        with self._timed("scroll"):
            sb.scroll(x, y, direction, amount)

    def hotkey(self, keys: str | list[str]) -> None:
        """``keys`` as "ctrl+c" or ["ctrl", "c"] (Daytona joins with '+', main key last)."""
        combo = "+".join(keys) if isinstance(keys, (list, tuple)) else keys
        sb = self._ensure_sandbox()
        with self._timed("hotkey"):
            sb.hotkey(combo)

    def key(self, key: str) -> None:
        sb = self._ensure_sandbox()
        with self._timed("key"):
            sb.key(key)

    def exec(self, command: str, timeout_s: float = 60) -> ExecResult:
        sb = self._ensure_sandbox()
        with self._timed("exec"):
            return sb.exec(command, timeout_s=timeout_s)

    def info(self) -> dict[str, Any]:
        return self._ensure_sandbox().info()

    # ---- RL step

    def _dispatch(self, action: dict[str, Any]) -> None:
        kind = action.get("type")
        if kind == "click":
            self.click(action["x"], action["y"], action.get("button", "left"))
        elif kind == "double_click":
            self.double_click(action["x"], action["y"])
        elif kind == "type":
            self.type(action["text"])
        elif kind == "scroll":
            self.scroll(action["x"], action["y"], action.get("direction", "down"), action.get("amount", 3))
        elif kind == "hotkey":
            self.hotkey(action["keys"])
        elif kind == "key":
            self.key(action["key"])
        elif kind == "move":
            self.move(action["x"], action["y"])
        elif kind == "wait":
            with self._timed("wait"):
                time.sleep(float(action.get("seconds", 1.0)))
        elif kind == "done":
            self.last_answer = action.get("answer")
        else:
            raise ValueError(f"unknown action type {kind!r}; expected one of {ACTION_TYPES}")

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.steps += 1
        info: dict[str, Any] = {"step": self.steps, "action": action.get("type")}
        t0 = time.perf_counter()
        try:
            self._dispatch(action)
        except (KeyError, ValueError, TypeError) as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
        info["action_latency_s"] = time.perf_counter() - t0
        if self.settle_s and action.get("type") not in ("wait", "done"):
            time.sleep(self.settle_s)

        max_steps = getattr(self.task, "max_steps", None)
        done = action.get("type") == "done" or (max_steps is not None and self.steps >= max_steps)
        if done and action.get("type") != "done":
            info["truncated"] = True
        reward = 0.0
        if done and self.task is not None:
            with self._timed("reward"):
                reward = float(self.task.reward(self, self.last_answer))
        obs = self.screenshot()
        return obs, reward, done, info


def summarize(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def pct(p: float) -> float:
        # nearest-rank percentile
        idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
        return s[idx]

    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "min": s[0],
        "p50": pct(50),
        "p95": pct(95),
        "max": s[-1],
    }
