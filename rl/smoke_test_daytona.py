"""Daytona computer-use smoke test: boot, latency, specs, optional VLM call.

  uv run python smoke_test_daytona.py --target linux --iters 10
  uv run python smoke_test_daytona.py --target macos --mac-reservation <id>
  uv run python smoke_test_daytona.py --vlm-only

Exit codes: 0 ok, 1 a measured step failed, 2 blocked (DAYTONA_API_KEY unset).
Always deletes the sandboxes it created and writes a JSON receipt.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import httpx

from daytona_env import DaytonaCUAEnv, MissingDaytonaKey, key_fingerprint, summarize

HERE = Path(__file__).resolve().parent
SPEC_CMDS = {
    "linux": "uname -a; echo nproc=$(nproc); free -m; df -h / | tail -n 1; cat /etc/os-release | head -n 2",
    "macos": "uname -a; sw_vers; echo ncpu=$(sysctl -n hw.ncpu); echo memsize=$(sysctl -n hw.memsize); df -h / | tail -n 1",
}
HOTKEY = {"linux": "ctrl+a", "macos": "cmd+a"}


def sdk_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("daytona")
    except Exception:
        return None


def models_url(chat_url: str) -> str:
    base = chat_url.split("/chat/completions")[0].rstrip("/")
    return f"{base}/models"


def synthetic_png() -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (640, 400), (30, 30, 30))
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 300, 120), fill=(40, 120, 220))
    d.text((60, 70), "OK button", fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def vlm_call(url: str, png: bytes, timeout: float = 120.0) -> dict[str, Any]:
    out: dict[str, Any] = {"url": url, "ok": False}
    try:
        with httpx.Client(timeout=timeout) as c:
            t0 = time.perf_counter()
            r = c.get(models_url(url), timeout=10.0)
            out["models_latency_s"] = time.perf_counter() - t0
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
            if not models:
                out["error"] = "GET /v1/models returned no models"
                return out
            out["model"] = models[0]
            body = {
                "model": models[0],
                "max_tokens": 256,  # reasoning models spend part of this on thinking
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "In one sentence, what is on this screen?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                            },
                        ],
                    }
                ],
            }
            t0 = time.perf_counter()
            r = c.post(url, json=body)
            out["chat_latency_s"] = time.perf_counter() - t0
            out["status_code"] = r.status_code
            r.raise_for_status()
            j = r.json()
            choice = j["choices"][0]
            out["reply"] = (choice["message"].get("content") or "")[:300]
            out["reasoning_chars"] = len(choice["message"].get("reasoning_content") or "")
            out["finish_reason"] = choice.get("finish_reason")
            out["usage"] = j.get("usage")
            out["image_bytes"] = len(png)
            out["ok"] = True
    except httpx.ConnectError as exc:
        out["error"] = f"connection error: {exc}"
    except httpx.TimeoutException as exc:
        out["error"] = f"timeout: {exc!r}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def measure(fn: Callable[[], Any], iters: int) -> tuple[dict[str, float], list[str]]:
    xs: list[float] = []
    errs: list[str] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            fn()
            xs.append(time.perf_counter() - t0)
        except Exception as exc:
            errs.append(f"{type(exc).__name__}: {exc}")
    return summarize(xs), errs[:3]


def run_target(target: str, args: argparse.Namespace, env_factory: Callable[..., DaytonaCUAEnv]) -> dict[str, Any]:
    rec: dict[str, Any] = {"target": target, "ok": False}
    env = None
    t_wall = time.perf_counter()
    try:
        env = env_factory(
            target=target,
            image=args.image,
            snapshot=args.snapshot,
            region=args.region,
            timeout=args.timeout,
            mac_reservation=args.mac_reservation,
            allow_mac_purchase=args.allow_mac_purchase,
        )
        obs = env.reset()
        rec["sandbox_id"] = env.sandbox.id if env.sandbox else None
        rec["boot"] = {k: env.timings[k][0] for k in ("create", "computer_use_start", "first_screenshot", "boot_total")}
        rec["screen"] = {"width": obs["width"], "height": obs["height"], "png_bytes": len(obs["screenshot"])}
        cx, cy = obs["width"] // 2, obs["height"] // 2
        rec["screenshot"], e1 = measure(lambda: env.screenshot(), args.iters)
        rec["click"], e2 = measure(lambda: env.click(cx, cy), args.iters)
        rec["type"], e3 = measure(lambda: env.type("a"), args.iters)
        rec["hotkey"], e4 = measure(lambda: env.hotkey(HOTKEY[target]), max(1, args.iters // 2))
        errs = {k: v for k, v in zip(("screenshot", "click", "type", "hotkey"), (e1, e2, e3, e4)) if v}
        if errs:
            rec["errors"] = errs
        try:
            rec["sdk_info"] = env.info()
        except Exception as exc:
            rec["sdk_info"] = f"{type(exc).__name__}: {exc}"
        try:
            r = env.exec(SPEC_CMDS[target], timeout_s=60)
            rec["specs"] = {"exit_code": r.exit_code, "output": r.output[-2000:]}
        except Exception as exc:
            rec["specs"] = {"error": f"{type(exc).__name__}: {exc}"}
        if not args.no_vlm:
            png = env.screenshot()["screenshot"]
            rec["vlm"] = vlm_call(args.vlm_url, png)
        rec["ok"] = not errs
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if env is not None:
            try:
                env.close()
                if env.timings.get("teardown"):
                    rec["teardown_s"] = env.timings["teardown"][0]
            except Exception as exc:
                rec["teardown_error"] = f"{type(exc).__name__}: {exc}"
                rec["ok"] = False
        rec["wall_s"] = time.perf_counter() - t_wall
    return rec


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["macos", "linux", "both"], default="linux")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--vlm-url", default="http://127.0.0.1:30001/v1/chat/completions")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--vlm-only", action="store_true", help="only test the VLM endpoint with a synthetic image")
    p.add_argument("--json-out", default=None, help="receipt path (default receipts/smoke_<utc>.json)")
    p.add_argument("--image", default=None, help="linux image or macOS image name (macOS default base-macos)")
    p.add_argument("--snapshot", default=None, help="linux memory snapshot name/id")
    p.add_argument("--region", default=None, help="v2 region (default $DAYTONA_TARGET)")
    p.add_argument("--timeout", type=int, default=600, help="create timeout seconds")
    p.add_argument("--mac-reservation", default=None, help="Mac reservation id (default $DAYTONA_MAC_RESERVATION)")
    p.add_argument("--allow-mac-purchase", action="store_true", help="let the macOS create buy a reservation")
    return p.parse_args(argv)


def write_receipt(receipt: dict[str, Any], path: str | None) -> Path:
    if path is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out = HERE / "receipts" / f"smoke_{receipt.get('mode', 'cua')}_{stamp}.json"
    else:
        out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str))
    return out


def main(argv: list[str] | None = None, env_factory: Callable[..., DaytonaCUAEnv] = DaytonaCUAEnv) -> int:
    args = parse_args(argv)
    key = (os.environ.get("DAYTONA_API_KEY") or "").strip()
    receipt: dict[str, Any] = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv[1:] if argv is None else argv,
        "daytona_sdk": sdk_version(),
        "api_key": key_fingerprint(key),
    }

    if args.vlm_only:
        receipt["mode"] = "vlm_only"
        receipt["vlm"] = vlm_call(args.vlm_url, synthetic_png())
        v = receipt["vlm"]
        if v["ok"]:
            print(
                f"VLM OK model={v['model']} chat={v['chat_latency_s']:.3f}s "
                f"finish={v['finish_reason']} reply={v['reply'][:120]!r}"
            )
        else:
            print(f"VLM FAIL {v['url']}: {v.get('error')}")
        print(f"receipt: {write_receipt(receipt, args.json_out)}")
        return 0 if v["ok"] else 1

    if not key:
        print("BLOCKED: DAYTONA_API_KEY not set")
        receipt["status"] = "blocked"
        receipt["reason"] = "DAYTONA_API_KEY not set"
        print(f"receipt: {write_receipt(receipt, args.json_out)}")
        return 2
    print(f"DAYTONA_API_KEY {key_fingerprint(key)}  sdk daytona=={receipt['daytona_sdk']}")

    targets = ["linux", "macos"] if args.target == "both" else [args.target]
    receipt["targets"] = {}
    try:
        for t in targets:
            print(f"[{t}] creating sandbox ...", flush=True)
            try:
                rec = run_target(t, args, env_factory)
            except MissingDaytonaKey as exc:  # defensive; key checked above
                print(f"BLOCKED: {exc}")
                return 2
            receipt["targets"][t] = rec
            if rec.get("boot"):
                b = rec["boot"]
                print(
                    f"[{t}] boot {b['boot_total']:.1f}s (create {b['create']:.1f} + cu_start "
                    f"{b['computer_use_start']:.1f} + first_shot {b['first_screenshot']:.1f})"
                )
                for k in ("screenshot", "click", "type", "hotkey"):
                    s = rec.get(k, {})
                    if s.get("n"):
                        print(f"[{t}] {k:<10} n={s['n']} p50={s['p50']*1000:.0f}ms p95={s['p95']*1000:.0f}ms")
                if "teardown_s" in rec:
                    print(f"[{t}] teardown {rec['teardown_s']:.1f}s")
                if "vlm" in rec:
                    v = rec["vlm"]
                    print(f"[{t}] vlm {'ok %.2fs' % v['chat_latency_s'] if v['ok'] else 'FAIL ' + str(v.get('error'))}")
            print(f"[{t}] {'OK' if rec['ok'] else 'FAIL ' + str(rec.get('error') or rec.get('errors'))}")
    finally:
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        print(f"receipt: {write_receipt(receipt, args.json_out)}")
    return 0 if all(r["ok"] for r in receipt["targets"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
