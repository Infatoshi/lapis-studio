"""Real MacBook CUA traces -> deduped, chronologically split grounding + planner data (strictly local).

Sources (~/.lapis-next/cua_traces, recorded live by Studio; the file keeps growing, so the row count read is frozen
into the manifest):
  human_traces.jsonl   shadow-recorded human clicks: app, AX title/description/role, AX bbox_1000, click point_1000
  traces.jsonl         Studio's own executor steps: goal, action, target, tier (L1_AX / L3_VLM / ...), point_1000
  planner_turns.jsonl  planner replies ([[ACTION:cua_...]] tags) to real goals

Rules:
  * dedupe by (app, target, round(x/10), round(y/10)) on the click point, keeping the OLDEST occurrence, before splitting;
  * strict time split on the deduped, eligible rows: oldest 80% -> mac_train, newest 20% -> mac_heldout_eval, cut at
    one timestamp so no heldout row predates any train row;
  * labels: grounding target = AX label, answer = AX box centre (Studio's grounder is asked for the element centre), only
    for small boxes (<= 4% of the screen) that contain the actual click; executor rows only when the point came from AX
    (L1_AX), never from a model's own L3_VLM guess; planner turns are unverified model output, kept for eval only;
  * privacy: chat / meeting / login apps are dropped; frames are referenced in place (never copied), outputs go under
    ~/cua_datasets/mac (refused if inside a git work tree). Nothing here is uploaded anywhere.

  uv run python data_prep/prepare_mac_traces.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

TRACES = os.path.expanduser("~/.lapis-next/cua_traces")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_APPS = {"Messages", "Discord", "Slack", "Zoom", "loginwindow", "Window Server", "Mail", "WhatsApp", "Telegram",
             "UserNotificationCenter", "1Password", "Keychain Access", "System Settings", "SecurityAgent"}
APP_ALIASES = {"Google Chrome": "Chrome", "Code": "VS Code"}
MAX_BOX_AREA = 40_000  # in 1000x1000 units: 4% of the screen
TRAIN_FRAC = 0.8
GROUND_MAX_WIDTH = 1536  # Studio captures the display JPEG at <= 1536 px wide before grounding (cua_executor.cpp)


def grounder_prompt(target: str) -> str:
    """Verbatim from buildInklingGroundRequest in src/mac/cua_executor.cpp (target.simplified())."""
    return ("Locate the exact center of UI element: \"%s\". Reply ONLY with JSON {\"point\":[x,y]} using integer "
            "coordinates 0-1000 (x=0 left, y=0 top), or {\"not_found\":true} if it is not clearly visible on screen."
            % " ".join(target.split()))


def studio_system_prompt() -> str | None:
    path = os.path.join(REPO, "src", "llm", "prefix.cpp")
    if not os.path.exists(path):
        return None
    src = open(path).read()
    start = src.index("kCompatSystemPrompt = QStringLiteral(")
    lits = re.findall(r'"((?:[^"\\]|\\.)*)"', src[start: src.index(");", start)])
    return "".join(lits).replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def clean_label(r: dict) -> str | None:
    lab = " ".join((r.get("title") or r.get("description") or "").split())
    if not (2 <= len(lab) <= 60):
        return None
    if "@" in lab or re.search(r"\d{6,}", lab):  # emails, phone / account numbers
        return None
    return lab


def inside_git(path: str) -> bool:
    d = path
    while not os.path.isdir(d):
        d = os.path.dirname(d)
    r = subprocess.run(["git", "-C", d, "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a half-written last line from the live recorder
    return rows


def image_dims(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def human_candidates(rows: list[dict], stats: collections.Counter) -> list[dict]:
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: r.get("ts", "")):
        app = r.get("app") or ""
        if app in SKIP_APPS:
            stats["drop_private_app"] += 1
            continue
        lab = clean_label(r)
        if lab is None:
            stats["drop_no_label"] += 1
            continue
        px, py = r.get("point_1000") or (None, None)
        box = r.get("bbox_1000")
        if px is None or not box:
            stats["drop_no_geometry"] += 1
            continue
        key = (app, lab, round(px / 10), round(py / 10))
        if key in seen:
            stats["drop_duplicate"] += 1
            continue
        seen.add(key)
        x0, y0, x1, y1 = box
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            stats["drop_click_outside_box"] += 1
            continue
        if (x1 - x0) * (y1 - y0) > MAX_BOX_AREA or (x1 - x0) < 3 or (y1 - y0) < 3:
            stats["drop_box_size"] += 1
            continue
        img = os.path.join(TRACES, r.get("image") or "")
        if not r.get("image") or not os.path.exists(img):
            stats["drop_missing_frame"] += 1
            continue
        out.append({"ts": r["ts"], "app": app, "target": lab, "role": r.get("role"), "bbox_1000": box,
                    "point_1000": [px, py], "image_path": img, "display_w": r.get("display_w"),
                    "display_h": r.get("display_h"), "source": "mac_human_click", "button": r.get("button")})
    return out


def agent_candidates(rows: list[dict], stats: collections.Counter) -> list[dict]:
    out = []
    for r in rows:
        if r.get("action") not in ("click", "type") or not r.get("ok"):
            continue
        if r.get("tier") != "L1_AX":
            stats["agent_drop_not_ax"] += 1  # L3_VLM points are the model's own guesses, not labels
            continue
        img = os.path.join(TRACES, r.get("before_image") or "")
        if not r.get("before_image") or not os.path.exists(img) or not r.get("point_1000"):
            stats["agent_drop_missing"] += 1
            continue
        target = str(r.get("target") or "")
        app, _, lab = target.partition("|") if "|" in target else (r.get("app") or "", "", target)
        lab = " ".join(lab.split())
        if not lab or (r.get("app") or app) in SKIP_APPS:
            stats["agent_drop_label_or_private"] += 1
            continue
        box = r.get("bbox_1000")
        out.append({"ts": r["ts"], "app": r.get("app") or app, "target": lab, "role": None,
                    "bbox_1000": box, "point_1000": r["point_1000"], "image_path": img, "display_w": r.get("display_w"),
                    "display_h": r.get("display_h"), "source": "mac_agent_ax", "goal": r.get("goal")})
    return out


def to_sample(c: dict, split: str) -> dict:
    if c.get("bbox_1000"):
        x0, y0, x1, y1 = c["bbox_1000"]
        center = [round((x0 + x1) / 2), round((y0 + y1) / 2)]
    else:
        center = list(c["point_1000"])
    w, h = image_dims(c["image_path"])
    app = APP_ALIASES.get(c["app"], c["app"])
    sid = hashlib.sha1(f"{c['source']}|{c['ts']}|{c['image_path']}".encode()).hexdigest()[:12]
    return {
        "id": f"mac_{sid}", "ts": c["ts"], "split": split, "source": c["source"],
        "image_path": c["image_path"], "width": w, "height": h, "display_w": c["display_w"], "display_h": c["display_h"],
        "app": c["app"], "instruction": c["target"], "role": c.get("role"),
        "bbox_1000": c.get("bbox_1000"), "center_1000": center, "click_1000": c["point_1000"],
        # Studio L3_VLM grounder turn (image resized to <= GROUND_MAX_WIDTH px wide as JPEG at train time)
        "grounder": {"prompt": grounder_prompt(c["target"]), "target": json.dumps({"point": center}, separators=(",", ":")),
                     "resize_max_width": GROUND_MAX_WIDTH},
        # Studio planner turn: the goal is synthetic for human clicks (the human's intent was not recorded)
        "planner": {"goal": c.get("goal") or f"Click {c['target']} in {app}.", "goal_synthetic": not c.get("goal"),
                    "target": f"[[ACTION:cua_click:{app}|{c['target']}]]"},
    }


def time_split(cands: list[dict], frac: float = TRAIN_FRAC) -> tuple[list[dict], list[dict], str]:
    cands = sorted(cands, key=lambda c: c["ts"])
    if not cands:
        return [], [], ""
    cut_ts = cands[min(len(cands) - 1, int(len(cands) * frac))]["ts"]
    train = [c for c in cands if c["ts"] < cut_ts]
    held = [c for c in cands if c["ts"] >= cut_ts]
    return train, held, cut_ts


def planner_rows(rows: list[dict], cut_ts: str) -> list[dict]:
    out = []
    for r in rows:
        if "raw_reply" not in r or not r.get("goal"):
            continue
        tags = re.findall(r"\[\[ACTION:([^\]]+)\]\]", r["raw_reply"])
        out.append({"ts": r["ts"], "split": "mac_train" if r["ts"] < cut_ts else "mac_heldout_eval", "goal": r["goal"],
                    "model_reply": r["raw_reply"], "actions": tags, "model": r.get("model"), "effort": r.get("effort"),
                    "label_quality": "unverified_model_output"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/cua_datasets/mac"))
    args = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(args.out))
    if inside_git(out):
        sys.exit(f"refusing to write personal frames' index inside a git work tree: {out}")
    os.makedirs(out, exist_ok=True)

    human = read_jsonl(os.path.join(TRACES, "human_traces.jsonl"))
    agent = read_jsonl(os.path.join(TRACES, "traces.jsonl"))
    planner = read_jsonl(os.path.join(TRACES, "planner_turns.jsonl"))
    stats: collections.Counter = collections.Counter()
    cands = human_candidates(human, stats) + agent_candidates(agent, stats)
    # cross-source dedupe on the same key, oldest wins
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: c["ts"]):
        k = (c["app"], c["target"], round(c["point_1000"][0] / 10), round(c["point_1000"][1] / 10))
        if k in seen:
            stats["drop_duplicate_cross_source"] += 1
            continue
        seen.add(k)
        uniq.append(c)
    train, held, cut_ts = time_split(uniq)
    files = {"mac_train.jsonl": [to_sample(c, "mac_train") for c in train],
             "mac_heldout_eval.jsonl": [to_sample(c, "mac_heldout_eval") for c in held]}
    for name, rows in files.items():
        with open(os.path.join(out, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    prow = planner_rows(planner, cut_ts)
    with open(os.path.join(out, "mac_planner_turns.jsonl"), "w") as f:
        for r in prow:
            f.write(json.dumps(r) + "\n")
    system = studio_system_prompt()
    tr, he = files["mac_train.jsonl"], files["mac_heldout_eval.jsonl"]
    key = lambda r: (r["app"], r["instruction"], round(r["click_1000"][0] / 10), round(r["click_1000"][1] / 10))  # noqa: E731
    manifest = {
        "frozen_inputs": {"human_traces_rows": len(human), "human_last_ts": human[-1]["ts"] if human else None,
                          "agent_trace_rows": len(agent), "planner_turn_rows": len(planner)},
        "cut_ts": cut_ts, "train_frac": TRAIN_FRAC,
        "counts": {"mac_train": len(tr), "mac_heldout_eval": len(he),
                   "planner_turns": dict(collections.Counter(r["split"] for r in prow))},
        "by_source": {s: dict(collections.Counter(r["source"] for r in rows)) for s, rows in
                      (("mac_train", tr), ("mac_heldout_eval", he))},
        "top_apps": {s: collections.Counter(r["app"] for r in rows).most_common(8) for s, rows in
                     (("mac_train", tr), ("mac_heldout_eval", he))},
        "checks": {"train_max_ts": max((r["ts"] for r in tr), default=None),
                   "heldout_min_ts": min((r["ts"] for r in he), default=None),
                   "key_overlap": len({key(r) for r in tr} & {key(r) for r in he}),
                   "image_overlap": len({r["image_path"] for r in tr} & {r["image_path"] for r in he})},
        "drops": dict(stats),
        "studio_system_prompt_sha1": hashlib.sha1(system.encode()).hexdigest() if system else None,
        "privacy": "frames referenced in place under ~/.lapis-next/cua_traces; never commit, never upload to Daytona",
    }
    if system:
        with open(os.path.join(out, "studio_system_prompt.txt"), "w") as f:
            f.write(system)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    assert manifest["checks"]["key_overlap"] == 0 and manifest["checks"]["image_overlap"] == 0
    assert not tr or not he or manifest["checks"]["train_max_ts"] < manifest["checks"]["heldout_min_ts"]
    print(json.dumps({k: manifest[k] for k in ("frozen_inputs", "cut_ts", "counts", "by_source", "checks", "drops")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
