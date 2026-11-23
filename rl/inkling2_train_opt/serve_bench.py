"""Stdlib-only SGLang client for the inference roofline and the DSPARK sweep (runs on the host).

  python3 serve_bench.py prefill --url http://127.0.0.1:30101 --out prefill.json
  python3 serve_bench.py decode  --url ... --conc 1,8,16,32 --out decode.json
  python3 serve_bench.py cua     --url ... --conc 1,16,32 --out cua.json

Every request gets a cache-busting token prefix and/or a unique screenshot, so radix-cache and
vision-embedding-cache hits cannot inflate the numbers.  Timings come from the server's
meta_info (prefill_latency, first_token_latency, decode_throughput, spec_accept_length) plus the
client wall clock for aggregate throughput.
"""

import argparse
import base64
import json
import os
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_assets")
POOL = None
PROMPTS = None
IMGS = []
_img_i = [0]


def load_assets():
    global POOL, PROMPTS, IMGS
    POOL = json.load(open(f"{ASSETS}/text_pool.json"))
    PROMPTS = json.load(open(f"{ASSETS}/prompts.json"))
    d = f"{ASSETS}/imgs"
    IMGS = [base64.b64encode(open(f"{d}/{f}", "rb").read()).decode() for f in sorted(os.listdir(d))]


def next_image():
    i = _img_i[0] % len(IMGS)
    _img_i[0] += 1
    return "data:image/png;base64," + IMGS[i]


def prefix(n, rng):
    if n <= 0:
        return []
    s = rng.randrange(0, len(POOL) - n)
    return POOL[s : s + n]


def post(url, body, timeout=900):
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["client_s"] = time.perf_counter() - t0
    return out


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": xs[len(xs) // 2], "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))],
            "min": xs[0], "max": xs[-1]}


def run_wave(url, make_body, conc, per_worker):
    """Closed loop: conc workers, each sends per_worker requests back to back."""
    results = []

    def worker(w):
        out = []
        for _ in range(per_worker):
            out.append(post(url, make_body()))
        return out

    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        for rs in ex.map(worker, range(conc)):
            results.extend(rs)
    wall = time.perf_counter() - t0
    m = [r["meta_info"] for r in results]
    comp = sum(x["completion_tokens"] for x in m)
    prompt = sum(x["prompt_tokens"] for x in m)
    cached = sum(x.get("cached_tokens") or 0 for x in m)
    return {
        "concurrency": conc,
        "requests": len(results),
        "wall_s": wall,
        "agg_output_tok_s": comp / wall,
        "agg_prompt_tok_s": (prompt - cached) / wall,
        "agg_total_tok_s": (comp + prompt - cached) / wall,
        "completion_tokens_mean": comp / len(results),
        "prompt_tokens_mean": prompt / len(results),
        "cached_prompt_tokens_mean": cached / len(results),
        "ttft_s": summary([x.get("first_token_latency") for x in m]),
        "prefill_latency_s": summary([x.get("prefill_latency") for x in m]),
        "queue_s": summary([x.get("queue_time") for x in m]),
        "e2e_s": summary([x.get("e2e_latency") for x in m]),
        "client_e2e_s": summary([r["client_s"] for r in results]),
        "per_req_decode_tok_s": summary([x.get("decode_throughput") for x in m]),
        "spec_accept_length": summary([x.get("spec_accept_length") for x in m]),
        "sample_text": results[0]["text"][:160],
    }


def cmd_prefill(args):
    rng = random.Random(1)
    cua = PROMPTS["cua_json"][0]
    out = []
    warm = {"input_ids": prefix(512, rng) + cua, "image_data": [next_image()], "sampling_params": {"max_new_tokens": 1}}
    for _ in range(3):
        post(args.url, dict(warm, input_ids=prefix(512, rng) + cua, image_data=[next_image()]))
    for n in (1024, 2048, 4096, 6144):
        for with_image in (True, False):
            lat, ttft, ptok = [], [], []
            for _ in range(args.reps):
                if with_image:
                    body = {"input_ids": prefix(n, rng) + cua, "image_data": [next_image()]}
                else:  # same total length; the 640 image rows become text rows
                    text_cua = [t for t in cua if t != -101]
                    body = {"input_ids": prefix(n + 640, rng) + text_cua}
                body["sampling_params"] = {"max_new_tokens": 1, "temperature": 0.0}
                r = post(args.url, body)
                m = r["meta_info"]
                lat.append(m["prefill_latency"])
                ttft.append(m["first_token_latency"])
                ptok.append(m["prompt_tokens"] - (m.get("cached_tokens") or 0))
            row = {"text_prefix_tokens": n, "image": with_image, "prompt_tokens": statistics.fmean(ptok),
                   "prefill_latency_s": summary(lat), "ttft_s": summary(ttft),
                   "prefill_tok_s_median": statistics.fmean(ptok) / sorted(lat)[len(lat) // 2]}
            print(json.dumps(row))
            out.append(row)
    json.dump({"mode": "prefill", "url": args.url, "rows": out}, open(args.out, "w"), indent=1)


def cmd_decode(args):
    rng = random.Random(2)
    lf = PROMPTS["longform"]
    rows = []

    def body():
        return {"input_ids": prefix(512, rng) + lf,
                "sampling_params": {"max_new_tokens": args.max_new, "temperature": 0.0, "ignore_eos": True}}

    post(args.url, body())
    for c in [int(x) for x in args.conc.split(",")]:
        row = run_wave(args.url, body, c, args.per_worker)
        print(json.dumps({k: row[k] for k in ("concurrency", "agg_output_tok_s", "per_req_decode_tok_s", "spec_accept_length")}))
        rows.append(row)
    json.dump({"mode": "decode", "url": args.url, "max_new": args.max_new, "rows": rows}, open(args.out, "w"), indent=1)


def cmd_cua(args):
    rng = random.Random(3)
    rows = []
    for kind, max_new in (("cua_json", 64), ("cua_micro", 256)):
        def body(kind=kind, max_new=max_new):
            return {"input_ids": PROMPTS[kind][rng.randrange(4)], "image_data": [next_image()],
                    "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0}}

        post(args.url, body())
        for c in [int(x) for x in args.conc.split(",")]:
            per = args.per_worker if c > 1 else max(args.per_worker, 8)
            row = run_wave(args.url, body, c, per)
            row["workload"] = kind
            row["max_new_tokens"] = max_new
            print(json.dumps({k: row[k] for k in ("workload", "concurrency", "agg_output_tok_s", "agg_total_tok_s", "ttft_s", "e2e_s", "per_req_decode_tok_s", "spec_accept_length")})[:600])
            rows.append(row)
    json.dump({"mode": "cua", "url": args.url, "label": args.label, "rows": rows}, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill", "decode", "cua"])
    ap.add_argument("--url", default="http://127.0.0.1:30101")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", default="1,8,16,32")
    ap.add_argument("--per-worker", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    load_assets()
    {"prefill": cmd_prefill, "decode": cmd_decode, "cua": cmd_cua}[args.mode](args)


if __name__ == "__main__":
    main()
