"""Hot-reload a merged LoRA checkpoint into a running SGLang server and show the logprobs move.

Host-side, stdlib only:
  python3 reload_check.py --url http://127.0.0.1:30101 --key-file admin_key \
      --samples train_samples.json --model-path /merged/ckpt --train-json train.json --out reload.json
"""

import argparse
import base64
import json
import time
import urllib.request

IMAGE_TOKENS = 640  # 1280x800 -> 32 x 20 tokens


def post(url, path, body, key=None, timeout=900):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def completion_logprobs(url, samples):
    per = []
    for s in samples:
        img = "data:image/png;base64," + base64.b64encode(open(s["image"].replace("/work/", ""), "rb").read()).decode()
        n_prompt = len(s["prompt"]) - 1 + IMAGE_TOKENS
        body = {"input_ids": s["prompt"] + s["completion"], "image_data": [img],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True, "logprob_start_len": n_prompt - 1}
        out, _ = post(url, "/generate", body)
        lps = [x[0] for x in out["meta_info"]["input_token_logprobs"]][1:]
        assert len(lps) == len(s["completion"]), (len(lps), len(s["completion"]))
        per.append(lps)
    flat = [x for p in per for x in p]
    return {"mean": sum(flat) / len(flat), "per_sample_mean": [sum(p) / len(p) for p in per], "tokens": len(flat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30101")
    ap.add_argument("--key-file", default="admin_key")
    ap.add_argument("--samples", default="train_samples.json")
    ap.add_argument("--model-path", default="/merged/ckpt")
    ap.add_argument("--train-json", default="train.json")
    ap.add_argument("--out", default="reload.json")
    args = ap.parse_args()
    key = open(args.key_file).read().strip()
    samples = json.load(open(args.samples))
    before = completion_logprobs(args.url, samples)
    print("before", before["mean"])
    upd, upd_s = post(args.url, "/update_weights_from_disk", {"model_path": args.model_path, "flush_cache": True}, key=key)
    print("update", upd, f"{upd_s:.1f}s")
    after = completion_logprobs(args.url, samples)
    print("after", after["mean"])
    sanity, _ = post(args.url, "/generate", {"text": "The capital of France is", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}})
    train = json.load(open(args.train_json))
    res = {"update_response": upd, "update_s": upd_s, "server_target_logprob_before": before,
           "server_target_logprob_after": after, "trainer_target_logprob_after": train.get("trained_target_logprob_mean"),
           "trainer_losses": train.get("losses"), "sanity_text_after_reload": sanity.get("text")}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("server_target_logprob_before", "server_target_logprob_after")}, indent=1))


if __name__ == "__main__":
    main()
