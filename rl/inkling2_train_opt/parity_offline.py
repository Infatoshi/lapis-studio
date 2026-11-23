"""Offline logprob parity check against recorded SGLang DSPARK8 server outputs.

Uses /work/parity_ref_cua.json (and /work/parity_ref_text.json) captured from the live
SGLang server so parity can be verified on GPU 3 without touching live serving GPUs.
"""

import argparse
import json
import time

import torch

from .model import load_model
from .prompt import TASKS, cua_messages, expand, render

MODEL_DIR = "/models/thinkingmachines--Inkling-2-Mini-heretic"


def compare_against_ref(model, ids, png, ref_entry, attn_impl, act_quant):
    out_ids = ref_entry["output_text_tokens"]
    inp, mask, images = expand(ids, [png] if png is not None else [])
    full = torch.cat([inp, torch.tensor(out_ids, device=inp.device)])
    mask = torch.cat([mask, torch.zeros(len(out_ids), dtype=torch.bool, device=inp.device)])
    first_text = int(torch.nonzero(mask)[-1]) + 1 if bool(mask.any()) else 0
    rows = torch.arange(first_text, full.shape[0] - 1, device=inp.device)
    per_row = ref_entry["per_row"]
    assert len(rows) == len(per_row), (len(rows), len(per_row))

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        logits = model(full, images=images, image_mask=mask, attn_impl=attn_impl, act_quant=act_quant, logit_rows=rows)
        torch.cuda.synchronize()
        fwd_s = time.time() - t0

    logp = torch.log_softmax(logits.float(), -1)
    targets = full[rows + 1]
    mine = logp.gather(1, targets[:, None])[:, 0].double().cpu()
    srv = torch.tensor([r["server"] for r in per_row], dtype=torch.float64)
    base_mine = torch.tensor([r["mine"] for r in per_row], dtype=torch.float64)
    n_out = len(out_ids)
    is_out_full = torch.zeros(len(rows), dtype=torch.bool)
    is_out_full[-n_out:] = True

    diff = (mine - srv).abs()
    diff_base = (mine - base_mine).abs()
    argmax = logp.argmax(-1).cpu()
    top1 = (argmax[is_out_full] == targets.cpu()[is_out_full]).float()
    return {
        "prompt": ref_entry["prompt"],
        "attn_impl": attn_impl,
        "act_quant": act_quant,
        "top1_match_output_rows": f"{int(top1.sum())}/{len(top1)}",
        "top1_match_count": int(top1.sum()),
        "top1_total": len(top1),
        "chosen_vs_server_mean": diff.mean().item(),
        "chosen_vs_server_max": diff.max().item(),
        "output_vs_server_mean": diff[is_out_full].mean().item(),
        "output_vs_server_max": diff[is_out_full].max().item(),
        "vs_baseline_eager_mean": diff_base.mean().item(),
        "vs_baseline_eager_max": diff_base.max().item(),
        "forward_s": fwd_s,
    }


def run_parity(model, cua_ref_path="/work/parity_ref_cua.json", text_ref_path="/work/parity_ref_text.json",
               image_path="/work/cua_1280x800.png", attn_impl="flex", act_quant=True):
    png = open(image_path, "rb").read()
    cua_ref = json.load(open(cua_ref_path))["results"]
    text_ref = json.load(open(text_ref_path))["results"]

    # Pick the eager:1 reference entry per prompt (all entries for same prompt have identical server logprobs)
    by_prompt = {}
    for r in cua_ref + text_ref:
        if r["prompt"] not in by_prompt and r["attn_impl"] == "eager" and r["act_quant"]:
            by_prompt[r["prompt"]] = r

    text_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain in two sentences why the sky is blue, then list three primes."},
    ]
    prompts = [
        ("cua_json_effort0", render(cua_messages(png, TASKS[0]), 0.0), png),
        ("cua_micro_think_effort0.05", render(cua_messages(png, TASKS[1]), 0.05), png),
        ("text_only_effort0", render(text_msgs, 0.0), None),
    ]
    results = []
    for name, ids, img in prompts:
        if name not in by_prompt:
            continue
        r = compare_against_ref(model, ids, img, by_prompt[name], attn_impl, act_quant)
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--attn-impl", default="flex")
    ap.add_argument("--no-act-quant", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model = load_model(args.model_dir)
    results = run_parity(model, attn_impl=args.attn_impl, act_quant=not args.no_act_quant)
    for r in results:
        print(json.dumps(r))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
