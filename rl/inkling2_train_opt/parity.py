"""Logprob parity: trainable PyTorch inkling2_loop vs a live SGLang server.

Sends the identical token ids + 1280x800 screenshot to SGLang /generate (greedy, 32 tokens,
prompt + output logprobs, top-20), then teacher-forces the same sequence through the PyTorch
port and compares log-probabilities of the same tokens.

usage (inside the sglang image, GPU 3):  python3 -m inkling2_train.parity --server http://127.0.0.1:30101
"""

import argparse
import json
import time
import urllib.request

import torch

from .model import load_model
from .prompt import TASKS, cua_messages, data_uri, expand, render

MODEL_DIR = "/models/thinkingmachines--Inkling-2-Mini-heretic"


def server_generate(server, ids, png, n_tokens, top_k=20):
    payload = {
        "input_ids": ids,
        **({"image_data": [data_uri(png)]} if png is not None else {}),
        "sampling_params": {"temperature": 0.0, "max_new_tokens": n_tokens, "ignore_eos": True,
                            "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": top_k,
    }
    req = urllib.request.Request(server + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def compare(model, ids, png, resp, attn_impl, act_quant):
    meta = resp["meta_info"]
    out_ids = [t[1] for t in meta["output_token_logprobs"]]
    inp, mask, images = expand(ids, [png] if png is not None else [])
    full = torch.cat([inp, torch.tensor(out_ids, device=inp.device)])
    mask = torch.cat([mask, torch.zeros(len(out_ids), dtype=torch.bool, device=inp.device)])
    n_prompt = inp.shape[0]
    assert len(meta["input_token_logprobs"]) == n_prompt, (len(meta["input_token_logprobs"]), n_prompt)
    # rows whose next token is a text token: prompt text rows after the image, and outputs
    first_text = int(torch.nonzero(mask)[-1]) + 1 if bool(mask.any()) else 0
    rows = torch.arange(first_text, full.shape[0] - 1, device=inp.device)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        logits = model(full, images=images, image_mask=mask, attn_impl=attn_impl, act_quant=act_quant, logit_rows=rows)
        torch.cuda.synchronize()
        fwd_s = time.time() - t0
    logp = torch.log_softmax(logits, -1)
    targets = full[rows + 1]
    mine = logp.gather(1, targets[:, None])[:, 0].double().cpu()
    srv_in = [x[0] for x in meta["input_token_logprobs"]]
    srv_out = [x[0] for x in meta["output_token_logprobs"]]
    srv_all = srv_in[first_text + 1:] + srv_out  # logprob of token (row+1)
    srv = torch.tensor(srv_all[: len(rows)], dtype=torch.float64)
    n_out = len(out_ids)
    is_out = torch.zeros(len(rows), dtype=torch.bool)
    is_out[-(n_out - 1):] = True  # rows predicting output tokens 1..n-1 (token 0 is predicted by the last prompt row)
    is_out_full = is_out.clone()
    is_out_full[-n_out] = True  # last prompt row predicts output token 0
    diff = (mine - srv).abs()
    # greedy top-1 agreement at output-producing rows: my argmax == server's greedy token
    argmax = logp.argmax(-1).cpu()
    top1 = (argmax[is_out_full] == targets.cpu()[is_out_full]).float()
    # top-20: logprob diffs over shared tokens, plus lse-centred logit diffs
    tops_srv = (meta["input_top_logprobs"][first_text + 1:] + meta["output_top_logprobs"])[: len(rows)]
    lp_d, logit_d, overlap = [], [], []
    for r, top in enumerate(tops_srv):
        if not top:
            continue
        toks = torch.tensor([t[1] for t in top])
        s_lp = torch.tensor([t[0] for t in top], dtype=torch.float64)
        m_lp = logp[r, toks.to(logp.device)].double().cpu()
        d = m_lp - s_lp
        lp_d.append(d.abs())
        logit_d.append((d - d.mean()).abs())
        mine_top = set(torch.topk(logp[r], len(top)).indices.tolist())
        overlap.append(len(mine_top & set(toks.tolist())) / len(top))
    lp_d, logit_d = torch.cat(lp_d), torch.cat(logit_d)
    return {
        "attn_impl": attn_impl,
        "act_quant": act_quant,
        "prompt_tokens": n_prompt,
        "image_tokens": int(mask.sum()),
        "compared_rows": len(rows),
        "output_tokens": n_out,
        "output_text_tokens": out_ids,
        "chosen_token_logprob_absdiff": {"max": diff.max().item(), "mean": diff.mean().item(),
                                         "output_rows_max": diff[is_out_full].max().item(),
                                         "output_rows_mean": diff[is_out_full].mean().item()},
        "top20_logprob_absdiff": {"max": lp_d.max().item(), "mean": lp_d.mean().item()},
        "top20_centered_logit_absdiff": {"max": logit_d.max().item(), "mean": logit_d.mean().item()},
        "top1_match_output_rows": f"{int(top1.sum())}/{len(top1)}",
        "top1_match_rate": top1.mean().item(),
        "top20_set_overlap_mean": sum(overlap) / len(overlap),
        "forward_s": fwd_s,
        "per_row": [{"row": int(r), "next_token": int(t), "mine": float(m), "server": float(s)}
                    for r, t, m, s in zip(rows.tolist(), targets.tolist(), mine.tolist(), srv.tolist())],
    }


def server_self_check(server, responses, n_tokens):
    """The same request again, now prefilled in one batch with 7 other prompts: how far the
    server's own logprobs move when only the batch composition (GEMM shapes) changes."""
    from concurrent.futures import ThreadPoolExecutor

    out = {}
    pool = json.load(open("/work/bench_assets/text_pool.json"))
    for name, ids, img, resp in responses:
        fillers = [pool[i * 997 : i * 997 + 700] for i in range(1, 8)]
        with ThreadPoolExecutor(8) as ex:
            futs = [ex.submit(server_generate, server, ids, img, n_tokens)]
            futs += [ex.submit(server_generate, server, f, None, n_tokens) for f in fillers]
            again = futs[0].result()
            [f.result() for f in futs[1:]]
        a, b = resp["meta_info"], again["meta_info"]
        # same rows as compare(): text rows after the image (or all rows if no image), and only up
        # to the first output token where the two greedy continuations diverge (identical context).
        start = (ids.index(-101) + 640 + 1) if img is not None else 1
        lp_a = a["input_token_logprobs"] + a["output_token_logprobs"]
        lp_b = b["input_token_logprobs"] + b["output_token_logprobs"]
        top_a = a["input_top_logprobs"] + a["output_top_logprobs"]
        top_b = b["input_top_logprobs"] + b["output_top_logprobs"]
        end = len(lp_a)
        for i in range(len(a["input_token_logprobs"]), len(lp_a)):
            if lp_a[i][1] != lp_b[i][1]:
                end = i + 1  # the diverging row itself still has identical context
                break
        d_lp = [abs(lp_a[i][0] - lp_b[i][0]) for i in range(start, end)]
        top_d = []
        for i in range(start, end):
            if top_a[i] and top_b[i]:
                mb = {t[1]: t[0] for t in top_b[i]}
                top_d += [abs(t[0] - mb[t[1]]) for t in top_a[i] if t[1] in mb]
        out[name] = {"same_greedy_tokens": end == len(lp_a), "compared_rows": end - start,
                     "chosen_logprob_absdiff_mean": sum(d_lp) / len(d_lp), "chosen_logprob_absdiff_max": max(d_lp),
                     "top20_logprob_absdiff_mean": sum(top_d) / len(top_d), "top20_logprob_absdiff_max": max(top_d)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:30101")
    ap.add_argument("--image", default="/work/cua_1280x800.png")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", default="/work/parity.json")
    ap.add_argument("--variants", default="eager:1,eager:0,flex:1")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--server-self-check", action="store_true")
    args = ap.parse_args()
    png = open(args.image, "rb").read()
    prompts = [("cua_json_effort0", render(cua_messages(png, TASKS[0]), 0.0), png),
               ("cua_micro_think_effort0.05", render(cua_messages(png, TASKS[1]), 0.05), png)]
    if args.text_only:
        text_msgs = [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": "Explain in two sentences why the sky is blue, then list three primes."}]
        prompts = [("text_only_effort0", render(text_msgs, 0.0), None)]
    responses = []
    for name, ids, img in prompts:
        t0 = time.time()
        resp = server_generate(args.server, ids, img, args.tokens)
        print(f"[server] {name}: {time.time() - t0:.2f}s text={resp['text'][:120]!r}")
        responses.append((name, ids, img, resp))
    self_check = server_self_check(args.server, responses, args.tokens) if args.server_self_check else None
    if self_check:
        print("[server self-check]", json.dumps(self_check))
    model = load_model(args.model_dir)
    results = []
    for name, ids, img, resp in responses:
        for v in args.variants.split(","):
            impl, aq = v.split(":")
            r = compare(model, ids, img, resp, impl, bool(int(aq)))
            r["prompt"] = name
            r["server_text"] = resp["text"]
            print(json.dumps({k: v for k, v in r.items() if k not in ("output_text_tokens", "server_text")}))
            results.append(r)
    json.dump({"model_dir": args.model_dir, "server": args.server, "results": results,
               "server_self_check": self_check}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
