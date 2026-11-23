"""LoRA r=128 training benchmark of the trainable inkling2_loop on real weights + CUA screenshots.

Synthetic CUA SFT: packed sequences of (1280x800 screenshot + CUA prompt -> JSON action) samples,
loss on the action tokens only.  Adapters: attention q/k/v/o (24 layers), MoE routers (24),
vision projector (proj_in, proj_out); everything else frozen.  Gradient checkpointing per layer
visit (48 visits), FlexAttention, emulated NVFP4/MXFP8 expert activations (straight-through).

python3 -m inkling2_train.train_bench --steps 10 --samples 4 --out /work/train.json --save-lora /work/lora.pt
"""

import argparse
import io
import json
import math
import os
import time

import torch
from PIL import Image

from .lora import lora_parameters
from .model import load_model
from .prompt import TASKS, cua_messages, expand, render

MODEL_DIR = "/models/thinkingmachines--Inkling-2-Mini-heretic"
TARGETS = (
    '{"action": "click", "x": 482, "y": 400}',
    '{"action": "done", "answer": "p009"}',
    '{"action": "click", "x": 471, "y": 632}',
    '{"action": "done", "answer": "p005"}',
)
END_SAMPLING = 200006


def build_samples(n, img_dir="/work/bench_assets/imgs"):
    from sglang.srt.parser.inkling_renderer import load_tml_renderers

    tok = load_tml_renderers("inkling2").tokenizer
    files = sorted(os.listdir(img_dir))
    samples = []
    for i in range(n):
        png = open(os.path.join(img_dir, files[i % len(files)]), "rb").read()
        prompt = render(cua_messages(png, TASKS[i % 4]), 0.0)
        completion = [200001, 200004] + list(tok.encode_ordinary(TARGETS[i % 4])) + [200010, END_SAMPLING]
        samples.append((prompt, completion, png))
    json.dump([{"prompt": p, "completion": c, "image": os.path.join(img_dir, files[i % len(files)])}
               for i, (p, c, _) in enumerate(samples)], open("/work/train_samples.json", "w"))
    return samples


def pack(samples, pad_to=128, device="cuda"):
    ids, mask, doc, pos, loss_rows, images = [], [], [], [], [], []
    for d, (prompt, completion, png) in enumerate(samples):
        inp, m, imgs = expand(prompt + completion, [png], device)
        start = sum(x.shape[0] for x in ids)
        n_prompt = inp.shape[0] - len(completion)
        loss_rows.extend(range(start + n_prompt - 1, start + inp.shape[0] - 1))
        ids.append(inp)
        mask.append(m)
        doc.append(torch.full_like(inp, d, dtype=torch.int32))
        pos.append(torch.arange(inp.shape[0], device=device))
        images.extend(imgs)
    t = sum(x.shape[0] for x in ids)
    pad = (-t) % pad_to
    if pad:
        ids.append(torch.zeros(pad, dtype=torch.long, device=device))
        mask.append(torch.zeros(pad, dtype=torch.bool, device=device))
        doc.append(torch.full((pad,), 10_000, dtype=torch.int32, device=device))
        pos.append(torch.arange(pad, device=device))
    ids, mask, doc, pos = (torch.cat(x) for x in (ids, mask, doc, pos))
    rows = torch.tensor(loss_rows, device=device)
    return {"input_ids": ids, "image_mask": mask, "doc": doc, "positions": pos, "images": images,
            "rows": rows, "targets": ids[rows + 1], "real_tokens": t}


def step_loss(model, batch, act_quant=True):
    logits = model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
                   image_mask=batch["image_mask"], attn_impl="flex", act_quant=act_quant, grad_ckpt=True,
                   logit_rows=batch["rows"])
    return torch.nn.functional.cross_entropy(logits, batch["targets"])


def analytic_flops(batch, model_cfg):
    """Forward FLOPs (2 x MACs) of the packed sequence; see the module docstring of the receipt."""
    d = 2304
    attn_proj = d * (3072 + 1024 + 1024 + 384 + 3072 + 24) + 3072 * d
    experts = 8 * (2304 * 2304 + 1152 * 2304)
    router = d * 128
    rel = 24 * 16 * 1024
    per_tok_visit = attn_proj + experts + router + rel
    doc = batch["doc"]
    real = doc < 10_000
    # causal context per token: position + 1 within its doc (local layers capped at 1024)
    ctx_full = (batch["positions"][real] + 1).float()
    ctx_local = ctx_full.clamp(max=1024)
    visits_local = 40  # 20 local layers: 16 outside the core... computed exactly below
    local_ids = set(model_cfg["local_layer_ids"])
    core = range(6, 18)
    visits = [i for i in range(24)] + [i for i in core] * 2
    n_local = sum(1 for i in visits if i in local_ids)
    n_global = len(visits) - n_local
    attn_core = 2 * 24 * 128 * (n_local * ctx_local.sum() + n_global * ctx_full.sum())  # QK + PV MACs
    controllers = 24 * 4 * (2 * d * 64 + 2 * 64 * d)
    tokens = int(real.sum())
    llm_macs = tokens * (len(visits) * per_tok_visit + controllers) + float(attn_core)
    n_img = len(batch["images"])
    vit_tokens = 2561
    vit_macs = n_img * (vit_tokens * 23 * (4 * 1024 * 1024 + 8 * 1024 * 1024) + 23 * 2 * vit_tokens**2 * 1024 + 2560 * 1200 * 1024)
    proj_macs = n_img * 640 * (4096 * 4096 + 4096 * 2304)
    head_macs = len(batch["rows"]) * d * 201024
    return {"llm_fwd_flops": 2 * llm_macs, "vit_fwd_flops": 2 * vit_macs, "projector_fwd_flops": 2 * proj_macs,
            "lm_head_fwd_flops": 2 * head_macs, "tokens": tokens, "layer_visits": len(visits),
            "expert_weight_bytes_bf16_per_visit": 128 * (2304 * 2304 + 2304 * 1152) * 2}


def categorize(name):
    n = name.lower()
    if "flex" in n or "tem_fused" in n:
        return "attention (flex fwd/bwd)"
    if "grouped" in n or "groupgemm" in n:
        return "experts grouped_mm"
    if "fmha" in n or "flash" in n or "cudnn" in n or "sdpa" in n or "attention" in n:
        return "vision SDPA"
    if "adam" in n or "multi_tensor" in n:
        return "optimizer"
    if "gemm" in n or "nvjet" in n or "cutlass" in n or "xmma" in n or "sm90" in n or "sm100" in n or "sm120" in n or "cublas" in n:
        return "dense GEMM (proj/LoRA/head/ViT)"
    if "bucketize" in n or "float8" in n or "fp8" in n:
        return "act-quant emulation"
    if "index" in n or "scatter" in n or "gather" in n or "sort" in n or "bincount" in n or "cumsum" in n:
        return "routing gather/scatter"
    if "reduce" in n or "norm" in n or "softmax" in n or "sum" in n or "mean" in n:
        return "reductions/norms"
    if "copy" in n or "cat" in n or "memcpy" in n or "memset" in n or "fill" in n:
        return "copies"
    return "elementwise (convs, attn-res, swiglu, quant)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=4, help="CUA samples packed per step")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-act-quant", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--compile-quant", action="store_true", help="torch.compile the NVFP4/MXFP8 activation fake-quant")
    ap.add_argument("--out", default="/work/train.json")
    ap.add_argument("--save-lora", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    model = load_model(MODEL_DIR, lora_rank=args.rank)
    load_s = time.time() - t0
    quant_exact = None
    if args.compile_quant:
        from . import model as model_mod
        from .quant import mxfp8_quantize_dequant, nvfp4_quantize_mantissa

        model_mod.compile_quant()
        quant_exact = {}
        for k in (2304, 1152):
            x = torch.randn(3001, k, device="cuda") * torch.rand(3001, 1, device="cuda") * 7
            m0, o0 = nvfp4_quantize_mantissa(x)
            m1, o1 = model_mod._quant_fns()[0](x)
            q0, q1 = mxfp8_quantize_dequant(x), model_mod._quant_fns()[1](x)
            quant_exact[f"K{k}"] = {"nvfp4_mantissa_mismatch": int((m0 != m1).sum()), "nvfp4_outer_mismatch": int((o0 != o1).sum()),
                                    "mxfp8_mismatch": int((q0 != q1).sum()), "elements": x.numel()}
        print("[train] compiled quant exactness", quant_exact)
    params = lora_parameters(model)
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99), fused=True)
    samples = build_samples(args.samples)
    batch = pack(samples)
    act_quant = not args.no_act_quant
    flops = analytic_flops(batch, model.cfg)
    print(f"[train] load {load_s:.1f}s; lora params {n_params / 1e6:.1f}M; seq {batch['input_ids'].shape[0]} "
          f"(real {batch['real_tokens']}), loss rows {len(batch['rows'])}, images {len(batch['images'])}")
    torch.cuda.reset_peak_memory_stats()
    # forward-only timing (no grad) for the recompute/backward split
    with torch.no_grad():
        for _ in range(2):
            torch.cuda.synchronize(); tf = time.time()
            model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
                  image_mask=batch["image_mask"], attn_impl="flex", act_quant=act_quant, logit_rows=batch["rows"])
            torch.cuda.synchronize(); fwd_s = time.time() - tf
    losses, times, nan_steps = [], [], 0
    for step in range(args.steps):
        torch.cuda.synchronize(); ts = time.time()
        opt.zero_grad(set_to_none=True)
        loss = step_loss(model, batch, act_quant)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        bad = (not math.isfinite(loss.item())) or (not math.isfinite(gnorm.item()))
        nan_steps += int(bad)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - ts)
        losses.append(loss.item())
        print(f"[train] step {step} loss {loss.item():.4f} gnorm {gnorm.item():.3f} {times[-1]:.2f}s")
    peak = torch.cuda.max_memory_allocated() / 2**30
    steady = sorted(times[2:]) if len(times) > 3 else sorted(times)
    step_s = steady[len(steady) // 2]
    tokens = batch["real_tokens"]
    llm_vit = flops["llm_fwd_flops"] + flops["vit_fwd_flops"] + flops["projector_fwd_flops"]
    # backward: frozen weights need only input-grads (~1x fwd); LoRA weight grads are negligible;
    # the ViT backbone runs without grad (fwd only); recompute adds the LLM forward again.
    useful = 2 * flops["llm_fwd_flops"] + flops["vit_fwd_flops"] + 2 * flops["projector_fwd_flops"]
    executed = useful + flops["llm_fwd_flops"]
    res = {
        "config": vars(args), "compiled_quant_exactness": quant_exact, "load_s": load_s, "lora_params": n_params, "seq_len": int(batch["input_ids"].shape[0]),
        "real_tokens": tokens, "images": len(batch["images"]), "loss_rows": len(batch["rows"]),
        "losses": losses, "loss_drop": losses[0] - losses[-1], "nan_or_inf_steps": nan_steps,
        "step_times_s": times, "step_s_median_steady": step_s, "forward_nograd_s": fwd_s,
        "tok_s": tokens / step_s, "peak_mem_gib": peak, "flops": flops,
        "useful_tflops_per_step": useful / 1e12, "executed_tflops_per_step": executed / 1e12,
        "achieved_useful_tflops": useful / step_s / 1e12, "achieved_executed_tflops": executed / step_s / 1e12,
        "forward_achieved_tflops": llm_vit / fwd_s / 1e12,
        "expert_weight_reads_gb_per_step": 3 * flops["layer_visits"] * flops["expert_weight_bytes_bf16_per_visit"] / 1e9,
    }
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            opt.zero_grad(set_to_none=True)
            loss = step_loss(model, batch, act_quant)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
        cats, total = {}, 0.0
        for e in prof.key_averages():
            t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
            if t <= 0 or e.device_type is None:
                continue
            if str(e.device_type).endswith("CPU"):
                continue
            c = categorize(e.key)
            cats[c] = cats.get(c, 0.0) + t / 1e3
            total += t / 1e3
        res["profile_gpu_ms_by_category"] = dict(sorted(cats.items(), key=lambda kv: -kv[1]))
        res["profile_gpu_ms_total"] = total
        top = sorted(((getattr(e, "self_device_time_total", 0), e.key) for e in prof.key_averages()
                      if not str(e.device_type).endswith("CPU")), reverse=True)[:80]
        res["profile_top_kernels_ms"] = [(k[:120], t / 1e3) for t, k in top if t > 0]
    if args.save_lora:
        sd = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
        torch.save({"rank": args.rank, "state": sd}, args.save_lora)
        # target-completion logprobs of the trained model (for the SGLang reload check)
        with torch.no_grad():
            logits = model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
                           image_mask=batch["image_mask"], attn_impl="eager", act_quant=act_quant, logit_rows=batch["rows"])
            lp = torch.log_softmax(logits, -1).gather(1, batch["targets"][:, None])[:, 0]
        res["trained_target_logprob_mean"] = lp.mean().item()
    print(json.dumps({k: v for k, v in res.items() if k not in ("profile_top_kernels_ms",)}, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
