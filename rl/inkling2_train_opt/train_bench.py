"""LoRA r=128 training benchmark for inkling2_train_opt on real weights + CUA screenshots.

Supports stage-by-stage ablations (--no-opt-moe, --no-opt-kernels, --no-opt-attn, --ckpt-frac),
offline parity checks (--check-parity), batch scaling (--samples), and torch.profiler breakdown.
"""

import argparse
import json
import math
import os
import time

import torch

from .lora import lora_parameters
from .model import load_model
from .parity_offline import run_parity
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


def step_loss(model, batch, act_quant=True, attn_impl=None, ckpt_frac=1.0, recompute_gate_up=False):
    logits = model(
        batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
        image_mask=batch["image_mask"], attn_impl=attn_impl, act_quant=act_quant,
        grad_ckpt=(ckpt_frac > 0.0), ckpt_frac=ckpt_frac, recompute_gate_up=recompute_gate_up,
        logit_rows=batch["rows"],
    )
    return torch.nn.functional.cross_entropy(logits, batch["targets"])


def analytic_flops(batch, model_cfg):
    d = 2304
    attn_proj = d * (3072 + 1024 + 1024 + 384 + 3072 + 24) + 3072 * d
    experts = 8 * (2304 * 2304 + 1152 * 2304)
    router = d * 128
    rel = 24 * 16 * 1024
    per_tok_visit = attn_proj + experts + router + rel
    doc = batch["doc"]
    real = doc < 10_000
    ctx_full = (batch["positions"][real] + 1).float()
    ctx_local = ctx_full.clamp(max=1024)
    local_ids = set(model_cfg["local_layer_ids"])
    core = range(6, 18)
    visits = [i for i in range(24)] + [i for i in core] * 2
    n_local = sum(1 for i in visits if i in local_ids)
    n_global = len(visits) - n_local
    attn_core = 2 * 24 * 128 * (n_local * ctx_local.sum() + n_global * ctx_full.sum())
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
    if "sm120blockscaledmoegemm" in n or "group_gemm" in n or "groupgemm" in n or "grouped" in n or "nvfp4" in n or "mxfp8" in n or "swiglu_bwd" in n or "routed_" in n or "post_reorder" in n or "groupproblemshape" in n:
        return "experts MoE (NVFP4/MXFP8/quant/swiglu)"
    if "flex" in n or "tem_fused" in n or "_attn_bwd_" in n or "_gqa_reduce_dkv" in n or "_triton_fa2_" in n or "flashattentionforward" in n or "shearingbias" in n or "gate_sink" in n or "fa4" in n:
        return "attention (fwd/bwd)"
    if "pytorch_flash" in n or "fmha" in n or "cudnn" in n or "sdpa" in n or "scaled_dot_product" in n:
        return "vision SDPA"
    if "adam" in n or "multi_tensor" in n:
        return "optimizer"
    if "gemm" in n or "nvjet" in n or "cutlass" in n or "xmma" in n or "sm90" in n or "sm100" in n or "sm120" in n or "cublas" in n or "bmm" in n:
        return "dense GEMM (proj/LoRA/head/ViT)"
    if "bucketize" in n or "float8" in n or "fp8" in n:
        return "act-quant emulation"
    if "index" in n or "scatter" in n or "gather" in n or "sort" in n or "bincount" in n or "cumsum" in n or "src2dst" in n or "expert_offsets" in n:
        return "routing gather/scatter"
    if "short_conv" in n or "attn_res" in n or "_bank_" in n or "rms_norm" in n or "rms_cat" in n or "l2_clip" in n or "controller_mix" in n or "add3_bf16" in n:
        return "fused Triton elementwise"
    if "reduce" in n or "norm" in n or "softmax" in n or "sum" in n or "mean" in n:
        return "reductions/norms"
    if "copy" in n or "cat" in n or "memcpy" in n or "memset" in n or "fill" in n:
        return "copies"
    return "unfused elementwise / other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-act-quant", action="store_true")
    ap.add_argument("--no-opt-moe", action="store_true")
    ap.add_argument("--no-opt-kernels", action="store_true")
    ap.add_argument("--no-opt-attn", action="store_true")
    ap.add_argument("--ckpt-frac", type=float, default=0.0, help="Fraction of 48 visits to grad-checkpoint (0.0=none, 1.0=all)")
    ap.add_argument("--recompute-gate-up", action="store_true", help="Selective MoE w13-only recompute (saves 11-23 GiB at large batch)")
    ap.add_argument("--compile-quant", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check-parity", action="store_true")
    ap.add_argument("--out", default="/work/train_opt.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    opt_moe = not args.no_opt_moe
    opt_kernels = not args.no_opt_kernels
    opt_attn = not args.no_opt_attn
    attn_impl = "fa4" if opt_attn else "flex"
    act_quant = not args.no_act_quant

    t0 = time.time()
    model = load_model(MODEL_DIR, lora_rank=args.rank, opt_moe=opt_moe, opt_kernels=opt_kernels, opt_attn=opt_attn)
    load_s = time.time() - t0
    if args.compile_quant and not opt_moe:
        from . import model as model_mod
        model_mod.compile_quant()

    parity_res = None
    if args.check_parity:
        parity_res = run_parity(model, attn_impl=attn_impl, act_quant=act_quant)
        for pr in parity_res:
            print(f"[parity] {pr['prompt']}: top1={pr['top1_match_output_rows']} vs_srv_mean={pr['chosen_vs_server_mean']:.4f} out_mean={pr['output_vs_server_mean']:.4f} fwd={pr['forward_s']:.3f}s")

    params = lora_parameters(model)
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99), fused=True)
    samples = build_samples(args.samples)
    batch = pack(samples)
    flops = analytic_flops(batch, model.cfg)
    print(f"[train] load {load_s:.1f}s; lora {n_params/1e6:.1f}M; seq {batch['input_ids'].shape[0]} (real {batch['real_tokens']}), "
          f"imgs {len(batch['images'])}, ckpt_frac={args.ckpt_frac}")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            torch.cuda.synchronize(); tf = time.time()
            model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
                  image_mask=batch["image_mask"], attn_impl=attn_impl, act_quant=act_quant, logit_rows=batch["rows"])
            torch.cuda.synchronize(); fwd_s = time.time() - tf

    losses, times, nan_steps = [], [], 0
    for step in range(args.steps):
        torch.cuda.synchronize(); ts = time.time()
        opt.zero_grad(set_to_none=True)
        loss = step_loss(model, batch, act_quant=act_quant, attn_impl=attn_impl, ckpt_frac=args.ckpt_frac, recompute_gate_up=args.recompute_gate_up)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        bad = (not math.isfinite(loss.item())) or (not math.isfinite(gnorm.item()))
        nan_steps += int(bad)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - ts)
        losses.append(loss.item())
        print(f"[train] step {step} loss {loss.item():.4f} gnorm {gnorm.item():.3f} {times[-1]:.3f}s ({batch['real_tokens']/times[-1]:.0f} tok/s)")

    peak = torch.cuda.max_memory_allocated() / 2**30
    steady = sorted(times[2:]) if len(times) > 3 else sorted(times)
    step_s = steady[len(steady) // 2]
    tokens = batch["real_tokens"]
    llm_vit = flops["llm_fwd_flops"] + flops["vit_fwd_flops"] + flops["projector_fwd_flops"]
    useful = 2 * flops["llm_fwd_flops"] + flops["vit_fwd_flops"] + 2 * flops["projector_fwd_flops"]
    executed = useful + args.ckpt_frac * flops["llm_fwd_flops"]

    res = {
        "config": vars(args),
        "parity": parity_res,
        "load_s": load_s,
        "lora_params": n_params,
        "seq_len": int(batch["input_ids"].shape[0]),
        "real_tokens": tokens,
        "images": len(batch["images"]),
        "loss_rows": len(batch["rows"]),
        "losses": losses,
        "loss_drop": losses[0] - losses[-1],
        "nan_or_inf_steps": nan_steps,
        "step_times_s": times,
        "step_s_median_steady": step_s,
        "forward_nograd_s": fwd_s,
        "tok_s": tokens / step_s,
        "peak_mem_gib": peak,
        "flops": flops,
        "useful_tflops_per_step": useful / 1e12,
        "executed_tflops_per_step": executed / 1e12,
        "achieved_useful_tflops": useful / step_s / 1e12,
        "achieved_executed_tflops": executed / step_s / 1e12,
        "pct_bf16_peak_410_useful": (useful / step_s / 1e12) / 410.095 * 100.0,
        "pct_bf16_peak_410_executed": (executed / step_s / 1e12) / 410.095 * 100.0,
        "forward_achieved_tflops": llm_vit / fwd_s / 1e12,
    }
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            opt.zero_grad(set_to_none=True)
            loss = step_loss(model, batch, act_quant=act_quant, attn_impl=attn_impl, ckpt_frac=args.ckpt_frac, recompute_gate_up=args.recompute_gate_up)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            torch.cuda.synchronize()
        cats, total = {}, 0.0
        for e in prof.key_averages():
            t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
            if t <= 0 or e.device_type is None or str(e.device_type).endswith("CPU"):
                continue
            c = categorize(e.key)
            cats[c] = cats.get(c, 0.0) + t / 1e3
            total += t / 1e3
        res["profile_gpu_ms_by_category"] = dict(sorted(cats.items(), key=lambda kv: -kv[1]))
        res["profile_gpu_ms_total"] = total
        top = sorted(((getattr(e, "self_device_time_total", 0), e.key) for e in prof.key_averages()
                      if not str(e.device_type).endswith("CPU")), reverse=True)[:50]
        res["profile_top_kernels_ms"] = [(k[:120], t / 1e3) for t, k in top if t > 0]
    print(json.dumps({k: v for k, v in res.items() if k not in ("profile_top_kernels_ms",)}, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
