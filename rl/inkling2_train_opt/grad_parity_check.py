"""Backward-pass gradient parity: inkling2_train (reference autograd) vs inkling2_train_opt (Triton/SM120).

Each engine runs in its own process (both models do not fit one 96 GB GPU together):

  python3 -m inkling2_train_opt.grad_parity_check run --engine ref --out /work/g_ref.pt --steps 20
  python3 -m inkling2_train_opt.grad_parity_check run --engine opt --out /work/g_opt.pt --steps 20
  python3 -m inkling2_train_opt.grad_parity_check run --engine opt --no-opt-moe --no-opt-attn --no-opt-kernels --out /work/g_opt_off.pt
  python3 -m inkling2_train_opt.grad_parity_check compare --ref /work/g_ref.pt --cand /work/g_opt.pt [--cand ...] --out /work/receipt.json

`run` loads the real weights with LoRA r=128 (attention q/k/v/o x 24 layers, 24 MoE routers, vision
proj_in/proj_out), overwrites every lora_A/lora_B with a name-seeded deterministic init (identical on
both engines, independent of construction order), runs forward+backward on the same packed 4-sample
CUA batch (real 1280x800 screenshots + JSON action targets, loss on action tokens), and saves every
LoRA gradient.  lora_B is nonzero for the gradient check (with the production B=0 init every lora_A
gradient is exactly zero and A-side backward paths would go untested).  It then optionally runs an
N-step AdamW trajectory from the production init (seeded A, B=0) on the same batch/LR/seed.
"""

import argparse
import importlib
import json
import math
import statistics
import time
import zlib

import torch

MODEL_DIR = "/models/thinkingmachines--Inkling-2-Mini-heretic"


def group_of(name):
    if "attn.wo." in name:
        return "attn_o"
    if "attn.w" in name:
        return "attn_qkv"
    if "router" in name:
        return "router"
    if "proj_in" in name or "proj_out" in name:
        return "vision_proj"
    return "other"


@torch.no_grad()
def seeded_lora_init(model, b_std):
    """Deterministic per-name init: A ~ U(-1/sqrt(in), 1/sqrt(in)) (kaiming a=sqrt(5)), B ~ N(0, b_std) or 0."""
    names = []
    for n, p in model.named_parameters():
        if not p.requires_grad or ("lora_A" not in n and "lora_B" not in n):
            continue
        g = torch.Generator().manual_seed(zlib.crc32(n.encode()))
        if "lora_A" in n:
            bound = 1.0 / math.sqrt(p.shape[1])
            v = (torch.rand(p.shape, generator=g, dtype=torch.float32) * 2 - 1) * bound
        else:
            v = torch.randn(p.shape, generator=g, dtype=torch.float32) * b_std if b_std > 0 else torch.zeros(p.shape)
        p.copy_(v.to(p.device))
        names.append(n)
    return names


def make_loss_fn(engine, model, batch, args):
    if engine == "ref":
        def f():
            logits = model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
                           image_mask=batch["image_mask"], attn_impl="flex", act_quant=True, grad_ckpt=True,
                           logit_rows=batch["rows"])
            return torch.nn.functional.cross_entropy(logits.float(), batch["targets"])
    else:
        def f():
            logits = model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
                           image_mask=batch["image_mask"], attn_impl=None, act_quant=True,
                           grad_ckpt=args.ckpt_frac > 0, ckpt_frac=args.ckpt_frac, logit_rows=batch["rows"])
            return torch.nn.functional.cross_entropy(logits.float(), batch["targets"])
    return f


def cmd_run(args):
    pkg = "inkling2_train" if args.engine == "ref" else "inkling2_train_opt"
    mmod = importlib.import_module(f"{pkg}.model")
    lmod = importlib.import_module(f"{pkg}.lora")
    tb = importlib.import_module("inkling2_train.train_bench")  # identical data prep for both engines
    torch.manual_seed(0)
    t0 = time.time()
    if args.engine == "ref":
        model = mmod.load_model(MODEL_DIR, lora_rank=args.rank)
    else:
        model = mmod.load_model(MODEL_DIR, lora_rank=args.rank, opt_moe=not args.no_opt_moe,
                                opt_kernels=not args.no_opt_kernels, opt_attn=not args.no_opt_attn)
    load_s = time.time() - t0
    names = seeded_lora_init(model, args.b_std)
    if args.perturb > 0:  # noise-floor probe: same engine, lora_A * (1 + perturb * N(0,1))
        with torch.no_grad():
            g = torch.Generator().manual_seed(1234)
            for n, p in model.named_parameters():
                if "lora_A" in n and p.requires_grad:
                    p.mul_(1 + args.perturb * torch.randn(p.shape, generator=g).to(p.device))
    params = lmod.lora_parameters(model)
    pname = {id(p): n for n, p in model.named_parameters()}
    batch = tb.pack(tb.build_samples(args.samples))
    loss_fn = make_loss_fn(args.engine, model, batch, args)
    print(f"[{args.engine}] load {load_s:.1f}s lora tensors {len(names)} params {sum(p.numel() for p in params)/1e6:.1f}M "
          f"seq {batch['input_ids'].shape[0]} real {batch['real_tokens']} rows {len(batch['rows'])} imgs {len(batch['images'])}")

    # warmup forward (no grad): the reference's lazy torch._grouped_mm probe would otherwise save extra
    # tensors in the first checkpointed forward only and break the recompute check (train_bench does the same)
    with torch.no_grad():
        loss_fn()
    grads, losses_rep = None, []
    for rep in range(args.reps):  # rep>0 measures same-engine run-to-run nondeterminism
        for p in params:
            p.grad = None
        torch.cuda.synchronize(); ts = time.time()
        loss = loss_fn()
        loss.backward()
        torch.cuda.synchronize()
        losses_rep.append(loss.item())
        g = {pname[id(p)]: (p.grad.detach().float().cpu().clone() if p.grad is not None else None) for p in params}
        print(f"[{args.engine}] grad rep {rep} loss {loss.item():.6f} fwd+bwd {time.time() - ts:.2f}s")
        if grads is None:
            grads = g
        else:
            rep_grads = g
    out = {"engine": args.engine, "flags": vars(args), "load_s": load_s, "grad_loss": losses_rep, "grads": grads,
           "seq_len": int(batch["input_ids"].shape[0]), "real_tokens": batch["real_tokens"],
           "loss_rows": len(batch["rows"]), "images": len(batch["images"]),
           "peak_mem_gib_grad": torch.cuda.max_memory_allocated() / 2**30}
    if args.reps > 1:
        out["grads_rep1"] = rep_grads

    if args.steps > 0:
        seeded_lora_init(model, 0.0)  # production init: seeded A, B = 0
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99), fused=True)
        traj = []
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            traj.append({"step": step, "loss": loss.item(), "gnorm": gnorm.item()})
            print(f"[{args.engine}] traj step {step} loss {loss.item():.5f} gnorm {gnorm.item():.4f}")
        out["trajectory"] = traj
    torch.save(out, args.out)
    print(f"[{args.engine}] saved {args.out}")


def _build_ref_layer(ref, cfg, i, f, rank):
    """A true inkling2_train DecoderLayer (+ experts/controllers) built from the checkpoint like its loader."""
    w = f.get_tensor
    layer = ref.DecoderLayer(cfg, i, w, rank, rank)
    p = f"model.llm.layers.{i}.mlp.experts."
    if f"{p}w13_weight_amax" in f.keys():
        w13 = ref._chunked(ref.dequant_nvfp4_mantissa, w(p + "w13_weight"), w(p + "w13_weight_scale"))
        w2 = ref._chunked(ref.dequant_nvfp4_mantissa, w(p + "w2_weight"), w(p + "w2_weight_scale"))
        o13 = w(p + "w13_weight_amax") / w(p + "w13_weight_outer_scale_denominator")
        o2 = w(p + "w2_weight_amax") / w(p + "w2_weight_outer_scale_denominator")
        layer.experts = ref.Experts("nvfp4", w13, w2, o13.float(), o2.float(), cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"])
    else:
        w13 = ref._chunked(ref.dequant_mxfp8, w(p + "w13_weight"), w(p + "w13_weight_scale"))
        w2 = ref._chunked(ref.dequant_mxfp8, w(p + "w2_weight"), w(p + "w2_weight_scale"))
        one = torch.ones(w13.shape[0], device=w13.device)
        layer.experts = ref.Experts("mxfp8", w13, w2, one, one, cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"])
    loop = cfg["loop_config"]
    if loop["core_start"] <= i < loop["core_end"]:
        cp = f"model.llm.layers.{i}.controllers."
        mods = ("attn", "k", "v", "mlp")
        layer.controllers = ref.Controllers(*(torch.stack([w(f"{cp}{m}.{k}_weight") for m in mods])
                                              for k in ("down", "gain", "forget")))
    return layer


VISITS = ((0, 0), (5, 0), (6, 0), (11, 1), (12, 2), (17, 2), (20, 0), (23, 0))


def cmd_visits(args):
    """Per-layer-visit unit gradient parity on real captured activations (well-conditioned, no 48-visit chaos).

    For each selected (layer, occurrence) visit: capture the opt engine's real inputs (blocks, partial, bank),
    then run that one visit forward+backward with a fixed seeded upstream gradient on the partial output in
    four arms and compare every input gradient and the layer's LoRA gradients against the true reference layer:
      ref      inkling2_train DecoderLayer + Experts (STE fake-quant, flex attention), TF32 off
      opt      inkling2_train_opt production path (SM120 MoE, Triton FA2 attention, fused kernels, bank kernel)
      opt_nok  inkling2_train_opt with reference primitives + flex but native SM120 MoE (isolates the MoE)
      ref_pert reference with lora_A * (1 + 1e-3 N(0,1)) (local conditioning / noise floor)
    plus a direct MoE unit test (ref Experts vs SM120Experts on the identical routed inputs).
    """
    from safetensors import safe_open

    ref = importlib.import_module("inkling2_train.model")
    om = importlib.import_module("inkling2_train_opt.model")
    from inkling2_train_opt.kernels import fused_attn_res_bank

    tb = importlib.import_module("inkling2_train.train_bench")
    torch.manual_seed(0)
    model = om.load_model(MODEL_DIR, lora_rank=args.rank)
    seeded_lora_init(model, args.b_std)
    cfg = model.cfg
    batch = tb.pack(tb.build_samples(args.samples))
    targets = set(VISITS)

    caps, occ = {}, {}
    orig = model._run_visit

    def wrapped(layer_idx, blocks, partial, bank, ctx, do_ckpt, bank_msgs=None):
        k = (layer_idx, occ.get(layer_idx, 0))
        occ[layer_idx] = k[1] + 1
        if k in targets:
            caps[k] = {"blocks": [b.detach().clone() for b in blocks], "partial": partial.detach().clone(),
                       "bank": None if bank is None else [b.detach().clone() for b in bank], "ctx": ctx}
        return orig(layer_idx, blocks, partial, bank, ctx, do_ckpt, bank_msgs)

    model._run_visit = wrapped
    pin = model.visual.proj_in
    pin_fwd = pin.forward
    pooled_cap = []
    pin.forward = lambda x: (pooled_cap.append(x.detach().clone()), pin_fwd(x))[1]
    with torch.no_grad():
        model(batch["input_ids"], positions=batch["positions"], doc=batch["doc"], images=batch["images"],
              image_mask=batch["image_mask"], act_quant=True, logit_rows=batch["rows"])
    model._run_visit = orig
    pin.forward = pin_fwd
    assert set(caps) == targets, sorted(caps)

    # vision projector unit test: real pooled ViT features -> proj_out(gelu(proj_in(x))), ref vs opt LoRALinear
    rlora_mod = importlib.import_module("inkling2_train.lora")
    pooled = torch.cat(pooled_cap)
    vis_unit = {}
    rp = {}
    for nm in ("proj_in", "proj_out"):
        om_lin = getattr(model.visual, nm)
        r_lin = rlora_mod.LoRALinear(om_lin.weight.detach().clone(), args.rank)
        with torch.no_grad():
            r_lin.lora_A.copy_(om_lin.lora_A)
            r_lin.lora_B.copy_(om_lin.lora_B)
        rp[nm] = r_lin
    gen = torch.Generator(device="cuda").manual_seed(77)
    Gv = None
    grads_v = {}
    for arm_name, pi, po in (("ref", rp["proj_in"], rp["proj_out"]), ("opt", model.visual.proj_in, model.visual.proj_out)):
        torch.backends.cuda.matmul.allow_tf32 = arm_name == "opt"
        for lin in (pi, po):
            lin.lora_A.grad = lin.lora_B.grad = None
        y = po(torch.nn.functional.gelu(pi(pooled)))
        if Gv is None:
            Gv = torch.randn(y.shape, generator=gen, device="cuda", dtype=torch.float32)
        (y.float() * Gv).sum().backward()
        grads_v[arm_name] = {f"visual.{nm}.lora_{ab}": getattr(lin, f"lora_{ab}").grad.detach().float().cpu()
                             for nm, lin in (("proj_in", pi), ("proj_out", po)) for ab in "AB"}
        grads_v[arm_name + "_y"] = y.detach().float().cpu()
    torch.backends.cuda.matmul.allow_tf32 = True
    vis_unit["fwd"] = _stats(grads_v["ref_y"], grads_v["opt_y"])
    for k in grads_v["ref"]:
        vis_unit[k] = _stats(grads_v["ref"][k], grads_v["opt"][k])
    print("[vision_proj unit] " + " ".join(f"{k}:cos{v['cos']:.5f}/rel{v['rel_l2']:.4f}" for k, v in vis_unit.items()))
    del rp, grads_v, pooled_cap

    doc, pos = batch["doc"], batch["positions"]
    window = cfg["sliding_window_size"] - 1
    ref_ctx = ref.Ctx(positions=pos, doc=doc, attn_impl="flex", act_quant=True,
                      block_masks=ref.flex_block_masks(doc, window))
    nok_ctx = om.Ctx(positions=pos, doc=doc, attn_impl="flex", act_quant=True,
                     block_masks=om.flex_block_masks(doc, window), opt_kernels=False)
    refm = ref.Inkling2LoopForTraining(cfg, None)
    refm.attn_q, refm.mlp_q, refm.loop_q = model.attn_q, model.mlp_q, model.loop_q
    for _ in range(cfg["num_hidden_layers"]):
        refm.layers.append(torch.nn.Identity())
    f = safe_open(f"{MODEL_DIR}/model-00001-of-00001.safetensors", framework="pt", device="cuda")
    core_start = refm.core.start
    report = {}

    for (li, oc) in VISITS:
        cap = caps[(li, oc)]
        rl = _build_ref_layer(ref, cfg, li, f, args.rank)
        ol = model.layers[li]
        olora = {n: p for n, p in ol.named_parameters() if "lora_" in n}
        rlora = {n: p for n, p in rl.named_parameters() if "lora_" in n}
        assert set(olora) == set(rlora)
        with torch.no_grad():
            for n in rlora:
                rlora[n].copy_(olora[n])
        refm.layers[li] = rl
        gen = torch.Generator(device="cuda").manual_seed(li * 10 + oc)
        G = torch.randn(cap["partial"].shape, generator=gen, device="cuda", dtype=torch.float32)
        moe_in = {}

        def arm(name):
            blocks = [b.clone().requires_grad_(True) for b in cap["blocks"]]
            partial = cap["partial"].clone().requires_grad_(True)
            bank = None if cap["bank"] is None else [b.clone().requires_grad_(True) for b in cap["bank"]]
            lora = rlora if name.startswith("ref") else olora
            for p in lora.values():
                p.grad = None
            torch.backends.cuda.matmul.allow_tf32 = not name.startswith("ref")
            saved = None
            if name == "ref_pert":
                saved = {n: p.detach().clone() for n, p in rlora.items() if "lora_A" in n}
                g2 = torch.Generator().manual_seed(4321)
                with torch.no_grad():
                    for n in sorted(saved):
                        rlora[n].mul_(1 + 1e-3 * torch.randn(rlora[n].shape, generator=g2).cuda())
            if name.startswith("ref"):
                _, out = refm._visit(li, blocks, partial, bank, ref_ctx)
            elif name == "opt":
                fwd = ol.experts.forward

                def grab(x, weights, ids, *a, **k):
                    moe_in.update(x=x.detach().clone(), w=weights.detach().clone(), ids=ids.clone())
                    return fwd(x, weights, ids, *a, **k)

                ol.experts.forward = grab
                if bank is not None:
                    c4 = (li - core_start) * 4
                    bm = fused_attn_res_bank(bank, model.loop_q_flat_bf16, model.loop_q_scale)[c4:c4 + 4]
                    _, out = model._visit(li, blocks, partial, None, cap["ctx"], *bm)
                else:
                    _, out = model._visit(li, blocks, partial, None, cap["ctx"])
                ol.experts.forward = fwd
            else:
                _, out = model._visit(li, blocks, partial, bank, nok_ctx)
            (out.float() * G).sum().backward()
            g = {"d_partial": partial.grad}
            g.update({f"d_block{j}": b.grad for j, b in enumerate(blocks)})
            if bank is not None:
                g.update({f"d_bank{j}": b.grad for j, b in enumerate(bank)})
            g.update({n: p.grad.clone() for n, p in lora.items()})
            if saved:
                with torch.no_grad():
                    for n, v in saved.items():
                        rlora[n].copy_(v)
            torch.backends.cuda.matmul.allow_tf32 = True
            return out.detach().float(), {k: (v.detach().float() if v is not None else None) for k, v in g.items()}

        res = {name: arm(name) for name in ("ref", "opt", "opt_nok", "ref_pert")}
        out_r, g_r = res["ref"]
        entry = {"local": bool(rl.local), "moe_fmt": rl.experts.fmt, "has_bank": cap["bank"] is not None,
                 "n_blocks": len(cap["blocks"]), "arms": {}}
        for name in ("opt", "opt_nok", "ref_pert"):
            out_c, g_c = res[name]
            groups = {}
            for k in g_r:
                if g_r[k] is None or g_c[k] is None:
                    if (g_r[k] is None) != (g_c[k] is None):
                        groups.setdefault("missing", []).append(k)
                    continue
                s = _stats(g_r[k].cpu(), g_c[k].cpu())
                grp = group_of(k) if "lora_" in k else "inputs"
                groups.setdefault(grp, []).append((k, s))
            summ = {}
            for grp, items in groups.items():
                if grp == "missing":
                    summ[grp] = items
                    continue
                summ[grp] = {"cos_min": min(s["cos"] for _, s in items), "cos_median": statistics.median(s["cos"] for _, s in items),
                             "rel_l2_max": max(s["rel_l2"] for _, s in items),
                             "rel_l2_median": statistics.median(s["rel_l2"] for _, s in items),
                             "nonfinite": sum(s["nonfinite"] for _, s in items),
                             "worst": min(items, key=lambda t: t[1]["cos"])[0],
                             "per_tensor": {k: {kk: s[kk] for kk in ("cos", "rel_l2", "norm_ratio")} for k, s in items}}
            fwd_rel = ((out_c - out_r).norm() / out_r.norm()).item()
            entry["arms"][name] = {"fwd_rel_l2": fwd_rel, "groups": summ}
            line = " ".join(f"{g}:cos{v['cos_min']:.4f}/rel{v['rel_l2_max']:.3f}" for g, v in summ.items() if g != "missing")
            print(f"[visit L{li}#{oc}] {name:8s} fwd_rel {fwd_rel:.4f} {line} {'MISSING ' + str(summ.get('missing')) if 'missing' in summ else ''}")

        # direct MoE unit test on the identical routed inputs
        x = moe_in["x"].clone().requires_grad_(True)
        wts_r = moe_in["w"].clone().requires_grad_(True)
        gen = torch.Generator(device="cuda").manual_seed(999 + li)
        G2 = torch.randn(x.shape, generator=gen, device="cuda", dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        o_r = rl.experts(x, wts_r, moe_in["ids"], True)
        (o_r.float() * G2).sum().backward()
        dx_r, dw_r = x.grad.clone(), wts_r.grad.clone()
        torch.backends.cuda.matmul.allow_tf32 = True
        x.grad = None
        wts_o = moe_in["w"].clone().requires_grad_(True)
        o_o = ol.experts(x, wts_o, moe_in["ids"], True)
        (o_o.float() * G2).sum().backward()
        dx_o, dw_o = x.grad.clone(), wts_o.grad.clone()
        # reference without activation fake-quant: how far is the quantized forward itself from bf16
        x.grad = None
        wts_n = moe_in["w"].clone().requires_grad_(True)
        o_n = rl.experts(x, wts_n, moe_in["ids"], False)
        (o_n.float() * G2).sum().backward()
        dx_n, dw_n = x.grad.clone(), wts_n.grad.clone()
        moe = {"fwd_opt_vs_ref": _stats(o_r.detach().cpu(), o_o.detach().cpu()),
               "dx_opt_vs_ref": _stats(dx_r.cpu(), dx_o.cpu()),
               "dweights_opt_vs_ref": _stats(dw_r.cpu(), dw_o.cpu()),
               "fwd_ref_noquant_vs_ref": _stats(o_r.detach().cpu(), o_n.detach().cpu()),
               "dx_ref_noquant_vs_ref": _stats(dx_r.cpu(), dx_n.cpu()),
               "dweights_ref_noquant_vs_ref": _stats(dw_r.cpu(), dw_n.cpu())}
        entry["moe_unit"] = moe
        print(f"[visit L{li}#{oc}] moe_unit " + " ".join(f"{k}:cos{v['cos']:.5f}/rel{v['rel_l2']:.4f}" for k, v in moe.items()))
        report[f"L{li}#{oc}"] = entry
        refm.layers[li] = torch.nn.Identity()
        del rl, res, moe_in
        torch.cuda.empty_cache()
    json.dump({"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "b_std": args.b_std,
               "vision_proj_unit": vis_unit, "visits": report}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


def _stats(gr, go):
    gr, go = gr.double().flatten(), go.double().flatten()
    nr, no = gr.norm().item(), go.norm().item()
    cos = (gr @ go).item() / (nr * no) if nr > 0 and no > 0 else float("nan")
    return {"cos": cos, "rel_l2": (go - gr).norm().item() / nr if nr > 0 else float("nan"),
            "max_abs": (go - gr).abs().max().item(), "norm_ratio": no / nr if nr > 0 else float("nan"),
            "nonfinite": int((~torch.isfinite(go)).sum()) + int((~torch.isfinite(gr)).sum())}


def compare_grads(gref, gcand):
    assert set(gref) == set(gcand), f"param name mismatch: {sorted(set(gref) ^ set(gcand))[:10]}"
    per, groups = {}, {}
    for n in sorted(gref):
        a, b = gref[n], gcand[n]
        if a is None or b is None:
            per[n] = {"missing_grad": [a is None, b is None]}
            continue
        s = _stats(a, b)
        per[n] = s
        kind = "A" if "lora_A" in n else "B"
        for key in (group_of(n), f"{group_of(n)}.{kind}"):
            groups.setdefault(key, []).append((n, s, a, b))
    summary = {}
    for key, items in sorted(groups.items()):
        coss = [s["cos"] for _, s, _, _ in items]
        rels = [s["rel_l2"] for _, s, _, _ in items]
        ra = torch.cat([a.flatten().double() for _, _, a, _ in items])
        rb = torch.cat([b.flatten().double() for _, _, _, b in items])
        worst = min(items, key=lambda t: t[1]["cos"])
        summary[key] = {
            "tensors": len(items),
            "cos_min": min(coss), "cos_median": statistics.median(coss), "cos_mean": statistics.fmean(coss),
            "rel_l2_group": (rb - ra).norm().item() / ra.norm().item(),
            "rel_l2_median": statistics.median(rels), "rel_l2_max": max(rels),
            "max_abs_diff": max(s["max_abs"] for _, s, _, _ in items),
            "ref_max_abs": ra.abs().max().item(),
            "norm_ratio_group": rb.norm().item() / ra.norm().item(),
            "nonfinite": sum(s["nonfinite"] for _, s, _, _ in items),
            "worst_tensor": worst[0], "worst_cos": worst[1]["cos"],
        }
    missing = [n for n, s in per.items() if "missing_grad" in s]
    return summary, per, missing


def cmd_compare(args):
    ref = torch.load(args.ref, weights_only=False)
    receipt = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ref": args.ref,
               "ref_flags": ref["flags"], "batch": {k: ref[k] for k in ("seq_len", "real_tokens", "loss_rows", "images")},
               "ref_grad_loss": ref["grad_loss"], "comparisons": {}}
    if "grads_rep1" in ref:
        s, _, _ = compare_grads(ref["grads"], ref["grads_rep1"])
        receipt["ref_run_to_run"] = s
    for path in args.cand:
        c = torch.load(path, weights_only=False)
        label = c["flags"].get("label") or path
        summary, per, missing = compare_grads(ref["grads"], c["grads"])
        entry = {"path": path, "flags": c["flags"], "grad_loss": c["grad_loss"],
                 "loss_abs_diff": abs(c["grad_loss"][0] - ref["grad_loss"][0]), "groups": summary,
                 "missing_grads": missing, "peak_mem_gib_grad": c.get("peak_mem_gib_grad")}
        if "grads_rep1" in c:
            entry["run_to_run"], _, _ = compare_grads(c["grads"], c["grads_rep1"])
        if args.per_tensor:
            entry["per_tensor"] = per
        if "trajectory" in c and "trajectory" in ref:
            lr_, lc = [t["loss"] for t in ref["trajectory"]], [t["loss"] for t in c["trajectory"]]
            entry["trajectory"] = {"ref_loss": lr_, "cand_loss": lc,
                                   "ref_gnorm": [t["gnorm"] for t in ref["trajectory"]],
                                   "cand_gnorm": [t["gnorm"] for t in c["trajectory"]],
                                   "abs_diff": [abs(a - b) for a, b in zip(lr_, lc)],
                                   "max_abs_diff": max(abs(a - b) for a, b in zip(lr_, lc)),
                                   "ref_drop": lr_[0] - lr_[-1], "cand_drop": lc[0] - lc[-1]}
        receipt["comparisons"][label] = entry
        print(f"== {label}  loss ref {ref['grad_loss'][0]:.5f} cand {c['grad_loss'][0]:.5f}")
        for k, v in summary.items():
            print(f"  {k:16s} n={v['tensors']:3d} cos min/med/mean {v['cos_min']:.5f}/{v['cos_median']:.5f}/{v['cos_mean']:.5f} "
                  f"relL2 {v['rel_l2_group']:.4f} (max {v['rel_l2_max']:.4f}) normratio {v['norm_ratio_group']:.4f} "
                  f"maxabs {v['max_abs_diff']:.3e} nonfinite {v['nonfinite']}  worst {v['worst_tensor']}")
        if "trajectory" in entry:
            t = entry["trajectory"]
            print("  traj ref ", " ".join(f"{x:.4f}" for x in t["ref_loss"]))
            print("  traj cand", " ".join(f"{x:.4f}" for x in t["cand_loss"]))
    json.dump(receipt, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", choices=("ref", "opt"), required=True)
    r.add_argument("--label", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--rank", type=int, default=128)
    r.add_argument("--samples", type=int, default=4)
    r.add_argument("--b-std", type=float, default=1e-3, help="std of the nonzero lora_B used for the gradient check")
    r.add_argument("--perturb", type=float, default=0.0, help="relative lora_A perturbation (noise-floor probe)")
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--ckpt-frac", type=float, default=0.0)
    r.add_argument("--no-opt-moe", action="store_true")
    r.add_argument("--no-opt-kernels", action="store_true")
    r.add_argument("--no-opt-attn", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", action="append", required=True)
    c.add_argument("--per-tensor", action="store_true")
    c.add_argument("--out", required=True)
    v = sub.add_parser("visits")
    v.add_argument("--out", required=True)
    v.add_argument("--rank", type=int, default=128)
    v.add_argument("--samples", type=int, default=4)
    v.add_argument("--b-std", type=float, default=1e-3)
    args = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "visits": cmd_visits}[args.cmd](args)


if __name__ == "__main__":
    main()
