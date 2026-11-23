"""Primitive-by-primitive check of the PyTorch port against SGLang's inkling2 kernels.

Run inside the lmsysorg/sglang:inkling-2-mini image:  python3 -m inkling2_train.kernel_check
"""

import json

import torch

from . import model as M

torch.manual_seed(0)
dev = "cuda"
T, W = 300, 2304


def rel(a, b):
    a, b = a.float(), b.float()
    return {"max_abs": (a - b).abs().max().item(), "rel_l2": ((a - b).norm() / (b.norm() + 1e-12)).item()}


def main():
    out = {}
    # attn_res -------------------------------------------------------------------------
    from sglang.kernels.ops.attention.inkling2_attn_res import attn_res as k_attn_res

    srcs = [torch.randn(T, W, device=dev).bfloat16() * s for s in (1.0, 3.0, 0.5, 2.0)]
    add = torch.randn(T, W, device=dev).bfloat16()
    q = torch.randn(3, W, device=dev) * W**-0.5
    for norm in (True, False):
        mine = M.attn_res(srcs, add, q, normalize_values=norm)
        ref = k_attn_res([s.contiguous() for s in srcs], add, q.contiguous(), normalize_values=norm)
        out[f"attn_res_norm{int(norm)}"] = rel(torch.stack(mine), torch.stack(ref))
    # rms_norm -------------------------------------------------------------------------
    from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm as k_rms

    x = torch.randn(T, W, device=dev) * 3
    wgt = torch.randn(W, device=dev).bfloat16()
    out["rms_norm_fp32in_bf16out"] = rel(M.rms_norm(x, wgt, 1e-6, torch.bfloat16), k_rms(x, wgt, 1e-6, output_dtype=torch.bfloat16))
    # l2 clip --------------------------------------------------------------------------
    from sglang.kernels.ops.layernorm.inkling2_qk_l2_clip import qk_l2_clip_fwd_cutedsl as k_clip

    qh = (torch.randn(T, 24, 128, device=dev) * 4).bfloat16()
    sc = (torch.rand(24, 128, device=dev) + 0.5).bfloat16()
    tau = torch.ones(T, device=dev)
    with torch.no_grad():
        out["l2_clip_q"] = rel(M.l2_clip(qh, 22.627416997969522, sc, tau), k_clip(qh, 22.627416997969522, 1e-6, scale=sc, output_scale=tau).view(qh.shape))
        out["l2_clip_k"] = rel(M.l2_clip(qh[:, :8], 22.627416997969522), k_clip(qh[:, :8].contiguous(), 22.627416997969522, 1e-6).view(T, 8, 128))
    # message controllers --------------------------------------------------------------
    from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers as k_mc

    dests = tuple(torch.randn(T, W, device=dev) for _ in range(3))
    msgs = torch.randn(3, T, W, device=dev)
    down = (torch.randn(3, 64, 2 * W, device=dev) * 0.02).bfloat16()
    gain = (torch.randn(3, W, 64, device=dev) * 0.1).bfloat16()
    forget = (torch.randn(3, W, 64, device=dev) * 0.1).bfloat16()
    ctl = M.Controllers(down, gain, forget)
    mine = torch.stack(ctl(list(dests), list(msgs.unbind(0)), (0, 1, 2)))
    ref = k_mc(dests, msgs, down, gain, forget)
    out["message_controllers"] = rel(mine, ref)
    # short conv (static + dynamic), single sequence, no cache ---------------------------
    from sglang.kernels.ops.mamba.inkling2_sconv import short_convolution as k_sconv

    doc = torch.zeros(T, dtype=torch.int32, device=dev)
    meta = {"safe_idx": torch.tensor([1], device=dev), "cache_mask": torch.zeros(1, 1, 1, dtype=torch.bool, device=dev),
            "cu": torch.tensor([0, T], device=dev), "si": torch.zeros(T, dtype=torch.int32, device=dev)}
    xs = (torch.randn(T, 1024, device=dev)).bfloat16()
    w_static = (torch.randn(1024, 1, 4, device=dev) * 0.5).bfloat16()
    cache = torch.zeros(4, 3, 1024, device=dev, dtype=torch.bfloat16)
    ref = k_sconv(xs, w_static[:, 0, :], cache.clone(), meta)
    out["sconv_static"] = rel(M.short_conv_static(xs, w_static, doc), ref)
    sw = (torch.randn(4, 1024, device=dev) * 0.5).bfloat16()
    dyn = (torch.randn(T, 8, 4, device=dev) * 0.5).bfloat16()
    ref = k_sconv(xs, sw, cache.clone(), meta, dynamic_weights=dyn)
    out["sconv_dynamic"] = rel(M.short_conv_dynamic(xs, sw, dyn, doc), ref)
    # long sequence uses the prefill kernel
    T2 = 3000
    xs2 = torch.randn(T2, 1024, device=dev).bfloat16()
    meta2 = dict(meta, cu=torch.tensor([0, T2], device=dev), si=torch.zeros(T2, dtype=torch.int32, device=dev))
    doc2 = torch.zeros(T2, dtype=torch.int32, device=dev)
    out["sconv_static_prefillkernel"] = rel(M.short_conv_static(xs2, w_static, doc2), k_sconv(xs2, w_static[:, 0, :], cache.clone(), meta2))
    dyn2 = (torch.randn(T2, 8, 4, device=dev) * 0.5).bfloat16()
    out["sconv_dynamic_prefillkernel"] = rel(M.short_conv_dynamic(xs2, sw, dyn2, doc2), k_sconv(xs2, sw, cache.clone(), meta2, dynamic_weights=dyn2))
    # router ---------------------------------------------------------------------------
    from sglang.kernels.ops.moe.inkling2_route import route_topk

    xr = torch.randn(T, W, device=dev).bfloat16()
    gw = (torch.randn(128, W, device=dev) * 0.02).bfloat16()
    gb = torch.randn(128, device=dev) * 0.1
    router = M.Router(gw, gb, 0)
    w_mine, ids_mine = router(xr)
    partials = (xr.float() @ gw.float().T)[None]
    _, ids_ref, w_ref = route_topk(partials.contiguous(), gb, 8, 8.0, torch.ones((), device=dev))
    same = (ids_mine.sort(-1).values == ids_ref.sort(-1).values.long()).all(-1).float().mean().item()
    order_m, order_r = ids_mine.argsort(-1), ids_ref.long().argsort(-1)
    out["router"] = {"same_expert_set": same, **rel(w_mine.gather(1, order_m), w_ref.gather(1, order_r))}
    # attention: FA4 sm120 with rel_bias + gate_sink vs eager --------------------------------
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_varlen_func
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink

    for T3, local in ((700, False), (1500, True)):
        qa = (torch.randn(T3, 24, 128, device=dev) * 1.5).bfloat16()
        ka = (torch.randn(T3, 8, 128, device=dev) * 1.5).bfloat16()
        va = torch.randn(T3, 8, 128, device=dev).bfloat16()
        rb = (torch.randn(T3, 24, 1024, device=dev) * 2).bfloat16()
        sinks = torch.randn(T3, 24, device=dev).bfloat16()
        gate = torch.randn(T3, 24 * 128, device=dev).bfloat16()
        cu = torch.tensor([0, T3], device=dev, dtype=torch.int32)
        o = torch.empty(T3, 24, 128, device=dev, dtype=torch.float32)
        res = flash_attn_varlen_func(qa, ka, va, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T3, max_seqlen_k=T3,
                                     softmax_scale=128**-0.5, causal=True,
                                     window_size=(1023, 0) if local else (-1, -1),
                                     rel_bias=rb, return_softmax_lse=True, out=o, num_splits=1, pack_gqa=False)
        o_ref, lse = res if isinstance(res, tuple) else (res, None)
        if lse.shape[0] != T3:
            lse = lse.reshape(24, T3).T if lse.numel() == T3 * 24 else lse
        ref = gate_sink(o_ref.reshape(T3, -1), gate, lse, sinks, output_dtype=torch.bfloat16)
        docz = torch.zeros(T3, dtype=torch.int32, device=dev)
        mine = M.attention_eager(qa, ka, va, rb, sinks, local=local, window=1023, doc=docz)
        mine = (mine.float() * torch.sigmoid(gate.float().view(T3, 24, 128))).bfloat16().reshape(T3, -1)
        out[f"fa4_relbias_gatesink_T{T3}_local{int(local)}"] = {**rel(mine, ref), "lse_shape": list(lse.shape)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
