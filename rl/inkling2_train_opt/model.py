"""Optimized trainable PyTorch port of Inkling2LoopForConditionalGeneration (Blackwell SM120).

Preserves exact mathematical semantics and LoRA checkpoint compatibility with inkling2_train,
while replacing unfused bottlenecks with:
  1. Native SM120 NVFP4 (L0-19) / MXFP8 (L20-23) experts forward + MXFP8 backward (moe_sm120.py).
  2. Custom fused differentiable Triton kernels for short convs, attn_res, rms_norm, l2_clip,
     message controllers, and residual adds (kernels.py), plus merged QKV+RGS GEMM and
     BF16->FP32 Tensor Core router/LM-head GEMM.
  3. Blackwell SM120 FA4 varlen forward + zero-atomic Triton FlashAttention backward (attn_sm120.py)
     saving compact r (16-dim) instead of materialized rel (1024-dim) to save 8+ GB VRAM.
  4. Batched, QKV-fused, rotary-cached, compiled frozen ViT backbone.
  5. Configurable gradient checkpointing policy (ckpt_frac: 0.0 = none, 0.5 = half, 1.0 = all).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attn_sm120 import attention_fa4_sm120, doc_to_cu_seqlens
from .kernels import (
    fused_attn_res,
    fused_attn_res_bank,
    fused_attn_res_rmsnorm,
    fused_controller1,
    fused_controller3,
    fused_controller_mix_rmsnorm,
    fused_controller_mix_rmsnorm3,
    fused_l2_clip,
    fused_residual_add3,
    fused_rms_cat,
    fused_rms_cat3,
    fused_rms_norm,
    fused_short_conv_dynamic,
    fused_short_conv_static,
)
from .lora import LoRALinear
from .moe_sm120 import SM120Experts, quantize_weight_to_mxfp8_cute
from .quant import (
    dequant_mxfp8,
    dequant_nvfp4_mantissa,
    mxfp8_quantize_dequant,
    nvfp4_quantize_mantissa,
    ste,
)

BF16 = torch.bfloat16
F32 = torch.float32

_QUANT = [nvfp4_quantize_mantissa, mxfp8_quantize_dequant]


def compile_quant():
    _QUANT[:] = [torch.compile(f, dynamic=True) for f in (nvfp4_quantize_mantissa, mxfp8_quantize_dequant)]


def _quant_fns():
    return _QUANT


# ----------------------------------------------------------------------------- differentiable BF16->FP32 linear


class _BF16LinearF32Out(torch.autograd.Function):
    """Differentiable y = x @ W.T with BF16 Tensor Core matmul and FP32 output (frozen W)."""

    @staticmethod
    def forward(ctx, x, weight):
        x_bf = x.to(BF16).contiguous()
        out = torch.mm(x_bf, weight.t(), out_dtype=F32)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(weight)
            ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, dy):
        (weight,) = ctx.saved_tensors
        dx = torch.mm(dy.to(BF16).contiguous(), weight).to(ctx.x_dtype)
        return dx, None


class _FusedAttnProjAllFn(torch.autograd.Function):
    """Fused [q, r, gate, sinks, k, v] = src @ W_all.T + LoRA(q, k, v).

    Forward uses in-place strided addmm_ (zero extra q/k/v buffers).
    Backward uses chained addmm_ into L2-resident dsrc (eliminates torch.cat + 3 LoRA BF16 adds).
    """

    @staticmethod
    def forward(ctx, src, w_all, splits, aq, bq, sq, ak, bk, sk, av, bv, sv):
        src_c = src.contiguous()
        all_y = torch.mm(src_c, w_all.t())
        nq, nr, ng, ns, nk, nv = splits
        or_, og, os_, ok, ov = nq, nq + nr, nq + nr + ng, nq + nr + ng + ns, nq + nr + ng + ns + nk
        hq = hk = hv = None
        if aq is not None:
            hq = torch.mm(src_c, aq.to(BF16).t())
            all_y[:, :nq].addmm_(hq, bq.to(BF16).t(), beta=1.0, alpha=sq)
        if ak is not None:
            hk = torch.mm(src_c, ak.to(BF16).t())
            all_y[:, ok:ov].addmm_(hk, bk.to(BF16).t(), beta=1.0, alpha=sk)
        if av is not None:
            hv = torch.mm(src_c, av.to(BF16).t())
            all_y[:, ov:].addmm_(hv, bv.to(BF16).t(), beta=1.0, alpha=sv)
        empty = torch.empty(0, device=src_c.device, dtype=BF16)
        ctx.save_for_backward(
            src_c, w_all,
            aq if aq is not None else empty, bq if bq is not None else empty, hq if hq is not None else empty,
            ak if ak is not None else empty, bk if bk is not None else empty, hk if hk is not None else empty,
            av if av is not None else empty, bv if bv is not None else empty, hv if hv is not None else empty,
        )
        ctx.splits = splits
        ctx.has_lora = (aq is not None, ak is not None, av is not None)
        ctx.scales = (sq, sk, sv)
        return (
            all_y[:, :nq],
            all_y[:, or_:og],
            all_y[:, og:os_],
            all_y[:, os_:ok],
            all_y[:, ok:ov],
            all_y[:, ov:],
        )

    @staticmethod
    def backward(ctx, dq, dr, dg, ds, dk, dv):
        src_c, w_all, aq, bq, hq, ak, bk, hk, av, bv, hv = ctx.saved_tensors
        wq, wr, wg, ws, wk, wv = w_all.split(ctx.splits, dim=0)
        dq = dq.to(BF16).contiguous()
        dr = dr.to(BF16).contiguous()
        dg = dg.to(BF16).contiguous()
        ds = ds.to(BF16).contiguous()
        dk = dk.to(BF16).contiguous()
        dv = dv.to(BF16).contiguous()

        need_src = ctx.needs_input_grad[0]
        if need_src:
            dsrc = torch.mm(dq, wq)
            dsrc.addmm_(dg, wg)
            dsrc.addmm_(dk, wk)
            dsrc.addmm_(dv, wv)
            dsrc.addmm_(dr, wr)
            dsrc.addmm_(ds, ws)
        else:
            dsrc = None

        has_q, has_k, has_v = ctx.has_lora
        sq, sk, sv = ctx.scales
        daq = dbq = dak = dbk = dav = dbv = None
        if has_q:
            dhq = torch.mm(dq, bq.to(BF16))
            if need_src:
                dsrc.addmm_(dhq, aq.to(BF16), beta=1.0, alpha=sq)
            if ctx.needs_input_grad[4]:
                dbq = torch.mm(dq.t(), hq, out_dtype=F32)
                if sq != 1.0:
                    dbq.mul_(sq)
            if ctx.needs_input_grad[3]:
                daq = torch.mm(dhq.t(), src_c, out_dtype=F32)
                if sq != 1.0:
                    daq.mul_(sq)
        if has_k:
            dhk = torch.mm(dk, bk.to(BF16))
            if need_src:
                dsrc.addmm_(dhk, ak.to(BF16), beta=1.0, alpha=sk)
            if ctx.needs_input_grad[7]:
                dbk = torch.mm(dk.t(), hk, out_dtype=F32)
                if sk != 1.0:
                    dbk.mul_(sk)
            if ctx.needs_input_grad[6]:
                dak = torch.mm(dhk.t(), src_c, out_dtype=F32)
                if sk != 1.0:
                    dak.mul_(sk)
        if has_v:
            dhv = torch.mm(dv, bv.to(BF16))
            if need_src:
                dsrc.addmm_(dhv, av.to(BF16), beta=1.0, alpha=sv)
            if ctx.needs_input_grad[10]:
                dbv = torch.mm(dv.t(), hv, out_dtype=F32)
                if sv != 1.0:
                    dbv.mul_(sv)
            if ctx.needs_input_grad[9]:
                dav = torch.mm(dhv.t(), src_c, out_dtype=F32)
                if sv != 1.0:
                    dav.mul_(sv)
        return dsrc, None, None, daq, dbq, None, dak, dbk, None, dav, dbv, None


class _FusedAttnProjQRGSFn(torch.autograd.Function):
    """Fused [q, r, gate, sinks] = src_q @ W_qrgs.T + LoRA(q) with zero torch.cat in backward."""

    @staticmethod
    def forward(ctx, src_q, w_qrgs, splits, aq, bq, sq):
        src_c = src_q.contiguous()
        y = torch.mm(src_c, w_qrgs.t())
        nq, nr, ng, ns = splits
        or_, og, os_ = nq, nq + nr, nq + nr + ng
        hq = None
        if aq is not None:
            hq = torch.mm(src_c, aq.to(BF16).t())
            y[:, :nq].addmm_(hq, bq.to(BF16).t(), beta=1.0, alpha=sq)
        empty = torch.empty(0, device=src_c.device, dtype=BF16)
        ctx.save_for_backward(
            src_c, w_qrgs,
            aq if aq is not None else empty, bq if bq is not None else empty, hq if hq is not None else empty,
        )
        ctx.splits = splits
        ctx.has_q = aq is not None
        ctx.sq = sq
        return y[:, :nq], y[:, or_:og], y[:, og:os_], y[:, os_:]

    @staticmethod
    def backward(ctx, dq, dr, dg, ds):
        src_c, w_qrgs, aq, bq, hq = ctx.saved_tensors
        wq, wr, wg, ws = w_qrgs.split(ctx.splits, dim=0)
        dq = dq.to(BF16).contiguous()
        dr = dr.to(BF16).contiguous()
        dg = dg.to(BF16).contiguous()
        ds = ds.to(BF16).contiguous()
        need_src = ctx.needs_input_grad[0]
        if need_src:
            dsrc = torch.mm(dq, wq)
            dsrc.addmm_(dg, wg)
            dsrc.addmm_(dr, wr)
            dsrc.addmm_(ds, ws)
        else:
            dsrc = None
        daq = dbq = None
        if ctx.has_q:
            sq = ctx.sq
            dhq = torch.mm(dq, bq.to(BF16))
            if need_src:
                dsrc.addmm_(dhq, aq.to(BF16), beta=1.0, alpha=sq)
            if ctx.needs_input_grad[4]:
                dbq = torch.mm(dq.t(), hq, out_dtype=F32)
                if sq != 1.0:
                    dbq.mul_(sq)
            if ctx.needs_input_grad[3]:
                daq = torch.mm(dhq.t(), src_c, out_dtype=F32)
                if sq != 1.0:
                    daq.mul_(sq)
        return dsrc, None, None, daq, dbq, None


# ----------------------------------------------------------------------------- reference primitives (for ablation)


def rms_norm(x, weight=None, eps=1e-6, out_dtype=None):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(out_dtype or x.dtype)


def attn_res(sources, addend, queries_scaled, normalize_values=True, eps=1e-6):
    xs = [s.float() for s in sources]
    if addend is not None:
        xs[-1] = xs[-1] + addend.float()
    x = torch.stack(xs)
    rstd = torch.rsqrt(x.pow(2).mean(-1) + eps)
    logits = torch.einsum("ntw,qw->qnt", x, queries_scaled) * rstd
    p = torch.softmax(logits, dim=1)
    if normalize_values:
        p = p * rstd
    return list(torch.einsum("qnt,ntw->qtw", p, x).unbind(0))


def shift_rows(x, lag, doc):
    if lag == 0:
        return x
    shifted = F.pad(x, (0, 0, lag, 0))[: x.shape[0]]
    same = F.pad(doc, (lag, 0), value=-1)[: x.shape[0]] == doc
    return shifted * same[:, None].to(x.dtype)


def short_conv_static(x, weight, doc):
    w = weight[:, 0, :].float()
    width = w.shape[1]
    xf = x.float()
    acc = None
    for tap in range(width):
        term = shift_rows(xf, width - 1 - tap, doc) * w[:, tap]
        acc = term if acc is None else acc + term
    return ((acc + xf) * (width + 1) ** -0.5).to(x.dtype)


def short_conv_dynamic(x, static_w, dyn, doc):
    width, d = static_w.shape
    heads = dyn.shape[1]
    xf = x.float()
    dynf = dyn.float().repeat_interleave(d // heads, dim=1)
    acc = None
    for lag in range(width):
        weight = static_w[lag].float()[None, :] + dynf[:, :, lag]
        term = shift_rows(xf, lag, doc) * weight
        acc = term if acc is None else acc + term
    return ((acc + xf) * (2 * width + 1) ** -0.5).to(x.dtype)


def l2_clip(x, threshold, scale=None, tau=None, eps=1e-6):
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = xf.pow(2).sum(-1, keepdim=True).sqrt()
    y = xf * torch.clamp(threshold / (norm + eps), max=1.0)
    if tau is not None:
        y = y.to(x.dtype).float() * tau.view(-1, *([1] * (x.ndim - 1)))
    return y.to(x.dtype)


def log_scaling_tau(positions, n_floor, alpha):
    effective = (positions + 1).float()
    return 1.0 + alpha * torch.log(torch.clamp(effective / float(n_floor), min=1.0))


# ----------------------------------------------------------------------------- reference attention kernels


def attention_eager(q, k, v, rel, sink, *, local, window, doc, chunk=256):
    t, h, d = q.shape
    group = h // k.shape[1]
    extent = rel.shape[-1]
    kf = k.float().repeat_interleave(group, 1)
    vf = v.float().repeat_interleave(group, 1)
    outs = []
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        k0 = max(0, s - window) if local else 0
        qi = torch.arange(s, e, device=q.device)
        ki = torch.arange(k0, e, device=q.device)
        scores = torch.einsum("qhd,khd->hqk", q[s:e].float(), kf[k0:e]) * d**-0.5
        dist = qi[:, None] - ki[None, :]
        idx = dist.clamp(0, extent - 1)
        bias = torch.gather(rel[s:e].float().permute(1, 0, 2), 2, idx.expand(h, -1, -1))
        inside = (dist >= 0) & (dist < extent)
        scores = scores + torch.where(inside, bias, torch.zeros_like(bias))
        allowed = (dist >= 0) & (doc[s:e, None] == doc[None, k0:e])
        if local:
            allowed = allowed & (dist <= window)
        scores = scores.masked_fill(~allowed, float("-inf"))
        lse = torch.logsumexp(scores, -1)
        o = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), vf[k0:e])
        factor = torch.sigmoid(lse - sink[s:e].float().T).T
        outs.append(o * factor[..., None])
    return torch.cat(outs)


_FLEX = None


def _flex_fn():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def flex_block_masks(doc, window):
    from torch.nn.attention.flex_attention import create_block_mask

    t = doc.shape[0]
    doc_k = torch.cat([doc[:1], doc])

    def make(local):
        def mask_mod(b, h, qi, ki):
            kj = ki - 1
            ok = (kj <= qi) & (doc_k[ki] == doc[qi])
            if local:
                ok = ok & (qi - kj <= window)
            return (ki == 0) | ok

        return create_block_mask(mask_mod, 1, 1, t, t + 1, device=doc.device, _compile=True)

    return {True: make(True), False: make(False)}


def attention_flex(q, k, v, rel, sink, block_mask):
    t, h, d = q.shape
    extent = rel.shape[-1]
    zeros = k.new_zeros(1, k.shape[1], d)
    kb = torch.cat([zeros, k]).permute(1, 0, 2)[None]
    vb = torch.cat([zeros, v]).permute(1, 0, 2)[None]
    qb = q.permute(1, 0, 2)[None]
    relh = rel.permute(1, 0, 2)
    sinkh = sink.permute(1, 0)

    def score_mod(score, b, hh, qi, ki):
        dist = qi - (ki - 1)
        bias = relh[hh, qi, torch.clamp(dist, 0, extent - 1)].to(score.dtype)
        bias = torch.where((dist >= 0) & (dist < extent), bias, torch.zeros_like(bias))
        return torch.where(ki == 0, sinkh[hh, qi].to(score.dtype), score + bias)

    o = _flex_fn()(qb, kb, vb, score_mod=score_mod, block_mask=block_mask, enable_gqa=True, scale=d**-0.5)
    return o[0].permute(1, 0, 2)


# ----------------------------------------------------------------------------- modules


@dataclass
class Ctx:
    positions: torch.Tensor
    doc: torch.Tensor
    attn_impl: str = "fa4"
    act_quant: bool = True
    block_masks: dict | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int = 0
    tau: torch.Tensor | None = None
    opt_kernels: bool = True
    recompute_gate_up: bool = False


class Attention(nn.Module):
    def __init__(self, cfg, layer_id, w, lora_rank, opt_kernels=True):
        super().__init__()
        self.local = layer_id in cfg["local_layer_ids"]
        self.heads, self.kv_heads, self.head_dim = 24, 8, 128
        self.d_rel = cfg["d_rel"]
        self.window = cfg["sliding_window_size"] - 1
        self.qk_thr, self.rel_thr = cfg["qk_l2_clip_threshold"], cfg["rel_l2_clip_threshold"]
        self.n_floor, self.alpha = cfg["log_scaling_n_floor"], cfg["log_scaling_alpha"]
        self.opt_kernels = opt_kernels
        p = f"model.llm.layers.{layer_id}.attn."

        wq_w = w(p + "wq_du.weight")
        wk_w = w(p + "wk_dv.weight")
        wv_w = w(p + "wv_dv.weight")
        wr_w = w(p + "wr_du.weight")
        wg_w = w(p + "gate_attn.weight")
        ws_w = w(p + "attn_sink_proj.weight")

        # Pack [wq, wr, gate, sink, wk, wv] contiguously so views share storage with 0 extra VRAM
        packed = torch.cat([wq_w, wr_w, wg_w, ws_w, wk_w, wv_w], dim=0).contiguous()
        self.register_buffer("w_qrgs_kv", packed, persistent=False)
        n_q = wq_w.shape[0]
        n_rgs = wr_w.shape[0] + wg_w.shape[0] + ws_w.shape[0]
        n_k = wk_w.shape[0]
        self._n_qrgs = n_q + n_rgs
        self._splits_all = [n_q, wr_w.shape[0], wg_w.shape[0], ws_w.shape[0], n_k, wv_w.shape[0]]
        self._splits_qrgs = [n_q, wr_w.shape[0], wg_w.shape[0], ws_w.shape[0]]

        self.wq = LoRALinear(self.w_qrgs_kv[:n_q], lora_rank)
        self.wk = LoRALinear(self.w_qrgs_kv[self._n_qrgs : self._n_qrgs + n_k], lora_rank)
        self.wv = LoRALinear(self.w_qrgs_kv[self._n_qrgs + n_k :], lora_rank)
        self.wo = LoRALinear(w(p + "wo_ud.weight"), lora_rank)
        self.register_buffer("w_rgs", self.w_qrgs_kv[n_q : self._n_qrgs], persistent=False)

        self.register_buffer("q_scale", w(p + "q_l2_clip.scale").contiguous(), persistent=False)
        self.register_buffer("rel_scale", w(p + "rel_l2_clip.scale").contiguous(), persistent=False)
        self.register_buffer("rel_proj", w(p + "rel_logits_proj.proj").contiguous(), persistent=False)
        if self.local:
            kc = w(p + "k_sconv.weight")
            vc = w(p + "v_sconv.weight")
            self.register_buffer("k_conv", kc, persistent=False)
            self.register_buffer("v_conv", vc, persistent=False)
            self.register_buffer("k_conv_4d", kc[:, 0, :].float().t().contiguous(), persistent=False)
            self.register_buffer("v_conv_4d", vc[:, 0, :].float().t().contiguous(), persistent=False)
        else:
            ks = w(p + "k_sconv.static_w")
            vs = w(p + "v_sconv.static_w")
            self.register_buffer("k_static", ks, persistent=False)
            self.register_buffer("v_static", vs, persistent=False)
            self.register_buffer("k_static_f", ks.float().contiguous(), persistent=False)
            self.register_buffer("v_static_f", vs.float().contiguous(), persistent=False)
            self.register_buffer("k_proj", w(p + "k_sconv.weight_proj").contiguous(), persistent=False)
            self.register_buffer("v_proj", w(p + "v_sconv.weight_proj").contiguous(), persistent=False)

    def forward(self, src_q, src_k, src_v, ctx: Ctx):
        t = src_q.shape[0]
        h, kvh, d, dr = self.heads, self.kv_heads, self.head_dim, self.d_rel
        use_k = ctx.opt_kernels

        if use_k:
            if src_q.data_ptr() == src_k.data_ptr() and src_q.data_ptr() == src_v.data_ptr():
                q, r, gate, sinks, k, v = _FusedAttnProjAllFn.apply(
                    src_q, self.w_qrgs_kv, self._splits_all,
                    self.wq.lora_A, self.wq.lora_B, self.wq.scale,
                    self.wk.lora_A, self.wk.lora_B, self.wk.scale,
                    self.wv.lora_A, self.wv.lora_B, self.wv.scale,
                )
            else:
                q, r, gate, sinks = _FusedAttnProjQRGSFn.apply(
                    src_q, self.w_qrgs_kv[: self._n_qrgs], self._splits_qrgs,
                    self.wq.lora_A, self.wq.lora_B, self.wq.scale,
                )
                k = self.wk(src_k)
                v = self.wv(src_v)
        else:
            q = self.wq(src_q)
            k = self.wk(src_k)
            v = self.wv(src_v)
            r, gate, sinks = F.linear(src_q, self.w_rgs).split([h * dr, h * d, h], dim=-1)

        if self.local:
            if use_k:
                k = fused_short_conv_static(k, self.k_conv_4d, ctx.doc)
                v = fused_short_conv_static(v, self.v_conv_4d, ctx.doc)
            else:
                k = short_conv_static(k, self.k_conv, ctx.doc)
                v = short_conv_static(v, self.v_conv, ctx.doc)
            tau = None
        else:
            dk = F.linear(src_k, self.k_proj).view(t, kvh, -1)
            dv = F.linear(src_v, self.v_proj).view(t, kvh, -1)
            if use_k:
                k = fused_short_conv_dynamic(k, self.k_static_f, dk, ctx.doc)
                v = fused_short_conv_dynamic(v, self.v_static_f, dv, ctx.doc)
            else:
                k = short_conv_dynamic(k, self.k_static, dk, ctx.doc)
                v = short_conv_dynamic(v, self.v_static, dv, ctx.doc)
            tau = ctx.tau if ctx.tau is not None else log_scaling_tau(ctx.positions, self.n_floor, self.alpha)

        if use_k:
            q = fused_l2_clip(q, self.qk_thr, self.q_scale, tau, heads=h, head_dim=d)
            k = fused_l2_clip(k.view(t, kvh, d), self.qk_thr)
            v = fused_rms_norm(v.view(t, kvh, d), None, 1e-6)
            r = fused_l2_clip(r, self.rel_thr, self.rel_scale, tau, heads=h, head_dim=dr)
        else:
            q = l2_clip(q.view(t, h, d), self.qk_thr, self.q_scale, tau)
            k = l2_clip(k.view(t, kvh, d), self.qk_thr)
            v = rms_norm(v.view(t, kvh, d), None, 1e-6)
            r = l2_clip(r.reshape(t, h, dr), self.rel_thr, self.rel_scale, tau)

        if ctx.attn_impl == "fa4":
            o = attention_fa4_sm120(
                q, k, v, r, self.rel_proj, sinks, gate,
                ctx.cu_seqlens, ctx.max_seqlen, local=self.local, window=self.window,
            )
            return self.wo(o)

        rel = torch.matmul(r, self.rel_proj) if use_k else torch.einsum("thd,de->the", r, self.rel_proj)
        if ctx.attn_impl == "eager":
            o = attention_eager(q, k, v, rel, sinks, local=self.local, window=self.window, doc=ctx.doc)
        else:
            o = attention_flex(q, k, v, rel, sinks, ctx.block_masks[self.local])
        o = (o.float() * torch.sigmoid(gate.float().view(t, h, d))).to(BF16).reshape(t, h * d)
        return self.wo(o)


class Router(nn.Module):
    def __init__(self, weight, bias, lora_rank, topk=8, route_scale=8.0):
        super().__init__()
        self.lin = LoRALinear(weight, lora_rank)
        self.register_buffer("bias", bias.float().contiguous(), persistent=False)
        self.topk, self.route_scale = topk, route_scale

    def forward(self, x, opt_kernels=True):
        if opt_kernels and _MM_OUT_DTYPE:
            logits = _BF16LinearF32Out.apply(x, self.lin.weight)
            if self.lin.lora_A is not None:
                a = self.lin.lora_A.to(BF16)
                b = self.lin.lora_B.to(BF16)
                logits = logits + (F.linear(F.linear(x, a), b) * self.lin.scale).float()
        else:
            logits = F.linear(x.float(), self.lin.weight.float())
            if self.lin.lora_A is not None:
                logits = logits + F.linear(F.linear(x.float(), self.lin.lora_A), self.lin.lora_B) * self.lin.scale
        selection = torch.sigmoid(logits.detach()) + self.bias
        ids = torch.topk(selection, self.topk, dim=-1, sorted=True).indices
        chosen = torch.gather(logits, 1, ids)
        weights = torch.softmax(F.logsigmoid(chosen), dim=-1) * self.route_scale
        return weights, ids


_GROUPED_MM_OK = None


def grouped_mm(x, w_t, offs, counts):
    global _GROUPED_MM_OK
    if _GROUPED_MM_OK is None:
        try:
            torch._grouped_mm(x[:16], w_t[:1], offs=torch.tensor([16], device=x.device, dtype=torch.int32))
            _GROUPED_MM_OK = True
        except Exception as exc:
            print(f"[inkling2_train_opt] torch._grouped_mm unavailable ({exc}); using per-expert loop")
            _GROUPED_MM_OK = False
    if _GROUPED_MM_OK:
        return torch._grouped_mm(x, w_t, offs=offs)
    out = x.new_empty(x.shape[0], w_t.shape[-1])
    start = 0
    for e, c in enumerate(counts.tolist()):
        if c:
            out[start : start + c] = x[start : start + c] @ w_t[e]
        start += c
    return out


class Experts(nn.Module):
    """Reference BF16-mantissa routed experts (used only when opt_moe=False)."""

    def __init__(self, fmt, w13, w2, outer13, outer2, gate_cap=4.0, up_cap=25.0):
        super().__init__()
        self.fmt = fmt
        self.register_buffer("w13", w13, persistent=False)
        self.register_buffer("w2", w2, persistent=False)
        self.register_buffer("outer13", outer13, persistent=False)
        self.register_buffer("outer2", outer2, persistent=False)
        self.gate_cap, self.up_cap = gate_cap, up_cap

    def _quant_in(self, x, act_quant):
        if not act_quant:
            return x.to(BF16), None
        if self.fmt == "nvfp4":
            mant, outer = _quant_fns()[0](x.detach())
            return ste(x / outer[:, None], mant), outer
        return ste(x, _quant_fns()[1](x.detach())), None

    def forward(self, x, weights, ids, act_quant):
        t, topk = ids.shape
        n_exp = self.w13.shape[0]
        flat = ids.flatten()
        order = torch.argsort(flat, stable=True)
        tok = order // topk
        row_e = flat[order]
        counts = torch.bincount(flat, minlength=n_exp)
        offs = torch.cumsum(counts, 0).to(torch.int32)
        gam = weights.flatten()[order]

        xq, x_outer = self._quant_in(x, act_quant)
        acc = grouped_mm(xq[tok], self.w13.transpose(-2, -1), offs, counts)
        h = acc.float()
        if self.fmt == "nvfp4":
            scale = self.outer13[row_e]
            if x_outer is not None:
                scale = scale * x_outer[tok]
            h = h * scale[:, None]
        gate, up = h[:, 0::2], h[:, 1::2]
        sig = torch.sigmoid(gate)
        if self.gate_cap > 0:
            gate = self.gate_cap * torch.tanh(gate / self.gate_cap)
        if self.up_cap > 0:
            up = self.up_cap * torch.tanh(up / self.up_cap)
        a = gate * sig * up * gam[:, None]
        aq, a_outer = self._quant_in(a, act_quant)
        d = grouped_mm(aq, self.w2.transpose(-2, -1), offs, counts).float()
        if self.fmt == "nvfp4":
            scale = self.outer2[row_e]
            if a_outer is not None:
                scale = scale * a_outer
            d = d * scale[:, None]
        d = d.to(BF16).float()
        out = torch.zeros(t, d.shape[1], device=x.device, dtype=F32).index_add(0, tok, d)
        return out.to(BF16)


class Controllers(nn.Module):
    """Stacked loop message controllers of one core layer, CONTROLLER_MODULES order."""

    def __init__(self, down, gain, forget):
        super().__init__()
        self.register_buffer("down", down.contiguous(), persistent=False)          # (4, R, 2W) bf16
        self.register_buffer("down_t", down.transpose(1, 2).contiguous(), persistent=False)  # (4, 2W, R) bf16
        gf = torch.cat([gain, forget], dim=1).contiguous()                         # (4, 2W, R) bf16
        self.register_buffer("gf", gf, persistent=False)
        self.register_buffer("gf_t", gf.transpose(1, 2).contiguous(), persistent=False)      # (4, R, 2W) bf16
        self._w = gain.shape[1]

    def forward(self, dests, messages, rows):
        """Reference path (when opt_kernels=False)."""
        out = []
        W = self._w
        for i, dest, msg in zip(rows, dests, messages):
            normed = torch.cat([rms_norm(dest, None, out_dtype=BF16), rms_norm(msg, None, out_dtype=BF16)], -1)
            hidden = F.linear(normed.float(), self.down[i].float()).to(BF16).float()
            gain = F.linear(hidden, self.gf[i, :W].float())
            forget = F.linear(hidden, self.gf[i, W:].float())
            out.append((1.0 - torch.tanh(forget)) * dest.float() + torch.tanh(gain) * msg.float())
        return out

    def forward_and_norm(self, dests, messages, rows, norms):
        """Fused whole-controller op: saves only hidden (R=64) instead of normed+gf (4W) and avoids BF16 adds."""
        if len(rows) == 3:
            shared = (dests[0].data_ptr() == dests[1].data_ptr()) and (dests[0].data_ptr() == dests[2].data_ptr())
            return fused_controller3(
                dests, messages,
                self.down[:3], self.down_t[:3], self.gf[:3], self.gf_t[:3],
                norms, shared_dest=shared,
            )
        idx = rows[0]
        return fused_controller1(
            dests[0], messages[0],
            self.down[idx], self.down_t[idx], self.gf[idx], self.gf_t[idx],
            norms[0],
        )


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_id, w, lora_rank, router_lora_rank, opt_kernels=True):
        super().__init__()
        self.layer_id = layer_id
        self.local = layer_id in cfg["local_layer_ids"]
        p = f"model.llm.layers.{layer_id}."
        self.attn = Attention(cfg, layer_id, w, lora_rank, opt_kernels=opt_kernels)
        if self.local:
            self.register_buffer("attn_norm", w(p + "attn_norm.weight").contiguous(), persistent=False)
        else:
            for n in ("q_input_norm", "k_input_norm", "v_input_norm"):
                self.register_buffer(n, w(p + n + ".weight").contiguous(), persistent=False)
        self.register_buffer("mlp_norm", w(p + "mlp_norm.weight").contiguous(), persistent=False)
        ac = w(p + "attn_sconv.weight")
        mc = w(p + "mlp_sconv.weight")
        self.register_buffer("attn_sconv", ac, persistent=False)
        self.register_buffer("mlp_sconv", mc, persistent=False)
        self.register_buffer("attn_sconv_4d", ac[:, 0, :].float().t().contiguous(), persistent=False)
        self.register_buffer("mlp_sconv_4d", mc[:, 0, :].float().t().contiguous(), persistent=False)
        self.router = Router(w(p + "mlp.gate.weight"), w(p + "mlp.gate.bias"), router_lora_rank)
        self.experts = None
        self.controllers = None


class VisionEncoder(nn.Module):
    def __init__(self, vcfg, w, lora_rank, opt_kernels=True):
        super().__init__()
        self.cfg = vcfg
        self.opt_kernels = opt_kernels
        pre = "model.visual."
        b = pre + "backbone."
        self.patch_w = nn.Parameter(w(b + "embeddings.patch_embedding.weight"), requires_grad=False)
        self.pos = nn.Parameter(w(b + "embeddings.position_embedding"), requires_grad=False)
        self.cls = nn.Parameter(
            w(b + "embeddings.class_embedding") + w(b + "embeddings.cls_position_embedding"), requires_grad=False
        )
        self.ln_pre = (w(b + "ln_pre.weight"), w(b + "ln_pre.bias"))
        self.layers = []
        for i in range(vcfg["num_hidden_layers"]):
            lp = f"{b}layers.{i}."
            d = {k: w(lp + k) for k in (
                "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "ls_1.gamma", "ls_2.gamma",
                "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias",
                "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
                "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                "self_attn.out_proj.weight", "self_attn.out_proj.bias")}
            # Fuse QKV weights and biases for batched backbone
            d["qkv_w"] = torch.cat([d["self_attn.q_proj.weight"], d["self_attn.k_proj.weight"], d["self_attn.v_proj.weight"]], dim=0).contiguous()
            d["qkv_b"] = torch.cat([d["self_attn.q_proj.bias"], d["self_attn.k_proj.bias"], d["self_attn.v_proj.bias"]], dim=0).contiguous()
            self.layers.append(d)
        self.proj_in = LoRALinear(w(pre + "proj_in.weight"), lora_rank)
        self.proj_out = LoRALinear(w(pre + "proj_out.weight"), lora_rank)
        dev = self.patch_w.device
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=F32, device=dev)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=F32, device=dev)
        self.register_buffer("clip_mean", mean.to(BF16), persistent=False)
        self.register_buffer("clip_std", std.to(BF16), persistent=False)
        self._geom_cache = {}
        self._compiled_backbone = None

    @staticmethod
    def _interp_weights(indices, targets, size):
        center = (size / targets.float()) * (indices.float() + 0.5)
        distance = torch.arange(size, device=indices.device)[None, :] + 0.5
        weights = (1 - (distance - center[:, None]).abs()).clamp(min=0)
        return weights / weights.sum(dim=1, keepdim=True)

    def _get_geom(self, hh, ww, device):
        key = (hh, ww)
        cached = self._geom_cache.get(key)
        if cached is not None:
            return cached
        m = self.cfg["adapter_merge_size"]
        g = self.cfg["posemb_grid_size"]
        heads = self.cfg["num_attention_heads"]
        hd = self.cfg["hidden_size"] // heads
        rows = torch.arange(hh * ww * m * m, device=device)
        coords = torch.stack((rows // (ww * m), rows % (ww * m)), -1)
        grid = self.pos.reshape(g, g, -1)
        wy = self._interp_weights(coords[:, 0], torch.full_like(rows, hh * m), g)
        wx = self._interp_weights(coords[:, 1], torch.full_like(rows, ww * m), g)
        pos = torch.einsum("ni,nj,ijd->nd", wy, wx, grid.float()).to(BF16)
        positions = torch.cat([coords.new_zeros(1, 2), coords + 1]).to(F32)
        freqs = 1.0 / (10000 ** (torch.arange(0, hd, 4, device=device, dtype=F32) / hd))
        angles = torch.cat((positions[:, 1, None] * freqs, positions[:, 0, None] * freqs), dim=-1)
        cos = torch.cos(angles)[None, None, :, :]  # (1, 1, N, hd/2)
        sin = torch.sin(angles)[None, None, :, :]
        cached = (pos, cos, sin)
        self._geom_cache[key] = cached
        return cached

    @staticmethod
    def _apply_rotary_batched(x, cos, sin):
        # x: (B, H, N, HD)
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(x.dtype)

    def _run_backbone_batched(self, hidden, cos, sin):
        # hidden: (B, N, C)
        B, N, C = hidden.shape
        heads = self.cfg["num_attention_heads"]
        hd = C // heads
        eps = self.cfg["layer_norm_eps"]
        scale = hd**-0.5
        for p in self.layers:
            x = F.layer_norm(hidden, (C,), p["ln_1.weight"], p["ln_1.bias"], eps)
            qkv = F.linear(x, p["qkv_w"], p["qkv_b"]).view(B, N, 3, heads, hd)
            q = qkv[:, :, 0].transpose(1, 2)  # (B, H, N, HD)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            q = self._apply_rotary_batched(q, cos, sin)
            k = self._apply_rotary_batched(k, cos, sin)
            o = F.scaled_dot_product_attention(q, k, v, scale=scale).transpose(1, 2).reshape(B, N, C)
            hidden = hidden + F.linear(o, p["self_attn.out_proj.weight"], p["self_attn.out_proj.bias"]) * p["ls_1.gamma"]
            x = F.layer_norm(hidden, (C,), p["ln_2.weight"], p["ln_2.bias"], eps)
            x = F.linear(F.gelu(F.linear(x, p["mlp.c_fc.weight"], p["mlp.c_fc.bias"])), p["mlp.c_proj.weight"], p["mlp.c_proj.bias"])
            hidden = hidden + x * p["ls_2.gamma"]
        return hidden

    def forward_images(self, images, grad_ckpt=False):
        """Batched vision encoding over a list of (patches_bthwc, grid_thw)."""
        if not self.opt_kernels:
            return torch.cat([self.forward(p, g, grad_ckpt=grad_ckpt) for p, g in images])

        m, p_sz = self.cfg["adapter_merge_size"], self.cfg["native_patch_size"]
        eps = self.cfg["layer_norm_eps"]
        # Check if all items are single-image with identical (hh, ww) (standard CUA batch fast path!)
        if all(len(g) == 1 and g[0] == images[0][1][0] for _, g in images):
            _, hh, ww = images[0][1][0]
            B = len(images)
            patches = torch.cat([pt for pt, _ in images], dim=0)  # (B*hh*ww, 2, 40, 40, 3)
            with torch.no_grad():
                x = patches.float() * self.clip_std.float() + self.clip_mean.float()
                x = (2.0 * x - 1.0).to(BF16)
                x = x.mean(dim=1).permute(0, 3, 1, 2).contiguous()
                native = x.reshape(B, hh, ww, 3, m, p_sz, m, p_sz).permute(0, 1, 4, 2, 6, 5, 7, 3).reshape(B, -1, p_sz * p_sz * 3)
                hidden = F.linear(native, self.patch_w)
                pos, cos, sin = self._get_geom(hh, ww, x.device)
                cls_tok = self.cls.view(1, 1, -1).expand(B, 1, -1)
                hidden = torch.cat([cls_tok, hidden + pos[None]], dim=1)
                hidden = F.layer_norm(hidden, (hidden.shape[-1],), self.ln_pre[0], self.ln_pre[1], eps)
                if self._compiled_backbone is None:
                    self._compiled_backbone = torch.compile(self._run_backbone_batched, dynamic=True)
                hidden = self._compiled_backbone(hidden, cos, sin)
                width = hidden.shape[-1]
                pooled = hidden[:, 1:].reshape(B, hh, m, ww, m, width).permute(0, 1, 3, 2, 4, 5).reshape(B * hh * ww, m * m * width)
            return self.proj_out(F.gelu(self.proj_in(pooled)))
        return torch.cat([self.forward(pt, g, grad_ckpt=grad_ckpt) for pt, g in images])

    def forward(self, patches_bthwc, grid_thw, grad_ckpt=False):
        m, p_sz = self.cfg["adapter_merge_size"], self.cfg["native_patch_size"]
        eps = self.cfg["layer_norm_eps"]
        x = patches_bthwc.float() * self.clip_std.float() + self.clip_mean.float()
        x = (2.0 * x - 1.0).to(BF16)
        x = x.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        outs = []
        offset = 0
        for _, hh, ww in grid_thw:
            image = x[offset : offset + hh * ww]
            offset += hh * ww
            with torch.no_grad():
                native = image.reshape(hh, ww, 3, m, p_sz, m, p_sz).permute(0, 3, 1, 5, 4, 6, 2).reshape(1, -1, p_sz * p_sz * 3)
                hidden = F.linear(native, self.patch_w)
                pos, cos, sin = self._get_geom(hh, ww, x.device)
                hidden = torch.cat([self.cls.view(1, 1, -1), hidden + pos[None]], dim=1)
                hidden = F.layer_norm(hidden, (hidden.shape[-1],), self.ln_pre[0], self.ln_pre[1], eps)
                if self.opt_kernels:
                    if self._compiled_backbone is None:
                        self._compiled_backbone = torch.compile(self._run_backbone_batched, dynamic=True)
                    hidden = self._compiled_backbone(hidden, cos, sin)
                else:
                    hidden = self._run_backbone_batched(hidden, cos, sin)
                width = hidden.shape[-1]
                pooled = hidden[0, 1:].reshape(hh, m, ww, m, width).permute(0, 2, 1, 3, 4).reshape(hh * ww, m * m * width)
            outs.append(self.proj_out(F.gelu(self.proj_in(pooled))))
        return torch.cat(outs)


class Inkling2LoopForTraining(nn.Module):
    def __init__(self, cfg, vision, opt_kernels=True, opt_attn=True):
        super().__init__()
        self.cfg = cfg
        self.visual = vision
        self.opt_kernels = opt_kernels
        self.opt_attn = opt_attn
        self.layers = nn.ModuleList()
        self.attn_q = []
        self.mlp_q = []
        self.out_q = None
        self.loop_q = None
        loop = cfg["loop_config"]
        self.core = range(loop["core_start"], loop["core_end"])
        self.num_passes = loop["num_passes"]
        self.block = cfg["n_layers_per_attn_res_block"]
        self.vocab = cfg["unpadded_vocab_size"]
        self.softcap = cfg["final_logit_softcapping"]

    def embed(self, input_ids, image_mask=None, image_embeds=None):
        safe = input_ids.clamp(min=0)
        e = F.embedding(safe, self.embed_w)
        x = fused_rms_norm(e, self.embed_norm, self.cfg["rms_norm_eps"]) if self.opt_kernels else rms_norm(e, self.embed_norm, self.cfg["rms_norm_eps"])
        if image_mask is not None and image_embeds is not None:
            x = x.masked_scatter(image_mask[:, None], image_embeds.to(x.dtype))
        return x

    def _visit(self, layer_idx, blocks, partial, bank, ctx: Ctx, bm0=None, bm1=None, bm2=None, bm3=None):
        L = self.layers[layer_idx]
        boundary = layer_idx % self.block == 0
        use_k = ctx.opt_kernels
        rn = fused_rms_norm if use_k else rms_norm
        core_i = layer_idx - self.core.start
        has_bank = (bank is not None) or (bm0 is not None)

        if has_bank:
            if not blocks:
                streams = [partial] if L.local else [partial] * 3
            else:
                streams = fused_attn_res(blocks + [partial], None, self.attn_q[layer_idx - 1], out_dtype=BF16) if use_k else attn_res(blocks + [partial], None, self.attn_q[layer_idx - 1])
            if bm0 is not None:
                msgs = (bm0, bm1, bm2)
            else:
                msgs = fused_attn_res(bank, None, self.loop_q[core_i, 0:3], out_dtype=BF16) if use_k else attn_res(bank, None, self.loop_q[core_i, 0:3])
            norms = (L.attn_norm,) * 3 if L.local else (L.q_input_norm, L.k_input_norm, L.v_input_norm)
            if use_k:
                src = L.controllers.forward_and_norm(streams * 3 if L.local else streams, msgs, (0, 1, 2), norms)
            else:
                mixed = L.controllers(streams * 3 if L.local else streams, msgs, (0, 1, 2))
                src = [rn(m, n, out_dtype=BF16) for m, n in zip(mixed, norms)]
        elif not blocks:
            if L.local:
                s = rn(partial, L.attn_norm, out_dtype=BF16)
                src = [s, s, s]
            else:
                src = [rn(partial, n, out_dtype=BF16) for n in (L.q_input_norm, L.k_input_norm, L.v_input_norm)]
        elif use_k:
            if L.local:
                s = fused_attn_res_rmsnorm(blocks + [partial], None, self.attn_q[layer_idx - 1], (L.attn_norm,))[0]
                src = [s, s, s]
            else:
                src = fused_attn_res_rmsnorm(
                    blocks + [partial], None, self.attn_q[layer_idx - 1],
                    (L.q_input_norm, L.k_input_norm, L.v_input_norm),
                )
        else:
            streams = attn_res(blocks + [partial], None, self.attn_q[layer_idx - 1])
            if L.local:
                s = rn(streams[0], L.attn_norm, out_dtype=BF16)
                src = [s, s, s]
            else:
                src = [rn(s, n, out_dtype=BF16) for s, n in zip(streams, (L.q_input_norm, L.k_input_norm, L.v_input_norm))]

        attn = L.attn(src[0], src[1], src[2], ctx)
        attn = fused_short_conv_static(attn, L.attn_sconv_4d, ctx.doc) if use_k else short_conv_static(attn, L.attn_sconv, ctx.doc)

        if has_bank:
            if boundary:
                mlp_in = fused_attn_res(blocks + [partial, attn], None, self.mlp_q[layer_idx][None], out_dtype=BF16)[0] if use_k else attn_res(blocks + [partial, attn], None, self.mlp_q[layer_idx][None])[0]
            else:
                mlp_in = fused_attn_res(blocks + [partial], attn, self.mlp_q[layer_idx][None], out_dtype=BF16)[0] if use_k else attn_res(blocks + [partial], attn, self.mlp_q[layer_idx][None])[0]
            if bm3 is not None:
                msg = (bm3,)
            else:
                msg = fused_attn_res(bank, None, self.loop_q[core_i, 3:4], out_dtype=BF16) if use_k else attn_res(bank, None, self.loop_q[core_i, 3:4])
            if use_k:
                mlp_in = L.controllers.forward_and_norm([mlp_in], msg, (3,), (L.mlp_norm,))
            else:
                mlp_in = rn(L.controllers([mlp_in], msg, (3,))[0], L.mlp_norm, out_dtype=BF16)
        elif use_k:
            if boundary:
                mlp_in = fused_attn_res_rmsnorm(blocks + [partial, attn], None, self.mlp_q[layer_idx][None], (L.mlp_norm,))[0]
            else:
                mlp_in = fused_attn_res_rmsnorm(blocks + [partial], attn, self.mlp_q[layer_idx][None], (L.mlp_norm,))[0]
        else:
            if boundary:
                mlp_in = attn_res(blocks + [partial, attn], None, self.mlp_q[layer_idx][None])[0]
            else:
                mlp_in = attn_res(blocks + [partial], attn, self.mlp_q[layer_idx][None])[0]
            mlp_in = rn(mlp_in, L.mlp_norm, out_dtype=BF16)

        weights, ids = L.router(mlp_in, opt_kernels=use_k)
        if isinstance(L.experts, SM120Experts):
            mlp = L.experts(mlp_in, weights, ids, ctx.act_quant, recompute_gate_up=ctx.recompute_gate_up)
        else:
            mlp = L.experts(mlp_in, weights, ids, ctx.act_quant)
        mlp = fused_short_conv_static(mlp, L.mlp_sconv_4d, ctx.doc) if use_k else short_conv_static(mlp, L.mlp_sconv, ctx.doc)
        if boundary:
            return blocks + [partial], attn + mlp
        if use_k:
            return blocks, fused_residual_add3(partial, attn, mlp)
        return blocks, (partial.float() + attn.float() + mlp.float()).to(BF16)

    def _run_visit(self, layer_idx, blocks, partial, bank, ctx, do_ckpt, bank_msgs=None):
        if do_ckpt and torch.is_grad_enabled():
            if bank_msgs is not None:
                return checkpoint(self._visit, layer_idx, blocks, partial, None, ctx, *bank_msgs, use_reentrant=False)
            return checkpoint(self._visit, layer_idx, blocks, partial, bank, ctx, use_reentrant=False)
        if bank_msgs is not None:
            return self._visit(layer_idx, blocks, partial, None, ctx, *bank_msgs)
        return self._visit(layer_idx, blocks, partial, bank, ctx)

    def hidden_states(self, x, ctx: Ctx, grad_ckpt=False, ckpt_frac=1.0):
        blocks, partial = [], x
        visit = 0

        def should_ckpt():
            nonlocal visit
            v = visit
            visit += 1
            if not grad_ckpt or ckpt_frac <= 0.0:
                return False
            if ckpt_frac >= 1.0:
                return True
            # Uniformly distribute checkpointed visits using Bresenham-style accumulator
            return int((v + 1) * ckpt_frac) > int(v * ckpt_frac)

        for i in range(self.core.start):
            blocks, partial = self._run_visit(i, blocks, partial, None, ctx, should_ckpt())
        prelude_partial, prelude_blocks = partial, list(blocks)
        bank = None
        bank_msgs = None
        for pass_idx in range(self.num_passes):
            if pass_idx:
                bank = [*blocks[len(prelude_blocks) + 1 :], partial]
                partial, blocks = prelude_partial, list(prelude_blocks)
                if ctx.opt_kernels:
                    bank_msgs = fused_attn_res_bank(bank, self.loop_q_flat_bf16, self.loop_q_scale)
            for i in self.core:
                if bank_msgs is not None:
                    c4 = (i - self.core.start) * 4
                    bm = bank_msgs[c4 : c4 + 4]
                else:
                    bm = None
                blocks, partial = self._run_visit(i, blocks, partial, bank, ctx, should_ckpt(), bm)
        for i in range(self.core.stop, len(self.layers)):
            blocks, partial = self._run_visit(i, blocks, partial, None, ctx, should_ckpt())

        if ctx.opt_kernels:
            return fused_attn_res_rmsnorm(blocks + [partial], None, self.out_q[None], (self.final_norm,))[0]
        out = attn_res(blocks + [partial], None, self.out_q[None])[0]
        return rms_norm(out, self.final_norm, out_dtype=BF16)

    def logits(self, hidden):
        if _MM_OUT_DTYPE and self.opt_kernels:
            logits = _BF16LinearF32Out.apply(hidden, self.unembed)
        elif _MM_OUT_DTYPE and not (torch.is_grad_enabled() and hidden.requires_grad):
            logits = torch.mm(hidden, self.unembed.t(), out_dtype=F32)
        else:
            logits = F.linear(hidden.float(), self.unembed.float())
        logits = self.softcap * torch.tanh(logits / self.softcap)
        logits[..., self.vocab :] = float("-inf")
        return logits

    def forward(self, input_ids, positions=None, doc=None, images=None, image_mask=None, *,
                attn_impl=None, act_quant=True, grad_ckpt=False, ckpt_frac=1.0,
                recompute_gate_up=False, logit_rows=None):
        t = input_ids.shape[0]
        dev = input_ids.device
        if attn_impl is None:
            attn_impl = "fa4" if self.opt_attn else "flex"
        if positions is None:
            positions = torch.arange(t, device=dev)
        if doc is None:
            doc = torch.zeros(t, dtype=torch.int32, device=dev)
        image_embeds = None
        if images:
            image_embeds = self.visual.forward_images(images, grad_ckpt=grad_ckpt)
        x = self.embed(input_ids, image_mask if images else None, image_embeds)
        ctx = Ctx(
            positions=positions,
            doc=doc,
            attn_impl=attn_impl,
            act_quant=act_quant,
            opt_kernels=self.opt_kernels,
            recompute_gate_up=recompute_gate_up,
        )
        if self.opt_kernels:
            ctx.tau = log_scaling_tau(positions, self.cfg["log_scaling_n_floor"], self.cfg["log_scaling_alpha"])
        if attn_impl == "flex":
            ctx.block_masks = flex_block_masks(doc, self.cfg["sliding_window_size"] - 1)
        elif attn_impl == "fa4":
            ctx.cu_seqlens, ctx.max_seqlen = doc_to_cu_seqlens(doc)
        hidden = self.hidden_states(x, ctx, grad_ckpt=grad_ckpt, ckpt_frac=ckpt_frac)
        if logit_rows is not None:
            hidden = hidden[logit_rows]
        return self.logits(hidden)


def _check_mm_out_dtype():
    try:
        a = torch.zeros(16, 16, dtype=BF16, device="cuda")
        return torch.mm(a, a, out_dtype=F32).dtype == F32
    except Exception:
        return False


_MM_OUT_DTYPE = False


# ----------------------------------------------------------------------------- loading


def _chunked(fn, values, scales, experts_per_chunk=8):
    out = None
    for s in range(0, values.shape[0], experts_per_chunk):
        part = fn(values[s : s + experts_per_chunk], scales[s : s + experts_per_chunk])
        if out is None:
            out = torch.empty((values.shape[0], *part.shape[1:]), dtype=part.dtype, device=part.device)
        out[s : s + part.shape[0]] = part
    return out


def _build_sm120_nvfp4_experts(w13_q, w13_s, w2_q, w2_s, w13_amax, w13_den, w2_amax, w2_den, gate_cap, up_cap):
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    o13 = (w13_amax / w13_den).float()
    o2 = (w2_amax / w2_den).float()
    # Dequantize chunked -> apply outer scale -> transpose -> quantize to MXFP8 for backward (frees temporary immediately)
    w13_m = _chunked(dequant_nvfp4_mantissa, w13_q, w13_s)
    w13_t = (w13_m.float() * o13[:, None, None]).to(BF16).transpose(-2, -1).contiguous()
    del w13_m
    w13_T_fp8, w13_T_sf = quantize_weight_to_mxfp8_cute(w13_t)
    del w13_t

    w2_m = _chunked(dequant_nvfp4_mantissa, w2_q, w2_s)
    w2_t = (w2_m.float() * o2[:, None, None]).to(BF16).transpose(-2, -1).contiguous()
    del w2_m
    w2_T_fp8, w2_T_sf = quantize_weight_to_mxfp8_cute(w2_t)
    del w2_t

    w13_sf_fwd = swizzle_blockscale(w13_s).flatten(1).contiguous()
    w2_sf_fwd = swizzle_blockscale(w2_s).flatten(1).contiguous()
    return SM120Experts(
        "nvfp4", w13_q.contiguous(), w13_sf_fwd, w2_q.contiguous(), w2_sf_fwd,
        w13_T_fp8, w13_T_sf, w2_T_fp8, w2_T_sf,
        w13_amax, w13_den, w2_amax, w2_den, gate_cap, up_cap,
    )


def _build_sm120_mxfp8_experts(w13_q, w13_s, w2_q, w2_s, gate_cap, up_cap):
    from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales

    w13_m = _chunked(dequant_mxfp8, w13_q, w13_s)
    w13_T_fp8, w13_T_sf = quantize_weight_to_mxfp8_cute(w13_m.transpose(-2, -1).contiguous())
    del w13_m

    w2_m = _chunked(dequant_mxfp8, w2_q, w2_s)
    w2_T_fp8, w2_T_sf = quantize_weight_to_mxfp8_cute(w2_m.transpose(-2, -1).contiguous())
    del w2_m

    return SM120Experts(
        "mxfp8", w13_q.contiguous(), mxfp8_weight_scales(w13_s), w2_q.contiguous(), mxfp8_weight_scales(w2_s),
        w13_T_fp8, w13_T_sf, w2_T_fp8, w2_T_sf,
        gate_cap=gate_cap, up_cap=up_cap,
    )


def load_model(
    model_dir,
    device="cuda",
    lora_rank=0,
    router_lora_rank=None,
    vision_lora_rank=None,
    verbose=True,
    opt_moe=True,
    opt_kernels=True,
    opt_attn=True,
):
    global _MM_OUT_DTYPE
    from safetensors import safe_open
    import time

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    cfg_all = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = cfg_all["text_config"]
    vcfg = cfg_all["vision_config"]
    router_lora_rank = lora_rank if router_lora_rank is None else router_lora_rank
    vision_lora_rank = lora_rank if vision_lora_rank is None else vision_lora_rank
    path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
    f = safe_open(path, framework="pt", device=device)
    used = set()

    def w(name):
        used.add(name)
        return f.get_tensor(name)

    vision = VisionEncoder(vcfg, w, vision_lora_rank, opt_kernels=opt_kernels)
    model = Inkling2LoopForTraining(cfg, vision, opt_kernels=opt_kernels, opt_attn=opt_attn)
    model.embed_w = w("model.llm.embed.weight")
    model.embed_norm = w("model.llm.embed_norm.weight")
    model.final_norm = w("model.llm.norm.weight")
    model.unembed = w("model.llm.unembed.weight")
    width = cfg["hidden_size"]
    scale = width**-0.5
    model.attn_q = [w(f"model.llm.attn_res.attn_queries.{i}").float().reshape(-1, width).contiguous() * scale for i in range(cfg["num_hidden_layers"] - 1)]
    model.mlp_q = [w(f"model.llm.attn_res.mlp_queries.{i}").float().contiguous() * scale for i in range(cfg["num_hidden_layers"])]
    model.out_q = w("model.llm.attn_res.output_query").float().contiguous() * scale
    qs = w("model.llm.middle_cycle_cross_pass_queries")
    kvq = w("model.llm.middle_cycle_cross_pass_kv_queries")
    loop_q_raw = torch.stack((qs[0::2], kvq[0::2], kvq[1::2], qs[1::2]), dim=1).contiguous()
    model.loop_q = (loop_q_raw.float() * scale).contiguous()
    model.loop_q_flat_bf16 = loop_q_raw.view(-1, width).to(BF16).contiguous()
    model.loop_q_scale = scale
    for i in range(cfg["num_hidden_layers"]):
        layer = DecoderLayer(cfg, i, w, lora_rank, router_lora_rank, opt_kernels=opt_kernels)
        p = f"model.llm.layers.{i}.mlp.experts."
        if f"{p}w13_weight_amax" in f.keys():
            w13_q, w13_s = w(p + "w13_weight"), w(p + "w13_weight_scale")
            w2_q, w2_s = w(p + "w2_weight"), w(p + "w2_weight_scale")
            w13_am, w13_dn = w(p + "w13_weight_amax"), w(p + "w13_weight_outer_scale_denominator")
            w2_am, w2_dn = w(p + "w2_weight_amax"), w(p + "w2_weight_outer_scale_denominator")
            if opt_moe:
                layer.experts = _build_sm120_nvfp4_experts(
                    w13_q, w13_s, w2_q, w2_s, w13_am, w13_dn, w2_am, w2_dn,
                    cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"],
                )
            else:
                w13 = _chunked(dequant_nvfp4_mantissa, w13_q, w13_s)
                w2 = _chunked(dequant_nvfp4_mantissa, w2_q, w2_s)
                layer.experts = Experts("nvfp4", w13, w2, (w13_am / w13_dn).float(), (w2_am / w2_dn).float(),
                                        cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"])
        else:
            w13_q, w13_s = w(p + "w13_weight"), w(p + "w13_weight_scale")
            w2_q, w2_s = w(p + "w2_weight"), w(p + "w2_weight_scale")
            if opt_moe:
                layer.experts = _build_sm120_mxfp8_experts(
                    w13_q, w13_s, w2_q, w2_s, cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"],
                )
            else:
                w13 = _chunked(dequant_mxfp8, w13_q, w13_s)
                w2 = _chunked(dequant_mxfp8, w2_q, w2_s)
                one = torch.ones(w13.shape[0], device=device)
                layer.experts = Experts("mxfp8", w13, w2, one, one, cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"])
        if i in model.core:
            cp = f"model.llm.layers.{i}.controllers."
            mods = ("attn", "k", "v", "mlp")
            layer.controllers = Controllers(
                torch.stack([w(f"{cp}{m}.down_weight") for m in mods]),
                torch.stack([w(f"{cp}{m}.gain_weight") for m in mods]),
                torch.stack([w(f"{cp}{m}.forget_weight") for m in mods]),
            )
        model.layers.append(layer)
        torch.cuda.empty_cache()
    missing = [k for k in f.keys() if k not in used]
    _MM_OUT_DTYPE = _check_mm_out_dtype()
    if verbose:
        print(f"[inkling2_train_opt] loaded in {time.time() - t0:.1f}s (opt_moe={opt_moe} opt_kernels={opt_kernels} opt_attn={opt_attn}); "
              f"unused: {len(missing)}; mm_f32: {_MM_OUT_DTYPE}; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    model.eval()
    return model
