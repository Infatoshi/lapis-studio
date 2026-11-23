"""Trainable PyTorch port of Inkling2LoopForConditionalGeneration (model_type inkling2_loop).

Mirrors the inference semantics of SGLang's inkling2 implementation (lmsysorg/sglang:inkling-2-mini,
sglang/srt/models/inkling2.py + models/inkling_common/* + kernels/ops/*), as plain differentiable
PyTorch over one packed sequence (no KV cache, no paging):

  text:  embed -> embed_norm; image token rows replaced by the vision tower output
  LLM:   24 physical layers, 20 sliding-window (1024) + 4 global; layers [6, 18) run 3 passes;
         later passes mix "messages" read from the previous pass through low-rank controllers.
         Cross-layer attention residuals over 6-layer blocks replace the plain residual stream.
  attn:  q/k/v/r/gate/sink projections; causal short convs on k, v (dynamic weights on global
         layers); L2-clipped q/k/r with learned scales and log-length tau; RMS-normed v;
         relative-position logits r @ proj added to the scores; per-token sink; sigmoid gate.
  MoE:   128 experts, top-8 sigmoid router with selection bias, renormalized x route_scale 8;
         capped SwiGLU; NVFP4 experts on layers 0-19, MXFP8 experts on layers 20-23 (frozen,
         stored as exact BF16 mantissas + FP32 outer scales, run with torch._grouped_mm).
  head:  final attention-residual readout -> RMSNorm -> unembed (FP32 out) -> softcap 30.

attn_impl="eager" is the FP32 reference attention used for parity; "flex" is FlexAttention for
training.  act_quant=True emulates the server's NVFP4/MXFP8 activation quantization in the
experts (straight-through gradients), so the trained policy sees the served numerics.
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

from .lora import LoRALinear
from .quant import (
    dequant_mxfp8,
    dequant_nvfp4_mantissa,
    mxfp8_quantize_dequant,
    nvfp4_quantize_mantissa,
    ste,
)

BF16 = torch.bfloat16
F32 = torch.float32

# Activation fake-quant: eager by default (reference); compile_quant() fuses each quantizer into
# a few Inductor kernels (bit-exact, checked by train_bench --compile-quant).
_QUANT = [nvfp4_quantize_mantissa, mxfp8_quantize_dequant]


def compile_quant():
    _QUANT[:] = [torch.compile(f, dynamic=True) for f in (nvfp4_quantize_mantissa, mxfp8_quantize_dequant)]


def _quant_fns():
    return _QUANT


# ----------------------------------------------------------------------------- primitives


def rms_norm(x, weight=None, eps=1e-6, out_dtype=None):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(out_dtype or x.dtype)


def attn_res(sources, addend, queries_scaled, normalize_values=True, eps=1e-6):
    """Softmax over sources of (rms-normalized) source . query; returns [Q] fp32 (T, W) mixes.

    sources: list of (T, W) tensors (same dtype); addend is added (fp32) to the last source.
    queries_scaled: (Q, W) fp32, already multiplied by W**-0.5.
    """
    xs = [s.float() for s in sources]
    if addend is not None:
        xs[-1] = xs[-1] + addend.float()
    x = torch.stack(xs)  # (N, T, W)
    rstd = torch.rsqrt(x.pow(2).mean(-1) + eps)  # (N, T)
    logits = torch.einsum("ntw,qw->qnt", x, queries_scaled) * rstd  # (Q, N, T)
    p = torch.softmax(logits, dim=1)
    if normalize_values:
        p = p * rstd
    return list(torch.einsum("qnt,ntw->qtw", p, x).unbind(0))


def shift_rows(x, lag, doc):
    """x[t - lag] within the same document, else 0 (causal conv history without cache)."""
    if lag == 0:
        return x
    shifted = F.pad(x, (0, 0, lag, 0))[: x.shape[0]]
    same = F.pad(doc, (lag, 0), value=-1)[: x.shape[0]] == doc
    return shifted * same[:, None].to(x.dtype)


def short_conv_static(x, weight, doc):
    """SGLang _sconv kernel, static weights (D, 1, W), residual, scale (W + 1) ** -0.5."""
    w = weight[:, 0, :].float()  # (D, W)
    width = w.shape[1]
    xf = x.float()
    acc = None
    for tap in range(width):
        term = shift_rows(xf, width - 1 - tap, doc) * w[:, tap]
        acc = term if acc is None else acc + term
    return ((acc + xf) * (width + 1) ** -0.5).to(x.dtype)


def short_conv_dynamic(x, static_w, dyn, doc):
    """Global-layer K/V conv: (static_w[lag] + dyn[t, head, lag]) * x[t - lag], residual,
    scale (2W + 1) ** -0.5.  static_w (W, D); dyn (T, heads, W)."""
    width, d = static_w.shape
    heads = dyn.shape[1]
    xf = x.float()
    dynf = dyn.float().repeat_interleave(d // heads, dim=1)  # (T, D, W)
    acc = None
    for lag in range(width):
        weight = static_w[lag].float()[None, :] + dynf[:, :, lag]
        term = shift_rows(xf, lag, doc) * weight
        acc = term if acc is None else acc + term
    return ((acc + xf) * (2 * width + 1) ** -0.5).to(x.dtype)


def l2_clip(x, threshold, scale=None, tau=None, eps=1e-6):
    """qk_l2_clip_fwd_cutedsl: (x * scale) * min(1, thr / (||x * scale|| + eps)) [-> bf16 * tau]."""
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


# ----------------------------------------------------------------------------- attention kernels


def attention_eager(q, k, v, rel, sink, *, local, window, doc, chunk=256):
    """FP32 reference: softmax(q.k * d**-0.5 + rel_bias) with causal (+window) mask, then the
    sink factor sigmoid(lse - sink).  q (T,H,D), k/v (T,KV,D), rel (T,H,E), sink (T,H)."""
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
        lse = torch.logsumexp(scores, -1)  # (H, q)
        o = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), vf[k0:e])
        factor = torch.sigmoid(lse - sink[s:e].float().T).T  # (q, H)
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
    """Block masks over T queries x (T + 1) keys, key 0 being the virtual sink key."""
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
    relh = rel.permute(1, 0, 2)  # (H, T, E)
    sinkh = sink.permute(1, 0)  # (H, T)

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
    attn_impl: str = "flex"
    act_quant: bool = True
    block_masks: dict | None = None


class Attention(nn.Module):
    def __init__(self, cfg, layer_id, w, lora_rank):
        super().__init__()
        self.local = layer_id in cfg["local_layer_ids"]
        self.heads, self.kv_heads, self.head_dim = 24, 8, 128
        self.d_rel = cfg["d_rel"]
        self.window = cfg["sliding_window_size"] - 1
        self.qk_thr, self.rel_thr = cfg["qk_l2_clip_threshold"], cfg["rel_l2_clip_threshold"]
        self.n_floor, self.alpha = cfg["log_scaling_n_floor"], cfg["log_scaling_alpha"]
        p = f"model.llm.layers.{layer_id}.attn."
        self.wq = LoRALinear(w(p + "wq_du.weight"), lora_rank)
        self.wk = LoRALinear(w(p + "wk_dv.weight"), lora_rank)
        self.wv = LoRALinear(w(p + "wv_dv.weight"), lora_rank)
        self.wo = LoRALinear(w(p + "wo_ud.weight"), lora_rank)
        # r / gate / sink projections stay frozen (merged into one GEMM).
        self.register_buffer(
            "w_rgs",
            torch.cat([w(p + "wr_du.weight"), w(p + "gate_attn.weight"), w(p + "attn_sink_proj.weight")]),
            persistent=False,
        )
        self.register_buffer("q_scale", w(p + "q_l2_clip.scale"), persistent=False)
        self.register_buffer("rel_scale", w(p + "rel_l2_clip.scale"), persistent=False)
        self.register_buffer("rel_proj", w(p + "rel_logits_proj.proj"), persistent=False)
        if self.local:
            self.register_buffer("k_conv", w(p + "k_sconv.weight"), persistent=False)
            self.register_buffer("v_conv", w(p + "v_sconv.weight"), persistent=False)
        else:
            self.register_buffer("k_static", w(p + "k_sconv.static_w"), persistent=False)
            self.register_buffer("v_static", w(p + "v_sconv.static_w"), persistent=False)
            self.register_buffer("k_proj", w(p + "k_sconv.weight_proj"), persistent=False)
            self.register_buffer("v_proj", w(p + "v_sconv.weight_proj"), persistent=False)

    def forward(self, src_q, src_k, src_v, ctx: Ctx):
        t = src_q.shape[0]
        h, kvh, d, dr = self.heads, self.kv_heads, self.head_dim, self.d_rel
        q = self.wq(src_q)
        k = self.wk(src_k)
        v = self.wv(src_v)
        r, gate, sinks = F.linear(src_q, self.w_rgs).split([h * dr, h * d, h], dim=-1)
        if self.local:
            k = short_conv_static(k, self.k_conv, ctx.doc)
            v = short_conv_static(v, self.v_conv, ctx.doc)
            tau = None
        else:
            dk = F.linear(src_k, self.k_proj).view(t, kvh, -1)
            dv = F.linear(src_v, self.v_proj).view(t, kvh, -1)
            k = short_conv_dynamic(k, self.k_static, dk, ctx.doc)
            v = short_conv_dynamic(v, self.v_static, dv, ctx.doc)
            tau = log_scaling_tau(ctx.positions, self.n_floor, self.alpha)
        q = l2_clip(q.view(t, h, d), self.qk_thr, self.q_scale, tau)
        k = l2_clip(k.view(t, kvh, d), self.qk_thr)
        v = rms_norm(v.view(t, kvh, d), None, 1e-6)
        r = l2_clip(r.reshape(t, h, dr), self.rel_thr, self.rel_scale, tau)
        rel = torch.einsum("thd,de->the", r, self.rel_proj)  # bf16, like the server's einsum
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
        self.register_buffer("bias", bias, persistent=False)
        self.topk, self.route_scale = topk, route_scale

    def forward(self, x):
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
    """x (R, K) rows sorted by expert, w_t (E, K, N), offs int32 cumulative ends."""
    global _GROUPED_MM_OK
    if _GROUPED_MM_OK is None:
        try:
            torch._grouped_mm(x[:16], w_t[:1], offs=torch.tensor([16], device=x.device, dtype=torch.int32))
            _GROUPED_MM_OK = True
        except Exception as exc:  # pragma: no cover
            print(f"[inkling2_train] torch._grouped_mm unavailable ({exc}); using per-expert loop")
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
    """Frozen routed experts; weights are exact BF16 mantissas + per-expert FP32 outer scales."""

    def __init__(self, fmt, w13, w2, outer13, outer2, gate_cap=4.0, up_cap=25.0):
        super().__init__()
        self.fmt = fmt
        self.register_buffer("w13", w13, persistent=False)  # (E, 2I, H) interleaved gate/up rows
        self.register_buffer("w2", w2, persistent=False)  # (E, H, I)
        self.register_buffer("outer13", outer13, persistent=False)
        self.register_buffer("outer2", outer2, persistent=False)
        self.gate_cap, self.up_cap = gate_cap, up_cap

    def _quant_in(self, x, act_quant):
        """-> (bf16 GEMM operand rows, fp32 per-row scale or None)."""
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
        d = d.to(BF16).float()  # routed_partial rounds each expert row to bf16
        out = torch.zeros(t, d.shape[1], device=x.device, dtype=F32).index_add(0, tok, d)
        return out.to(BF16)


class Controllers(nn.Module):
    """Stacked loop message controllers of one core layer, CONTROLLER_MODULES order."""

    def __init__(self, down, gain, forget):
        super().__init__()
        self.register_buffer("down", down, persistent=False)  # (4, R, 2W)
        self.register_buffer("gain", gain, persistent=False)  # (4, W, R)
        self.register_buffer("forget", forget, persistent=False)

    def forward(self, dests, messages, rows):
        out = []
        for i, dest, msg in zip(rows, dests, messages):
            normed = torch.cat([rms_norm(dest, None, out_dtype=BF16), rms_norm(msg, None, out_dtype=BF16)], -1)
            hidden = F.linear(normed.float(), self.down[i].float()).to(BF16).float()
            gain = F.linear(hidden, self.gain[i].float())
            forget = F.linear(hidden, self.forget[i].float())
            out.append((1.0 - torch.tanh(forget)) * dest.float() + torch.tanh(gain) * msg.float())
        return out


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_id, w, lora_rank, router_lora_rank):
        super().__init__()
        self.layer_id = layer_id
        self.local = layer_id in cfg["local_layer_ids"]
        p = f"model.llm.layers.{layer_id}."
        self.attn = Attention(cfg, layer_id, w, lora_rank)
        if self.local:
            self.register_buffer("attn_norm", w(p + "attn_norm.weight"), persistent=False)
        else:
            for n in ("q_input_norm", "k_input_norm", "v_input_norm"):
                self.register_buffer(n, w(p + n + ".weight"), persistent=False)
        self.register_buffer("mlp_norm", w(p + "mlp_norm.weight"), persistent=False)
        self.register_buffer("attn_sconv", w(p + "attn_sconv.weight"), persistent=False)
        self.register_buffer("mlp_sconv", w(p + "mlp_sconv.weight"), persistent=False)
        self.router = Router(w(p + "mlp.gate.weight"), w(p + "mlp.gate.bias"), router_lora_rank)
        self.experts = None  # filled by the loader
        self.controllers = None


class VisionEncoder(nn.Module):
    def __init__(self, vcfg, w, lora_rank):
        super().__init__()
        self.cfg = vcfg
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
            self.layers.append({k: w(lp + k) for k in (
                "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "ls_1.gamma", "ls_2.gamma",
                "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias",
                "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
                "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                "self_attn.out_proj.weight", "self_attn.out_proj.bias")})
        self.proj_in = LoRALinear(w(pre + "proj_in.weight"), lora_rank)
        self.proj_out = LoRALinear(w(pre + "proj_out.weight"), lora_rank)
        dev = self.patch_w.device
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=F32, device=dev)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=F32, device=dev)
        self.register_buffer("clip_mean", mean.to(BF16), persistent=False)
        self.register_buffer("clip_std", std.to(BF16), persistent=False)

    @staticmethod
    def _interp_weights(indices, targets, size):
        center = (size / targets.float()) * (indices.float() + 0.5)
        distance = torch.arange(size, device=indices.device)[None, :] + 0.5
        weights = (1 - (distance - center[:, None]).abs()).clamp(min=0)
        return weights / weights.sum(dim=1, keepdim=True)

    @staticmethod
    def _rotary(x, positions):
        dim = x.shape[-1]
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 4, device=x.device, dtype=F32) / dim))
        angles = torch.cat((positions[:, 1, None] * freqs, positions[:, 0, None] * freqs), dim=-1)
        cos, sin = torch.cos(angles)[:, None, :], torch.sin(angles)[:, None, :]
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(x.dtype)

    def _layer(self, p, hidden, positions):
        n = hidden.shape[0]
        heads = self.cfg["num_attention_heads"]
        hd = hidden.shape[1] // heads
        x = F.layer_norm(hidden, (hidden.shape[1],), p["ln_1.weight"], p["ln_1.bias"], self.cfg["layer_norm_eps"])
        q = F.linear(x, p["self_attn.q_proj.weight"], p["self_attn.q_proj.bias"]).view(n, heads, hd)
        k = F.linear(x, p["self_attn.k_proj.weight"], p["self_attn.k_proj.bias"]).view(n, heads, hd)
        v = F.linear(x, p["self_attn.v_proj.weight"], p["self_attn.v_proj.bias"]).view(n, heads, hd)
        q, k = self._rotary(q, positions), self._rotary(k, positions)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], scale=hd**-0.5
        )[0].transpose(0, 1).reshape(n, -1)
        hidden = hidden + F.linear(o, p["self_attn.out_proj.weight"], p["self_attn.out_proj.bias"]) * p["ls_1.gamma"]
        x = F.layer_norm(hidden, (hidden.shape[1],), p["ln_2.weight"], p["ln_2.bias"], self.cfg["layer_norm_eps"])
        x = F.linear(F.gelu(F.linear(x, p["mlp.c_fc.weight"], p["mlp.c_fc.bias"])), p["mlp.c_proj.weight"], p["mlp.c_proj.bias"])
        return hidden + x * p["ls_2.gamma"]

    def forward(self, patches_bthwc, grid_thw, grad_ckpt=False):
        """patches (N, 2, 40, 40, 3) bf16 from Inkling2ImageProcessor; grid [[1, h, w], ...]."""
        m, p = self.cfg["adapter_merge_size"], self.cfg["native_patch_size"]
        x = patches_bthwc.float() * self.clip_std.float() + self.clip_mean.float()
        x = (2.0 * x - 1.0).to(BF16)  # torch.compile'd _renormalize: one rounding
        x = x.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        outs = []
        offset = 0
        for _, hh, ww in grid_thw:
            image = x[offset : offset + hh * ww]
            offset += hh * ww
            native = image.reshape(hh, ww, 3, m, p, m, p).permute(0, 3, 1, 5, 4, 6, 2).reshape(-1, p * p * 3)
            rows = torch.arange(hh * ww * m * m, device=x.device)
            coords = torch.stack((rows // (ww * m), rows % (ww * m)), -1)
            hidden = F.linear(native, self.patch_w)
            g = self.cfg["posemb_grid_size"]
            grid = self.pos.reshape(g, g, -1)
            wy = self._interp_weights(coords[:, 0], torch.full_like(rows, hh * m), g)
            wx = self._interp_weights(coords[:, 1], torch.full_like(rows, ww * m), g)
            pos = torch.einsum("ni,nj,ijd->nd", wy, wx, grid.float()).to(BF16)
            hidden = torch.cat([self.cls[None], hidden + pos])
            positions = torch.cat([coords.new_zeros(1, 2), coords + 1]).to(F32)
            hidden = F.layer_norm(hidden, (hidden.shape[1],), self.ln_pre[0], self.ln_pre[1], self.cfg["layer_norm_eps"])
            # The backbone is frozen and has no adapters: no activations to keep for backward.
            with torch.no_grad():
                for lp in self.layers:
                    hidden = self._layer(lp, hidden, positions)
            width = hidden.shape[-1]
            pooled = hidden[1:].reshape(hh, m, ww, m, width).permute(0, 2, 1, 3, 4).reshape(hh * ww, m * m * width)
            outs.append(self.proj_out(F.gelu(self.proj_in(pooled))))
        return torch.cat(outs)


class Inkling2LoopForTraining(nn.Module):
    def __init__(self, cfg, vision):
        super().__init__()
        self.cfg = cfg
        self.visual = vision
        self.layers = nn.ModuleList()
        self.attn_q = []  # per layer >= 1: (Q, W) fp32 scaled
        self.mlp_q = []
        self.out_q = None
        self.loop_q = None  # (core, 4, W) fp32 scaled
        loop = cfg["loop_config"]
        self.core = range(loop["core_start"], loop["core_end"])
        self.num_passes = loop["num_passes"]
        self.block = cfg["n_layers_per_attn_res_block"]
        self.vocab = cfg["unpadded_vocab_size"]
        self.softcap = cfg["final_logit_softcapping"]

    # ------------------------------------------------------------------ text model
    def embed(self, input_ids, image_mask=None, image_embeds=None):
        safe = input_ids.clamp(min=0)
        x = rms_norm(F.embedding(safe, self.embed_w), self.embed_norm, self.cfg["rms_norm_eps"])
        if image_mask is not None and image_embeds is not None:
            x = x.masked_scatter(image_mask[:, None], image_embeds.to(x.dtype))
        return x

    def _visit(self, layer_idx, blocks, partial, bank, ctx: Ctx):
        L = self.layers[layer_idx]
        boundary = layer_idx % self.block == 0
        if not blocks:
            streams = [partial] if L.local else [partial] * 3
        else:
            streams = attn_res(blocks + [partial], None, self.attn_q[layer_idx - 1])
        core_i = layer_idx - self.core.start
        if bank is not None:
            msgs = attn_res(bank, None, self.loop_q[core_i, 0:3])
            mixed = L.controllers(streams * 3 if L.local else streams, msgs, (0, 1, 2))
            norms = (L.attn_norm,) * 3 if L.local else (L.q_input_norm, L.k_input_norm, L.v_input_norm)
            src = [rms_norm(m, n, out_dtype=BF16) for m, n in zip(mixed, norms)]
        elif L.local:
            s = rms_norm(streams[0], L.attn_norm, out_dtype=BF16)
            src = [s, s, s]
        else:
            src = [rms_norm(s, n, out_dtype=BF16) for s, n in zip(streams, (L.q_input_norm, L.k_input_norm, L.v_input_norm))]
        attn = L.attn(src[0], src[1], src[2], ctx)
        attn = short_conv_static(attn, L.attn_sconv, ctx.doc)
        if boundary:
            mlp_in = attn_res(blocks + [partial, attn], None, self.mlp_q[layer_idx][None])[0]
        else:
            mlp_in = attn_res(blocks + [partial], attn, self.mlp_q[layer_idx][None])[0]
        if bank is not None:
            msg = attn_res(bank, None, self.loop_q[core_i, 3:4])
            mlp_in = L.controllers([mlp_in], msg, (3,))[0]
        mlp_in = rms_norm(mlp_in, L.mlp_norm, out_dtype=BF16)
        weights, ids = L.router(mlp_in)
        mlp = L.experts(mlp_in, weights, ids, ctx.act_quant)
        mlp = short_conv_static(mlp, L.mlp_sconv, ctx.doc)
        if boundary:
            return blocks + [partial], attn + mlp
        return blocks, (partial.float() + attn.float() + mlp.float()).to(BF16)

    def _run_visit(self, layer_idx, blocks, partial, bank, ctx, grad_ckpt):
        if grad_ckpt and torch.is_grad_enabled():
            new_blocks, new_partial = checkpoint(
                self._visit, layer_idx, blocks, partial, bank, ctx, use_reentrant=False
            )
            return new_blocks, new_partial
        return self._visit(layer_idx, blocks, partial, bank, ctx)

    def hidden_states(self, x, ctx: Ctx, grad_ckpt=False):
        blocks, partial = [], x
        for i in range(self.core.start):
            blocks, partial = self._run_visit(i, blocks, partial, None, ctx, grad_ckpt)
        prelude_partial, prelude_blocks = partial, list(blocks)
        bank = None
        for pass_idx in range(self.num_passes):
            if pass_idx:
                bank = [*blocks[len(prelude_blocks) + 1 :], partial]
                partial, blocks = prelude_partial, list(prelude_blocks)
            for i in self.core:
                blocks, partial = self._run_visit(i, blocks, partial, bank, ctx, grad_ckpt)
        for i in range(self.core.stop, len(self.layers)):
            blocks, partial = self._run_visit(i, blocks, partial, None, ctx, grad_ckpt)
        out = attn_res(blocks + [partial], None, self.out_q[None])[0]
        return rms_norm(out, self.final_norm, out_dtype=BF16)

    def logits(self, hidden):
        """FP32 logits (unembed output fp32), softcapped, padded vocab masked."""
        if _MM_OUT_DTYPE and not (torch.is_grad_enabled() and hidden.requires_grad):
            logits = torch.mm(hidden, self.unembed.t(), out_dtype=F32)
        else:  # aten::mm(out_dtype) has no derivative; fp32 GEMM on the (few) loss rows
            logits = F.linear(hidden.float(), self.unembed.float())
        logits = self.softcap * torch.tanh(logits / self.softcap)
        logits[..., self.vocab :] = float("-inf")
        return logits

    def forward(self, input_ids, positions=None, doc=None, images=None, image_mask=None, *,
                attn_impl="flex", act_quant=True, grad_ckpt=False, logit_rows=None):
        """input_ids (T,); image_mask (T,) bool marks image rows (filled in order from images,
        a list of (patches_bthwc, grid_thw list)).  Returns fp32 logits for logit_rows (or all)."""
        t = input_ids.shape[0]
        dev = input_ids.device
        if positions is None:
            positions = torch.arange(t, device=dev)
        if doc is None:
            doc = torch.zeros(t, dtype=torch.int32, device=dev)
        image_embeds = None
        if images:
            image_embeds = torch.cat([self.visual(p, g, grad_ckpt=grad_ckpt) for p, g in images])
        x = self.embed(input_ids, image_mask if images else None, image_embeds)
        ctx = Ctx(positions=positions, doc=doc, attn_impl=attn_impl, act_quant=act_quant)
        if attn_impl == "flex":
            ctx.block_masks = flex_block_masks(doc, self.cfg["sliding_window_size"] - 1)
        hidden = self.hidden_states(x, ctx, grad_ckpt)
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
    """Dequantize expert by expert-chunk to bound the temporary memory."""
    out = None
    for s in range(0, values.shape[0], experts_per_chunk):
        part = fn(values[s : s + experts_per_chunk], scales[s : s + experts_per_chunk])
        if out is None:
            out = torch.empty((values.shape[0], *part.shape[1:]), dtype=part.dtype, device=part.device)
        out[s : s + part.shape[0]] = part
    return out


def load_model(model_dir, device="cuda", lora_rank=0, router_lora_rank=None, vision_lora_rank=None, verbose=True):
    """Build the trainable model from the HF checkpoint dir (NVFP4/MXFP8 experts dequantized
    to exact BF16 mantissas on the GPU, everything else as stored)."""
    global _MM_OUT_DTYPE
    from safetensors import safe_open
    import time

    t0 = time.time()
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

    vision = VisionEncoder(vcfg, w, vision_lora_rank)
    model = Inkling2LoopForTraining(cfg, vision)
    model.embed_w = w("model.llm.embed.weight")
    model.embed_norm = w("model.llm.embed_norm.weight")
    model.final_norm = w("model.llm.norm.weight")
    model.unembed = w("model.llm.unembed.weight")
    width = cfg["hidden_size"]
    scale = width**-0.5
    model.attn_q = [w(f"model.llm.attn_res.attn_queries.{i}").float().reshape(-1, width) * scale for i in range(cfg["num_hidden_layers"] - 1)]
    model.mlp_q = [w(f"model.llm.attn_res.mlp_queries.{i}").float() * scale for i in range(cfg["num_hidden_layers"])]
    model.out_q = w("model.llm.attn_res.output_query").float() * scale
    qs = w("model.llm.middle_cycle_cross_pass_queries")
    kvq = w("model.llm.middle_cycle_cross_pass_kv_queries")
    model.loop_q = torch.stack((qs[0::2], kvq[0::2], kvq[1::2], qs[1::2]), dim=1).float() * scale
    for i in range(cfg["num_hidden_layers"]):
        layer = DecoderLayer(cfg, i, w, lora_rank, router_lora_rank)
        p = f"model.llm.layers.{i}.mlp.experts."
        if f"{p}w13_weight_amax" in f.keys():
            w13 = _chunked(dequant_nvfp4_mantissa, w(p + "w13_weight"), w(p + "w13_weight_scale"))
            w2 = _chunked(dequant_nvfp4_mantissa, w(p + "w2_weight"), w(p + "w2_weight_scale"))
            o13 = w(p + "w13_weight_amax") / w(p + "w13_weight_outer_scale_denominator")
            o2 = w(p + "w2_weight_amax") / w(p + "w2_weight_outer_scale_denominator")
            # SGLang applies x_amax * w_amax / (1536 * denominator); x mantissas carry 1/1536.
            layer.experts = Experts("nvfp4", w13, w2, o13.float(), o2.float(), cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"])
        else:
            w13 = _chunked(dequant_mxfp8, w(p + "w13_weight"), w(p + "w13_weight_scale"))
            w2 = _chunked(dequant_mxfp8, w(p + "w2_weight"), w(p + "w2_weight_scale"))
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
        print(f"[inkling2_train] loaded in {time.time() - t0:.1f}s; unused tensors: {len(missing)} {missing[:4]}; "
              f"mm out_dtype fp32: {_MM_OUT_DTYPE}; mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    model.eval()
    return model
