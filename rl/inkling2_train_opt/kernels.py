"""Fused differentiable Triton kernels for Inkling-2-Loop (Blackwell SM120).

Eliminates unfused FP32 elementwise ops and tensor copies across all 48 layer visits:
  - fused_short_conv_static (fwd + bwd, zero saved activation)
  - fused_short_conv_dynamic (fwd + bwd, zero repeat_interleave)
  - fused_rms_norm (fwd + bwd)
  - fused_l2_clip (fwd + bwd)
  - fused_attn_res (fwd + bwd, zero torch.stack / float copies)
  - fused_rms_cat & fused_controller_mix_rmsnorm (fwd + bwd)
  - fused_residual_add3 (fwd + bwd)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BF16 = torch.bfloat16
F32 = torch.float32


# =============================================================================
# 1. Fused Short Conv Static (W=4, causal within doc, residual, scale 5**-0.5)
# =============================================================================


@triton.jit
def _short_conv_static_kernel(
    x_ptr,      # (T, D) bf16 (possibly strided along dim 0 with stride_xt)
    w_ptr,      # (4, D) fp32  row 0=lag3, row 1=lag2, row 2=lag1, row 3=lag0
    doc_ptr,    # (T,) int32
    y_ptr,      # (T, D) bf16 contiguous
    T,
    D,
    stride_xt,
    scale,
    DIR: tl.constexpr,      # -1 for forward (t - lag), +1 for backward (t + lag)
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_offs < T
    d_mask = d_offs < D

    w0 = tl.load(w_ptr + 0 * D + d_offs, mask=d_mask, other=0.0)
    w1 = tl.load(w_ptr + 1 * D + d_offs, mask=d_mask, other=0.0)
    w2 = tl.load(w_ptr + 2 * D + d_offs, mask=d_mask, other=0.0)
    w3 = tl.load(w_ptr + 3 * D + d_offs, mask=d_mask, other=0.0)

    doc0 = tl.load(doc_ptr + t_offs, mask=t_mask, other=-1)

    t1 = t_offs + DIR * 1
    t2 = t_offs + DIR * 2
    t3 = t_offs + DIR * 3
    m1 = t_mask & (t1 >= 0) & (t1 < T)
    m2 = t_mask & (t2 >= 0) & (t2 < T)
    m3 = t_mask & (t3 >= 0) & (t3 < T)

    doc1 = tl.load(doc_ptr + tl.where(m1, t1, 0), mask=m1, other=-2)
    doc2 = tl.load(doc_ptr + tl.where(m2, t2, 0), mask=m2, other=-2)
    doc3 = tl.load(doc_ptr + tl.where(m3, t3, 0), mask=m3, other=-2)
    m1 = m1 & (doc1 == doc0)
    m2 = m2 & (doc2 == doc0)
    m3 = m3 & (doc3 == doc0)

    D_i64 = D.to(tl.int64)
    sxt = stride_xt.to(tl.int64)
    x0 = tl.load(x_ptr + t_offs[:, None].to(tl.int64) * sxt + d_offs[None, :], mask=t_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + tl.where(m1, t1, 0)[:, None].to(tl.int64) * sxt + d_offs[None, :], mask=m1[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + tl.where(m2, t2, 0)[:, None].to(tl.int64) * sxt + d_offs[None, :], mask=m2[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + tl.where(m3, t3, 0)[:, None].to(tl.int64) * sxt + d_offs[None, :], mask=m3[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    acc = (((x3 * w0[None, :] + x2 * w1[None, :]) + x1 * w2[None, :]) + x0 * w3[None, :] + x0) * scale
    tl.store(y_ptr + t_offs[:, None].to(tl.int64) * D_i64 + d_offs[None, :], acc.to(tl.bfloat16), mask=t_mask[:, None] & d_mask[None, :])


class _ShortConvStaticFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_4d, doc):
        x_in = x if x.stride(-1) == 1 else x.contiguous()
        T, D = x_in.shape
        y = torch.empty((T, D), device=x_in.device, dtype=x_in.dtype)
        scale = 5.0**-0.5
        grid = (triton.cdiv(T, 32), triton.cdiv(D, 128))
        _short_conv_static_kernel[grid](x_in, w_4d, doc, y, T, D, x_in.stride(0), scale, DIR=-1, BLOCK_T=32, BLOCK_D=128, num_warps=4)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(w_4d, doc)
        return y

    @staticmethod
    def backward(ctx, dy):
        w_4d, doc = ctx.saved_tensors
        dy_c = dy.to(BF16).contiguous()
        T, D = dy_c.shape
        dx = torch.empty_like(dy_c)
        scale = 5.0**-0.5
        grid = (triton.cdiv(T, 32), triton.cdiv(D, 128))
        _short_conv_static_kernel[grid](dy_c, w_4d, doc, dx, T, D, D, scale, DIR=1, BLOCK_T=32, BLOCK_D=128, num_warps=4)
        return dx, None, None


def fused_short_conv_static(x: torch.Tensor, w_4d: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    return _ShortConvStaticFn.apply(x, w_4d, doc)


# =============================================================================
# 2. Fused Short Conv Dynamic (global layers, W=4, H=8, HD=128, scale 9**-0.5)
# =============================================================================


@triton.jit
def _short_conv_dyn_fwd_kernel(
    x_ptr,      # (T, D) bf16
    sw_ptr,     # (4, D) fp32  row lag = 0..3
    dyn_ptr,    # (T, H, 4) bf16
    doc_ptr,    # (T,) int32
    y_ptr,      # (T, D) bf16
    T,
    D,
    H: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = h * HD + tl.arange(0, HD)
    t_mask = t_offs < T

    sw0 = tl.load(sw_ptr + 0 * D + d_offs)
    sw1 = tl.load(sw_ptr + 1 * D + d_offs)
    sw2 = tl.load(sw_ptr + 2 * D + d_offs)
    sw3 = tl.load(sw_ptr + 3 * D + d_offs)

    dyn_base = (t_offs * H + h) * 4
    d0 = tl.load(dyn_ptr + dyn_base + 0, mask=t_mask, other=0.0).to(tl.float32)
    d1 = tl.load(dyn_ptr + dyn_base + 1, mask=t_mask, other=0.0).to(tl.float32)
    d2 = tl.load(dyn_ptr + dyn_base + 2, mask=t_mask, other=0.0).to(tl.float32)
    d3 = tl.load(dyn_ptr + dyn_base + 3, mask=t_mask, other=0.0).to(tl.float32)

    doc0 = tl.load(doc_ptr + t_offs, mask=t_mask, other=-1)
    t1, t2, t3 = t_offs - 1, t_offs - 2, t_offs - 3
    m1 = t_mask & (t1 >= 0)
    m2 = t_mask & (t2 >= 0)
    m3 = t_mask & (t3 >= 0)
    m1 = m1 & (tl.load(doc_ptr + tl.where(m1, t1, 0), mask=m1, other=-2) == doc0)
    m2 = m2 & (tl.load(doc_ptr + tl.where(m2, t2, 0), mask=m2, other=-2) == doc0)
    m3 = m3 & (tl.load(doc_ptr + tl.where(m3, t3, 0), mask=m3, other=-2) == doc0)

    D_i64 = D.to(tl.int64)
    x0 = tl.load(x_ptr + t_offs[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + tl.where(m1, t1, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=m1[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + tl.where(m2, t2, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=m2[:, None], other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + tl.where(m3, t3, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=m3[:, None], other=0.0).to(tl.float32)

    acc = (
        x0 * (sw0[None, :] + d0[:, None])
        + x1 * (sw1[None, :] + d1[:, None])
        + x2 * (sw2[None, :] + d2[:, None])
        + x3 * (sw3[None, :] + d3[:, None])
        + x0
    ) * (1.0 / 3.0)
    tl.store(y_ptr + t_offs[:, None].to(tl.int64) * D_i64 + d_offs[None, :], acc.to(tl.bfloat16), mask=t_mask[:, None])


@triton.jit
def _short_conv_dyn_bwd_kernel(
    dy_ptr,     # (T, D) bf16
    x_ptr,      # (T, D) bf16
    sw_ptr,     # (4, D) fp32
    dyn_ptr,    # (T, H, 4) bf16
    doc_ptr,    # (T,) int32
    dx_ptr,     # (T, D) bf16 out
    ddyn_ptr,   # (T, H, 4) bf16 out
    T,
    D,
    H: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = h * HD + tl.arange(0, HD)
    t_mask = t_offs < T
    D_i64 = D.to(tl.int64)
    scale = 1.0 / 3.0

    doc0 = tl.load(doc_ptr + t_offs, mask=t_mask, other=-1)
    dy0 = tl.load(dy_ptr + t_offs[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)

    # Part A: ddyn[t, h, lag] = scale * sum_d(dy[t, d] * x[t - lag, d] * same_doc)
    tm1, tm2, tm3 = t_offs - 1, t_offs - 2, t_offs - 3
    mm1 = t_mask & (tm1 >= 0)
    mm2 = t_mask & (tm2 >= 0)
    mm3 = t_mask & (tm3 >= 0)
    mm1 = mm1 & (tl.load(doc_ptr + tl.where(mm1, tm1, 0), mask=mm1, other=-2) == doc0)
    mm2 = mm2 & (tl.load(doc_ptr + tl.where(mm2, tm2, 0), mask=mm2, other=-2) == doc0)
    mm3 = mm3 & (tl.load(doc_ptr + tl.where(mm3, tm3, 0), mask=mm3, other=-2) == doc0)

    x0 = tl.load(x_ptr + t_offs[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    xm1 = tl.load(x_ptr + tl.where(mm1, tm1, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=mm1[:, None], other=0.0).to(tl.float32)
    xm2 = tl.load(x_ptr + tl.where(mm2, tm2, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=mm2[:, None], other=0.0).to(tl.float32)
    xm3 = tl.load(x_ptr + tl.where(mm3, tm3, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=mm3[:, None], other=0.0).to(tl.float32)

    dyn_base = (t_offs * H + h) * 4
    tl.store(ddyn_ptr + dyn_base + 0, (tl.sum(dy0 * x0, axis=1) * scale).to(tl.bfloat16), mask=t_mask)
    tl.store(ddyn_ptr + dyn_base + 1, (tl.sum(dy0 * xm1, axis=1) * scale).to(tl.bfloat16), mask=t_mask)
    tl.store(ddyn_ptr + dyn_base + 2, (tl.sum(dy0 * xm2, axis=1) * scale).to(tl.bfloat16), mask=t_mask)
    tl.store(ddyn_ptr + dyn_base + 3, (tl.sum(dy0 * xm3, axis=1) * scale).to(tl.bfloat16), mask=t_mask)

    # Part B: dx[t, d] = scale * (dy[t] + sum_lag dy[t+lag] * (sw[lag] + dyn[t+lag, h, lag]) * same_doc)
    sw0 = tl.load(sw_ptr + 0 * D + d_offs)
    sw1 = tl.load(sw_ptr + 1 * D + d_offs)
    sw2 = tl.load(sw_ptr + 2 * D + d_offs)
    sw3 = tl.load(sw_ptr + 3 * D + d_offs)

    tp1, tp2, tp3 = t_offs + 1, t_offs + 2, t_offs + 3
    mp1 = t_mask & (tp1 < T)
    mp2 = t_mask & (tp2 < T)
    mp3 = t_mask & (tp3 < T)
    mp1 = mp1 & (tl.load(doc_ptr + tl.where(mp1, tp1, 0), mask=mp1, other=-2) == doc0)
    mp2 = mp2 & (tl.load(doc_ptr + tl.where(mp2, tp2, 0), mask=mp2, other=-2) == doc0)
    mp3 = mp3 & (tl.load(doc_ptr + tl.where(mp3, tp3, 0), mask=mp3, other=-2) == doc0)

    dp0 = tl.load(dyn_ptr + dyn_base + 0, mask=t_mask, other=0.0).to(tl.float32)
    dp1 = tl.load(dyn_ptr + (tl.where(mp1, tp1, 0) * H + h) * 4 + 1, mask=mp1, other=0.0).to(tl.float32)
    dp2 = tl.load(dyn_ptr + (tl.where(mp2, tp2, 0) * H + h) * 4 + 2, mask=mp2, other=0.0).to(tl.float32)
    dp3 = tl.load(dyn_ptr + (tl.where(mp3, tp3, 0) * H + h) * 4 + 3, mask=mp3, other=0.0).to(tl.float32)

    dyp1 = tl.load(dy_ptr + tl.where(mp1, tp1, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=mp1[:, None], other=0.0).to(tl.float32)
    dyp2 = tl.load(dy_ptr + tl.where(mp2, tp2, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=mp2[:, None], other=0.0).to(tl.float32)
    dyp3 = tl.load(dy_ptr + tl.where(mp3, tp3, 0)[:, None].to(tl.int64) * D_i64 + d_offs[None, :], mask=mp3[:, None], other=0.0).to(tl.float32)

    dx = (
        dy0 * (1.0 + sw0[None, :] + dp0[:, None])
        + dyp1 * (sw1[None, :] + dp1[:, None])
        + dyp2 * (sw2[None, :] + dp2[:, None])
        + dyp3 * (sw3[None, :] + dp3[:, None])
    ) * scale
    tl.store(dx_ptr + t_offs[:, None].to(tl.int64) * D_i64 + d_offs[None, :], dx.to(tl.bfloat16), mask=t_mask[:, None])


class _ShortConvDynFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sw_4d, dyn, doc):
        x_c = x.contiguous()
        dyn_c = dyn.contiguous()
        T, D = x_c.shape
        H = dyn_c.shape[1]
        HD = D // H
        y = torch.empty_like(x_c)
        grid = (triton.cdiv(T, 32), H)
        _short_conv_dyn_fwd_kernel[grid](x_c, sw_4d, dyn_c, doc, y, T, D, H=H, HD=HD, BLOCK_T=32, num_warps=4)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(x_c, sw_4d, dyn_c, doc)
        return y

    @staticmethod
    def backward(ctx, dy):
        x_c, sw_4d, dyn_c, doc = ctx.saved_tensors
        dy_c = dy.to(BF16).contiguous()
        T, D = dy_c.shape
        H = dyn_c.shape[1]
        HD = D // H
        dx = torch.empty_like(x_c)
        ddyn = torch.empty_like(dyn_c)
        grid = (triton.cdiv(T, 32), H)
        _short_conv_dyn_bwd_kernel[grid](dy_c, x_c, sw_4d, dyn_c, doc, dx, ddyn, T, D, H=H, HD=HD, BLOCK_T=32, num_warps=4)
        return dx, None, ddyn, None


def fused_short_conv_dynamic(x: torch.Tensor, sw_4d: torch.Tensor, dyn: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    return _ShortConvDynFn.apply(x, sw_4d, dyn, doc)


# =============================================================================
# 3. Fused RMSNorm (frozen optional weight, bf16/fp32 in -> bf16/fp32 out)
# =============================================================================


@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    N, D, eps,
    HAS_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = row.to(tl.int64) * D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    tl.store(rstd_ptr + row, rstd)
    y = x * rstd
    if HAS_W:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(y_ptr + base + offs, y, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
    N, D,
    HAS_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = row.to(tl.int64) * D
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    g = dy
    if HAS_W:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = g * w
    dot = tl.sum(g * x, axis=0) * (rstd * rstd / D)
    dx = rstd * (g - x * dot)
    tl.store(dx_ptr + base + offs, dx, mask=mask)


@triton.jit
def _rms_norm_small_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    N, D: tl.constexpr, eps,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    rmask = rows < N
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    tl.store(rstd_ptr + rows, rstd, mask=rmask)
    y = x * rstd[:, None]
    if HAS_W:
        w = tl.load(w_ptr + cols).to(tl.float32)
        y = y * w[None, :]
    tl.store(y_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], y, mask=rmask[:, None])


@triton.jit
def _rms_norm_small_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
    N, D: tl.constexpr,
    HAS_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    rmask = rows < N
    dy = tl.load(dy_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    g = dy
    if HAS_W:
        w = tl.load(w_ptr + cols).to(tl.float32)
        g = g * w[None, :]
    dot = tl.sum(g * x, axis=1) * (rstd * rstd / D)
    dx = rstd[:, None] * (g - x * dot[:, None])
    tl.store(dx_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], dx, mask=rmask[:, None])


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype):
        shape = x.shape
        D = shape[-1]
        x2 = x.contiguous().view(-1, D)
        N = x2.shape[0]
        od = out_dtype if out_dtype is not None else x.dtype
        y = torch.empty((N, D), device=x.device, dtype=od)
        rstd = torch.empty(N, device=x.device, dtype=F32)
        has_w = weight is not None
        dummy = rstd
        if D <= 128 and (D & (D - 1)) == 0:
            _rms_norm_small_fwd_kernel[(triton.cdiv(N, 8),)](
                x2, weight if has_w else dummy, y, rstd, N, D=D, eps=eps, HAS_W=has_w, BLOCK_M=8, num_warps=4
            )
        else:
            BD = triton.next_power_of_2(D)
            _rms_norm_fwd_kernel[(N,)](
                x2, weight if has_w else dummy, y, rstd, N, D, eps, HAS_W=has_w, BLOCK_D=BD, num_warps=8
            )
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(x2, weight if has_w else torch.empty(0, device=x.device), rstd)
            ctx.has_w = has_w
            ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        N, D = x2.shape
        dy2 = dy.contiguous().view(N, D)
        dx = torch.empty_like(x2)
        if D <= 128 and (D & (D - 1)) == 0:
            _rms_norm_small_bwd_kernel[(triton.cdiv(N, 8),)](
                dy2, x2, w, rstd, dx, N, D=D, HAS_W=ctx.has_w, BLOCK_M=8, num_warps=4
            )
        else:
            BD = triton.next_power_of_2(D)
            _rms_norm_bwd_kernel[(N,)](
                dy2, x2, w, rstd, dx, N, D, HAS_W=ctx.has_w, BLOCK_D=BD, num_warps=8
            )
        return dx.view(ctx.shape), None, None, None


def fused_rms_norm(x: torch.Tensor, weight=None, eps: float = 1e-6, out_dtype=None) -> torch.Tensor:
    return _RMSNormFn.apply(x, weight, eps, out_dtype)


# =============================================================================
# 4. Fused L2 Clip (q: (T,24,128), k: (T,8,128), r: (T,24,16))
# =============================================================================


@triton.jit
def _l2_clip_fwd_kernel(
    x_ptr, scale_ptr, tau_ptr, y_ptr, norm_ptr,
    T, stride_xt, H: tl.constexpr, D: tl.constexpr, threshold, eps,
    HAS_SCALE: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N = T * H
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    rmask = rows < N
    t_idx = (rows // H).to(tl.int64)
    h_idx = rows % H

    xf = tl.load(x_ptr + t_idx[:, None] * stride_xt + h_idx[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    if HAS_SCALE:
        sc = tl.load(scale_ptr + h_idx[:, None] * D + cols[None, :], mask=rmask[:, None], other=1.0).to(tl.float32)
        xf = xf * sc
    norm = tl.sqrt(tl.sum(xf * xf, axis=1))
    tl.store(norm_ptr + rows, norm, mask=rmask)
    clip = tl.minimum(threshold / (norm + eps), 1.0)
    y = xf * clip[:, None]
    if HAS_TAU:
        tau = tl.load(tau_ptr + t_idx, mask=rmask, other=1.0)
        y = y.to(tl.bfloat16).to(tl.float32) * tau[:, None]
    tl.store(y_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], y.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _l2_clip_bwd_kernel(
    dy_ptr, x_ptr, scale_ptr, tau_ptr, norm_ptr, dx_ptr,
    T, stride_xt, H: tl.constexpr, D: tl.constexpr, threshold, eps,
    HAS_SCALE: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N = T * H
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    rmask = rows < N
    t_idx = (rows // H).to(tl.int64)
    h_idx = rows % H

    dy = tl.load(dy_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    xf = tl.load(x_ptr + t_idx[:, None] * stride_xt + h_idx[:, None] * D + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    if HAS_SCALE:
        sc = tl.load(scale_ptr + h_idx[:, None] * D + cols[None, :], mask=rmask[:, None], other=1.0).to(tl.float32)
        xf = xf * sc
    if HAS_TAU:
        tau = tl.load(tau_ptr + t_idx, mask=rmask, other=1.0)
        dy = dy * tau[:, None]
    norm = tl.load(norm_ptr + rows, mask=rmask, other=0.0)
    denom = norm + eps
    clipped = threshold < denom
    clip = tl.where(clipped, threshold / denom, 1.0)
    dot = tl.sum(dy * xf, axis=1)
    safe_norm = tl.maximum(norm, 1e-30)
    coeff = tl.where(clipped, threshold / (denom * denom * safe_norm), 0.0)
    dxf = clip[:, None] * dy - (coeff * dot)[:, None] * xf
    if HAS_SCALE:
        dxf = dxf * sc
    tl.store(dx_ptr + rows[:, None].to(tl.int64) * D + cols[None, :], dxf.to(tl.bfloat16), mask=rmask[:, None])


class _L2ClipFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, H, D, threshold, scale, tau, eps):
        # Accept either contiguous (T, H, D) or strided 2D (T, H*D) with stride(-1)==1 (zero copy!)
        T = x.shape[0]
        if x.ndim == 3:
            x_in = x if x.is_contiguous() else x.contiguous()
            stride_xt = H * D
            in_2d = False
        else:
            x_in = x if x.stride(-1) == 1 else x.contiguous()
            stride_xt = x_in.stride(0)
            in_2d = True
        y = torch.empty((T, H, D), device=x.device, dtype=BF16)
        norm = torch.empty((T, H), device=x.device, dtype=F32)
        has_scale = scale is not None
        has_tau = tau is not None
        dummy = norm
        BM = 8 if D >= 64 else 32
        _l2_clip_fwd_kernel[(triton.cdiv(T * H, BM),)](
            x_in, scale if has_scale else dummy, tau if has_tau else dummy,
            y, norm, T, stride_xt, H=H, D=D, threshold=float(threshold), eps=float(eps),
            HAS_SCALE=has_scale, HAS_TAU=has_tau, BLOCK_M=BM, num_warps=4,
        )
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(
                x_in,
                scale if has_scale else torch.empty(0, device=x.device),
                tau if has_tau else torch.empty(0, device=x.device),
                norm,
            )
            ctx.has_scale = has_scale
            ctx.has_tau = has_tau
            ctx.threshold = float(threshold)
            ctx.eps = float(eps)
            ctx.H = H
            ctx.D = D
            ctx.stride_xt = stride_xt
            ctx.in_2d = in_2d
        return y

    @staticmethod
    def backward(ctx, dy):
        x_in, scale, tau, norm = ctx.saved_tensors
        dy_c = dy.to(BF16).contiguous()
        T = x_in.shape[0]
        H, D = ctx.H, ctx.D
        dx = torch.empty((T, H, D), device=x_in.device, dtype=BF16)
        BM = 8 if D >= 64 else 32
        _l2_clip_bwd_kernel[(triton.cdiv(T * H, BM),)](
            dy_c, x_in, scale, tau, norm, dx,
            T, ctx.stride_xt, H=H, D=D, threshold=ctx.threshold, eps=ctx.eps,
            HAS_SCALE=ctx.has_scale, HAS_TAU=ctx.has_tau, BLOCK_M=BM, num_warps=4,
        )
        if ctx.in_2d:
            dx = dx.view(T, H * D)
        return dx, None, None, None, None, None, None


def fused_l2_clip(x: torch.Tensor, threshold: float, scale=None, tau=None, eps: float = 1e-6, heads: int | None = None, head_dim: int | None = None) -> torch.Tensor:
    if x.ndim == 3:
        H, D = x.shape[1], x.shape[2]
    else:
        assert heads is not None and head_dim is not None
        H, D = heads, head_dim
    return _L2ClipFn.apply(x, H, D, threshold, scale, tau, eps)


# =============================================================================
# 5. Fused AttnRes (N <= 6 bf16 sources, optional bf16 addend, Q in {1, 3})
# =============================================================================


@triton.jit
def _attn_res_fwd_kernel(
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    add_ptr,
    q_ptr,       # (Q, W) fp32
    out_ptr,     # (Q, T, W) fp32
    rstd_ptr,    # (N, T) fp32
    prob_ptr,    # (Q, N, T) fp32 (pre-rstd softmax prob)
    dot_ptr,     # (Q, N, T) fp32 (raw dot product x_n . q_k)
    T, W, eps,
    N: tl.constexpr,
    Q: tl.constexpr,
    HAS_ADD: tl.constexpr,
    NORM_VAL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_off = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)

    q0 = tl.load(q_ptr + 0 * W + offs, mask=mask, other=0.0)
    if Q >= 2:
        q1 = tl.load(q_ptr + 1 * W + offs, mask=mask, other=0.0)
    if Q >= 3:
        q2 = tl.load(q_ptr + 2 * W + offs, mask=mask, other=0.0)

    if HAS_ADD:
        add_v = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)

    NPAD: tl.constexpr = 8
    n_idx = tl.arange(0, NPAD)
    n_mask = n_idx < N
    rstds = tl.zeros((NPAD,), dtype=tl.float32)
    dots0 = tl.zeros((NPAD,), dtype=tl.float32)
    dots1 = tl.zeros((NPAD,), dtype=tl.float32)
    dots2 = tl.zeros((NPAD,), dtype=tl.float32)

    # Pass 1: compute rstd_n and dots in registers (zero global read-after-write)
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        rn = tl.rsqrt(tl.sum(xn * xn, axis=0) / W + eps)
        rstds = tl.where(n_idx == n, rn, rstds)
        d0 = tl.sum(xn * q0, axis=0)
        dots0 = tl.where(n_idx == n, d0, dots0)
        if Q >= 2:
            d1 = tl.sum(xn * q1, axis=0)
            dots1 = tl.where(n_idx == n, d1, dots1)
        if Q >= 3:
            d2 = tl.sum(xn * q2, axis=0)
            dots2 = tl.where(n_idx == n, d2, dots2)

    # Softmax in registers
    tl.store(rstd_ptr + n_idx * T_i64 + t_i64, rstds, mask=n_mask)
    tl.store(dot_ptr + (0 * N + n_idx) * T_i64 + t_i64, dots0, mask=n_mask)
    l0 = tl.where(n_mask, dots0 * rstds, -1e30)
    e0 = tl.where(n_mask, tl.exp(l0 - tl.max(l0, axis=0)), 0.0)
    p0 = e0 / tl.sum(e0, axis=0)
    tl.store(prob_ptr + (0 * N + n_idx) * T_i64 + t_i64, p0, mask=n_mask)
    w0_vec = p0 * (rstds if NORM_VAL else 1.0)

    if Q >= 2:
        tl.store(dot_ptr + (1 * N + n_idx) * T_i64 + t_i64, dots1, mask=n_mask)
        l1 = tl.where(n_mask, dots1 * rstds, -1e30)
        e1 = tl.where(n_mask, tl.exp(l1 - tl.max(l1, axis=0)), 0.0)
        p1 = e1 / tl.sum(e1, axis=0)
        tl.store(prob_ptr + (1 * N + n_idx) * T_i64 + t_i64, p1, mask=n_mask)
        w1_vec = p1 * (rstds if NORM_VAL else 1.0)

    if Q >= 3:
        tl.store(dot_ptr + (2 * N + n_idx) * T_i64 + t_i64, dots2, mask=n_mask)
        l2 = tl.where(n_mask, dots2 * rstds, -1e30)
        e2 = tl.where(n_mask, tl.exp(l2 - tl.max(l2, axis=0)), 0.0)
        p2 = e2 / tl.sum(e2, axis=0)
        tl.store(prob_ptr + (2 * N + n_idx) * T_i64 + t_i64, p2, mask=n_mask)
        w2_vec = p2 * (rstds if NORM_VAL else 1.0)

    # Pass 2: weighted sum into outputs (re-reading hot L1 sources, weights from registers)
    acc0 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if Q >= 2:
        acc1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if Q >= 3:
        acc2 = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        w0 = tl.sum(tl.where(n_idx == n, w0_vec, 0.0), axis=0)
        acc0 += w0 * xn
        if Q >= 2:
            w1 = tl.sum(tl.where(n_idx == n, w1_vec, 0.0), axis=0)
            acc1 += w1 * xn
        if Q >= 3:
            w2 = tl.sum(tl.where(n_idx == n, w2_vec, 0.0), axis=0)
            acc2 += w2 * xn

    tl.store(out_ptr + (0 * T_i64 + t_i64) * W + offs, acc0, mask=mask)
    if Q >= 2:
        tl.store(out_ptr + (1 * T_i64 + t_i64) * W + offs, acc1, mask=mask)
    if Q >= 3:
        tl.store(out_ptr + (2 * T_i64 + t_i64) * W + offs, acc2, mask=mask)


@triton.jit
def _attn_res_bwd_kernel(
    dy_ptr,      # (Q, T, W) fp32
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    add_ptr,
    q_ptr,       # (Q, W) fp32
    rstd_ptr,    # (N, T) fp32
    prob_ptr,    # (Q, N, T) fp32
    dot_ptr,     # (Q, N, T) fp32
    d0_ptr, d1_ptr, d2_ptr, d3_ptr, d4_ptr, d5_ptr,
    T, W,
    N: tl.constexpr,
    Q: tl.constexpr,
    HAS_ADD: tl.constexpr,
    NORM_VAL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_off = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)

    dy0 = tl.load(dy_ptr + (0 * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32)
    q0 = tl.load(q_ptr + 0 * W + offs, mask=mask, other=0.0)
    if Q >= 2:
        dy1 = tl.load(dy_ptr + (1 * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + 1 * W + offs, mask=mask, other=0.0)
    if Q >= 3:
        dy2 = tl.load(dy_ptr + (2 * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + 2 * W + offs, mask=mask, other=0.0)

    if HAS_ADD:
        add_v = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)

    NPAD: tl.constexpr = 8
    n_idx = tl.arange(0, NPAD)
    n_mask = n_idx < N
    rstds = tl.load(rstd_ptr + n_idx * T_i64 + t_i64, mask=n_mask, other=0.0)
    r_eff = rstds if NORM_VAL else tl.full((NPAD,), 1.0, dtype=tl.float32)

    dw0_vec = tl.zeros((NPAD,), dtype=tl.float32)
    dw1_vec = tl.zeros((NPAD,), dtype=tl.float32)
    dw2_vec = tl.zeros((NPAD,), dtype=tl.float32)

    # Pass 1: compute dw_{q, n} = dy_q . x_n in registers
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        dw0_vec = tl.where(n_idx == n, tl.sum(dy0 * xn, axis=0), dw0_vec)
        if Q >= 2:
            dw1_vec = tl.where(n_idx == n, tl.sum(dy1 * xn, axis=0), dw1_vec)
        if Q >= 3:
            dw2_vec = tl.where(n_idx == n, tl.sum(dy2 * xn, axis=0), dw2_vec)

    p0_vec = tl.load(prob_ptr + (0 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
    dot0_vec = tl.load(dot_ptr + (0 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
    s_pdp0 = tl.sum(p0_vec * (dw0_vec * r_eff), axis=0)
    dl0_vec = p0_vec * (dw0_vec * r_eff - s_pdp0)
    w0_vec = p0_vec * r_eff
    drstd_vec = dl0_vec * dot0_vec + (dw0_vec * p0_vec if NORM_VAL else 0.0)

    if Q >= 2:
        p1_vec = tl.load(prob_ptr + (1 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        dot1_vec = tl.load(dot_ptr + (1 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        s_pdp1 = tl.sum(p1_vec * (dw1_vec * r_eff), axis=0)
        dl1_vec = p1_vec * (dw1_vec * r_eff - s_pdp1)
        w1_vec = p1_vec * r_eff
        drstd_vec = drstd_vec + dl1_vec * dot1_vec + (dw1_vec * p1_vec if NORM_VAL else 0.0)

    if Q >= 3:
        p2_vec = tl.load(prob_ptr + (2 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        dot2_vec = tl.load(dot_ptr + (2 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        s_pdp2 = tl.sum(p2_vec * (dw2_vec * r_eff), axis=0)
        dl2_vec = p2_vec * (dw2_vec * r_eff - s_pdp2)
        w2_vec = p2_vec * r_eff
        drstd_vec = drstd_vec + dl2_vec * dot2_vec + (dw2_vec * p2_vec if NORM_VAL else 0.0)

    x_coeff_vec = drstd_vec * rstds * rstds * rstds * (1.0 / W)
    q0_coeff_vec = dl0_vec * rstds
    if Q >= 2:
        q1_coeff_vec = dl1_vec * rstds
    if Q >= 3:
        q2_coeff_vec = dl2_vec * rstds

    # Pass 2: compute dx_n for each n=0..N-1 from register vectors
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v

        w0 = tl.sum(tl.where(n_idx == n, w0_vec, 0.0), axis=0)
        qc0 = tl.sum(tl.where(n_idx == n, q0_coeff_vec, 0.0), axis=0)
        xc = tl.sum(tl.where(n_idx == n, x_coeff_vec, 0.0), axis=0)
        dx = w0 * dy0 + qc0 * q0 - xc * xn
        if Q >= 2:
            w1 = tl.sum(tl.where(n_idx == n, w1_vec, 0.0), axis=0)
            qc1 = tl.sum(tl.where(n_idx == n, q1_coeff_vec, 0.0), axis=0)
            dx = dx + w1 * dy1 + qc1 * q1
        if Q >= 3:
            w2 = tl.sum(tl.where(n_idx == n, w2_vec, 0.0), axis=0)
            qc2 = tl.sum(tl.where(n_idx == n, q2_coeff_vec, 0.0), axis=0)
            dx = dx + w2 * dy2 + qc2 * q2

        if n == 0:
            tl.store(d0_ptr + row_off + offs, dx, mask=mask)
        elif n == 1:
            tl.store(d1_ptr + row_off + offs, dx, mask=mask)
        elif n == 2:
            tl.store(d2_ptr + row_off + offs, dx, mask=mask)
        elif n == 3:
            tl.store(d3_ptr + row_off + offs, dx, mask=mask)
        elif n == 4:
            tl.store(d4_ptr + row_off + offs, dx, mask=mask)
        else:
            tl.store(d5_ptr + row_off + offs, dx, mask=mask)


class _AttnResFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries_scaled, normalize_values, eps, has_addend, addend, out_dtype, *sources):
        N = len(sources)
        assert 1 <= N <= 6, f"Expected 1..6 sources, got {N}"
        Q, W = queries_scaled.shape
        assert Q in (1, 2, 3), f"Expected Q in 1..3, got {Q}"
        srcs = [s.contiguous() for s in sources]
        T = srcs[0].shape[0]
        dev = srcs[0].device
        add_c = addend.contiguous() if has_addend else srcs[0]
        q_c = queries_scaled.contiguous()

        out = torch.empty((Q, T, W), device=dev, dtype=out_dtype)
        rstd = torch.empty((N, T), device=dev, dtype=F32)
        prob = torch.empty((Q, N, T), device=dev, dtype=F32)
        dot = torch.empty((Q, N, T), device=dev, dtype=F32)
        padded_s = list(srcs) + [srcs[0]] * (6 - N)
        BW = triton.next_power_of_2(W)
        _attn_res_fwd_kernel[(T,)](
            *padded_s, add_c, q_c, out, rstd, prob, dot,
            T, W, float(eps),
            N=N, Q=Q, HAS_ADD=has_addend, NORM_VAL=normalize_values, BLOCK_W=BW, num_warps=8,
        )
        ctx.save_for_backward(q_c, rstd, prob, dot, add_c if has_addend else torch.empty(0, device=dev), *srcs)
        ctx.N = N
        ctx.Q = Q
        ctx.has_addend = has_addend
        ctx.normalize_values = normalize_values
        ctx.add_dtype = addend.dtype if has_addend else None
        return out

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        q_c, rstd, prob, dot, add_c = saved[:5]
        srcs = saved[5:]
        N, Q = ctx.N, ctx.Q
        T, W = srcs[0].shape
        dy_c = dy.contiguous()

        d_srcs = [torch.empty_like(s) for s in srcs]
        padded_s = list(srcs) + [srcs[0]] * (6 - N)
        padded_d = list(d_srcs) + [d_srcs[0]] * (6 - N)
        BW = triton.next_power_of_2(W)
        _attn_res_bwd_kernel[(T,)](
            dy_c, *padded_s, add_c if ctx.has_addend else srcs[0], q_c,
            rstd, prob, dot, *padded_d,
            T, W, N=N, Q=Q, HAS_ADD=ctx.has_addend, NORM_VAL=ctx.normalize_values, BLOCK_W=BW, num_warps=8,
        )
        d_add = d_srcs[-1].to(ctx.add_dtype) if ctx.has_addend else None
        return (None, None, None, None, d_add, None, *d_srcs)


def fused_attn_res(sources, addend, queries_scaled, normalize_values: bool = True, eps: float = 1e-6, out_dtype=F32):
    """Drop-in fused Triton replacement for model.attn_res; returns list of Q (T, W) tensors."""
    has_add = addend is not None
    out = _AttnResFn.apply(queries_scaled, normalize_values, eps, has_add, addend, out_dtype, *sources)
    return list(out.unbind(0))


# =============================================================================
# 5b. Fused AttnRes + RMSNorm (zero FP32 intermediate, bf16 output directly)
# =============================================================================


@triton.jit
def _attn_res_rmsnorm_fwd_kernel(
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    add_ptr,
    q_ptr,        # (Q, W) fp32
    nw0_ptr, nw1_ptr, nw2_ptr,  # bf16 norm weights (W,)
    out_ptr,      # (Q, T, W) bf16
    rstd_ptr,     # (N, T) fp32
    prob_ptr,     # (Q, N, T) fp32
    dot_ptr,      # (Q, N, T) fp32
    out_rstd_ptr, # (Q, T) fp32
    T, W, eps,
    N: tl.constexpr,
    Q: tl.constexpr,
    HAS_ADD: tl.constexpr,
    NORM_VAL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_off = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)

    q0 = tl.load(q_ptr + 0 * W + offs, mask=mask, other=0.0)
    if Q >= 2:
        q1 = tl.load(q_ptr + 1 * W + offs, mask=mask, other=0.0)
    if Q >= 3:
        q2 = tl.load(q_ptr + 2 * W + offs, mask=mask, other=0.0)
    if HAS_ADD:
        add_v = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)

    NPAD: tl.constexpr = 8
    n_idx = tl.arange(0, NPAD)
    n_mask = n_idx < N
    rstds = tl.zeros((NPAD,), dtype=tl.float32)
    dots0 = tl.zeros((NPAD,), dtype=tl.float32)
    dots1 = tl.zeros((NPAD,), dtype=tl.float32)
    dots2 = tl.zeros((NPAD,), dtype=tl.float32)

    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        rn = tl.rsqrt(tl.sum(xn * xn, axis=0) / W + eps)
        rstds = tl.where(n_idx == n, rn, rstds)
        dots0 = tl.where(n_idx == n, tl.sum(xn * q0, axis=0), dots0)
        if Q >= 2:
            dots1 = tl.where(n_idx == n, tl.sum(xn * q1, axis=0), dots1)
        if Q >= 3:
            dots2 = tl.where(n_idx == n, tl.sum(xn * q2, axis=0), dots2)

    tl.store(rstd_ptr + n_idx * T_i64 + t_i64, rstds, mask=n_mask)
    tl.store(dot_ptr + (0 * N + n_idx) * T_i64 + t_i64, dots0, mask=n_mask)
    l0 = tl.where(n_mask, dots0 * rstds, -1e30)
    e0 = tl.where(n_mask, tl.exp(l0 - tl.max(l0, axis=0)), 0.0)
    p0 = e0 / tl.sum(e0, axis=0)
    tl.store(prob_ptr + (0 * N + n_idx) * T_i64 + t_i64, p0, mask=n_mask)
    w0_vec = p0 * (rstds if NORM_VAL else 1.0)

    if Q >= 2:
        tl.store(dot_ptr + (1 * N + n_idx) * T_i64 + t_i64, dots1, mask=n_mask)
        l1 = tl.where(n_mask, dots1 * rstds, -1e30)
        e1 = tl.where(n_mask, tl.exp(l1 - tl.max(l1, axis=0)), 0.0)
        p1 = e1 / tl.sum(e1, axis=0)
        tl.store(prob_ptr + (1 * N + n_idx) * T_i64 + t_i64, p1, mask=n_mask)
        w1_vec = p1 * (rstds if NORM_VAL else 1.0)
    if Q >= 3:
        tl.store(dot_ptr + (2 * N + n_idx) * T_i64 + t_i64, dots2, mask=n_mask)
        l2 = tl.where(n_mask, dots2 * rstds, -1e30)
        e2 = tl.where(n_mask, tl.exp(l2 - tl.max(l2, axis=0)), 0.0)
        p2 = e2 / tl.sum(e2, axis=0)
        tl.store(prob_ptr + (2 * N + n_idx) * T_i64 + t_i64, p2, mask=n_mask)
        w2_vec = p2 * (rstds if NORM_VAL else 1.0)

    acc0 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if Q >= 2:
        acc1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if Q >= 3:
        acc2 = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        acc0 += tl.sum(tl.where(n_idx == n, w0_vec, 0.0), axis=0) * xn
        if Q >= 2:
            acc1 += tl.sum(tl.where(n_idx == n, w1_vec, 0.0), axis=0) * xn
        if Q >= 3:
            acc2 += tl.sum(tl.where(n_idx == n, w2_vec, 0.0), axis=0) * xn

    # Fused RMSNorm -> bf16 out
    ro0 = tl.rsqrt(tl.sum(acc0 * acc0, axis=0) / W + eps)
    tl.store(out_rstd_ptr + 0 * T_i64 + t_i64, ro0)
    nw0 = tl.load(nw0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + (0 * T_i64 + t_i64) * W + offs, (acc0 * ro0 * nw0).to(tl.bfloat16), mask=mask)
    if Q >= 2:
        ro1 = tl.rsqrt(tl.sum(acc1 * acc1, axis=0) / W + eps)
        tl.store(out_rstd_ptr + 1 * T_i64 + t_i64, ro1)
        nw1 = tl.load(nw1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + (1 * T_i64 + t_i64) * W + offs, (acc1 * ro1 * nw1).to(tl.bfloat16), mask=mask)
    if Q >= 3:
        ro2 = tl.rsqrt(tl.sum(acc2 * acc2, axis=0) / W + eps)
        tl.store(out_rstd_ptr + 2 * T_i64 + t_i64, ro2)
        nw2 = tl.load(nw2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + (2 * T_i64 + t_i64) * W + offs, (acc2 * ro2 * nw2).to(tl.bfloat16), mask=mask)


@triton.jit
def _attn_res_rmsnorm_bwd_kernel(
    dy_ptr,       # (Q, T, W) bf16
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    add_ptr,
    q_ptr,        # (Q, W) fp32
    nw0_ptr, nw1_ptr, nw2_ptr,
    rstd_ptr,     # (N, T) fp32
    prob_ptr,     # (Q, N, T) fp32
    dot_ptr,      # (Q, N, T) fp32
    out_rstd_ptr, # (Q, T) fp32
    d0_ptr, d1_ptr, d2_ptr, d3_ptr, d4_ptr, d5_ptr,
    T, W,
    N: tl.constexpr,
    Q: tl.constexpr,
    HAS_ADD: tl.constexpr,
    NORM_VAL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_off = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)
    inv_W = 1.0 / W

    if HAS_ADD:
        add_v = tl.load(add_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)

    NPAD: tl.constexpr = 8
    n_idx = tl.arange(0, NPAD)
    n_mask = n_idx < N
    rstds = tl.load(rstd_ptr + n_idx * T_i64 + t_i64, mask=n_mask, other=0.0)
    r_eff = rstds if NORM_VAL else tl.full((NPAD,), 1.0, dtype=tl.float32)

    p0_vec = tl.load(prob_ptr + (0 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
    w0_vec = p0_vec * r_eff
    if Q >= 2:
        p1_vec = tl.load(prob_ptr + (1 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        w1_vec = p1_vec * r_eff
    if Q >= 3:
        p2_vec = tl.load(prob_ptr + (2 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        w2_vec = p2_vec * r_eff

    # Pass 1: reconstruct pre-norm y_q in registers (sources stay hot in L1)
    y0 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if Q >= 2:
        y1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if Q >= 3:
        y2 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        y0 += tl.sum(tl.where(n_idx == n, w0_vec, 0.0), axis=0) * xn
        if Q >= 2:
            y1 += tl.sum(tl.where(n_idx == n, w1_vec, 0.0), axis=0) * xn
        if Q >= 3:
            y2 += tl.sum(tl.where(n_idx == n, w2_vec, 0.0), axis=0) * xn

    # Apply RMSNorm backward to get dy_q (pre-norm gradient)
    ro0 = tl.load(out_rstd_ptr + 0 * T_i64 + t_i64)
    g0 = tl.load(dy_ptr + (0 * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32) * tl.load(nw0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy0 = ro0 * (g0 - y0 * (tl.sum(g0 * y0, axis=0) * (ro0 * ro0 * inv_W)))
    q0 = tl.load(q_ptr + 0 * W + offs, mask=mask, other=0.0)
    if Q >= 2:
        ro1 = tl.load(out_rstd_ptr + 1 * T_i64 + t_i64)
        g1 = tl.load(dy_ptr + (1 * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32) * tl.load(nw1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy1 = ro1 * (g1 - y1 * (tl.sum(g1 * y1, axis=0) * (ro1 * ro1 * inv_W)))
        q1 = tl.load(q_ptr + 1 * W + offs, mask=mask, other=0.0)
    if Q >= 3:
        ro2 = tl.load(out_rstd_ptr + 2 * T_i64 + t_i64)
        g2 = tl.load(dy_ptr + (2 * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32) * tl.load(nw2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy2 = ro2 * (g2 - y2 * (tl.sum(g2 * y2, axis=0) * (ro2 * ro2 * inv_W)))
        q2 = tl.load(q_ptr + 2 * W + offs, mask=mask, other=0.0)

    # Pass 2: compute dw_{q, n} = dy_q . x_n
    dw0_vec = tl.zeros((NPAD,), dtype=tl.float32)
    dw1_vec = tl.zeros((NPAD,), dtype=tl.float32)
    dw2_vec = tl.zeros((NPAD,), dtype=tl.float32)
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        dw0_vec = tl.where(n_idx == n, tl.sum(dy0 * xn, axis=0), dw0_vec)
        if Q >= 2:
            dw1_vec = tl.where(n_idx == n, tl.sum(dy1 * xn, axis=0), dw1_vec)
        if Q >= 3:
            dw2_vec = tl.where(n_idx == n, tl.sum(dy2 * xn, axis=0), dw2_vec)

    dot0_vec = tl.load(dot_ptr + (0 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
    s_pdp0 = tl.sum(p0_vec * (dw0_vec * r_eff), axis=0)
    dl0_vec = p0_vec * (dw0_vec * r_eff - s_pdp0)
    drstd_vec = dl0_vec * dot0_vec + (dw0_vec * p0_vec if NORM_VAL else 0.0)
    if Q >= 2:
        dot1_vec = tl.load(dot_ptr + (1 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        s_pdp1 = tl.sum(p1_vec * (dw1_vec * r_eff), axis=0)
        dl1_vec = p1_vec * (dw1_vec * r_eff - s_pdp1)
        drstd_vec = drstd_vec + dl1_vec * dot1_vec + (dw1_vec * p1_vec if NORM_VAL else 0.0)
    if Q >= 3:
        dot2_vec = tl.load(dot_ptr + (2 * N + n_idx) * T_i64 + t_i64, mask=n_mask, other=0.0)
        s_pdp2 = tl.sum(p2_vec * (dw2_vec * r_eff), axis=0)
        dl2_vec = p2_vec * (dw2_vec * r_eff - s_pdp2)
        drstd_vec = drstd_vec + dl2_vec * dot2_vec + (dw2_vec * p2_vec if NORM_VAL else 0.0)

    x_coeff_vec = drstd_vec * rstds * rstds * rstds * inv_W
    q0_coeff_vec = dl0_vec * rstds
    if Q >= 2:
        q1_coeff_vec = dl1_vec * rstds
    if Q >= 3:
        q2_coeff_vec = dl2_vec * rstds

    # Pass 3: write dx_n
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD and n == N - 1:
            xn = xn + add_v
        w0 = tl.sum(tl.where(n_idx == n, w0_vec, 0.0), axis=0)
        qc0 = tl.sum(tl.where(n_idx == n, q0_coeff_vec, 0.0), axis=0)
        xc = tl.sum(tl.where(n_idx == n, x_coeff_vec, 0.0), axis=0)
        dx = w0 * dy0 + qc0 * q0 - xc * xn
        if Q >= 2:
            dx = dx + tl.sum(tl.where(n_idx == n, w1_vec, 0.0), axis=0) * dy1 + tl.sum(tl.where(n_idx == n, q1_coeff_vec, 0.0), axis=0) * q1
        if Q >= 3:
            dx = dx + tl.sum(tl.where(n_idx == n, w2_vec, 0.0), axis=0) * dy2 + tl.sum(tl.where(n_idx == n, q2_coeff_vec, 0.0), axis=0) * q2
        if n == 0:
            tl.store(d0_ptr + row_off + offs, dx, mask=mask)
        elif n == 1:
            tl.store(d1_ptr + row_off + offs, dx, mask=mask)
        elif n == 2:
            tl.store(d2_ptr + row_off + offs, dx, mask=mask)
        elif n == 3:
            tl.store(d3_ptr + row_off + offs, dx, mask=mask)
        elif n == 4:
            tl.store(d4_ptr + row_off + offs, dx, mask=mask)
        else:
            tl.store(d5_ptr + row_off + offs, dx, mask=mask)


class _AttnResRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries_scaled, nw0, nw1, nw2, normalize_values, eps, has_addend, addend, *sources):
        N = len(sources)
        Q, W = queries_scaled.shape
        srcs = [s.contiguous() for s in sources]
        T = srcs[0].shape[0]
        dev = srcs[0].device
        add_c = addend.contiguous() if has_addend else srcs[0]
        q_c = queries_scaled.contiguous()

        out = torch.empty((Q, T, W), device=dev, dtype=BF16)
        rstd = torch.empty((N, T), device=dev, dtype=F32)
        prob = torch.empty((Q, N, T), device=dev, dtype=F32)
        dot = torch.empty((Q, N, T), device=dev, dtype=F32)
        out_rstd = torch.empty((Q, T), device=dev, dtype=F32)
        padded_s = list(srcs) + [srcs[0]] * (6 - N)
        BW = triton.next_power_of_2(W)
        _attn_res_rmsnorm_fwd_kernel[(T,)](
            *padded_s, add_c, q_c, nw0, nw1, nw2, out, rstd, prob, dot, out_rstd,
            T, W, float(eps),
            N=N, Q=Q, HAS_ADD=has_addend, NORM_VAL=normalize_values, BLOCK_W=BW, num_warps=8,
        )
        ctx.save_for_backward(q_c, nw0, nw1, nw2, rstd, prob, dot, out_rstd, add_c if has_addend else torch.empty(0, device=dev), *srcs)
        ctx.N = N
        ctx.Q = Q
        ctx.has_addend = has_addend
        ctx.normalize_values = normalize_values
        ctx.add_dtype = addend.dtype if has_addend else None
        return out

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        q_c, nw0, nw1, nw2, rstd, prob, dot, out_rstd, add_c = saved[:9]
        srcs = saved[9:]
        N, Q = ctx.N, ctx.Q
        T, W = srcs[0].shape
        dy_c = dy.to(BF16).contiguous()
        d_srcs = [torch.empty_like(s) for s in srcs]
        padded_s = list(srcs) + [srcs[0]] * (6 - N)
        padded_d = list(d_srcs) + [d_srcs[0]] * (6 - N)
        BW = triton.next_power_of_2(W)
        _attn_res_rmsnorm_bwd_kernel[(T,)](
            dy_c, *padded_s, add_c if ctx.has_addend else srcs[0], q_c, nw0, nw1, nw2,
            rstd, prob, dot, out_rstd, *padded_d,
            T, W, N=N, Q=Q, HAS_ADD=ctx.has_addend, NORM_VAL=ctx.normalize_values, BLOCK_W=BW, num_warps=8,
        )
        d_add = d_srcs[-1].to(ctx.add_dtype) if ctx.has_addend else None
        return (None, None, None, None, None, None, None, d_add, *d_srcs)


def fused_attn_res_rmsnorm(sources, addend, queries_scaled, norm_weights, normalize_values: bool = True, eps: float = 1e-6):
    """Fused AttnRes + RMSNorm -> list of Q (T, W) bf16 tensors (zero FP32 intermediate)."""
    has_add = addend is not None
    nw0 = norm_weights[0]
    nw1 = norm_weights[1] if len(norm_weights) > 1 else nw0
    nw2 = norm_weights[2] if len(norm_weights) > 2 else nw0
    out = _AttnResRMSNormFn.apply(queries_scaled, nw0, nw1, nw2, normalize_values, eps, has_add, addend, *sources)
    return list(out.unbind(0))


# =============================================================================
# 6. Fused Controllers Helpers: RMSNorm+Cat and Tanh-Mix
# =============================================================================


@triton.jit
def _rms_cat_fwd_kernel(
    a_ptr, b_ptr, out_ptr, rstd_a_ptr, rstd_b_ptr,
    T, W, eps,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    row_2w = t.to(tl.int64) * (2 * W)
    a = tl.load(a_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    ra = tl.rsqrt(tl.sum(a * a, axis=0) / W + eps)
    rb = tl.rsqrt(tl.sum(b * b, axis=0) / W + eps)
    tl.store(rstd_a_ptr + t, ra)
    tl.store(rstd_b_ptr + t, rb)
    tl.store(out_ptr + row_2w + offs, (a * ra).to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + row_2w + W + offs, (b * rb).to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_cat_bwd_kernel(
    dy_ptr, a_ptr, b_ptr, rstd_a_ptr, rstd_b_ptr, da_ptr, db_ptr,
    T, W,
    ACCUM: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    row_2w = t.to(tl.int64) * (2 * W)
    dya = tl.load(dy_ptr + row_2w + offs, mask=mask, other=0.0).to(tl.float32)
    dyb = tl.load(dy_ptr + row_2w + W + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    ra = tl.load(rstd_a_ptr + t)
    rb = tl.load(rstd_b_ptr + t)
    da = ra * (dya - a * (tl.sum(dya * a, axis=0) * (ra * ra / W)))
    db = rb * (dyb - b * (tl.sum(dyb * b, axis=0) * (rb * rb / W)))
    if ACCUM:
        da = da + tl.load(da_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        db = db + tl.load(db_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(da_ptr + row_w + offs, da, mask=mask)
    tl.store(db_ptr + row_w + offs, db, mask=mask)


class _RMSCatFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, eps=1e-6):
        a_c, b_c = a.contiguous(), b.contiguous()
        T, W = a_c.shape
        out = torch.empty((T, 2 * W), device=a.device, dtype=BF16)
        ra = torch.empty(T, device=a.device, dtype=F32)
        rb = torch.empty(T, device=a.device, dtype=F32)
        BW = triton.next_power_of_2(W)
        _rms_cat_fwd_kernel[(T,)](a_c, b_c, out, ra, rb, T, W, float(eps), BLOCK_W=BW, num_warps=8)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(a_c, b_c, ra, rb)
        return out

    @staticmethod
    def backward(ctx, dy):
        a_c, b_c, ra, rb = ctx.saved_tensors
        T, W = a_c.shape
        dy_c = dy.contiguous()
        da = torch.empty_like(a_c)
        db = torch.empty_like(b_c)
        BW = triton.next_power_of_2(W)
        _rms_cat_bwd_kernel[(T,)](dy_c, a_c, b_c, ra, rb, da, db, T, W, ACCUM=False, BLOCK_W=BW, num_warps=8)
        return da, db, None


def fused_rms_cat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _RMSCatFn.apply(a, b, eps)


@triton.jit
def _rms_cat3_fwd_kernel(
    d0_ptr, d1_ptr, d2_ptr,
    m0_ptr, m1_ptr, m2_ptr,
    out_ptr,      # (3, T, 2*W) bf16
    rd_ptr,       # (3, T) fp32 (or (1, T) when SHARED_DEST)
    rm_ptr,       # (3, T) fp32
    T, W, eps,
    SHARED_DEST: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    T_i64 = T.to(tl.int64)
    t_i64 = t.to(tl.int64)

    if SHARED_DEST:
        d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        rd = tl.rsqrt(tl.sum(d * d, axis=0) / W + eps)
        tl.store(rd_ptr + t_i64, rd)
        dn = (d * rd).to(tl.bfloat16)

    for k in tl.static_range(3):
        if not SHARED_DEST:
            if k == 0:
                d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            elif k == 1:
                d = tl.load(d1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                d = tl.load(d2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            rd = tl.rsqrt(tl.sum(d * d, axis=0) / W + eps)
            tl.store(rd_ptr + k * T_i64 + t_i64, rd)
            dn = (d * rd).to(tl.bfloat16)

        if k == 0:
            m = tl.load(m0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        elif k == 1:
            m = tl.load(m1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m = tl.load(m2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        rm = tl.rsqrt(tl.sum(m * m, axis=0) / W + eps)
        tl.store(rm_ptr + k * T_i64 + t_i64, rm)
        out_base = (k * T_i64 + t_i64) * (2 * W)
        tl.store(out_ptr + out_base + offs, dn, mask=mask)
        tl.store(out_ptr + out_base + W + offs, (m * rm).to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_cat3_bwd_kernel(
    dy_ptr,       # (3, T, 2*W) bf16
    d0_ptr, d1_ptr, d2_ptr,
    m0_ptr, m1_ptr, m2_ptr,
    rd_ptr, rm_ptr,
    dd0_ptr, dd1_ptr, dd2_ptr,
    dm0_ptr, dm1_ptr, dm2_ptr,
    T, W,
    SHARED_DEST: tl.constexpr,
    ACCUM: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    T_i64 = T.to(tl.int64)
    t_i64 = t.to(tl.int64)
    inv_W = 1.0 / W

    if SHARED_DEST:
        d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        rd = tl.load(rd_ptr + t_i64)
        dy_d_sum = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for k in tl.static_range(3):
        dy_base = (k * T_i64 + t_i64) * (2 * W)
        dyd = tl.load(dy_ptr + dy_base + offs, mask=mask, other=0.0).to(tl.float32)
        dym = tl.load(dy_ptr + dy_base + W + offs, mask=mask, other=0.0).to(tl.float32)
        if SHARED_DEST:
            dy_d_sum += dyd
        else:
            if k == 0:
                d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            elif k == 1:
                d = tl.load(d1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                d = tl.load(d2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            rd = tl.load(rd_ptr + k * T_i64 + t_i64)
            dd = rd * (dyd - d * (tl.sum(dyd * d, axis=0) * (rd * rd * inv_W)))
            if k == 0:
                if ACCUM:
                    dd = dd + tl.load(dd0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
                tl.store(dd0_ptr + row_w + offs, dd, mask=mask)
            elif k == 1:
                if ACCUM:
                    dd = dd + tl.load(dd1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
                tl.store(dd1_ptr + row_w + offs, dd, mask=mask)
            else:
                if ACCUM:
                    dd = dd + tl.load(dd2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
                tl.store(dd2_ptr + row_w + offs, dd, mask=mask)

        if k == 0:
            m = tl.load(m0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        elif k == 1:
            m = tl.load(m1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m = tl.load(m2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        rm = tl.load(rm_ptr + k * T_i64 + t_i64)
        dm = rm * (dym - m * (tl.sum(dym * m, axis=0) * (rm * rm * inv_W)))
        if k == 0:
            if ACCUM:
                dm = dm + tl.load(dm0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(dm0_ptr + row_w + offs, dm, mask=mask)
        elif k == 1:
            if ACCUM:
                dm = dm + tl.load(dm1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(dm1_ptr + row_w + offs, dm, mask=mask)
        else:
            if ACCUM:
                dm = dm + tl.load(dm2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(dm2_ptr + row_w + offs, dm, mask=mask)

    if SHARED_DEST:
        dd = rd * (dy_d_sum - d * (tl.sum(dy_d_sum * d, axis=0) * (rd * rd * inv_W)))
        if ACCUM:
            dd = dd + tl.load(dd0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dd0_ptr + row_w + offs, dd, mask=mask)


class _RMSCat3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d0, d1, d2, m0, m1, m2, shared_dest, eps=1e-6):
        d0_c = d0.contiguous()
        d1_c = d0_c if shared_dest else d1.contiguous()
        d2_c = d0_c if shared_dest else d2.contiguous()
        m0_c, m1_c, m2_c = m0.contiguous(), m1.contiguous(), m2.contiguous()
        T, W = d0_c.shape
        out = torch.empty((3, T, 2 * W), device=d0.device, dtype=BF16)
        rd = torch.empty((1 if shared_dest else 3, T), device=d0.device, dtype=F32)
        rm = torch.empty((3, T), device=d0.device, dtype=F32)
        BW = triton.next_power_of_2(W)
        _rms_cat3_fwd_kernel[(T,)](
            d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, out, rd, rm,
            T, W, float(eps), SHARED_DEST=shared_dest, BLOCK_W=BW, num_warps=8,
        )
        ctx.save_for_backward(d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, rd, rm)
        ctx.shared_dest = shared_dest
        return out

    @staticmethod
    def backward(ctx, dy):
        d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, rd, rm = ctx.saved_tensors
        T, W = d0_c.shape
        dy_c = dy.contiguous()
        dd0 = torch.empty_like(d0_c)
        dd1 = dd0 if ctx.shared_dest else torch.empty_like(d1_c)
        dd2 = dd0 if ctx.shared_dest else torch.empty_like(d2_c)
        dm0, dm1, dm2 = torch.empty_like(m0_c), torch.empty_like(m1_c), torch.empty_like(m2_c)
        BW = triton.next_power_of_2(W)
        _rms_cat3_bwd_kernel[(T,)](
            dy_c, d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, rd, rm,
            dd0, dd1, dd2, dm0, dm1, dm2,
            T, W, SHARED_DEST=ctx.shared_dest, ACCUM=False, BLOCK_W=BW, num_warps=8,
        )
        if ctx.shared_dest:
            return dd0, None, None, dm0, dm1, dm2, None, None
        return dd0, dd1, dd2, dm0, dm1, dm2, None, None


def fused_rms_cat3(dests, msgs, shared_dest: bool, eps: float = 1e-6) -> torch.Tensor:
    if shared_dest:
        return _RMSCat3Fn.apply(dests[0], dests[0], dests[0], msgs[0], msgs[1], msgs[2], True, eps)
    return _RMSCat3Fn.apply(dests[0], dests[1], dests[2], msgs[0], msgs[1], msgs[2], False, eps)


@triton.jit
def _controller_mix_rmsnorm3_fwd_kernel(
    d0_ptr, d1_ptr, d2_ptr,
    m0_ptr, m1_ptr, m2_ptr,
    gf_ptr,       # (3, T, 2*W) bf16
    nw0_ptr, nw1_ptr, nw2_ptr,
    out_ptr,      # (3, T, W) bf16
    rstd_ptr,     # (3, T) fp32
    T, W, eps,
    SHARED_DEST: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    T_i64 = T.to(tl.int64)
    t_i64 = t.to(tl.int64)

    if SHARED_DEST:
        d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)

    for k in tl.static_range(3):
        if not SHARED_DEST:
            if k == 0:
                d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            elif k == 1:
                d = tl.load(d1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                d = tl.load(d2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        if k == 0:
            m = tl.load(m0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            nw = tl.load(nw0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        elif k == 1:
            m = tl.load(m1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            nw = tl.load(nw1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m = tl.load(m2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            nw = tl.load(nw2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gf_base = (k * T_i64 + t_i64) * (2 * W)
        gain = tl.load(gf_ptr + gf_base + offs, mask=mask, other=0.0).to(tl.float32)
        forget = tl.load(gf_ptr + gf_base + W + offs, mask=mask, other=0.0).to(tl.float32)
        mixed = (1.0 - libdevice.tanh(forget)) * d + libdevice.tanh(gain) * m
        rstd = tl.rsqrt(tl.sum(mixed * mixed, axis=0) / W + eps)
        tl.store(rstd_ptr + k * T_i64 + t_i64, rstd)
        tl.store(out_ptr + (k * T_i64 + t_i64) * W + offs, (mixed * rstd * nw).to(tl.bfloat16), mask=mask)


@triton.jit
def _controller_mix_rmsnorm3_bwd_kernel(
    dy_ptr,       # (3, T, W) bf16
    d0_ptr, d1_ptr, d2_ptr,
    m0_ptr, m1_ptr, m2_ptr,
    gf_ptr,       # (3, T, 2*W) bf16
    nw0_ptr, nw1_ptr, nw2_ptr,
    rstd_ptr,     # (3, T) fp32
    dd0_ptr, dd1_ptr, dd2_ptr,
    dm0_ptr, dm1_ptr, dm2_ptr,
    dgf_ptr,      # (3, T, 2*W) bf16
    T, W,
    SHARED_DEST: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    T_i64 = T.to(tl.int64)
    t_i64 = t.to(tl.int64)
    inv_W = 1.0 / W

    if SHARED_DEST:
        d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        dd_acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for k in tl.static_range(3):
        if not SHARED_DEST:
            if k == 0:
                d = tl.load(d0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            elif k == 1:
                d = tl.load(d1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                d = tl.load(d2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        if k == 0:
            m = tl.load(m0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            nw = tl.load(nw0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        elif k == 1:
            m = tl.load(m1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            nw = tl.load(nw1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m = tl.load(m2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            nw = tl.load(nw2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + (k * T_i64 + t_i64) * W + offs, mask=mask, other=0.0).to(tl.float32)
        gf_base = (k * T_i64 + t_i64) * (2 * W)
        gain = tl.load(gf_ptr + gf_base + offs, mask=mask, other=0.0).to(tl.float32)
        forget = tl.load(gf_ptr + gf_base + W + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + k * T_i64 + t_i64)

        tg = libdevice.tanh(gain)
        tf = libdevice.tanh(forget)
        mixed = (1.0 - tf) * d + tg * m
        g = dy * nw
        dmixed = rstd * (g - mixed * (tl.sum(g * mixed, axis=0) * (rstd * rstd * inv_W)))

        dd = dmixed * (1.0 - tf)
        dm = dmixed * tg
        dgain = dmixed * m * (1.0 - tg * tg)
        dforget = -dmixed * d * (1.0 - tf * tf)

        if SHARED_DEST:
            dd_acc += dd
        else:
            if k == 0:
                tl.store(dd0_ptr + row_w + offs, dd, mask=mask)
            elif k == 1:
                tl.store(dd1_ptr + row_w + offs, dd, mask=mask)
            else:
                tl.store(dd2_ptr + row_w + offs, dd, mask=mask)

        if k == 0:
            tl.store(dm0_ptr + row_w + offs, dm, mask=mask)
        elif k == 1:
            tl.store(dm1_ptr + row_w + offs, dm, mask=mask)
        else:
            tl.store(dm2_ptr + row_w + offs, dm, mask=mask)
        tl.store(dgf_ptr + gf_base + offs, dgain.to(tl.bfloat16), mask=mask)
        tl.store(dgf_ptr + gf_base + W + offs, dforget.to(tl.bfloat16), mask=mask)

    if SHARED_DEST:
        tl.store(dd0_ptr + row_w + offs, dd_acc, mask=mask)


class _ControllerMixRMSNorm3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d0, d1, d2, m0, m1, m2, gf3, nw0, nw1, nw2, shared_dest, eps=1e-6):
        d0_c = d0.contiguous()
        d1_c = d0_c if shared_dest else d1.contiguous()
        d2_c = d0_c if shared_dest else d2.contiguous()
        m0_c, m1_c, m2_c = m0.contiguous(), m1.contiguous(), m2.contiguous()
        gf3_c = gf3.contiguous()
        T, W = d0_c.shape
        out = torch.empty((3, T, W), device=d0.device, dtype=BF16)
        rstd = torch.empty((3, T), device=d0.device, dtype=F32)
        BW = triton.next_power_of_2(W)
        _controller_mix_rmsnorm3_fwd_kernel[(T,)](
            d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, gf3_c, nw0, nw1, nw2, out, rstd,
            T, W, float(eps), SHARED_DEST=shared_dest, BLOCK_W=BW, num_warps=8,
        )
        ctx.save_for_backward(d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, gf3_c, nw0, nw1, nw2, rstd)
        ctx.shared_dest = shared_dest
        return out

    @staticmethod
    def backward(ctx, dy):
        d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, gf3_c, nw0, nw1, nw2, rstd = ctx.saved_tensors
        T, W = d0_c.shape
        dy_c = dy.to(BF16).contiguous()
        dd0 = torch.empty_like(d0_c)
        dd1 = dd0 if ctx.shared_dest else torch.empty_like(d1_c)
        dd2 = dd0 if ctx.shared_dest else torch.empty_like(d2_c)
        dm0, dm1, dm2 = torch.empty_like(m0_c), torch.empty_like(m1_c), torch.empty_like(m2_c)
        dgf = torch.empty_like(gf3_c)
        BW = triton.next_power_of_2(W)
        _controller_mix_rmsnorm3_bwd_kernel[(T,)](
            dy_c, d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, gf3_c, nw0, nw1, nw2, rstd,
            dd0, dd1, dd2, dm0, dm1, dm2, dgf,
            T, W, SHARED_DEST=ctx.shared_dest, BLOCK_W=BW, num_warps=8,
        )
        if ctx.shared_dest:
            return dd0, None, None, dm0, dm1, dm2, dgf, None, None, None, None, None
        return dd0, dd1, dd2, dm0, dm1, dm2, dgf, None, None, None, None, None


def fused_controller_mix_rmsnorm3(dests, msgs, gf3, norms, shared_dest: bool, eps: float = 1e-6):
    if shared_dest:
        out = _ControllerMixRMSNorm3Fn.apply(dests[0], dests[0], dests[0], msgs[0], msgs[1], msgs[2], gf3, norms[0], norms[1], norms[2], True, eps)
    else:
        out = _ControllerMixRMSNorm3Fn.apply(dests[0], dests[1], dests[2], msgs[0], msgs[1], msgs[2], gf3, norms[0], norms[1], norms[2], False, eps)
    return list(out.unbind(0))


@triton.jit
def _controller_mix_rmsnorm_fwd_kernel(
    dest_ptr, msg_ptr, gf_ptr, nw_ptr, out_ptr, rstd_ptr,
    T, W, eps,
    BLOCK_W: tl.constexpr,
):
    """Fused (1 - tanh(forget)) * dest + tanh(gain) * msg -> rms_norm(weight=nw, out=bf16)."""
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    row_2w = t.to(tl.int64) * (2 * W)
    dest = tl.load(dest_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    msg = tl.load(msg_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    gain = tl.load(gf_ptr + row_2w + offs, mask=mask, other=0.0).to(tl.float32)
    forget = tl.load(gf_ptr + row_2w + W + offs, mask=mask, other=0.0).to(tl.float32)
    mixed = (1.0 - libdevice.tanh(forget)) * dest + libdevice.tanh(gain) * msg
    rstd = tl.rsqrt(tl.sum(mixed * mixed, axis=0) / W + eps)
    tl.store(rstd_ptr + t, rstd)
    nw = tl.load(nw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_w + offs, (mixed * rstd * nw).to(tl.bfloat16), mask=mask)


@triton.jit
def _controller_mix_rmsnorm_bwd_kernel(
    dy_ptr, dest_ptr, msg_ptr, gf_ptr, nw_ptr, rstd_ptr,
    ddest_ptr, dmsg_ptr, dgf_ptr,
    T, W,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    row_2w = t.to(tl.int64) * (2 * W)
    dy = tl.load(dy_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    nw = tl.load(nw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dest = tl.load(dest_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    msg = tl.load(msg_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    gain = tl.load(gf_ptr + row_2w + offs, mask=mask, other=0.0).to(tl.float32)
    forget = tl.load(gf_ptr + row_2w + W + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + t)

    tg = libdevice.tanh(gain)
    tf = libdevice.tanh(forget)
    mixed = (1.0 - tf) * dest + tg * msg
    g = dy * nw
    dmixed = rstd * (g - mixed * (tl.sum(g * mixed, axis=0) * (rstd * rstd / W)))

    ddest = dmixed * (1.0 - tf)
    dmsg = dmixed * tg
    dgain = dmixed * msg * (1.0 - tg * tg)
    dforget = -dmixed * dest * (1.0 - tf * tf)

    tl.store(ddest_ptr + row_w + offs, ddest, mask=mask)
    tl.store(dmsg_ptr + row_w + offs, dmsg, mask=mask)
    tl.store(dgf_ptr + row_2w + offs, dgain.to(tl.bfloat16), mask=mask)
    tl.store(dgf_ptr + row_2w + W + offs, dforget.to(tl.bfloat16), mask=mask)


class _ControllerMixRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dest, msg, gf, nw, eps=1e-6):
        dest_c, msg_c, gf_c = dest.contiguous(), msg.contiguous(), gf.contiguous()
        T, W = dest_c.shape
        out = torch.empty((T, W), device=dest.device, dtype=BF16)
        rstd = torch.empty(T, device=dest.device, dtype=F32)
        BW = triton.next_power_of_2(W)
        _controller_mix_rmsnorm_fwd_kernel[(T,)](
            dest_c, msg_c, gf_c, nw, out, rstd, T, W, float(eps), BLOCK_W=BW, num_warps=8
        )
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(dest_c, msg_c, gf_c, nw, rstd)
        return out

    @staticmethod
    def backward(ctx, dy):
        dest_c, msg_c, gf_c, nw, rstd = ctx.saved_tensors
        T, W = dest_c.shape
        dy_c = dy.to(BF16).contiguous()
        ddest = torch.empty_like(dest_c)
        dmsg = torch.empty_like(msg_c)
        dgf = torch.empty_like(gf_c)
        BW = triton.next_power_of_2(W)
        _controller_mix_rmsnorm_bwd_kernel[(T,)](
            dy_c, dest_c, msg_c, gf_c, nw, rstd, ddest, dmsg, dgf, T, W, BLOCK_W=BW, num_warps=8
        )
        return ddest, dmsg, dgf, None, None


def fused_controller_mix_rmsnorm(dest, msg, gf, nw, eps: float = 1e-6):
    return _ControllerMixRMSNormFn.apply(dest, msg, gf, nw, eps)


# =============================================================================
# 7. Fused Residual Add3: (a.float() + b.float() + c.float()).to(bf16)
# =============================================================================


@triton.jit
def _add3_bf16_kernel(a_ptr, b_ptr, c_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (a + b + c).to(tl.bfloat16), mask=mask)


class _ResidualAdd3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c):
        a_c, b_c, c_c = a.contiguous(), b.contiguous(), c.contiguous()
        out = torch.empty_like(a_c, dtype=BF16)
        N = a_c.numel()
        _add3_bf16_kernel[(triton.cdiv(N, 1024),)](a_c, b_c, c_c, out, N, BLOCK=1024, num_warps=4)
        return out

    @staticmethod
    def backward(ctx, dy):
        return dy, dy, dy


def fused_residual_add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return _ResidualAdd3Fn.apply(a, b, c)


# =============================================================================
# 8. Fused Memory-Light Controllers (saves hidden (R=64) instead of normed+gf (4W))
# =============================================================================


class _FusedController3Fn(torch.autograd.Function):
    """Whole 3-row controller (rms_cat3 -> down -> gf -> mix_rmsnorm3) in one autograd op.

    Saves only hidden (3, T, 64) instead of normed (3, T, 2W) + gf (3, T, 2W) (144x smaller),
    and accumulates rms_cat3 backward directly into ddest/dmsg via ACCUM=True (zero BF16 adds).
    """

    @staticmethod
    def forward(ctx, d0, d1, d2, m0, m1, m2, down3, down_t3, gf3_w, gf_t3, nw0, nw1, nw2, shared_dest, eps=1e-6):
        d0_c = d0.contiguous()
        d1_c = d0_c if shared_dest else d1.contiguous()
        d2_c = d0_c if shared_dest else d2.contiguous()
        m0_c, m1_c, m2_c = m0.contiguous(), m1.contiguous(), m2.contiguous()
        T, W = d0_c.shape
        dev = d0_c.device
        BW = triton.next_power_of_2(W)

        normed = torch.empty((3, T, 2 * W), device=dev, dtype=BF16)
        rd = torch.empty((1 if shared_dest else 3, T), device=dev, dtype=F32)
        rm = torch.empty((3, T), device=dev, dtype=F32)
        _rms_cat3_fwd_kernel[(T,)](
            d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, normed, rd, rm,
            T, W, float(eps), SHARED_DEST=shared_dest, BLOCK_W=BW, num_warps=8,
        )
        hidden = torch.bmm(normed, down_t3)
        del normed
        gf = torch.bmm(hidden, gf_t3)

        out = torch.empty((3, T, W), device=dev, dtype=BF16)
        rstd = torch.empty((3, T), device=dev, dtype=F32)
        _controller_mix_rmsnorm3_fwd_kernel[(T,)](
            d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, gf, nw0, nw1, nw2, out, rstd,
            T, W, float(eps), SHARED_DEST=shared_dest, BLOCK_W=BW, num_warps=8,
        )
        del gf

        ctx.save_for_backward(d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, hidden, rd, rm, rstd, down3, gf3_w, gf_t3, nw0, nw1, nw2)
        ctx.shared_dest = shared_dest
        return out

    @staticmethod
    def backward(ctx, dy):
        d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, hidden, rd, rm, rstd, down3, gf3_w, gf_t3, nw0, nw1, nw2 = ctx.saved_tensors
        T, W = d0_c.shape
        BW = triton.next_power_of_2(W)
        dy_c = dy.to(BF16).contiguous()

        gf = torch.bmm(hidden, gf_t3)
        dd0 = torch.empty_like(d0_c)
        dd1 = dd0 if ctx.shared_dest else torch.empty_like(d1_c)
        dd2 = dd0 if ctx.shared_dest else torch.empty_like(d2_c)
        dm0, dm1, dm2 = torch.empty_like(m0_c), torch.empty_like(m1_c), torch.empty_like(m2_c)
        dgf = torch.empty_like(gf)
        _controller_mix_rmsnorm3_bwd_kernel[(T,)](
            dy_c, d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, gf, nw0, nw1, nw2, rstd,
            dd0, dd1, dd2, dm0, dm1, dm2, dgf,
            T, W, SHARED_DEST=ctx.shared_dest, BLOCK_W=BW, num_warps=8,
        )
        del gf
        dhidden = torch.bmm(dgf, gf3_w)
        del dgf
        dnormed = torch.bmm(dhidden, down3)
        del dhidden
        _rms_cat3_bwd_kernel[(T,)](
            dnormed, d0_c, d1_c, d2_c, m0_c, m1_c, m2_c, rd, rm,
            dd0, dd1, dd2, dm0, dm1, dm2,
            T, W, SHARED_DEST=ctx.shared_dest, ACCUM=True, BLOCK_W=BW, num_warps=8,
        )
        if ctx.shared_dest:
            return (dd0, None, None, dm0, dm1, dm2) + (None,) * 9
        return (dd0, dd1, dd2, dm0, dm1, dm2) + (None,) * 9


def fused_controller3(dests, msgs, down3, down_t3, gf3_w, gf_t3, norms, shared_dest: bool, eps: float = 1e-6):
    if shared_dest:
        out = _FusedController3Fn.apply(
            dests[0], dests[0], dests[0], msgs[0], msgs[1], msgs[2],
            down3, down_t3, gf3_w, gf_t3, norms[0], norms[1], norms[2], True, eps,
        )
    else:
        out = _FusedController3Fn.apply(
            dests[0], dests[1], dests[2], msgs[0], msgs[1], msgs[2],
            down3, down_t3, gf3_w, gf_t3, norms[0], norms[1], norms[2], False, eps,
        )
    return list(out.unbind(0))


class _FusedController1Fn(torch.autograd.Function):
    """Whole 1-row controller (rms_cat -> down -> gf -> mix_rmsnorm) in one autograd op."""

    @staticmethod
    def forward(ctx, dest, msg, down1, down_t1, gf1_w, gf_t1, nw, eps=1e-6):
        dest_c, msg_c = dest.contiguous(), msg.contiguous()
        T, W = dest_c.shape
        dev = dest_c.device
        BW = triton.next_power_of_2(W)

        normed = torch.empty((T, 2 * W), device=dev, dtype=BF16)
        ra = torch.empty(T, device=dev, dtype=F32)
        rb = torch.empty(T, device=dev, dtype=F32)
        _rms_cat_fwd_kernel[(T,)](dest_c, msg_c, normed, ra, rb, T, W, float(eps), BLOCK_W=BW, num_warps=8)
        hidden = torch.mm(normed, down_t1)
        del normed
        gf = torch.mm(hidden, gf_t1)

        out = torch.empty((T, W), device=dev, dtype=BF16)
        rstd = torch.empty(T, device=dev, dtype=F32)
        _controller_mix_rmsnorm_fwd_kernel[(T,)](
            dest_c, msg_c, gf, nw, out, rstd, T, W, float(eps), BLOCK_W=BW, num_warps=8,
        )
        del gf

        ctx.save_for_backward(dest_c, msg_c, hidden, ra, rb, rstd, down1, gf1_w, gf_t1, nw)
        return out

    @staticmethod
    def backward(ctx, dy):
        dest_c, msg_c, hidden, ra, rb, rstd, down1, gf1_w, gf_t1, nw = ctx.saved_tensors
        T, W = dest_c.shape
        BW = triton.next_power_of_2(W)
        dy_c = dy.to(BF16).contiguous()

        gf = torch.mm(hidden, gf_t1)
        ddest = torch.empty_like(dest_c)
        dmsg = torch.empty_like(msg_c)
        dgf = torch.empty_like(gf)
        _controller_mix_rmsnorm_bwd_kernel[(T,)](
            dy_c, dest_c, msg_c, gf, nw, rstd, ddest, dmsg, dgf, T, W, BLOCK_W=BW, num_warps=8,
        )
        del gf
        dhidden = torch.mm(dgf, gf1_w)
        del dgf
        dnormed = torch.mm(dhidden, down1)
        del dhidden
        _rms_cat_bwd_kernel[(T,)](
            dnormed, dest_c, msg_c, ra, rb, ddest, dmsg, T, W, ACCUM=True, BLOCK_W=BW, num_warps=8,
        )
        return ddest, dmsg, None, None, None, None, None, None


def fused_controller1(dest, msg, down1, down_t1, gf1_w, gf_t1, nw, eps: float = 1e-6):
    return _FusedController1Fn.apply(dest, msg, down1, down_t1, gf1_w, gf_t1, nw, eps)


# =============================================================================
# 9. Batched Bank AttnRes (all 48 cross-pass loop queries in ONE fwd + ONE bwd)
# =============================================================================


@triton.jit
def _bank_rstd_kernel(
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    rstd_ptr,  # (N, T) fp32
    T, W, eps,
    N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)
    for n in tl.static_range(N):
        if n == 0:
            xn = tl.load(s0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 1:
            xn = tl.load(s1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 2:
            xn = tl.load(s2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 3:
            xn = tl.load(s3_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        elif n == 4:
            xn = tl.load(s4_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            xn = tl.load(s5_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        rn = tl.rsqrt(tl.sum(xn * xn, axis=0) / W + eps)
        tl.store(rstd_ptr + n * T_i64 + t_i64, rn)


@triton.jit
def _bank_mix_fwd_kernel(
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    weff_ptr,   # (T, Q, N) fp32
    out_ptr,    # (Q, T, W) bf16
    T, W, Q,
    N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)
    Q_i64 = Q.to(tl.int64)

    x0 = tl.load(s0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    if N >= 2:
        x1 = tl.load(s1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    if N >= 3:
        x2 = tl.load(s2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    if N >= 4:
        x3 = tl.load(s3_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    if N >= 5:
        x4 = tl.load(s4_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    if N >= 6:
        x5 = tl.load(s5_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)

    w_base = t_i64 * Q_i64 * N
    for q in range(Q):
        qb = w_base + q.to(tl.int64) * N
        acc = tl.load(weff_ptr + qb + 0) * x0
        if N >= 2:
            acc = acc + tl.load(weff_ptr + qb + 1) * x1
        if N >= 3:
            acc = acc + tl.load(weff_ptr + qb + 2) * x2
        if N >= 4:
            acc = acc + tl.load(weff_ptr + qb + 3) * x3
        if N >= 5:
            acc = acc + tl.load(weff_ptr + qb + 4) * x4
        if N >= 6:
            acc = acc + tl.load(weff_ptr + qb + 5) * x5
        tl.store(out_ptr + (q.to(tl.int64) * T_i64 + t_i64) * W + offs, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _bank_attn_res_bwd_kernel(
    d_ptrs_ptr,  # (Q,) int64 pointers to (T, W) bf16 d_out_q
    s0_ptr, s1_ptr, s2_ptr, s3_ptr, s4_ptr, s5_ptr,
    rstd_ptr,    # (N, T) fp32
    prob_ptr,    # (T, Q, N) fp32
    dot_ptr,     # (T, Q, N) fp32
    d0_ptr, d1_ptr, d2_ptr, d3_ptr, d4_ptr, d5_ptr,
    q_coeff_ptr, # (N, T, Q) bf16
    T, W, Q,
    N: tl.constexpr,
    NORM_VAL: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    row_w = t.to(tl.int64) * W
    t_i64 = t.to(tl.int64)
    T_i64 = T.to(tl.int64)
    Q_i64 = Q.to(tl.int64)
    inv_W = 1.0 / W

    x0 = tl.load(s0_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(rstd_ptr + 0 * T_i64 + t_i64)
    re0 = r0 if NORM_VAL else 1.0
    d0 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    dr0 = 0.0
    if N >= 2:
        x1 = tl.load(s1_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(rstd_ptr + 1 * T_i64 + t_i64)
        re1 = r1 if NORM_VAL else 1.0
        d1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        dr1 = 0.0
    if N >= 3:
        x2 = tl.load(s2_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(rstd_ptr + 2 * T_i64 + t_i64)
        re2 = r2 if NORM_VAL else 1.0
        d2 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        dr2 = 0.0
    if N >= 4:
        x3 = tl.load(s3_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        r3 = tl.load(rstd_ptr + 3 * T_i64 + t_i64)
        re3 = r3 if NORM_VAL else 1.0
        d3 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        dr3 = 0.0
    if N >= 5:
        x4 = tl.load(s4_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        r4 = tl.load(rstd_ptr + 4 * T_i64 + t_i64)
        re4 = r4 if NORM_VAL else 1.0
        d4 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        dr4 = 0.0
    if N >= 6:
        x5 = tl.load(s5_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
        r5 = tl.load(rstd_ptr + 5 * T_i64 + t_i64)
        re5 = r5 if NORM_VAL else 1.0
        d5 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        dr5 = 0.0

    tq_base = t_i64 * Q_i64 * N
    for q in range(Q):
        p_i64 = tl.load(d_ptrs_ptr + q)
        if p_i64 != 0:
            dy_ptr = p_i64.to(tl.pointer_type(tl.bfloat16))
            dy = tl.load(dy_ptr + row_w + offs, mask=mask, other=0.0).to(tl.float32)
            qb = tq_base + q.to(tl.int64) * N
            q_i64 = q.to(tl.int64)

            p0 = tl.load(prob_ptr + qb + 0)
            dw0 = tl.sum(dy * x0, axis=0)
            d0 = d0 + (p0 * re0) * dy
            s_pdp = p0 * dw0 * re0
            if N >= 2:
                p1 = tl.load(prob_ptr + qb + 1)
                dw1 = tl.sum(dy * x1, axis=0)
                d1 = d1 + (p1 * re1) * dy
                s_pdp = s_pdp + p1 * dw1 * re1
            if N >= 3:
                p2 = tl.load(prob_ptr + qb + 2)
                dw2 = tl.sum(dy * x2, axis=0)
                d2 = d2 + (p2 * re2) * dy
                s_pdp = s_pdp + p2 * dw2 * re2
            if N >= 4:
                p3 = tl.load(prob_ptr + qb + 3)
                dw3 = tl.sum(dy * x3, axis=0)
                d3 = d3 + (p3 * re3) * dy
                s_pdp = s_pdp + p3 * dw3 * re3
            if N >= 5:
                p4 = tl.load(prob_ptr + qb + 4)
                dw4 = tl.sum(dy * x4, axis=0)
                d4 = d4 + (p4 * re4) * dy
                s_pdp = s_pdp + p4 * dw4 * re4
            if N >= 6:
                p5 = tl.load(prob_ptr + qb + 5)
                dw5 = tl.sum(dy * x5, axis=0)
                d5 = d5 + (p5 * re5) * dy
                s_pdp = s_pdp + p5 * dw5 * re5

            dl0 = p0 * (dw0 * re0 - s_pdp)
            dr0 = dr0 + dl0 * tl.load(dot_ptr + qb + 0) + (dw0 * p0 if NORM_VAL else 0.0)
            tl.store(q_coeff_ptr + (0 * T_i64 + t_i64) * Q_i64 + q_i64, (dl0 * r0).to(tl.bfloat16))
            if N >= 2:
                dl1 = p1 * (dw1 * re1 - s_pdp)
                dr1 = dr1 + dl1 * tl.load(dot_ptr + qb + 1) + (dw1 * p1 if NORM_VAL else 0.0)
                tl.store(q_coeff_ptr + (1 * T_i64 + t_i64) * Q_i64 + q_i64, (dl1 * r1).to(tl.bfloat16))
            if N >= 3:
                dl2 = p2 * (dw2 * re2 - s_pdp)
                dr2 = dr2 + dl2 * tl.load(dot_ptr + qb + 2) + (dw2 * p2 if NORM_VAL else 0.0)
                tl.store(q_coeff_ptr + (2 * T_i64 + t_i64) * Q_i64 + q_i64, (dl2 * r2).to(tl.bfloat16))
            if N >= 4:
                dl3 = p3 * (dw3 * re3 - s_pdp)
                dr3 = dr3 + dl3 * tl.load(dot_ptr + qb + 3) + (dw3 * p3 if NORM_VAL else 0.0)
                tl.store(q_coeff_ptr + (3 * T_i64 + t_i64) * Q_i64 + q_i64, (dl3 * r3).to(tl.bfloat16))
            if N >= 5:
                dl4 = p4 * (dw4 * re4 - s_pdp)
                dr4 = dr4 + dl4 * tl.load(dot_ptr + qb + 4) + (dw4 * p4 if NORM_VAL else 0.0)
                tl.store(q_coeff_ptr + (4 * T_i64 + t_i64) * Q_i64 + q_i64, (dl4 * r4).to(tl.bfloat16))
            if N >= 6:
                dl5 = p5 * (dw5 * re5 - s_pdp)
                dr5 = dr5 + dl5 * tl.load(dot_ptr + qb + 5) + (dw5 * p5 if NORM_VAL else 0.0)
                tl.store(q_coeff_ptr + (5 * T_i64 + t_i64) * Q_i64 + q_i64, (dl5 * r5).to(tl.bfloat16))
        else:
            q_i64 = q.to(tl.int64)
            tl.store(q_coeff_ptr + (0 * T_i64 + t_i64) * Q_i64 + q_i64, tl.full((), 0.0, dtype=tl.bfloat16))
            if N >= 2:
                tl.store(q_coeff_ptr + (1 * T_i64 + t_i64) * Q_i64 + q_i64, tl.full((), 0.0, dtype=tl.bfloat16))
            if N >= 3:
                tl.store(q_coeff_ptr + (2 * T_i64 + t_i64) * Q_i64 + q_i64, tl.full((), 0.0, dtype=tl.bfloat16))
            if N >= 4:
                tl.store(q_coeff_ptr + (3 * T_i64 + t_i64) * Q_i64 + q_i64, tl.full((), 0.0, dtype=tl.bfloat16))
            if N >= 5:
                tl.store(q_coeff_ptr + (4 * T_i64 + t_i64) * Q_i64 + q_i64, tl.full((), 0.0, dtype=tl.bfloat16))
            if N >= 6:
                tl.store(q_coeff_ptr + (5 * T_i64 + t_i64) * Q_i64 + q_i64, tl.full((), 0.0, dtype=tl.bfloat16))

    d0 = d0 - (dr0 * r0 * r0 * r0 * inv_W) * x0
    tl.store(d0_ptr + row_w + offs, d0.to(tl.bfloat16), mask=mask)
    if N >= 2:
        d1 = d1 - (dr1 * r1 * r1 * r1 * inv_W) * x1
        tl.store(d1_ptr + row_w + offs, d1.to(tl.bfloat16), mask=mask)
    if N >= 3:
        d2 = d2 - (dr2 * r2 * r2 * r2 * inv_W) * x2
        tl.store(d2_ptr + row_w + offs, d2.to(tl.bfloat16), mask=mask)
    if N >= 4:
        d3 = d3 - (dr3 * r3 * r3 * r3 * inv_W) * x3
        tl.store(d3_ptr + row_w + offs, d3.to(tl.bfloat16), mask=mask)
    if N >= 5:
        d4 = d4 - (dr4 * r4 * r4 * r4 * inv_W) * x4
        tl.store(d4_ptr + row_w + offs, d4.to(tl.bfloat16), mask=mask)
    if N >= 6:
        d5 = d5 - (dr5 * r5 * r5 * r5 * inv_W) * x5
        tl.store(d5_ptr + row_w + offs, d5.to(tl.bfloat16), mask=mask)


class _BankAttnResFn(torch.autograd.Function):
    """Precompute all Q=48 cross-pass bank queries in one fused op (eliminates 46 bank reads + 192 BF16 adds)."""

    @staticmethod
    def forward(ctx, loop_q_bf16, scale, normalize_values, eps, *sources):
        N = len(sources)
        srcs = [s.contiguous() for s in sources]
        T, W = srcs[0].shape
        Q = loop_q_bf16.shape[0]
        dev = srcs[0].device
        BW = triton.next_power_of_2(W)
        padded_s = list(srcs) + [srcs[0]] * (6 - N)

        rstd = torch.empty((N, T), device=dev, dtype=F32)
        _bank_rstd_kernel[(T,)](*padded_s, rstd, T, W, float(eps), N=N, BLOCK_W=BW, num_warps=8)

        # Exact BF16->FP32 Tensor Core dot products: dots[n] = (srcs[n] @ loop_q_bf16.T) * scale
        q_t = loop_q_bf16.t()
        dots = torch.empty((N, T, Q), device=dev, dtype=F32)
        for n in range(N):
            dots[n] = torch.mm(srcs[n], q_t, out_dtype=F32)
        dots.mul_(scale)

        logits = dots * rstd[:, :, None]
        prob = torch.softmax(logits, dim=0)
        weff = (prob * rstd[:, :, None]) if normalize_values else prob

        weff_tqn = weff.permute(1, 2, 0).contiguous()
        out = torch.empty((Q, T, W), device=dev, dtype=BF16)
        _bank_mix_fwd_kernel[(T,)](*padded_s, weff_tqn, out, T, W, Q, N=N, BLOCK_W=BW, num_warps=8)

        if any(ctx.needs_input_grad[4:]):
            prob_tqn = prob.permute(1, 2, 0).contiguous()
            dot_tqn = dots.permute(1, 2, 0).contiguous()
            ctx.save_for_backward(loop_q_bf16, rstd, prob_tqn, dot_tqn, *srcs)
            ctx.N = N
            ctx.Q = Q
            ctx.scale = scale
            ctx.normalize_values = normalize_values
        return tuple(out.unbind(0))

    @staticmethod
    def backward(ctx, *d_outs):
        saved = ctx.saved_tensors
        loop_q_bf16, rstd, prob_tqn, dot_tqn = saved[:4]
        srcs = saved[4:]
        N, Q = ctx.N, ctx.Q
        T, W = srcs[0].shape
        dev = srcs[0].device
        BW = triton.next_power_of_2(W)

        d_keep = []
        ptrs = []
        for d in d_outs:
            if d is None:
                ptrs.append(0)
            else:
                dc = d.to(BF16).contiguous()
                d_keep.append(dc)
                ptrs.append(dc.data_ptr())
        d_ptrs = torch.tensor(ptrs, device=dev, dtype=torch.int64)

        d_srcs = [torch.empty_like(s) for s in srcs]
        q_coeff = torch.empty((N, T, Q), device=dev, dtype=BF16)
        padded_s = list(srcs) + [srcs[0]] * (6 - N)
        padded_d = list(d_srcs) + [d_srcs[0]] * (6 - N)

        _bank_attn_res_bwd_kernel[(T,)](
            d_ptrs, *padded_s, rstd, prob_tqn, dot_tqn, *padded_d, q_coeff,
            T, W, Q, N=N, NORM_VAL=ctx.normalize_values, BLOCK_W=BW, num_warps=8,
        )
        # Keep d_keep alive through kernel launch (already ensured by local scope)
        del d_keep
        for n in range(N):
            d_srcs[n].addmm_(q_coeff[n], loop_q_bf16, beta=1.0, alpha=ctx.scale)
        return (None, None, None, None, *d_srcs)


def fused_attn_res_bank(bank, loop_q_flat_bf16, scale: float, normalize_values: bool = True, eps: float = 1e-6):
    """Compute all Q=48 cross-pass bank attn_res outputs -> tuple of Q (T, W) bf16 tensors."""
    return _BankAttnResFn.apply(loop_q_flat_bf16, float(scale), normalize_values, float(eps), *bank)
