"""Native SM120 Blackwell MoE (NVFP4 L0-19 / MXFP8 L20-23 forward + MXFP8 backward).

Forward uses SGLang's exact SM120 NVFP4 (CUTLASS) and MXFP8 (CuTe) routed kernels so
forward numerics match the served SGLang engine bit-for-bit on the experts.
Backward uses pre-transposed MXFP8 frozen expert weights (W2^T, W13^T) with a fused
Triton SwiGLU-backward + d_gamma + CuTe-MXFP8-quantize kernel and zero-atomic unpermute.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped_sm120
from sglang.kernels.ops.moe import inkling2_experts_sm120 as experts_sm120
from sglang.kernels.ops.moe.inkling2_moe import routed_partial, unweighted_post_reorder
from sglang.kernels.ops.moe.inkling2_route import ROUTE_MAX_IDS, route_metadata
from sglang.kernels.ops.moe.inkling_moe import _compute_expert_offsets_kernel, get_src2dst
from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales
from sglang.srt.layers.quantization.utils import swizzle_blockscale


def _nvfp4_matmul_pro6000(values, scales, weight, weight_scales, offsets, tokens):
    """RTX PRO 6000 Blackwell (188 SM) tuned tile selection for FlashInfer SM120 NVFP4 grouped GEMM."""
    experts, n, packed_width = weight.shape
    out = torch.empty((values.shape[0], n), dtype=torch.bfloat16, device=values.device)
    if values.shape[0]:
        if tokens <= 1536:
            swap_ab, tile_n, tile_k = True, 32, 256
        else:
            swap_ab, tile_n, tile_k = True, 128, 128
        group_gemm_nvfp4_nt_groupwise(
            values,
            weight,
            scales,
            weight_scales.view(torch.uint8).view(experts, n, packed_width // 8),
            offsets,
            tile_n=tile_n,
            tile_k=tile_k,
            swap_ab=swap_ab,
            out=out,
        )
    return out


def sm120_route(ids: torch.Tensor, weights: torch.Tensor, num_experts: int):
    """Fast SM120 routing metadata: (src2dst, dst2src, gammas_f32, offsets_i32)."""
    flat = ids.flatten()
    if flat.numel() <= ROUTE_MAX_IDS:
        src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(
            flat, weights.flatten(), 0, num_experts, num_experts
        )
        return src2dst, dst2src, gammas, offsets
    sorted_ids, dst2src = torch.sort(flat.to(torch.int16), stable=True)
    dst2src = dst2src.int()
    offsets = torch.empty(num_experts + 1, device=ids.device, dtype=torch.int32)
    _compute_expert_offsets_kernel[(num_experts,)](sorted_ids, offsets, sorted_ids.numel())
    src2dst = get_src2dst(dst2src)
    gammas = weights.flatten().float()[dst2src]
    return src2dst, dst2src, gammas, offsets


@triton.jit
def _mxfp8_quant_block8x32(vals_8x32):
    """Quantize an (8, 32) FP32 block to MXFP8 E4M3 values + 8 UE8M0 uint8 exponents."""
    amax = tl.max(tl.abs(vals_8x32), axis=1)
    raw = amax * (1.0 / 448.0)
    min_scale = tl.full((), 2.0**-127, tl.float32)
    bits = raw.to(tl.int32, bitcast=True)
    exp = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exp = tl.where(raw <= min_scale, 0, tl.minimum(exp, 254))
    exp = tl.where(raw != raw, 255, exp)
    scale = tl.maximum((exp << 23).to(tl.float32, bitcast=True), min_scale)
    scale = tl.where(exp == 255, float("nan"), scale)
    inv = tl.where(exp == 0, 2.0**127, libdevice.rcp_rn(scale))
    inv = tl.where(exp == 255, 0.0, inv)
    q = vals_8x32 * inv[:, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    return q.to(tl.float8e4nv), exp.to(tl.uint8)


@triton.jit
def _swiglu_bwd_mxfp8_quant_kernel(
    da_ptr,          # (R, I) bf16
    gu_ptr,          # (R, 2*I) bf16 raw w13 output (interleaved gate, up)
    gammas_ptr,      # (R,) fp32
    offsets_ptr,     # (E+1,) int32
    x_amax_ptr,      # (R,) fp32 (NVFP4 only)
    w13_scale_ptr,   # (E,) fp32 = w13_amax / (1536 * w13_denom) (NVFP4 only)
    d_gammas_ptr,    # (R,) fp32 out
    out_fp8_ptr,     # (R, 2*I) fp8_e4m3fn out
    out_exps_u8_ptr, # (2*I // 128, m_padded, 4) uint8 view of int32 CuTe exponents
    R,
    m_padded,
    gate_cap,
    up_cap,
    I: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    LOG2_E: tl.constexpr,
    IS_NVFP4: tl.constexpr,
    HAS_GATE_CAP: tl.constexpr,
    HAS_UP_CAP: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= R:
        return

    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(LOG2_E):
        mid = (lo + hi) >> 1
        off_mid = tl.load(offsets_ptr + mid)
        lo = tl.where(off_mid <= row, mid, lo)
        hi = tl.where(off_mid <= row, hi, mid)
    expert = lo
    off_e = tl.load(offsets_ptr + expert)
    scale_row = ((off_e + 3 * expert) // 4) * 4 + (row - off_e)

    gamma = tl.load(gammas_ptr + row)
    if IS_NVFP4:
        row_scale = tl.load(x_amax_ptr + row) * tl.load(w13_scale_ptr + expert)
    else:
        row_scale = 1.0

    inv_gcap = 1.0 / gate_cap if HAS_GATE_CAP else 0.0
    inv_ucap = 1.0 / up_cap if HAS_UP_CAP else 0.0

    d_gamma_acc = tl.zeros((), dtype=tl.float32)
    offs_128 = tl.arange(0, 128)
    offs_256 = tl.arange(0, 256)
    offs_4 = tl.arange(0, 4)

    N_CHUNKS: tl.constexpr = I // 128  # 1152 // 128 = 9 chunks of 128 pairs (= 256 cols = 8 groups of 32 = 2 int32 words)
    row_i64 = row.to(tl.int64)
    row_I = row_i64 * I
    row_2I = row_i64 * (2 * I)
    m_pad_i64 = m_padded.to(tl.int64)
    srow_i64 = scale_row.to(tl.int64)

    for c in range(N_CHUNKS):
        j = c * 128 + offs_128
        da = tl.load(da_ptr + row_I + j).to(tl.float32)
        gu = tl.load(gu_ptr + row_2I + c * 256 + offs_256).to(tl.float32)
        g_raw, u_raw = tl.split(tl.reshape(gu, (128, 2)))
        if IS_NVFP4:
            g = g_raw * row_scale
            u = u_raw * row_scale
        else:
            g = g_raw
            u = u_raw

        sig = tl.sigmoid(g)
        if HAS_GATE_CAP:
            tg = libdevice.tanh(g * inv_gcap)
            gc = gate_cap * tg
            dgc = 1.0 - tg * tg
        else:
            gc = g
            dgc = 1.0
        if HAS_UP_CAP:
            tu = libdevice.tanh(u * inv_ucap)
            uc = up_cap * tu
            duc = 1.0 - tu * tu
        else:
            uc = u
            duc = 1.0

        gsig = gc * sig
        swiglu_nogam = gsig * uc
        d_gamma_acc += tl.sum(da * swiglu_nogam, axis=0)

        g_eff = da * gamma
        dg = g_eff * uc * (dgc * sig + gsig * (1.0 - sig))
        du = g_eff * gsig * duc

        d_gu = tl.reshape(tl.join(dg, du), (8, 32))
        q_fp8, exp_u8 = _mxfp8_quant_block8x32(d_gu)
        tl.store(out_fp8_ptr + row_2I + c * 256 + offs_256, tl.reshape(q_fp8, (256,)))
        exp_lo, exp_hi = tl.split(tl.reshape(exp_u8, (2, 4)).trans())  # split into two (4,)
        c0 = 2 * c
        c1 = c0 + 1
        tl.store(out_exps_u8_ptr + (c0 * m_pad_i64 + srow_i64) * 4 + offs_4, exp_lo)
        tl.store(out_exps_u8_ptr + (c1 * m_pad_i64 + srow_i64) * 4 + offs_4, exp_hi)

    tl.store(d_gammas_ptr + row, d_gamma_acc)


def swiglu_bwd_mxfp8_quant(
    da: torch.Tensor,
    gate_up: torch.Tensor,
    gammas: torch.Tensor,
    offsets: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    x_amax: torch.Tensor | None = None,
    w13_scale_eff: torch.Tensor | None = None,
):
    """Fused SwiGLU backward + d_gammas reduction + CuTe MXFP8 quantization of d_gate_up."""
    R, I = da.shape
    E = offsets.numel() - 1
    m_padded = (R + 3 * E) // 4 * 4
    d_gammas = torch.empty(R, device=da.device, dtype=torch.float32)
    out_fp8 = torch.empty((R, 2 * I), device=da.device, dtype=torch.float8_e4m3fn)
    out_exps = torch.zeros(((2 * I) // 128, m_padded), device=da.device, dtype=torch.int32)
    if R > 0:
        is_nvfp4 = x_amax is not None
        dummy = gammas
        _swiglu_bwd_mxfp8_quant_kernel[(R,)](
            da,
            gate_up,
            gammas,
            offsets,
            x_amax if is_nvfp4 else dummy,
            w13_scale_eff if is_nvfp4 else dummy,
            d_gammas,
            out_fp8,
            out_exps.view(torch.uint8),
            R,
            m_padded,
            float(gate_cap),
            float(up_cap),
            I=I,
            NUM_EXPERTS=E,
            LOG2_E=(E - 1).bit_length(),
            IS_NVFP4=is_nvfp4,
            HAS_GATE_CAP=gate_cap > 0.0,
            HAS_UP_CAP=up_cap > 0.0,
            num_warps=4,
        )
    return d_gammas, out_fp8, out_exps.t()


@torch.no_grad()
def quantize_weight_to_mxfp8_cute(w_bf16: torch.Tensor, experts_per_chunk: int = 16):
    """Quantize (E, N, K) BF16 tensor along K (block 32) -> (fp8 (E, N, K), cute_scales)."""
    E, N, K = w_bf16.shape
    assert K % 32 == 0
    vals_out = torch.empty((E, N, K), dtype=torch.float8_e4m3fn, device=w_bf16.device)
    exps_out = torch.empty((E, N, K // 32), dtype=torch.uint8, device=w_bf16.device)
    min_scale = 2.0**-127
    for s in range(0, E, experts_per_chunk):
        e = min(E, s + experts_per_chunk)
        xf = w_bf16[s:e].float().reshape(e - s, N, K // 32, 32)
        amax = xf.abs().amax(-1)
        raw = amax * (1.0 / 448.0)
        bits = raw.view(torch.int32)
        exp = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
        exp = torch.where(raw <= min_scale, torch.zeros_like(exp), exp.clamp(max=254))
        scale = (exp << 23).view(torch.float32).clamp(min=min_scale)
        inv = torch.where(exp == 0, torch.full_like(scale, 2.0**127), 1.0 / scale)
        q = (xf * inv[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        vals_out[s:e] = q.reshape(e - s, N, K)
        exps_out[s:e] = exp.to(torch.uint8)
    return vals_out, mxfp8_weight_scales(exps_out)


class _SM120ExpertsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weights,
        ids,
        is_nvfp4,
        w13_fwd,
        w13_sf_fwd,
        w2_fwd,
        w2_sf_fwd,
        w13_amax,
        w13_denom,
        w2_amax,
        w2_denom,
        w13_scale_eff,
        w13_T_fp8,
        w13_T_sf,
        w2_T_fp8,
        w2_T_sf,
        gate_cap,
        up_cap,
        recompute_gate_up,
    ):
        tokens, top_k = ids.shape
        num_experts = w13_fwd.shape[0]
        x_c = x.contiguous()
        src2dst, dst2src, gammas, offsets = sm120_route(ids, weights, num_experts)

        if is_nvfp4:
            x_vals, x_sf, x_amax = experts_sm120.nvfp4_routed_quantize(
                x=x_c, dst2src=dst2src, top_k=top_k, offsets=offsets
            )
            gate_up = _nvfp4_matmul_pro6000(
                x_vals, x_sf, w13_fwd, w13_sf_fwd, offsets, tokens,
            )
            a_vals, a_sf, a_amax = experts_sm120.nvfp4_routed_swiglu_quantize(
                acc=gate_up,
                acc_amax=x_amax,
                w_amax=w13_amax,
                w_denominator=w13_denom,
                gammas=gammas,
                offsets=offsets,
                gate_cap=gate_cap,
                up_cap=up_cap,
            )
            down = _nvfp4_matmul_pro6000(
                a_vals, a_sf, w2_fwd, w2_sf_fwd, offsets, tokens,
            )
            nvfp4_scales = (a_amax, w2_amax, w2_denom)
        else:
            x_amax = None
            x_vals, x_sf = experts_sm120.mxfp8_routed_quantize(
                x=x_c, dst2src=dst2src, top_k=top_k, offsets=offsets
            )
            gate_up = grouped_sm120.mxfp8_expert_matmul(
                values=x_vals,
                scales=x_sf,
                weight=w13_fwd,
                weight_scales=w13_sf_fwd,
                offsets=offsets,
            )
            a_vals, a_sf = experts_sm120.mxfp8_routed_swiglu_quantize(
                acc=gate_up,
                gammas=gammas,
                offsets=offsets,
                gate_cap=gate_cap,
                up_cap=up_cap,
            )
            down = grouped_sm120.mxfp8_expert_matmul(
                values=a_vals,
                scales=a_sf,
                weight=w2_fwd,
                weight_scales=w2_sf_fwd,
                offsets=offsets,
            )
            nvfp4_scales = None

        out = routed_partial(down, offsets, src2dst, top_k, nvfp4_scales=nvfp4_scales).to(x.dtype)

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            empty = torch.empty(0, device=x.device)
            ctx.save_for_backward(
                x_c if recompute_gate_up else gate_up,
                gammas,
                offsets,
                src2dst,
                dst2src,
                x_amax if (is_nvfp4 and not recompute_gate_up) else empty,
                w13_scale_eff if is_nvfp4 else empty,
                w13_fwd if recompute_gate_up else empty,
                w13_sf_fwd if recompute_gate_up else empty,
                w13_T_fp8,
                w13_T_sf,
                w2_T_fp8,
                w2_T_sf,
            )
            ctx.is_nvfp4 = is_nvfp4
            ctx.recompute_gate_up = recompute_gate_up
            ctx.top_k = top_k
            ctx.tokens = tokens
            ctx.gate_cap = gate_cap
            ctx.up_cap = up_cap
            ctx.x_dtype = x.dtype
            ctx.w_dtype = weights.dtype

        return out

    @staticmethod
    def backward(ctx, dout):
        (
            saved_x_or_gu,
            gammas,
            offsets,
            src2dst,
            dst2src,
            x_amax,
            w13_scale_eff,
            w13_fwd,
            w13_sf_fwd,
            w13_T_fp8,
            w13_T_sf,
            w2_T_fp8,
            w2_T_sf,
        ) = ctx.saved_tensors
        top_k = ctx.top_k
        dout_bf16 = dout.to(torch.bfloat16).contiguous()

        if ctx.recompute_gate_up:
            if ctx.is_nvfp4:
                x_vals, x_sf, x_amax = experts_sm120.nvfp4_routed_quantize(
                    x=saved_x_or_gu, dst2src=dst2src, top_k=top_k, offsets=offsets
                )
                gate_up = _nvfp4_matmul_pro6000(
                    x_vals, x_sf, w13_fwd, w13_sf_fwd, offsets, ctx.tokens,
                )
            else:
                x_vals, x_sf = experts_sm120.mxfp8_routed_quantize(
                    x=saved_x_or_gu, dst2src=dst2src, top_k=top_k, offsets=offsets
                )
                gate_up = grouped_sm120.mxfp8_expert_matmul(
                    values=x_vals, scales=x_sf, weight=w13_fwd, weight_scales=w13_sf_fwd, offsets=offsets,
                )
        else:
            gate_up = saved_x_or_gu

        # Step 1: da = dout[tok] @ W2_true  (via MXFP8 routed quantize + CuTe MXFP8 grouped GEMM)
        do_vals, do_sf = experts_sm120.mxfp8_routed_quantize(
            x=dout_bf16, dst2src=dst2src, top_k=top_k, offsets=offsets
        )
        da = grouped_sm120.mxfp8_expert_matmul(
            values=do_vals,
            scales=do_sf,
            weight=w2_T_fp8,
            weight_scales=w2_T_sf,
            offsets=offsets,
        )

        # Step 2: Fused SwiGLU backward + d_gammas + CuTe MXFP8 quantization of d_gate_up
        d_gammas, dgu_fp8, dgu_sf = swiglu_bwd_mxfp8_quant(
            da=da,
            gate_up=gate_up,
            gammas=gammas,
            offsets=offsets,
            gate_cap=ctx.gate_cap,
            up_cap=ctx.up_cap,
            x_amax=x_amax if ctx.is_nvfp4 else None,
            w13_scale_eff=w13_scale_eff if ctx.is_nvfp4 else None,
        )

        # Step 3: dx_routed = d_gate_up @ W13_true (CuTe MXFP8 grouped GEMM)
        dx_routed = grouped_sm120.mxfp8_expert_matmul(
            values=dgu_fp8,
            scales=dgu_sf,
            weight=w13_T_fp8,
            weight_scales=w13_T_sf,
            offsets=offsets,
        )

        # Step 4: Unpermute + sum across top-k experts per token (writes BF16 directly, zero FP32 copy!)
        dx = unweighted_post_reorder(dx_routed, src2dst, top_k)
        if dx.dtype != ctx.x_dtype:
            dx = dx.to(ctx.x_dtype)
        d_weights = d_gammas[src2dst].view(ctx.tokens, top_k).to(ctx.w_dtype)

        return (dx, d_weights) + (None,) * 18


class SM120Experts(torch.nn.Module):
    """Frozen SM120 NVFP4 (L0-19) / MXFP8 (L20-23) routed experts with MXFP8 backward."""

    def __init__(
        self,
        fmt: str,
        w13_fwd: torch.Tensor,
        w13_sf_fwd: torch.Tensor,
        w2_fwd: torch.Tensor,
        w2_sf_fwd: torch.Tensor,
        w13_T_fp8: torch.Tensor,
        w13_T_sf: torch.Tensor,
        w2_T_fp8: torch.Tensor,
        w2_T_sf: torch.Tensor,
        w13_amax: torch.Tensor | None = None,
        w13_denom: torch.Tensor | None = None,
        w2_amax: torch.Tensor | None = None,
        w2_denom: torch.Tensor | None = None,
        gate_cap: float = 4.0,
        up_cap: float = 25.0,
    ):
        super().__init__()
        self.fmt = fmt
        self.is_nvfp4 = fmt == "nvfp4"
        self.register_buffer("w13_fwd", w13_fwd, persistent=False)
        self.register_buffer("w13_sf_fwd", w13_sf_fwd, persistent=False)
        self.register_buffer("w2_fwd", w2_fwd, persistent=False)
        self.register_buffer("w2_sf_fwd", w2_sf_fwd, persistent=False)
        self.register_buffer("w13_T_fp8", w13_T_fp8, persistent=False)
        self.register_buffer("w13_T_sf", w13_T_sf, persistent=False)
        self.register_buffer("w2_T_fp8", w2_T_fp8, persistent=False)
        self.register_buffer("w2_T_sf", w2_T_sf, persistent=False)
        if self.is_nvfp4:
            self.register_buffer("w13_amax", w13_amax.float().contiguous(), persistent=False)
            self.register_buffer("w13_denom", w13_denom.float().contiguous(), persistent=False)
            self.register_buffer("w2_amax", w2_amax.float().contiguous(), persistent=False)
            self.register_buffer("w2_denom", w2_denom.float().contiguous(), persistent=False)
            self.register_buffer(
                "w13_scale_eff",
                (w13_amax.float() / (1536.0 * w13_denom.float())).contiguous(),
                persistent=False,
            )
        else:
            self.w13_amax = self.w13_denom = self.w2_amax = self.w2_denom = self.w13_scale_eff = None
        self.gate_cap = gate_cap
        self.up_cap = up_cap

    def forward(self, x, weights, ids, act_quant=True, recompute_gate_up=False):
        return _SM120ExpertsFn.apply(
            x,
            weights,
            ids,
            self.is_nvfp4,
            self.w13_fwd,
            self.w13_sf_fwd,
            self.w2_fwd,
            self.w2_sf_fwd,
            self.w13_amax,
            self.w13_denom,
            self.w2_amax,
            self.w2_denom,
            self.w13_scale_eff,
            self.w13_T_fp8,
            self.w13_T_sf,
            self.w2_T_fp8,
            self.w2_T_sf,
            self.gate_cap,
            self.up_cap,
            recompute_gate_up,
        )
