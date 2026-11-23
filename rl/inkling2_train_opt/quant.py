"""NVFP4 / MXFP8 weight dequantization and activation fake-quantization for Inkling-2.

Matches SGLang's inkling2 numerics (sglang/kernels/ops/moe/*, test_inkling2_moe_kernels.py):
  * NVFP4 weights: E2M1 codes (low nibble first) x 16-wide FP8-E4M3 block scales, with a
    per-expert outer scale amax / outer_scale_denominator applied to the GEMM output row.
  * NVFP4 activations: per-token amax, outer = amax / 1536, 16-wide E4M3 block scales chosen
    per block between max/6 and max/4 by lower absolute error ("fouroversix_mae").
  * MXFP8 weights and activations: E4M3 values x 32-wide UE8M0 power-of-two block scales.

Every "mantissa" tensor returned here (grid value x block scale) is exact in bfloat16, so a
BF16 GEMM with FP32 accumulation reproduces the block-scaled tensor-core product up to
summation order; outer scales are applied to the output rows in FP32 like the server does.
"""

import torch

_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_nvfp4_mantissa(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., K/2], scales e4m3 [..., K/16] -> bf16 [..., K] (exact)."""
    grid = torch.tensor(_FP4_GRID, device=packed.device, dtype=torch.float32)
    lo, hi = packed & 15, packed >> 4
    codes = torch.stack((lo, hi), -1).flatten(-2).long()
    values = grid[codes & 7]
    values = torch.where(codes >= 8, -values, values)
    out = values.view(*values.shape[:-1], -1, 16) * scales.float().unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """values e4m3 [..., K], scales uint8 (UE8M0) [..., K/32] -> bf16 [..., K] (exact)."""
    exps = torch.exp2(scales.float() - 127.0)
    out = values.float().view(*values.shape[:-1], -1, 32) * exps.unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


@torch.no_grad()
def nvfp4_quantize_mantissa(x: torch.Tensor):
    """Per-row NVFP4 fake quantization (SGLang's _quantize_reference).

    Returns (mantissa bf16 [..., K], outer fp32 [...]) with x ~= mantissa * outer[..., None].
    """
    x = x.float()
    amax = x.abs().amax(-1)
    encoder = (1536.0 / amax).clamp(max=torch.finfo(torch.float32).max)
    decoder = encoder.reciprocal()
    outer = amax * (1.0 / 1536.0)
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    maximum = blocks.abs().amax(-1)
    grid = x.new_tensor(_FP4_GRID)
    boundaries = (grid[:-1] + grid[1:]) / 2
    best_vals = best_err = None
    for inverse_max in (1.0 / 6.0, 0.25):
        scale = (maximum * (encoder[..., None] * inverse_max)).clamp(max=448)
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = (scale * decoder[..., None]).reciprocal().clamp(max=torch.finfo(torch.float32).max)
        values = blocks * inverse[..., None]
        magnitude = values.abs()
        code = torch.bucketize(magnitude, boundaries)
        ties = magnitude == boundaries[code.clamp(max=6)]
        code = code + (ties & (code % 2 == 1)).long()
        rounded = grid[code] * torch.where(values.signbit(), -1.0, 1.0)
        mant = rounded * scale[..., None]
        err = ((rounded * scale[..., None]) * outer[..., None, None] - blocks).abs()
        for _ in range(4):  # the reference's pairwise reduction order
            err = err[..., ::2] + err[..., 1::2]
        err = err.squeeze(-1)
        if best_vals is None:
            best_vals, best_err = mant, err
        else:
            choose = err < best_err
            best_vals = torch.where(choose[..., None], mant, best_vals)
    return best_vals.reshape(x.shape).to(torch.bfloat16), outer


@torch.no_grad()
def mxfp8_quantize_dequant(x: torch.Tensor) -> torch.Tensor:
    """Per-32-block MXFP8 fake quantization (SGLang's _mxfp8_block); returns bf16 (exact)."""
    xf = x.float()
    groups = xf.reshape(*xf.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1)
    raw = amax * (1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    min_scale = 2.0 ** -127
    exponent = torch.where(raw <= min_scale, torch.zeros_like(exponent), exponent.clamp(max=254))
    scale = (exponent << 23).view(torch.float32).clamp(min=min_scale)
    inverse = torch.where(exponent == 0, torch.full_like(scale, 2.0 ** 127), 1.0 / scale)
    values = (groups * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (values * scale[..., None]).reshape(x.shape).to(torch.bfloat16)


class _STE(torch.autograd.Function):
    """Forward: quantized value. Backward: identity (straight-through) times d(out)/d(x)."""

    @staticmethod
    def forward(ctx, x, q):
        return q

    @staticmethod
    def backward(ctx, g):
        return g, None


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return _STE.apply(x, q) if x.requires_grad else q
