"""Minimal LoRA for frozen BF16 linears, with fused forward/backward addmm and merge-back helpers."""

import math

import torch
from torch import nn
import torch.nn.functional as F

BF16 = torch.bfloat16
F32 = torch.float32


class _FusedLoRALinearFn(torch.autograd.Function):
    """y = x @ W.T + scale * (x @ A.T) @ B.T with fused addmm in both fwd and bwd (zero BF16 add)."""

    @staticmethod
    def forward(ctx, x, weight, lora_A, lora_B, scale):
        x_c = x.contiguous()
        a_bf = lora_A.to(x_c.dtype)
        b_bf = lora_B.to(x_c.dtype)
        y = torch.mm(x_c, weight.t())
        h = torch.mm(x_c, a_bf.t())
        y.addmm_(h, b_bf.t(), beta=1.0, alpha=scale)
        ctx.save_for_backward(x_c, weight, lora_A, lora_B, h)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, dy):
        x_c, weight, lora_A, lora_B, h = ctx.saved_tensors
        scale = ctx.scale
        dy_c = dy.to(x_c.dtype).contiguous()
        a_bf = lora_A.to(x_c.dtype)
        b_bf = lora_B.to(x_c.dtype)
        dh = torch.mm(dy_c, b_bf)
        if ctx.needs_input_grad[0]:
            dx = torch.mm(dy_c, weight)
            dx.addmm_(dh, a_bf, beta=1.0, alpha=scale)
        else:
            dx = None
        if lora_B.requires_grad:
            db = torch.mm(dy_c.t(), h, out_dtype=F32)
            if scale != 1.0:
                db.mul_(scale)
        else:
            db = None
        if lora_A.requires_grad:
            da = torch.mm(dh.t(), x_c, out_dtype=F32)
            if scale != 1.0:
                da.mul_(scale)
        else:
            da = None
        return dx, None, da, db, None


class LoRALinear(nn.Module):
    """y = x @ W.T + scale * (x @ A.T) @ B.T with W frozen (bf16) and A, B trainable."""

    def __init__(self, weight: torch.Tensor, rank: int = 0, alpha: float | None = None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.rank = rank
        if rank > 0:
            out_f, in_f = weight.shape
            self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32, device=weight.device))
            self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32, device=weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scale = (alpha if alpha is not None else rank) / rank
        else:
            self.lora_A = self.lora_B = None
            self.scale = 0.0

    def add_lora(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.lora_A is None:
            return base_out
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        h = F.linear(x, a)
        if base_out.ndim == 2 and h.ndim == 2:
            return torch.addmm(base_out, h, b.t(), beta=1.0, alpha=self.scale)
        return base_out + F.linear(h, b) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_A is not None and x.ndim == 2 and x.dtype == BF16:
            return _FusedLoRALinearFn.apply(x, self.weight, self.lora_A, self.lora_B, self.scale)
        y = F.linear(x, self.weight)
        return self.add_lora(x, y)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        if self.lora_A is None:
            return self.weight.data
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
        return (self.weight.data.float() + delta).to(self.weight.dtype)


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
