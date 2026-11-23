"""Minimal LoRA for frozen BF16 linears, plus merge-back helpers."""

import math

import torch
from torch import nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.lora_A is not None:
            a = self.lora_A.to(x.dtype)
            b = self.lora_B.to(x.dtype)
            y = y + F.linear(F.linear(x, a), b) * self.scale
        return y

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        if self.lora_A is None:
            return self.weight.data
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
        return (self.weight.data.float() + delta).to(self.weight.dtype)


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
