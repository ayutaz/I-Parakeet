"""Integer LayerNorm following I-BERT: integer mean, integer sqrt (Newton), integer reciprocal."""

import math

import torch

from iparakeet.quant.fixed_point import adaptive_multiplier, qmax


def isqrt(x: torch.Tensor, iterations: int = 64) -> torch.Tensor:
    """floor(sqrt(x)) for non-negative int64 tensors by Newton's method from above."""
    x = x.clamp(min=0)
    y = x.clone()
    for _ in range(iterations):
        nxt = torch.div(y + torch.div(x, y.clamp(min=1), rounding_mode="floor"), 2, rounding_mode="floor")
        y = torch.where(nxt < y, nxt, y)
    return y


class IntLayerNorm:
    """y = gamma * (x - mean) / std + beta on integer inputs of scale in_scale (eps is negligible)."""

    def __init__(self, gamma, beta, in_scale: float, out_scale: float, out_bits: int, factor_bits: int = 23) -> None:
        gamma = torch.as_tensor(gamma, dtype=torch.float64)
        beta = torch.as_tensor(beta, dtype=torch.float64)
        self.N = gamma.numel()
        self.factor_bits = factor_bits
        # normalized u = z * sqrt(N) / 2^F with z = y * floor(2^F / std_int)
        ratio = gamma * math.sqrt(self.N) / (2.0**factor_bits * out_scale)
        self.m, self.n = adaptive_multiplier(ratio)
        self.bias = torch.round(beta / out_scale * 2.0**self.n).to(torch.int64)
        self.out_bits = out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        mean = torch.div(q.sum(-1, keepdim=True) + self.N // 2, self.N, rounding_mode="floor")
        y = q - mean
        std = isqrt((y * y).sum(-1, keepdim=True)).clamp(min=1)
        z = y * torch.div(torch.full_like(std, 1 << self.factor_bits), std, rounding_mode="floor")
        out = (z * self.m + self.bias + (1 << (self.n - 1))) >> self.n
        return out.clamp(-qmax(self.out_bits), qmax(self.out_bits))
