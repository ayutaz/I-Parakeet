"""Integer softmax following I-BERT's i-exp: exp(x) = 2^-k * exp(r) with a second-order polynomial.

range_reduction:
  "ibert" - I-BERT as published: k = floor(q / floor(-ln2 / S)). Accurate when the input grid S
            is fine (I-BERT feeds INT32 scores); coarse for INT8 scores, where floor(-ln2/S) is off
            by up to one step of ~7 (S ~ 0.1).
  "log2"  - same polynomial and shift, but x/ln2 is computed with a fixed-point multiplier, so the
            range reduction stays exact for coarse INT8 score grids (default; the paper does not state
            the score bit width).
"""

import math

import torch

from iparakeet.quant.fixed_point import qmax

_A, _B, _C = 0.35815147, 0.96963238, 1.0  # exp(r) ~ A r^2 + B r + C on r in (-ln2, 0]
_LN2 = math.log(2.0)


class IntSoftmax:
    """Softmax over the last axis; output on [0, 2^(b-1)-1] with scale 1/(2^(b-1)-1)."""

    def __init__(self, in_scale: float, out_bits: int = 8, range_reduction: str = "log2", frac_bits: int = 16, sum_bits: int = 40) -> None:
        s = float(in_scale)
        self.mode = range_reduction
        self.sum_bits = sum_bits
        self.q_out = qmax(out_bits)
        if range_reduction == "ibert":
            self.const = 30
            self.x0 = min(-1, math.floor(-_LN2 / s))
            self.b_int = math.floor(_B / _A / s)
            self.c_int = math.floor(_C / _A / s**2)
            self.shift_e = max(0, (self.c_int << self.const).bit_length() - 16)
        elif range_reduction == "log2":
            self.F = frac_bits
            self.T = round(s / _LN2 * 2**frac_bits)  # q * T = x / ln2 in units of 2^-F
            g = 15  # polynomial of 2^-f (f in [0, 1)) with 2^-15 resolution
            self.a2 = round(_A * _LN2**2 * 2**g)
            self.a1 = round(-_B * _LN2 * 2**g)
            self.a0 = round(_C * 2**g)
        else:
            raise ValueError(f"unknown range reduction {range_reduction!r}")

    def _exp_ibert(self, q: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(q, min=self.const * self.x0)
        k = torch.div(q, self.x0, rounding_mode="floor")
        r = q - self.x0 * k
        z = ((r + self.b_int) * r + self.c_int).clamp(min=0)
        return torch.bitwise_left_shift(z, self.const - k) >> self.shift_e

    def _exp_log2(self, q: torch.Tensor) -> torch.Tensor:
        v = -(q * self.T)  # >= 0
        k = v >> self.F
        frac = v - (k << self.F)
        p = ((((self.a2 * frac) >> self.F) + self.a1) * frac >> self.F) + self.a0
        return p.clamp(min=0) >> k.clamp(max=62)

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        q = q - q.amax(dim=-1, keepdim=True)
        e = self._exp_ibert(q) if self.mode == "ibert" else self._exp_log2(q)
        total = e.sum(-1, keepdim=True).clamp(min=1)
        factor = torch.div(torch.full_like(total, 1 << self.sum_bits), total, rounding_mode="floor")
        return (e * factor * self.q_out + (1 << (self.sum_bits - 1))) >> self.sum_bits
