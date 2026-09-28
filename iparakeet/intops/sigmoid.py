"""Integer sigmoid for the GLU gate; output on the unsigned grid [0, 2^(b-1)-1] with scale 1/(2^(b-1)-1)."""

import torch

from iparakeet.intops.swish import TanhPolySigmoid
from iparakeet.quant.fixed_point import adaptive_multiplier, qmax, requantize


class IntSigmoidLUT:
    def __init__(self, in_scale: float, in_bits: int, out_bits: int = 8) -> None:
        self.offset = qmax(in_bits)
        x = torch.arange(-self.offset, self.offset + 1, dtype=torch.float64) * in_scale
        self.table = torch.round(torch.sigmoid(x) * qmax(out_bits)).to(torch.int64)

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return self.table[q + self.offset]


class IntSigmoidPoly:
    def __init__(self, a: float, c: float, in_scale: float, out_bits: int = 8) -> None:
        self.sigmoid = TanhPolySigmoid(a, c, in_scale)
        self.m, self.n = adaptive_multiplier(self.sigmoid.scale * qmax(out_bits))
        self.out_bits = out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return requantize(self.sigmoid(q), self.m, self.n, self.out_bits, lower=0)
