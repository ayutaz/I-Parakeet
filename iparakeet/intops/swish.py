"""Integer Swish sw(x) = x * sigma(x) (paper Sec. 3.2).

sigma(x) = (1 + tanh(x/2)) / 2 with tanh approximated by Eq. (12):
    tanh^(u) = sgn(u) * [a * (min(|u|, c) - c)^2 + 1]
evaluated in integer arithmetic like I-BERT's i-GELU. Coefficients:
  linf_swish  - L_inf fit to the Swish output (paper, a* = -0.1240, c* = 2.4632)
  l2_swish, linf_tanh, l2_tanh - the Table 3 alternatives, fitted by scripts/fit_swish_approx.py
"""

import torch

from iparakeet.quant.fixed_point import adaptive_multiplier, qmax, requantize

SWISH_COEFFS: dict[str, tuple[float, float]] = {
    "linf_swish": (-0.1240, 2.4632),
    "l2_swish": (-0.1381, 2.3728),
    "linf_tanh": (-0.2182, 2.1032),
    "l2_tanh": (-0.2304, 2.0506),
}


class TanhPolySigmoid:
    """sigma^(x) as an integer `sig` with scale |a| S_u^2 / 2 * 2^shift (sig in [0, 2|B| >> shift])."""

    def __init__(self, a: float, c: float, in_scale, sig_bits: int = 15) -> None:
        s = torch.as_tensor(in_scale, dtype=torch.float64)
        s_u = s / 2
        self.c_int = torch.floor(c / s_u).to(torch.int64)
        s_t = a * s_u**2
        self.B = torch.floor(1.0 / s_t).to(torch.int64)  # negative: 1 on the tanh grid
        max_sig = int((-2 * self.B).max())
        self.shift = max(0, max_sig.bit_length() - sig_bits)
        self.scale = (s_t.abs() / 2) * 2.0**self.shift

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        t = torch.minimum(q.abs(), self.c_int) - self.c_int
        poly = t * t + self.B
        sig = -(torch.sign(q) * poly + self.B)
        return sig.clamp(min=0) >> self.shift


class IntSwishPoly:
    def __init__(self, a: float, c: float, in_scale, out_scale, out_bits: int, sig_bits: int = 15) -> None:
        self.sigmoid = TanhPolySigmoid(a, c, in_scale, sig_bits)
        ratio = torch.as_tensor(in_scale, dtype=torch.float64) * self.sigmoid.scale / out_scale
        self.m, self.n = adaptive_multiplier(ratio)
        self.out_bits = out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return requantize(q * self.sigmoid(q), self.m, self.n, self.out_bits)


class IntHardSwish:
    """x * ReLU6(x + 3) / 6 (MobileNetV3), the Table 3 baseline."""

    def __init__(self, in_scale: float, out_scale: float, out_bits: int) -> None:
        self.three = round(3.0 / in_scale)
        self.six = round(6.0 / in_scale)
        self.m, self.n = adaptive_multiplier(in_scale * in_scale / 6.0 / out_scale)
        self.out_bits = out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        r = torch.clamp(q + self.three, 0, self.six)
        return requantize(q * r, self.m, self.n, self.out_bits)


class IntSwishLUT:
    """x * LUT_sigma(x): the NPU evaluates the sigmoid with a hardware lookup table (paper Sec. 4.1)."""

    def __init__(self, in_scale: float, in_bits: int, out_scale: float, out_bits: int) -> None:
        self.offset = qmax(in_bits)
        lut_bits = min(16, 31 - in_bits)
        levels = 2**lut_bits - 1
        x = torch.arange(-self.offset, self.offset + 1, dtype=torch.float64) * in_scale
        self.table = torch.round(torch.sigmoid(x) * levels).to(torch.int64)
        self.m, self.n = adaptive_multiplier(in_scale / levels / out_scale)
        self.out_bits = out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return requantize(q * self.table[q + self.offset], self.m, self.n, self.out_bits)


def make_swish(spec: str, in_scale, in_bits: int, out_scale: float, out_bits: int):
    if spec.startswith("poly:"):
        name = spec.split(":", 1)[1]
        if name not in SWISH_COEFFS:
            raise ValueError(f"unknown Swish coefficients {name!r}")
        return IntSwishPoly(*SWISH_COEFFS[name], in_scale=in_scale, out_scale=out_scale, out_bits=out_bits)
    if spec == "hardswish":
        return IntHardSwish(float(in_scale), out_scale, out_bits)
    if spec == "lut":
        if isinstance(in_scale, torch.Tensor) and in_scale.numel() > 1:
            raise ValueError("a LUT needs a per-tensor input scale")
        return IntSwishLUT(float(in_scale), in_bits, out_scale, out_bits)
    raise ValueError(f"unknown Swish approximation {spec!r}")
