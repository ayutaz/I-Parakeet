"""Integer linear / convolution layers (Eq. 7), residual addition and GLU.

Weights: per-output-channel symmetric INT8. Bias: INT32 on the accumulator grid S_W * S_x.
Output: fixed-point multiplier m = round(2^n S_W S_x / S_y) with n = 16 (paper) and a shift.
"""

import torch

from iparakeet.quant.fixed_point import adaptive_multiplier, multiplier, qmax, quantize, requantize
from iparakeet.quant.intmm import int_conv1d, int_conv2d, int_matmul


def quantize_weight_per_channel(w: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.detach().double()
    alpha = w.abs().amax(dim=tuple(range(1, w.dim()))).clamp_min(1e-12)
    scale = alpha / qmax(bits)
    return quantize(w, scale.view(-1, *([1] * (w.dim() - 1))), bits), scale


def _bias_int(bias, in_scale, w_scale) -> torch.Tensor:
    if bias is None:
        return torch.zeros_like(w_scale, dtype=torch.int64)
    return torch.round(bias.detach().double() / (in_scale * w_scale)).to(torch.int64)


class IntLinear:
    def __init__(self, weight, bias, in_scale: float, out_scale, out_bits: int, n: int = 16, w_bits: int = 8, extra: float = 1.0, relu: bool = False) -> None:
        w_q, s_w = quantize_weight_per_channel(weight, w_bits)
        self.w_t = w_q.T.contiguous()
        self.bias = _bias_int(bias, in_scale, s_w)
        self.ratio = in_scale * s_w / torch.as_tensor(out_scale, dtype=torch.float64)
        self.m = multiplier(self.ratio * extra, n)
        self.n, self.out_bits, self.relu = n, out_bits, relu

    def accumulate(self, q: torch.Tensor) -> torch.Tensor:
        return int_matmul(q, self.w_t) + self.bias

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        out = requantize(self.accumulate(q), self.m, self.n, self.out_bits)
        return out.clamp(min=0) if self.relu else out


class IntConv2d:
    def __init__(self, conv: torch.nn.Conv2d, in_scale: float, out_scale, out_bits: int, relu: bool = False, n: int = 16, w_bits: int = 8) -> None:
        self.w_q, s_w = quantize_weight_per_channel(conv.weight, w_bits)
        self.bias = _bias_int(conv.bias, in_scale, s_w).view(-1, 1, 1)
        self.ratio = in_scale * s_w / torch.as_tensor(out_scale, dtype=torch.float64)
        self.m = multiplier(self.ratio, n).view(-1, 1, 1)
        self.stride, self.padding, self.groups = conv.stride[0], conv.padding[0], conv.groups
        self.n, self.out_bits, self.relu = n, out_bits, relu

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        acc = int_conv2d(q, self.w_q, self.stride, self.padding, self.groups) + self.bias
        out = requantize(acc, self.m, self.n, self.out_bits)
        return out.clamp(min=0) if self.relu else out


class IntConv1dDepthwise:
    """Depthwise Conv1d with BatchNorm already folded into (weight, bias); output grid may be per-channel."""

    def __init__(self, weight, bias, in_scale: float, out_scale, out_bits: int, n: int = 16, w_bits: int = 8) -> None:
        self.w_q, s_w = quantize_weight_per_channel(weight, w_bits)
        self.bias = _bias_int(bias, in_scale, s_w).view(-1, 1)
        self.ratio = in_scale * s_w / torch.as_tensor(out_scale, dtype=torch.float64)
        self.m = multiplier(self.ratio, n).view(-1, 1)
        self.padding = (weight.shape[-1] - 1) // 2
        self.channels = weight.shape[0]
        self.n, self.out_bits = n, out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        acc = int_conv1d(q, self.w_q, padding=self.padding, groups=self.channels) + self.bias
        return requantize(acc, self.m, self.n, self.out_bits)


class IntAdd:
    """sum_i f_i * x_i on a common output grid: round(2^-n * sum_i m_i q_i), m_i = round(2^n f_i S_i / S_out)."""

    def __init__(self, in_scales: list[float], factors: list[float], out_scale: float, out_bits: int, n: int = 16) -> None:
        self.ms = [multiplier(f * s / out_scale, n) for f, s in zip(factors, in_scales)]
        self.ratios = [f * s / out_scale for f, s in zip(factors, in_scales)]
        self.n, self.out_bits = n, out_bits

    def __call__(self, *qs: torch.Tensor) -> torch.Tensor:
        acc = sum(q * m for q, m in zip(qs, self.ms))
        out = (acc + (1 << (self.n - 1))) >> self.n
        return out.clamp(-qmax(self.out_bits), qmax(self.out_bits))


class IntGLU:
    """a * sigmoid(b) for [a, b] = split(x); the gate is an integer sigmoid on [0, 2^(g-1)-1]."""

    def __init__(self, in_scale: float, out_scale: float, out_bits: int, gate, gate_bits: int = 8) -> None:
        self.gate = gate
        self.m, self.n = adaptive_multiplier(in_scale / qmax(gate_bits) / out_scale)
        self.out_bits = out_bits

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        a, b = q.chunk(2, dim=-1)
        return requantize(a * self.gate(b), self.m, self.n, self.out_bits)
