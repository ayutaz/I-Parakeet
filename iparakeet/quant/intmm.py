"""Exact integer GEMM / convolution: emulation of the NPU's INT8 x INT8 -> INT32 primitives.

Products and partial sums are integers below 2^53, so float64 BLAS computes them exactly on CPU;
on CUDA, torch._int_mm is used for 2-D INT8 operands. Results are always int64 tensors.
"""

import torch
import torch.nn.functional as F

from iparakeet.quant.nofloat import exact_float_emulation

_EXACT = 2**53


def _check_exact(a: torch.Tensor, b: torch.Tensor, k: int) -> None:
    bound = int(a.abs().max()) * int(b.abs().max()) * k if a.numel() and b.numel() else 0
    if bound >= _EXACT:
        raise OverflowError(f"integer GEMM bound {bound} is not exactly representable")


def _int8_range(t: torch.Tensor) -> bool:
    return t.numel() > 0 and int(t.abs().max()) <= 127


def int_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    k = a.shape[-1]
    if (
        a.is_cuda and a.dim() == 2 and b.dim() == 2 and a.shape[0] > 16
        and k % 8 == 0 and b.shape[1] % 8 == 0 and _int8_range(a) and _int8_range(b)
    ):
        return torch._int_mm(a.to(torch.int8), b.to(torch.int8)).to(torch.int64)
    _check_exact(a, b, k)
    with exact_float_emulation():
        return torch.matmul(a.double(), b.double()).round().to(torch.int64)


def int_conv1d(x: torch.Tensor, w: torch.Tensor, padding: int = 0, groups: int = 1) -> torch.Tensor:
    _check_exact(x, w, w.shape[1] * w.shape[2])
    with exact_float_emulation():
        return F.conv1d(x.double(), w.double(), padding=padding, groups=groups).round().to(torch.int64)


def int_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0, groups: int = 1) -> torch.Tensor:
    _check_exact(x, w, w.shape[1] * w.shape[2] * w.shape[3])
    with exact_float_emulation():
        return F.conv2d(x.double(), w.double(), stride=stride, padding=padding, groups=groups).round().to(torch.int64)
