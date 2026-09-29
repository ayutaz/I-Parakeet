"""Symmetric uniform quantization (paper Eq. 6) and fixed-point requantization (Eq. 7)."""

import contextlib

import torch

_CHECK = {"int32": False}


@contextlib.contextmanager
def int32_check(enabled: bool = True):
    """Raise OverflowError when an accumulator entering requantization leaves the INT32 range."""
    previous = _CHECK["int32"]
    _CHECK["int32"] = enabled
    try:
        yield
    finally:
        _CHECK["int32"] = previous


def check_int32(t: torch.Tensor, what: str = "accumulator") -> None:
    if _CHECK["int32"] and t.numel() and int(t.abs().max()) >= 2**31:
        raise OverflowError(f"{what} exceeds INT32 (max |x| = {int(t.abs().max())})")


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def scale_from_alpha(alpha, bits: int):
    """S = alpha / (2^(b-1) - 1); alpha may be a float or a per-channel tensor."""
    if isinstance(alpha, torch.Tensor):
        return alpha.double().clamp_min(1e-12) / qmax(bits)
    return max(float(alpha), 1e-12) / qmax(bits)


def quantize(x: torch.Tensor, scale, bits: int) -> torch.Tensor:
    """q = round(clip(x, -alpha, alpha) / S) as int64."""
    q = torch.round(x.double() / torch.as_tensor(scale, dtype=torch.float64))
    return q.clamp(-qmax(bits), qmax(bits)).to(torch.int64)


def dequantize(q: torch.Tensor, scale) -> torch.Tensor:
    return q.double() * torch.as_tensor(scale, dtype=torch.float64)


def multiplier(ratio, n: int = 16) -> torch.Tensor:
    """Fixed-point multiplier m = round(2^n * ratio), computed offline."""
    return torch.round(torch.as_tensor(ratio, dtype=torch.float64) * 2.0**n).to(torch.int64)


def requantize(acc: torch.Tensor, m: torch.Tensor, n: int, bits: int, rounding: str = "half_up", lower: int | None = None) -> torch.Tensor:
    """q_y = round(2^-n * m * acc), clamped to the b-bit symmetric grid (integer multiply + shift only)."""
    check_int32(acc)
    prod = acc * m
    if rounding == "half_up":
        prod = prod + (1 << (n - 1))
    elif rounding != "floor":
        raise ValueError(f"unknown rounding {rounding!r}")
    out = prod >> n
    return out.clamp(-qmax(bits) if lower is None else lower, qmax(bits))


def adaptive_multiplier(ratio, m_bits: int = 15) -> tuple[torch.Tensor, int]:
    """(m, n) with m = round(2^n * ratio) holding m_bits bits for the largest ratio.

    Eq. (7) fixes n = 16 for linear layers; kernels whose internal rescale is not specified in the
    paper (Swish, LayerNorm, Softmax, GLU, LUTs) use this to keep the multiplier precise.
    """
    ratio_t = torch.as_tensor(ratio, dtype=torch.float64)
    r_max = float(ratio_t.abs().max())
    if r_max <= 0:
        return torch.zeros_like(ratio_t, dtype=torch.int64), 0
    n = max(1, min(62, m_bits - 1 - int(torch.floor(torch.log2(torch.tensor(r_max))))))
    return multiplier(ratio_t, n), n
