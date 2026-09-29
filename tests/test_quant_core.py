import numpy as np
import pytest
import torch
import torch.nn.functional as F

from iparakeet.quant.fixed_point import (
    dequantize,
    int32_check,
    multiplier,
    qmax,
    quantize,
    requantize,
    scale_from_alpha,
)
from iparakeet.quant.intmm import int_conv1d, int_conv2d, int_matmul
from iparakeet.quant.nofloat import FloatLeakError, NoFloatMode, exact_float_emulation


# ---- Eq. (6): symmetric uniform quantization ----

def test_quantize_is_symmetric_round_and_clip():
    scale = scale_from_alpha(1.0, 8)
    assert scale == pytest.approx(1 / 127)
    q = quantize(torch.tensor([0.0, 0.5, -0.5, 2.0, -2.0]), scale, 8)
    assert q.dtype == torch.int64
    assert q.tolist() == [0, 64, -64, 127, -127]


def test_dequantize_error_is_at_most_half_step_inside_range():
    x = torch.linspace(-3, 3, 1001, dtype=torch.float64)
    scale = scale_from_alpha(3.0, 16)
    err = (dequantize(quantize(x, scale, 16), scale) - x).abs().max()
    assert err <= scale / 2 + 1e-12


def test_percentile_clipping_restores_resolution_of_the_bulk_on_heavy_tails():
    # Sec. 3.3: a few outliers spread a min-max range and degrade rounding precision for the 99.9%.
    x = torch.from_numpy(np.random.default_rng(0).standard_t(df=2, size=200_000))
    p999 = torch.quantile(x.abs()[:100_000], 0.999).item()
    bulk = x[x.abs() <= p999]
    err = {}
    for name, alpha in {"minmax": x.abs().max().item(), "p99.9": p999}.items():
        s = scale_from_alpha(alpha, 8)
        err[name] = (dequantize(quantize(bulk, s, 8), s) - bulk).abs().mean().item()
    assert err["p99.9"] * 10 < err["minmax"]


# ---- Eq. (7): fixed-point requantization ----

def test_requantize_matches_real_rescale_within_one_step():
    gen = torch.Generator().manual_seed(0)
    acc = torch.randint(-(2**20), 2**20, (1000,), generator=gen)
    ratio = 3.7e-4
    m = multiplier(ratio, 16)
    out = requantize(acc, m, 16, bits=16)
    expected = torch.round(acc.double() * m.double() / 2**16)
    assert torch.all((out.double() - expected).abs() <= 1)
    assert torch.all((out.double() - acc.double() * ratio).abs() <= 1 + acc.abs().double() * abs(m.item() / 2**16 - ratio))


def test_requantize_per_channel_and_clamp():
    acc = torch.tensor([[1000, 1000], [-100000, 100000]])
    m = multiplier(torch.tensor([0.5, 0.01]), 16)
    out = requantize(acc, m, 16, bits=8)
    assert out.tolist() == [[127, 10], [-127, 127]]


def test_requantize_floor_mode():
    acc = torch.tensor([3, -3])
    assert requantize(acc, multiplier(0.5, 16), 16, bits=8, rounding="floor").tolist() == [1, -2]
    assert requantize(acc, multiplier(0.5, 16), 16, bits=8).tolist() == [2, -1]


def test_qmax():
    assert qmax(8) == 127 and qmax(16) == 32767


def test_int32_check_flags_accumulator_overflow():
    big = torch.tensor([2**31])
    requantize(big, multiplier(1e-6, 16), 16, bits=8)  # no check by default
    with int32_check(), pytest.raises(OverflowError):
        requantize(big, multiplier(1e-6, 16), 16, bits=8)


# ---- float leak guard ----

def test_no_float_mode_rejects_float_results():
    q = torch.tensor([1, 2, 3])
    with NoFloatMode():
        assert (q * 2 + 1).tolist() == [3, 5, 7]
        with pytest.raises(FloatLeakError):
            _ = q * 0.5
        with pytest.raises(FloatLeakError):
            torch.sigmoid(q.double())


def test_exact_float_emulation_is_exempt():
    q = torch.tensor([1, 2, 3])
    with NoFloatMode():
        with exact_float_emulation():
            y = (q.double() * 2).to(torch.int64)
        assert y.tolist() == [2, 4, 6]


# ---- exact integer GEMM / conv (INT8 x INT8 -> INT32 hardware emulation) ----

def test_int_matmul_is_exact():
    gen = torch.Generator().manual_seed(0)
    a = torch.randint(-127, 128, (2, 3, 5, 4096), generator=gen)
    b = torch.randint(-127, 128, (4096, 7), generator=gen)
    got = int_matmul(a, b)
    assert got.dtype == torch.int64
    assert torch.equal(got, torch.matmul(a, b))


def test_int_matmul_rejects_non_exact_ranges():
    a = torch.tensor([[2**40]])
    with pytest.raises(OverflowError):
        int_matmul(a, torch.tensor([[2**20]]))


def test_int_matmul_runs_inside_no_float_mode():
    a = torch.randint(-127, 128, (3, 8))
    b = torch.randint(-127, 128, (8, 2))
    with NoFloatMode():
        assert torch.equal(int_matmul(a, b), a @ b)


def _loop_conv1d(x, w, padding, groups):
    b, cin, t = x.shape
    cout, cin_g, k = w.shape
    xp = F.pad(x, (padding, padding))
    out = torch.zeros(b, cout, t + 2 * padding - k + 1, dtype=torch.int64)
    per_group = cout // groups
    for o in range(cout):
        g = o // per_group
        for s in range(out.shape[-1]):
            out[:, o, s] = (xp[:, g * cin_g : (g + 1) * cin_g, s : s + k] * w[o]).sum(dim=(1, 2))
    return out


def test_int_conv1d_depthwise_is_exact():
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(-127, 128, (1, 4, 12), generator=gen)
    w = torch.randint(-127, 128, (4, 1, 9), generator=gen)
    assert torch.equal(int_conv1d(x, w, padding=4, groups=4), _loop_conv1d(x, w, 4, 4))


def test_int_conv2d_matches_int64_reference():
    gen = torch.Generator().manual_seed(2)
    x = torch.randint(-127, 128, (1, 1, 9, 10), generator=gen)
    w = torch.randint(-127, 128, (3, 1, 3, 3), generator=gen)
    got = int_conv2d(x, w, stride=2, padding=1)
    ref = F.conv2d(x.double(), w.double(), stride=2, padding=1).to(torch.int64)
    assert got.dtype == torch.int64 and torch.equal(got, ref)
