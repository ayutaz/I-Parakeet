import math

import pytest
import torch

from iparakeet.intops.layernorm import IntLayerNorm, isqrt
from iparakeet.intops.sigmoid import IntSigmoidLUT, IntSigmoidPoly
from iparakeet.intops.softmax import IntSoftmax
from iparakeet.intops.swish import SWISH_COEFFS, IntHardSwish, IntSwishLUT, IntSwishPoly, make_swish
from iparakeet.quant.fixed_point import dequantize, int32_check, qmax, quantize, scale_from_alpha
from iparakeet.quant.nofloat import NoFloatMode


def swish_hat(x, a, c):
    u = x / 2
    t = torch.sign(u) * (a * (torch.clamp(u.abs(), max=c) - c) ** 2 + 1)
    return x * (1 + t) / 2


def _grid(alpha, bits):
    s = scale_from_alpha(alpha, bits)
    q = torch.arange(-qmax(bits), qmax(bits) + 1)
    return q, s


# ---- Swish (Sec. 3.2) ----

def test_paper_coefficients_are_registered():
    assert SWISH_COEFFS["linf_swish"] == (-0.1240, 2.4632)


@pytest.mark.parametrize("bits,alpha", [(8, 8.0), (16, 60.0)])
def test_int_swish_poly_implements_eq12_within_one_output_step(bits, alpha):
    a, c = SWISH_COEFFS["linf_swish"]
    q, s = _grid(alpha, bits)
    s_out = scale_from_alpha(alpha, 8)
    kernel = IntSwishPoly(a, c, in_scale=s, out_scale=s_out, out_bits=8)
    with NoFloatMode(), int32_check():
        out = kernel(q)
    x = dequantize(q, s)
    err = (dequantize(out, s_out) - swish_hat(x, a, c)).abs().max().item()
    assert err <= 1.5 * s_out
    true_err = (dequantize(out, s_out) - x * torch.sigmoid(x)).abs().max().item()
    assert true_err <= 0.0387 + 1.5 * s_out


def test_int_swish_poly_supports_per_channel_input_scales():
    a, c = SWISH_COEFFS["linf_swish"]
    alphas = torch.tensor([0.5, 3.0, 6.0], dtype=torch.float64)
    scales = scale_from_alpha(alphas, 16)
    x = (torch.rand(200, 3, dtype=torch.float64) * 2 - 1) * alphas
    q = quantize(x, scales, 16)
    s_out = scale_from_alpha(6.0, 8)
    kernel = IntSwishPoly(a, c, in_scale=scales, out_scale=s_out, out_bits=8)
    with NoFloatMode():
        out = kernel(q)
    assert (dequantize(out, s_out) - swish_hat(dequantize(q, scales), a, c)).abs().max() <= 1.5 * s_out


def test_int_hard_swish_matches_float_hard_swish():
    q, s = _grid(8.0, 8)
    s_out = scale_from_alpha(8.0, 8)
    kernel = IntHardSwish(in_scale=s, out_scale=s_out, out_bits=8)
    with NoFloatMode():
        out = kernel(q)
    x = dequantize(q, s)
    ref = x * torch.clamp(x + 3, 0, 6) / 6
    assert (dequantize(out, s_out) - ref).abs().max() <= 1.5 * s_out + 3 * s


def test_int_swish_lut_is_close_to_exact_swish():
    q, s = _grid(8.0, 8)
    s_out = scale_from_alpha(8.0, 8)
    kernel = IntSwishLUT(in_scale=s, in_bits=8, out_scale=s_out, out_bits=8)
    with NoFloatMode():
        out = kernel(q)
    x = dequantize(q, s)
    assert (dequantize(out, s_out) - x * torch.sigmoid(x)).abs().max() <= 1.5 * s_out


def test_make_swish_dispatches_recipe_strings():
    assert isinstance(make_swish("poly:l2_tanh", 0.1, 8, 0.1, 8), IntSwishPoly)
    assert isinstance(make_swish("hardswish", 0.1, 8, 0.1, 8), IntHardSwish)
    assert isinstance(make_swish("lut", 0.1, 8, 0.1, 8), IntSwishLUT)
    with pytest.raises(ValueError):
        make_swish("poly:unknown", 0.1, 8, 0.1, 8)


# ---- sigmoid for GLU ----

def test_sigmoid_lut_outputs_unsigned_grid():
    q, s = _grid(10.0, 8)
    kernel = IntSigmoidLUT(in_scale=s, in_bits=8, out_bits=8)
    with NoFloatMode():
        out = kernel(q)
    assert int(out.min()) >= 0 and int(out.max()) <= 127
    assert torch.equal(out, torch.round(torch.sigmoid(dequantize(q, s)) * 127).to(torch.int64))


def test_sigmoid_poly_tracks_sigmoid():
    q, s = _grid(10.0, 8)
    a, c = SWISH_COEFFS["linf_tanh"]
    kernel = IntSigmoidPoly(a, c, in_scale=s, out_bits=8)
    with NoFloatMode():
        out = kernel(q)
    assert (out.double() / 127 - torch.sigmoid(dequantize(q, s))).abs().max() <= 0.0675 / 2 + 1.5 / 127


# ---- LayerNorm (I-BERT style) ----

def test_isqrt_matches_math_isqrt():
    gen = torch.Generator().manual_seed(0)
    values = torch.cat([torch.tensor([0, 1, 2, 3, 4, 15, 16, 17, 2**62 - 1]), torch.randint(0, 2**62, (500,), generator=gen)])
    with NoFloatMode():
        got = isqrt(values)
    assert got.tolist() == [math.isqrt(v) for v in values.tolist()]


def test_int_layernorm_matches_float_layernorm():
    torch.manual_seed(0)
    d = 64
    x = torch.randn(5, d, dtype=torch.float64) * 3 + 0.5
    gamma = torch.rand(d, dtype=torch.float64) + 0.5
    beta = torch.randn(d, dtype=torch.float64) * 0.1
    s_in = scale_from_alpha(x.abs().max().item(), 8)
    q = quantize(x, s_in, 8)
    ref = torch.nn.functional.layer_norm(dequantize(q, s_in), (d,), gamma, beta, eps=1e-5)
    s_out = scale_from_alpha(ref.abs().max().item(), 8)
    ln = IntLayerNorm(gamma, beta, in_scale=s_in, out_scale=s_out, out_bits=8)
    with NoFloatMode():
        out = ln(q)
    assert (dequantize(out, s_out) - ref).abs().max() <= 1.5 * s_out


# ---- Softmax (I-BERT) ----

def test_int_softmax_matches_float_softmax():
    torch.manual_seed(0)
    scores = torch.randn(4, 3, 50, dtype=torch.float64) * 3
    s_in = scale_from_alpha(scores.abs().max().item(), 8)
    q = quantize(scores, s_in, 8)
    kernel = IntSoftmax(in_scale=s_in, out_bits=8)
    with NoFloatMode():
        p = kernel(q)
    ref = torch.softmax(dequantize(q, s_in), dim=-1)
    assert int(p.min()) >= 0 and int(p.max()) <= 127
    assert (p.double() / 127 - ref).abs().max() <= 2.5 / 127


def test_int_softmax_rows_sum_to_one_within_rounding():
    q = torch.randint(-127, 128, (10, 30))
    p = IntSoftmax(in_scale=0.05, out_bits=8)(q)
    assert torch.all((p.sum(-1) - 127).abs() <= 30)


def test_ibert_range_reduction_is_accurate_on_fine_grids_only():
    torch.manual_seed(1)
    scores = torch.randn(2, 40, dtype=torch.float64) * 3
    ref = torch.softmax(scores, dim=-1)
    fine = IntSoftmax(in_scale=1e-4, out_bits=8, range_reduction="ibert")
    q_fine = quantize(scores, 1e-4, 32)
    assert (fine(q_fine).double() / 127 - ref).abs().max() <= 2.5 / 127
    with pytest.raises(ValueError):
        IntSoftmax(in_scale=0.1, range_reduction="unknown")
