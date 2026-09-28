import math

import pytest
import torch
import torch.nn.functional as F

from conftest import randomize_
from iparakeet.intops.linear import IntAdd, IntConv1dDepthwise, IntConv2d, IntGLU, IntLinear, quantize_weight_per_channel
from iparakeet.intops.relpos_mhsa import IntRelPosMHSA, fused_scores
from iparakeet.intops.sigmoid import IntSigmoidLUT
from iparakeet.model.layers import RelPositionMultiHeadAttention, fold_batchnorm
from iparakeet.model.parakeet import collect_taps
from iparakeet.model.relpos import rel_pos_emb, rel_pos_table
from iparakeet.quant.fixed_point import dequantize, int32_check, quantize, scale_from_alpha
from iparakeet.quant.nofloat import NoFloatMode


def sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref, test = ref.detach().double(), test.detach().double()
    return 10 * math.log10(float((ref**2).sum()) / max(float(((ref - test) ** 2).sum()), 1e-30))


def test_per_channel_weight_quantization():
    w = torch.tensor([[0.5, -1.0], [0.01, 0.02]], dtype=torch.float64)
    q, s = quantize_weight_per_channel(w, 8)
    assert q.tolist() == [[64, -127], [64, 127]]
    assert torch.allclose(s, torch.tensor([1.0 / 127, 0.02 / 127], dtype=torch.float64))


def test_int_linear_matches_float_with_quantized_weights():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 16)
    x = torch.randn(10, 64, dtype=torch.float64)
    s_x = scale_from_alpha(x.abs().max().item(), 8)
    q_x = quantize(x, s_x, 8)
    ref = F.linear(dequantize(q_x, s_x), lin.weight.double(), lin.bias.double())
    s_y = scale_from_alpha(ref.abs().max().item(), 8)
    layer = IntLinear(lin.weight, lin.bias, in_scale=s_x, out_scale=s_y, out_bits=8)
    with NoFloatMode(), int32_check():
        out = layer(q_x)
    assert out.dtype == torch.int64
    assert sqnr_db(ref, dequantize(out, s_y)) > 30


def test_int_linear_extra_factor_and_relu():
    lin = torch.nn.Linear(4, 3)
    q_x = torch.randint(-127, 128, (5, 4))
    base = IntLinear(lin.weight, lin.bias, 0.01, 0.02, 8)
    scaled = IntLinear(lin.weight, lin.bias, 0.01, 0.02, 8, extra=0.5, relu=True)
    expected = torch.clamp(torch.round(base.accumulate(q_x).double() * base.ratio * 0.5), -127, 127).clamp(min=0)
    assert torch.all((scaled(q_x).double() - expected).abs() <= 1)


def test_int_conv2d_stride2_relu_matches_float():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(1, 4, 3, 2, 1)
    x = torch.randn(1, 1, 20, 16, dtype=torch.float64)
    s_x = scale_from_alpha(x.abs().max().item(), 8)
    q_x = quantize(x, s_x, 8)
    ref = F.relu(F.conv2d(dequantize(q_x, s_x), conv.weight.double(), conv.bias.double(), stride=2, padding=1))
    s_y = scale_from_alpha(ref.abs().max().item(), 8)
    layer = IntConv2d(conv, in_scale=s_x, out_scale=s_y, out_bits=8, relu=True)
    with NoFloatMode():
        out = layer(q_x)
    assert int(out.min()) >= 0
    assert sqnr_db(ref, dequantize(out, s_y)) > 30


def test_int_depthwise_conv_with_folded_bn_and_int16_output():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(8, 8, 9, padding=4, groups=8)
    bn = randomize_(torch.nn.BatchNorm1d(8)).eval()
    with torch.no_grad():
        bn.weight[0] = 500.0  # extreme per-channel BN scale (Fig. 2a)
    w, b = fold_batchnorm(conv.weight, conv.bias, bn)
    x = torch.randn(1, 8, 30, dtype=torch.float64)
    s_x = scale_from_alpha(x.abs().max().item(), 8)
    q_x = quantize(x, s_x, 8)
    ref = F.conv1d(dequantize(q_x, s_x), w.double(), b.double(), padding=4, groups=8)
    sq = {}
    for bits in (8, 16):
        s_y = scale_from_alpha(ref.abs().max().item(), bits)
        layer = IntConv1dDepthwise(w, b, in_scale=s_x, out_scale=s_y, out_bits=bits)
        with NoFloatMode():
            out = layer(q_x)
        sq[bits] = sqnr_db(ref[:, 1:], dequantize(out, s_y)[:, 1:])  # the ordinary channels
    assert sq[16] > sq[8] + 20


def test_int_add_with_macaron_half_factor():
    q_a = torch.tensor([100, -50])
    q_b = torch.tensor([40, 40])
    add = IntAdd(in_scales=[0.1, 0.2], factors=[1.0, 0.5], out_scale=0.1, out_bits=8)
    assert add(q_a, q_b).tolist() == [127, -10]  # 10 + 0.5*8 = 14 -> 140 (clamped); -5 + 4 = -1 -> -10


def test_int_glu_gates_first_half_with_sigmoid_of_second_half():
    q = torch.tensor([[50, -50, 100, -100]])
    s_in, s_out = 0.02, 0.01
    glu = IntGLU(in_scale=s_in, out_scale=s_out, out_bits=8, gate=IntSigmoidLUT(s_in, 8, 8))
    x = dequantize(q, s_in)
    ref = x[:, :2] * torch.sigmoid(x[:, 2:])
    with NoFloatMode():
        out = glu(q)
    assert (dequantize(out, s_out) - ref).abs().max() <= 1.5 * s_out + 0.01


# ---- integer relative-position MHSA (Sec. 3.1) ----

def test_fused_scores_equal_rounded_real_sum():
    gen = torch.Generator().manual_seed(0)
    qc = torch.randint(-(2**15), 2**15, (2, 5, 5), generator=gen)
    qp = torch.randint(-(2**15), 2**15, (2, 5, 5), generator=gen)
    s_c, s_p, s_s, dk = 1e-4, 3e-4, 0.05, 64
    from iparakeet.quant.fixed_point import multiplier

    m_c = multiplier(s_c / (s_s * math.sqrt(dk)), 16)
    m_p = multiplier(s_p / (s_s * math.sqrt(dk)), 16)
    out = fused_scores(qc, qp, m_c, m_p, 16, bits=16)
    real = (s_c * qc.double() + s_p * qp.double()) / math.sqrt(dk) / s_s
    assert torch.all((out.double() - real).abs() <= 1 + real.abs() * 2e-3)


def _calibrated_scales(att, x, pos_emb):
    seen = {}
    with collect_taps(att, lambda n, t: seen.__setitem__(n.split(".att.")[1], float(t.abs().max()))):
        att(x, pos_emb, None)
    return seen


def test_int_rel_pos_mhsa_tracks_float_module():
    torch.manual_seed(0)
    d, h, L, L_max = 32, 4, 12, 20
    att = randomize_(RelPositionMultiHeadAttention(h, d, "L0")).eval()
    table = rel_pos_table(64, d)
    x = torch.randn(1, L, d)
    with torch.no_grad():
        ref = att(x, rel_pos_emb(table, L)[None], None)
        alphas = _calibrated_scales(att, x, rel_pos_emb(table, L)[None])
    s_in = scale_from_alpha(float(x.abs().max()), 8)
    scales = {k: scale_from_alpha(v, 8) for k, v in alphas.items()}
    scales["q"] = scale_from_alpha(max(alphas["qu"], alphas["qv"]), 8)
    int_att = IntRelPosMHSA(att, in_scale=s_in, scales=scales, pos_table=table, max_len=L_max)
    q_x = quantize(x.double(), s_in, 8)
    with NoFloatMode(), int32_check():
        out = int_att(q_x)
    assert out.shape == (1, L, d)
    assert sqnr_db(ref, dequantize(out, scales["out"])) > 15


def test_position_constant_is_one_int8_tensor_sliced_per_length():
    torch.manual_seed(0)
    d, h = 32, 4
    att = randomize_(RelPositionMultiHeadAttention(h, d, "L0")).eval()
    table = rel_pos_table(64, d)
    scales = {k: 0.05 for k in ("q", "k", "v", "scores", "ctx", "out")}
    int_att = IntRelPosMHSA(att, in_scale=0.05, scales=scales, pos_table=table, max_len=20)
    assert int_att.q_P.dtype == torch.int64 and int_att.q_P.shape == (h, 39, d // h)
    full = int_att.position_rows(20)
    short = int_att.position_rows(7)
    assert torch.equal(short, full[:, 20 - 7 : 20 + 7 - 1])
    for L in (3, 7, 20):
        assert int_att(torch.randint(-127, 128, (1, L, d))).shape == (1, L, d)
