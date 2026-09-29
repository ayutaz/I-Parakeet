import math

import pytest
import torch
import torch.nn.functional as F

from conftest import randomize_, tiny_config
from iparakeet.model.layers import (
    ConformerConvolution,
    ConvSubsampling,
    RelPositionMultiHeadAttention,
    fold_batchnorm,
    subsampled_length,
)
from iparakeet.model.parakeet import ParakeetCTC
from iparakeet.model.relpos import rel_pos_emb, rel_pos_table


def test_fold_batchnorm_matches_conv_then_bn():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(6, 6, 9, padding=4, groups=6)
    bn = randomize_(torch.nn.BatchNorm1d(6)).eval()
    x = torch.randn(2, 6, 20)
    w, b = fold_batchnorm(conv.weight, conv.bias, bn)
    folded = F.conv1d(x, w, b, padding=4, groups=6)
    assert torch.allclose(folded, bn(conv(x)), atol=1e-5)


@pytest.mark.parametrize("T", [1, 7, 8, 9, 64, 101, 3500])
def test_subsampled_length_is_three_stride2_convs(T):
    expected = T
    for _ in range(3):
        expected = (expected - 1) // 2 + 1
    assert subsampled_length(T, 8) == expected


def test_subsampling_output_shape_matches_length_formula():
    cfg = tiny_config()
    sub = ConvSubsampling(cfg).eval()
    for T in (9, 50, 101):
        out, lengths = sub(torch.randn(1, T, cfg.feat_in), torch.tensor([T]))
        assert out.shape == (1, subsampled_length(T, 8), cfg.d_model)
        assert lengths.item() == subsampled_length(T, 8)


def test_rel_pos_attention_matches_explicit_formula():
    torch.manual_seed(0)
    d, h, L = 16, 2, 5
    dk = d // h
    att = randomize_(RelPositionMultiHeadAttention(n_heads=h, d_model=d, prefix="L0")).eval()
    x = torch.randn(1, L, d)
    pos_emb = rel_pos_emb(rel_pos_table(64, d), L)[None]
    out = att(x, pos_emb, mask=None)

    q = att.linear_q(x).view(L, h, dk)
    k = att.linear_k(x).view(L, h, dk)
    v = att.linear_v(x).view(L, h, dk)
    p = att.linear_pos(pos_emb[0]).view(2 * L - 1, h, dk)
    ctx = torch.zeros(L, h, dk)
    for head in range(h):
        scores = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                rel = L - 1 - i + j  # row of P holding relative position i - j
                ac = (q[i, head] + att.pos_bias_u[head]) @ k[j, head]
                bd = (q[i, head] + att.pos_bias_v[head]) @ p[rel, head]
                scores[i, j] = (ac + bd) / math.sqrt(dk)
        ctx[:, head] = torch.softmax(scores, dim=-1) @ v[:, head]
    expected = att.linear_out(ctx.reshape(L, d))
    assert torch.allclose(out[0], expected, atol=1e-5)


def test_conv_module_is_pw1_glu_dw_bn_swish_pw2():
    torch.manual_seed(0)
    d = 8
    conv = randomize_(ConformerConvolution(d_model=d, kernel_size=9, prefix="L0")).eval()
    x = torch.randn(1, 12, d)
    y = conv(x, pad_mask=None)
    t = F.conv1d(x.transpose(1, 2), conv.pointwise_conv1.weight, conv.pointwise_conv1.bias)
    t = F.glu(t, dim=1)
    t = conv.batch_norm(conv.depthwise_conv(t))
    t = t * torch.sigmoid(t)
    t = F.conv1d(t, conv.pointwise_conv2.weight, conv.pointwise_conv2.bias)
    assert torch.allclose(y, t.transpose(1, 2), atol=1e-5)


def test_padded_batch_matches_individual_utterances():
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    t1, t2 = 70, 101
    f1 = torch.randn(1, 80, t1)
    f2 = torch.randn(1, 80, t2)
    batch = torch.zeros(2, 80, t2)
    batch[0, :, :t1] = f1[0]
    batch[1] = f2[0]
    with torch.no_grad():
        logits, lengths = model.forward_features(batch, torch.tensor([t1, t2]))
        alone1, len1 = model.forward_features(f1, torch.tensor([t1]))
        alone2, _ = model.forward_features(f2, torch.tensor([t2]))
    n1 = len1.item()
    assert lengths[0].item() == n1
    assert torch.allclose(logits[0, :n1], alone1[0], atol=1e-4)
    assert torch.allclose(logits[1], alone2[0], atol=1e-4)
