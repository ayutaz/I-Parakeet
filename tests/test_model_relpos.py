import math

import pytest
import torch
import torch.nn.functional as F

from iparakeet.model.relpos import rel_pos_emb, rel_pos_table, rel_shift, relshift_index


def nemo_rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Reference: NeMo RelPositionMultiHeadAttention.rel_shift followed by the key-length slice."""
    b, h, qlen, pos_len = x.size()
    x = F.pad(x, pad=(1, 0))
    x = x.view(b, h, -1, qlen)
    x = x[:, :, 1:].view(b, h, qlen, pos_len)
    return x[..., :qlen]


@pytest.mark.parametrize("L", [1, 2, 5, 17, 64])
def test_gather_rel_shift_matches_nemo_pad_reshape(L):
    x = torch.randn(2, 3, L, 2 * L - 1)
    assert torch.equal(rel_shift(x), nemo_rel_shift(x))


def test_rel_shift_is_arithmetic_free_on_integers():
    x = torch.randint(-128, 128, (1, 2, 7, 13), dtype=torch.int32)
    out = rel_shift(x)
    assert out.dtype == torch.int32
    assert torch.equal(out, nemo_rel_shift(x))


def test_relshift_index_formula():
    L = 6
    idx = relshift_index(L)
    for i in range(L):
        for j in range(L):
            assert idx[i, j] == L - 1 - i + j


def test_rel_pos_table_rows_encode_descending_positions():
    max_len, d = 10, 8
    table = rel_pos_table(max_len, d)
    assert table.shape == (2 * max_len - 1, d)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    for row in (0, max_len - 1, 2 * max_len - 2):
        pos = max_len - 1 - row
        assert torch.allclose(table[row, 0::2], torch.sin(pos * div))
        assert torch.allclose(table[row, 1::2], torch.cos(pos * div))


def test_rel_pos_emb_is_centered_slice_for_any_length():
    max_len, d = 40, 16
    table = rel_pos_table(max_len, d)
    for L in (1, 3, 40):
        emb = rel_pos_emb(table, L)
        assert emb.shape == (2 * L - 1, d)
        direct = rel_pos_table(L, d)  # positions L-1 .. -(L-1)
        assert torch.allclose(emb, direct, atol=1e-6)


def test_projected_positions_for_short_length_are_slice_of_longest():
    # P = linear_pos(pos_emb) depends on L only through slicing, so one constant can serve every bucket.
    max_len, d, L = 30, 16, 7
    table = rel_pos_table(max_len, d)
    lin = torch.nn.Linear(d, d, bias=False)
    p_long = lin(rel_pos_emb(table, max_len))
    p_short = lin(rel_pos_emb(table, L))
    assert torch.allclose(p_short, p_long[max_len - L : max_len + L - 1], atol=1e-6)
