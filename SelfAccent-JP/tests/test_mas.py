import itertools

import numpy as np
import pytest
import torch

from selfaccent.backbones.mas import maximum_path, monotonic_alignment


def _brute(value):
    tx, ty = value.shape
    best, best_d = -np.inf, None
    # durations: positive ints summing to ty
    for cuts in itertools.combinations(range(1, ty), tx - 1):
        d = np.diff((0, *cuts, ty))
        x = np.repeat(np.arange(tx), d)
        s = value[x, np.arange(ty)].sum()
        if s > best:
            best, best_d = s, d
    return best, best_d


@pytest.mark.parametrize("tx, ty", [(1, 4), (2, 5), (3, 6), (4, 7), (3, 3)])
def test_maximum_path_matches_brute_force(tx, ty):
    rng = np.random.default_rng(tx * 100 + ty)
    value = rng.normal(size=(tx, ty))
    path = maximum_path(value)
    assert path.shape == (tx, ty)
    assert (path.sum(0) == 1).all()
    durations = path.sum(1)
    assert (durations >= 1).all()
    best, best_d = _brute(value)
    assert value[path.astype(bool)].sum() == pytest.approx(best)
    np.testing.assert_array_equal(durations, best_d)


def test_batched_alignment_respects_lengths():
    torch.manual_seed(0)
    log_p = torch.randn(2, 4, 9)
    x_len = torch.tensor([4, 2])
    y_len = torch.tensor([9, 5])
    attn = monotonic_alignment(log_p, x_len, y_len)
    assert attn.shape == (2, 4, 9)
    assert attn[1, 2:].sum() == 0 and attn[1, :, 5:].sum() == 0
    assert torch.equal(attn.sum(1)[0], torch.ones(9))
    assert attn[1].sum() == 5
    assert (attn[0].sum(1) >= 1).all()
