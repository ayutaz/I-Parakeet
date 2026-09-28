import numpy as np
import pytest
import torch

from conftest import randomize_, tiny_config
from iparakeet.analysis.range import (
    AbsHistogram,
    CalibrationStats,
    bn_scale_spread,
    calibrate,
    fp16_overflow,
    max_over_percentile,
)
from iparakeet.model.parakeet import ParakeetCTC, collect_taps


@pytest.fixture
def model():
    torch.manual_seed(0)
    return randomize_(ParakeetCTC(tiny_config())).eval()


def _inputs(n=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, 80, 40 + 17 * i, generator=gen), torch.tensor([40 + 17 * i])) for i in range(n)]


def test_bn_scale_spread_is_max_over_median_per_layer(model):
    bn = model.encoder.layers[0].conv.batch_norm
    with torch.no_grad():
        bn.running_var.fill_(1.0 - bn.eps)
        bn.weight.fill_(1.0)
        bn.weight[0] = 1000.0
        bn.weight[1] = -2000.0  # magnitude matters
    spread = bn_scale_spread(model)
    assert len(spread) == 2
    assert spread[0] == pytest.approx(2000.0, rel=1e-4)


def test_histogram_percentile_matches_numpy_on_heavy_tail():
    rng = np.random.default_rng(0)
    chunks = [rng.standard_t(df=2, size=5000) for _ in range(4)]
    absmax = max(np.abs(c).max() for c in chunks)
    hist = AbsHistogram(absmax, n_bins=8192)
    for c in chunks:
        hist.update(torch.from_numpy(c))
    expected = np.percentile(np.abs(np.concatenate(chunks)), 99.9)
    assert abs(hist.percentile(99.9) - expected) <= absmax / 8192 * 2


def test_calibrate_collects_every_tap_with_exact_max(model):
    inputs = _inputs()
    seen: dict[str, float] = {}

    def record(name, x):
        seen[name] = max(seen.get(name, 0.0), float(x.abs().max()))

    with collect_taps(model, record), torch.no_grad():
        for feats, lengths in inputs:
            model.forward_features(feats, lengths)
    stats = calibrate(model, lambda: iter(inputs), n_bins=512)
    assert set(stats.tensors) == set(seen)
    for name, value in seen.items():
        assert stats.tensors[name].absmax == pytest.approx(value, rel=1e-6)
        assert stats.alpha(name, "p99.9") <= stats.alpha(name, "minmax") + 1e-6


def test_calibrate_tracks_per_channel_max_for_bn_output(model):
    inputs = _inputs()
    stats = calibrate(model, lambda: iter(inputs), n_bins=64, per_channel=("conv.dw",))
    channel = stats.tensors["L0.conv.dw"].channel_absmax
    assert len(channel) == 32
    assert max(channel) == pytest.approx(stats.tensors["L0.conv.dw"].absmax, rel=1e-6)
    assert stats.tensors["L0.ff1.lin1"].channel_absmax is None


def test_max_over_percentile_splits_pre_encoder_and_encoder(model):
    stats = calibrate(model, lambda: iter(_inputs()), n_bins=256)
    ratios = max_over_percentile(stats, 99.9)
    assert set(ratios) == {"pre", "encoder"}
    assert "pre.conv0" in ratios["pre"]
    assert "L1.conv.dw" in ratios["encoder"]
    assert all(r >= 1.0 for group in ratios.values() for r in group.values())


def test_fp16_overflow_flags_tensors_beyond_fp16_range(model):
    inputs = [(feats * 1e6, lengths) for feats, lengths in _inputs(1)]
    stats = calibrate(model, lambda: iter(inputs), n_bins=16)
    flagged = fp16_overflow(stats)
    assert "pre.in" in flagged
    assert all(stats.tensors[n].absmax > 65504 for n in flagged)


def test_stats_json_roundtrip(model, tmp_path):
    stats = calibrate(model, lambda: iter(_inputs(2)), n_bins=32, per_channel=("conv.dw",))
    stats.save(tmp_path / "stats.json")
    loaded = CalibrationStats.load(tmp_path / "stats.json")
    assert loaded.tensors == stats.tensors
    assert loaded.alpha("L0.res1", "p99.9") == stats.alpha("L0.res1", "p99.9")
