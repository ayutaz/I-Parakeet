import math

import pytest
import torch

from conftest import randomize_, tiny_config
from iparakeet.analysis.range import calibrate
from iparakeet.model.parakeet import ParakeetCTC, collect_taps
from iparakeet.quant.fixed_point import int32_check
from iparakeet.quant.nofloat import NoFloatMode
from iparakeet.sim.int_parakeet import IntParakeet
from iparakeet.sim.recipe import PAPER_TABLE2, PAPER_TABLE3, RECIPES, Recipe

MAX_FRAMES = 120


def _inputs(n=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    lengths = [60, 90, 120, 75][:n]
    return [(torch.randn(1, 80, t, generator=gen), torch.tensor([t])) for t in lengths]


@pytest.fixture(scope="module")
def fp_and_stats():
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    inputs = _inputs()
    stats = calibrate(model, lambda: iter(inputs), n_bins=512, per_channel=("conv.dw",))
    return model, stats


def sqnr_db(ref, test):
    ref, test = ref.detach().double(), test.detach().double()
    return 10 * math.log10(float((ref**2).sum()) / max(float(((ref - test) ** 2).sum()), 1e-30))


def test_recipes_cover_paper_tables():
    assert set(PAPER_TABLE2) == {"ibert_recipe", "naive_int8", "iparakeet"}
    assert len(PAPER_TABLE3) == 11
    assert [r.group for r in PAPER_TABLE3].count("swish") == 5
    assert sum(r.recipe == "iparakeet" for r in PAPER_TABLE3) == 3
    for name in list(PAPER_TABLE2) + [r.recipe for r in PAPER_TABLE3]:
        assert name in RECIPES
    assert PAPER_TABLE2["iparakeet"] == (2.61, 5.32, 14.70)
    ip = RECIPES["iparakeet"]
    assert (ip.bn_bits, ip.pre_calib, ip.calib, ip.swish) == (16, "p99.9", "minmax", "poly:linf_swish")
    assert RECIPES["ibert_recipe"].swish == "poly:l2_tanh" and RECIPES["ibert_recipe"].bn_bits == 8
    assert RECIPES["naive_int8"].pre_calib == "minmax" and RECIPES["naive_int8"].bn_bits == 8


def test_forward_is_integer_only(fp_and_stats):
    model, stats = fp_and_stats
    sim = IntParakeet(model, stats, RECIPES["iparakeet"], MAX_FRAMES)
    feats, _ = _inputs(1)[0]
    q_in = sim.quantize_input(feats)
    with NoFloatMode(), int32_check():
        acc = sim.forward_int(q_in)
    assert acc.dtype == torch.int64
    assert acc.shape == (1, 8, model.cfg.vocab_size + 1)


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_every_recipe_builds_and_runs(fp_and_stats, name):
    model, stats = fp_and_stats
    sim = IntParakeet(model, stats, RECIPES[name], MAX_FRAMES)
    feats, _ = _inputs(1)[0]
    q_in = sim.quantize_input(feats)  # model boundary: float features -> INT grid
    with NoFloatMode():
        ids = sim.forward_int(q_in).argmax(-1)
    assert ids.shape == (1, 8)


def test_near_lossless_recipe_tracks_fp32(fp_and_stats):
    model, stats = fp_and_stats
    sim = IntParakeet(model, stats, RECIPES["lossless_int16"], MAX_FRAMES)
    for feats, lengths in _inputs():
        with torch.no_grad():
            ref, _ = model.forward_features(feats, lengths)
        got = sim.logits(feats)
        assert sqnr_db(ref, got) > 25
        assert (ref.argmax(-1) == got.argmax(-1)).double().mean() > 0.9


def test_trace_reports_every_quantized_tensor(fp_and_stats):
    model, stats = fp_and_stats
    fp_names = set()
    with collect_taps(model, lambda n, x: fp_names.add(n)), torch.no_grad():
        model.forward_features(*_inputs(1)[0])
    sim = IntParakeet(model, stats, RECIPES["iparakeet"], MAX_FRAMES)
    traced = {}
    sim.forward_int(sim.quantize_input(_inputs(1)[0][0]), trace=lambda n, q: traced.setdefault(n, q))
    missing = fp_names - set(traced) - {"head.logits", "L0.att.p", "L1.att.p", "L0.conv.gate", "L1.conv.gate"}
    assert not missing, sorted(missing)
    for name, q in traced.items():
        assert q.dtype == torch.int64, name


def test_int16_bn_output_preserves_ordinary_channels_under_extreme_bn_spread():
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    with torch.no_grad():
        model.encoder.layers[0].conv.batch_norm.weight[0] = 2000.0  # Fig. 2a: max/median ~ 10^3-10^4
    inputs = _inputs()
    stats = calibrate(model, lambda: iter(inputs), n_bins=512, per_channel=("conv.dw",))
    fp = {}
    with collect_taps(model, lambda n, x: fp.__setitem__(n, x)), torch.no_grad():
        model.forward_features(*inputs[0])
    result = {}
    for name in ("iparakeet", "bn_int8"):
        sim = IntParakeet(model, stats, RECIPES[name], MAX_FRAMES)
        traced = {}
        sim.forward_int(sim.quantize_input(inputs[0][0]), trace=lambda n, q: traced.setdefault(n, q))
        deq = sim.dequantize("L0.conv.dw", traced["L0.conv.dw"])
        result[name] = sqnr_db(fp["L0.conv.dw"][..., 1:], deq[..., 1:])
    assert result["iparakeet"] > result["bn_int8"] + 10


def test_multiplier_report_and_shift_sensitivity(fp_and_stats):
    model, stats = fp_and_stats
    err16 = IntParakeet(model, stats, RECIPES["iparakeet"], MAX_FRAMES).multiplier_report()["max_rel_error"]
    wide = Recipe(**{**RECIPES["iparakeet"].__dict__, "name": "n24", "requant_shift": 24})
    err24 = IntParakeet(model, stats, wide, MAX_FRAMES).multiplier_report()["max_rel_error"]
    assert err24 < err16


def test_one_build_serves_all_input_lengths(fp_and_stats):
    model, stats = fp_and_stats
    sim = IntParakeet(model, stats, RECIPES["iparakeet"], MAX_FRAMES)
    for t in (9, 64, MAX_FRAMES):
        assert sim.forward_int(sim.quantize_input(torch.randn(1, 80, t))).shape[1] == (((t - 1) // 2) // 2) // 2 + 1
    with pytest.raises(ValueError):
        sim.forward_int(sim.quantize_input(torch.randn(1, 80, MAX_FRAMES + 16)))
