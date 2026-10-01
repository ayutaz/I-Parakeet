import numpy as np
import pytest

from selfaccent.audio import MelConfig
from selfaccent.frontend import OracleRenderer, analyze
from selfaccent.metrics import (
    Utterance,
    build_reference_set,
    dtw_path,
    f0_distance,
    forced_choice,
    mel_distance,
    utterance_from_wav,
)
from selfaccent.tags import AccentPhrase


def _brute_dtw(a, b):
    ta, tb = len(a), len(b)
    c = np.linalg.norm(a[:, None] - b[None], axis=-1)
    d = np.full((ta + 1, tb + 1), np.inf)
    d[0, 0] = 0
    for i in range(1, ta + 1):
        for j in range(1, tb + 1):
            d[i, j] = c[i - 1, j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[ta, tb]


def test_dtw_identity_is_zero_cost_diagonal():
    a = np.random.default_rng(0).normal(size=(7, 3))
    cost, path = dtw_path(a, a)
    assert cost == pytest.approx(0.0)
    assert path == [(i, i) for i in range(7)]


@pytest.mark.parametrize("ta, tb", [(5, 5), (6, 9), (11, 4), (1, 3)])
def test_dtw_cost_matches_brute_force(ta, tb):
    rng = np.random.default_rng(ta * 10 + tb)
    a, b = rng.normal(size=(ta, 2)), rng.normal(size=(tb, 2))
    cost, path = dtw_path(a, b)
    assert cost == pytest.approx(_brute_dtw(a, b))
    assert path[0] == (0, 0) and path[-1] == (ta - 1, tb - 1)
    steps = {(i2 - i1, j2 - j1) for (i1, j1), (i2, j2) in zip(path, path[1:])}
    assert steps <= {(1, 0), (0, 1), (1, 1)}


def test_f0_distance_ignores_global_pitch_offset():
    t = np.linspace(0, 1, 50)
    mel = np.stack([np.sin(t * k) for k in range(1, 5)], axis=1)
    rise = Utterance(mel, np.log(150 + 50 * t))
    rise_shifted = Utterance(mel, np.log(150 + 50 * t) + 0.3)
    fall = Utterance(mel, np.log(200 - 50 * t))
    assert f0_distance(rise_shifted, rise) == pytest.approx(0.0, abs=1e-9)
    assert f0_distance(rise_shifted, fall) > 0.05


def test_forced_choice_picks_matching_contour():
    t = np.linspace(0, 1, 60)
    mel = np.stack([np.cos(t * k) for k in range(1, 5)], axis=1)
    cands = [Utterance(mel, np.log(150 + 40 * np.sin(np.pi * t * (k + 1)))) for k in range(3)]
    noisy = Utterance(mel, cands[1].logf0 + np.random.default_rng(0).normal(0, 0.01, 60))
    choice, dists = forced_choice(noisy, build_reference_set(cands))
    assert choice == 1
    assert len(dists) == 3


def test_unvoiced_frames_are_nan_and_skipped():
    t = np.linspace(0, 1, 20)
    mel = np.stack([t, t ** 2], axis=1)
    f0 = np.log(150 + 10 * t)
    f0[5:10] = np.nan
    u = Utterance(mel, f0)
    assert np.isfinite(f0_distance(u, u))


def test_oracle_renderings_are_classified_as_themselves():
    """Self-consistency of the automatic accent judge on HTS references."""
    cfg = MelConfig()
    oracle = OracleRenderer(cfg.sample_rate)
    text = "昨日、橋を見ました。"
    idx = next(w.index for w in analyze(text) if w.surface == "橋")
    refs = [
        utterance_from_wav(oracle.render_with_override(text, idx, [AccentPhrase(("ハ", "シ"), k)]), cfg)
        for k in range(3)
    ]
    refset = build_reference_set(refs)
    assert 0 < refset.region.sum() < len(refset.region)
    for k in range(3):
        choice, _ = forced_choice(refs[k], refset)
        assert choice == k


def test_mel_distance_is_small_for_same_sentence():
    cfg = MelConfig()
    oracle = OracleRenderer(cfg.sample_rate)
    a = utterance_from_wav(oracle.render("橋を渡る。"), cfg)
    b = utterance_from_wav(oracle.render("箸で食べる。"), cfg)
    assert mel_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert mel_distance(a, b) > 0.1


def test_region_covers_frames_where_references_disagree():
    t = np.linspace(0, 1, 40)
    mel = np.stack([np.sin(3 * t), np.cos(5 * t), t], axis=1)
    flat = np.log(150 + 0 * t)
    bump = flat.copy()
    bump[10:20] += 0.3
    refset = build_reference_set([Utterance(mel, flat), Utterance(mel, bump)])
    assert refset.region[10:20].all()
    assert not refset.region[25:].any()


def test_identical_references_fall_back_to_all_voiced_frames():
    t = np.linspace(0, 1, 30)
    mel = np.stack([np.sin(3 * t), t], axis=1)
    u = Utterance(mel, np.log(150 + 20 * t))
    refset = build_reference_set([u, u])
    assert refset.region.all()
