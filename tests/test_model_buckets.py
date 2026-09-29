import pytest
import torch

from iparakeet.model.buckets import BucketSet, pad_features, padding_ratio, search_uniform_step, uniform_buckets


def test_route_picks_smallest_bucket_that_fits():
    buckets = BucketSet([3.0, 5.0, 10.0])
    assert buckets.frames == [300, 500, 1000]
    assert buckets.route(1) == 0
    assert buckets.route(300) == 0
    assert buckets.route(301) == 1
    assert buckets.route(1000) == 2


def test_route_rejects_utterances_longer_than_largest_bucket():
    with pytest.raises(ValueError):
        BucketSet([3.0]).route(301)


def test_pad_features_appends_silence_frames():
    feats = torch.ones(80, 5)
    silence = torch.full((80,), -2.0)
    out = pad_features(feats, 8, silence)
    assert out.shape == (80, 8)
    assert torch.all(out[:, :5] == 1.0)
    assert torch.all(out[:, 5:] == -2.0)


def test_padding_ratio_counts_extra_frames_over_real_frames():
    buckets = BucketSet([3.0, 5.0])
    # 2.0 s -> padded to 3.0 s, 4.0 s -> padded to 5.0 s: (1 + 1) / (2 + 4)
    assert padding_ratio([2.0, 4.0], buckets) == pytest.approx(2.0 / 6.0)


def test_uniform_buckets_cover_range_inclusively():
    assert uniform_buckets(3.0, 35.0, 8.0).seconds == [3.0, 11.0, 19.0, 27.0, 35.0]
    assert uniform_buckets(3.0, 35.0, 10.0).seconds == [3.0, 13.0, 23.0, 33.0, 35.0]


def test_search_uniform_step_returns_step_closest_to_target_ratio():
    durations = [float(d) for d in range(3, 36)]
    step, ratio = search_uniform_step(durations, target_ratio=0.0, steps=[1.0, 2.0, 4.0])
    assert step == 1.0 and ratio == pytest.approx(0.0)
    step, ratio = search_uniform_step(durations, target_ratio=1.0, steps=[1.0, 2.0, 4.0])
    assert step == 4.0


def test_bucket_names_are_stable_identifiers():
    assert BucketSet([3.0, 10.0, 0.5]).names == ["b3s", "b10s", "b0.5s"]
