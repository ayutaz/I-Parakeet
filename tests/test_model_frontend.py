import librosa
import numpy as np
import pytest
import torch

from iparakeet.model.frontend import FilterbankFeatures, mel_filterbank


def test_mel_filterbank_matches_librosa_slaney():
    ours = mel_filterbank(sample_rate=16000, n_fft=512, n_mels=80)
    ref = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0.0, fmax=8000.0, norm="slaney")
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-7)


def _numpy_reference(audio: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Independent re-implementation of NeMo FilterbankFeatures (eval mode, per_feature)."""
    n_fft, hop, win = 512, 160, 400
    x = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]])
    window = np.zeros(n_fft)
    window[(n_fft - win) // 2 : (n_fft - win) // 2 + win] = np.hanning(win)  # symmetric Hann
    x = np.pad(x, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[t * hop : t * hop + n_fft] * window for t in range(n_frames)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (T, 257)
    logmel = np.log(fb @ power.T + 2.0**-24)  # (80, T)
    seq_len = len(audio) // hop
    valid = logmel[:, :seq_len]
    mean = valid.mean(axis=1, keepdims=True)
    std = valid.std(axis=1, ddof=1, keepdims=True) + 1e-5
    out = (logmel - mean) / std
    out[:, seq_len:] = 0.0
    return out


def test_features_match_independent_numpy_reference():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000 + 37).astype(np.float32) * 0.1
    featurizer = FilterbankFeatures()
    feats, lengths = featurizer(torch.from_numpy(audio)[None], torch.tensor([len(audio)]))
    ref = _numpy_reference(audio.astype(np.float64), mel_filterbank(16000, 512, 80).astype(np.float64))
    assert lengths.item() == len(audio) // 160
    assert feats.shape == (1, 80, ref.shape[1])
    np.testing.assert_allclose(feats[0].numpy(), ref, atol=2e-3)


def test_per_feature_normalization_on_valid_frames():
    audio = torch.randn(1, 24000) * 0.1
    feats, lengths = FilterbankFeatures()(audio, torch.tensor([24000]))
    valid = feats[0, :, : lengths.item()]
    assert torch.allclose(valid.mean(dim=1), torch.zeros(80), atol=1e-4)
    assert torch.allclose(valid.std(dim=1), torch.ones(80), atol=1e-3)
    assert torch.all(feats[0, :, lengths.item() :] == 0)


def test_batching_with_padding_does_not_change_valid_frames():
    a = torch.randn(12000) * 0.1
    b = torch.randn(20000) * 0.1
    featurizer = FilterbankFeatures()
    batch = torch.zeros(2, 20000)
    batch[0, :12000] = a
    batch[1] = b
    feats, lengths = featurizer(batch, torch.tensor([12000, 20000]))
    alone, alone_len = featurizer(a[None], torch.tensor([12000]))
    n = alone_len.item()
    assert lengths[0].item() == n
    assert torch.allclose(feats[0, :, :n], alone[0, :, :n], atol=1e-5)


def test_filterbank_buffers_use_nemo_names_and_shapes():
    featurizer = FilterbankFeatures()
    state = featurizer.state_dict()
    assert state["fb"].shape == (1, 80, 257)
    assert state["window"].shape == (400,)


@pytest.mark.parametrize("n_samples", [160, 16000, 16159])
def test_feature_length_is_samples_floor_div_hop(n_samples):
    feats, lengths = FilterbankFeatures()(torch.randn(1, n_samples), torch.tensor([n_samples]))
    assert lengths.item() == n_samples // 160
