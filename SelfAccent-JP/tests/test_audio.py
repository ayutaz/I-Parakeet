import numpy as np
import torch

from selfaccent.audio import MelConfig, MelExtractor, extract_f0, griffin_lim, hz_to_mel, mel_filterbank


def _tone(f0, sr=22050, seconds=1.0, harmonics=8):
    t = np.arange(int(sr * seconds)) / sr
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1)).astype(np.float32) * 0.3


def test_filterbank_shape_and_ordering():
    cfg = MelConfig()
    fb = mel_filterbank(cfg)
    assert fb.shape == (cfg.n_mels, cfg.n_fft // 2 + 1)
    assert (fb >= 0).all()
    peaks = fb.argmax(dim=1)
    assert (peaks[1:] >= peaks[:-1]).all()


def test_mel_frame_count_matches_hop():
    cfg = MelConfig()
    ext = MelExtractor(cfg)
    wav = torch.zeros(cfg.hop_length * 50)
    assert ext(wav).shape == (cfg.n_mels, 50)
    assert ext(wav.unsqueeze(0)).shape == (1, cfg.n_mels, 50)


def test_sine_energy_lands_in_matching_mel_bin():
    cfg = MelConfig()
    ext = MelExtractor(cfg)
    sr = cfg.sample_rate
    t = torch.arange(sr) / sr
    mel = ext(torch.sin(2 * torch.pi * 1000.0 * t))
    peak = int(mel.mean(dim=1).argmax())
    centers = np.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_mels + 2)[1:-1]
    expected = int(np.abs(centers - hz_to_mel(1000.0)).argmin())
    assert abs(peak - expected) <= 1


def test_griffin_lim_reconstruction_preserves_mel():
    cfg = MelConfig()
    ext = MelExtractor(cfg)
    wav = torch.from_numpy(_tone(150.0))
    mel = ext(wav)
    rec = griffin_lim(mel, cfg, n_iter=32)
    assert rec.ndim == 1 and abs(len(rec) - len(wav)) <= cfg.hop_length
    mel2 = ext(torch.from_numpy(rec))
    n = min(mel.shape[1], mel2.shape[1])
    assert (mel[:, :n] - mel2[:, :n]).abs().mean() < 0.5


def test_extract_f0_on_harmonic_tone():
    cfg = MelConfig()
    f0 = extract_f0(_tone(200.0), cfg)
    voiced = f0[f0 > 0]
    assert len(voiced) > 0.8 * len(f0)
    assert abs(np.median(voiced) - 200.0) < 5.0
    assert abs(len(f0) - 22050 // cfg.hop_length) <= 2
