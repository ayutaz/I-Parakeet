"""Log-mel features, Griffin-Lim inversion and F0 extraction.

The mel settings follow HiFi-GAN / Matcha-TTS (22.05 kHz, 1024/256, 80 bins,
Slaney mel), so a pretrained vocoder for those settings can replace
Griffin-Lim without changing the backbone.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch

with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
    warnings.simplefilter("ignore")
    import pyworld


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    log_floor: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


def hz_to_mel(f):
    """Slaney mel scale (linear below 1 kHz, logarithmic above)."""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    mel = f / f_sp
    return np.where(f >= min_log_hz, min_log_hz / f_sp + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mel)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(cfg: MelConfig) -> torch.Tensor:
    freqs = np.linspace(0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_mels + 2))
    fb = np.zeros((cfg.n_mels, len(freqs)))
    for i in range(cfg.n_mels):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return torch.tensor(fb, dtype=torch.float32)


class MelExtractor:
    def __init__(self, cfg: MelConfig = MelConfig()) -> None:
        self.cfg = cfg
        self.fb = mel_filterbank(cfg)
        self.window = torch.hann_window(cfg.win_length)

    def magnitude(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        pad = (cfg.n_fft - cfg.hop_length) // 2
        x = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            x, cfg.n_fft, cfg.hop_length, cfg.win_length, self.window, center=False, return_complex=True
        )
        return spec.abs()

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.unsqueeze(0)
        mag = self.magnitude(wav.float())
        mel = torch.log(torch.clamp(self.fb @ mag, min=self.cfg.log_floor))
        return mel[0] if squeeze else mel


def griffin_lim(log_mel: torch.Tensor, cfg: MelConfig = MelConfig(), n_iter: int = 32, power: float = 1.2) -> np.ndarray:
    """Waveform from a log-mel spectrogram ([n_mels, frames]) for quick listening and F0 analysis."""
    fb = mel_filterbank(cfg)
    mel = torch.exp(log_mel.detach().float())
    mag = torch.clamp(torch.linalg.pinv(fb) @ mel, min=0.0) ** power
    window = torch.hann_window(cfg.win_length)
    length = mel.shape[-1] * cfg.hop_length
    stft = lambda y: torch.stft(y, cfg.n_fft, cfg.hop_length, cfg.win_length, window, center=True, return_complex=True)
    istft = lambda s: torch.istft(s, cfg.n_fft, cfg.hop_length, cfg.win_length, window, center=True, length=length)
    gen = torch.Generator().manual_seed(0)
    angles = torch.exp(2j * torch.pi * torch.rand(mag.shape, generator=gen))
    y = istft(mag * angles)
    for _ in range(n_iter):
        s = stft(y)[..., : mag.shape[-1]]
        if s.shape[-1] < mag.shape[-1]:
            s = torch.nn.functional.pad(s, (0, mag.shape[-1] - s.shape[-1]))
        y = istft(mag * torch.exp(1j * torch.angle(s)))
    y = y / max(1e-6, float(y.abs().max())) * 0.9
    return y.numpy().astype(np.float32)


def extract_f0(wav: np.ndarray, cfg: MelConfig = MelConfig(), f0_floor: float = 60.0, f0_ceil: float = 600.0) -> np.ndarray:
    """F0 (Hz, 0 for unvoiced) on the mel frame grid, using WORLD Harvest."""
    x = np.asarray(wav, dtype=np.float64)
    period_ms = 1000.0 * cfg.hop_length / cfg.sample_rate
    f0, _ = pyworld.harvest(x, cfg.sample_rate, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=period_ms)
    return f0[: len(x) // cfg.hop_length + 1]
