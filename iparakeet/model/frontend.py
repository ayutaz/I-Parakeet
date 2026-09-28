"""NeMo-compatible log-mel front end (AudioToMelSpectrogramPreprocessor, eval mode)."""

import numpy as np
import torch
from torch import nn

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asanyarray(f, dtype=np.float64)
    mel = f / _F_SP
    log_region = f >= _MIN_LOG_HZ
    mel = np.where(log_region, _MIN_LOG_MEL + np.log(np.maximum(f, 1e-10) / _MIN_LOG_HZ) / _LOGSTEP, mel)
    return mel


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asanyarray(m, dtype=np.float64)
    hz = _F_SP * m
    return np.where(m >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), hz)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Slaney-style mel filterbank with Slaney area normalization (librosa defaults)."""
    fmax = sample_rate / 2.0 if fmax is None else fmax
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


class FilterbankFeatures(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        preemph: float = 0.97,
        log_zero_guard_value: float = 2.0**-24,
    ) -> None:
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.preemph = preemph
        self.log_zero_guard_value = log_zero_guard_value
        self.register_buffer("window", torch.hann_window(win_length, periodic=False))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(sample_rate, n_fft, n_mels)).unsqueeze(0))

    def get_seq_len(self, lengths: torch.Tensor) -> torch.Tensor:
        pad_amount = self.n_fft // 2 * 2
        return torch.div(lengths + pad_amount - self.n_fft, self.hop_length, rounding_mode="floor").long()

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.get_seq_len(lengths)
        x = audio.float()
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(torch.float32),
            return_complex=True,
            pad_mode="constant",
        )
        mag = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1))
        power = mag.pow(2.0)
        mel = torch.matmul(self.fb.to(power.dtype), power)
        logmel = torch.log(mel + self.log_zero_guard_value)
        feats = _normalize_per_feature(logmel, seq_len)
        mask = torch.arange(feats.shape[-1], device=feats.device).unsqueeze(0) >= seq_len.unsqueeze(1)
        return feats.masked_fill(mask.unsqueeze(1), 0.0), seq_len


def _normalize_per_feature(x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """Per-utterance, per-mel-bin mean/std normalization over valid frames (NeMo normalize_batch)."""
    batch, _, max_time = x.shape
    valid = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch, max_time) < seq_len.unsqueeze(1)
    count = valid.sum(dim=1)
    reference = x[:, :, 0]
    centered = torch.where(valid.unsqueeze(1), x - reference.unsqueeze(2), 0.0)
    mean = reference + centered.sum(dim=2) / count.clamp_min(1).unsqueeze(1)
    sq = torch.where(valid.unsqueeze(1), x - mean.unsqueeze(2), 0.0) ** 2
    std = torch.sqrt(sq.sum(dim=2) / (count.unsqueeze(1) - 1.0))
    std = std.masked_fill(std.isnan(), 0.0) + 1e-5
    return (x - mean.unsqueeze(2)) / std.unsqueeze(2)
