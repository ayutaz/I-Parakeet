"""Static-shape input-length buckets with silence padding (paper Sec. 4.1, after NPUsper)."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BucketSet:
    seconds: list[float]
    hop_seconds: float = 0.01

    @property
    def names(self) -> list[str]:
        return [f"b{s:g}s" for s in self.seconds]

    @property
    def frames(self) -> list[int]:
        return [round(s / self.hop_seconds) for s in self.seconds]

    def route(self, n_frames: int) -> int:
        for i, frames in enumerate(self.frames):
            if n_frames <= frames:
                return i
        raise ValueError(f"{n_frames} frames exceed the largest bucket ({self.frames[-1]} frames)")

    def padded_frames(self, n_frames: int) -> int:
        return self.frames[self.route(n_frames)]


def duration_to_frames(seconds: float, hop_seconds: float = 0.01) -> int:
    return math.floor(seconds / hop_seconds + 1e-9)


def pad_features(feats: torch.Tensor, target_frames: int, silence: torch.Tensor | None = None) -> torch.Tensor:
    """Append silence feature frames to (n_mels, T) features up to target_frames."""
    n_mels, t = feats.shape
    if t > target_frames:
        raise ValueError(f"{t} frames do not fit into {target_frames}")
    fill = torch.zeros(n_mels, dtype=feats.dtype) if silence is None else silence.to(feats.dtype)
    pad = fill.unsqueeze(1).expand(n_mels, target_frames - t)
    return torch.cat([feats, pad], dim=1)


def padding_ratio(durations: list[float], buckets: BucketSet) -> float:
    real = padded = 0
    for d in durations:
        n = duration_to_frames(d, buckets.hop_seconds)
        real += n
        padded += buckets.padded_frames(n)
    return (padded - real) / real if real else 0.0


def uniform_buckets(lo: float, hi: float, step: float) -> BucketSet:
    seconds, s = [], lo
    while s < hi - 1e-9:
        seconds.append(round(s, 6))
        s += step
    seconds.append(hi)
    return BucketSet(seconds)


def search_uniform_step(
    durations: list[float], target_ratio: float, steps: list[float], lo: float = 3.0, hi: float = 35.0
) -> tuple[float, float]:
    """Uniform bucket step whose padding overhead is closest to target_ratio."""
    best = None
    for step in steps:
        ratio = padding_ratio(durations, uniform_buckets(lo, hi, step))
        if best is None or abs(ratio - target_ratio) < abs(best[1] - target_ratio):
            best = (step, ratio)
    return best
