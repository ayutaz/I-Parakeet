"""Layer-wise activation range analysis and calibration statistics (paper Sec. 3.3, Fig. 2).

Statistics are collected in two passes over the calibration set: pass 1 records max|x| (and
per-channel max for selected tensors), pass 2 fills a fixed-bin histogram of |x| on [0, max|x|]
from which percentiles are read. The same statistics define the clipping ranges alpha_x of the
integer model (Eq. 6).
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from iparakeet.model.parakeet import collect_taps

FP16_MAX = 65504.0


class AbsHistogram:
    def __init__(self, max_value: float, n_bins: int = 4096) -> None:
        self.max_value = max(float(max_value), 1e-12)
        self.counts = torch.zeros(n_bins, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        values = x.detach().abs().flatten().double().clamp(max=self.max_value)
        self.counts += torch.histc(values, bins=len(self.counts), min=0.0, max=self.max_value)

    def percentile(self, q: float) -> float:
        return _hist_percentile(self.counts.tolist(), self.max_value, q)


def _hist_percentile(counts: list[float], max_value: float, q: float) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    width = max_value / len(counts)
    target = total * q / 100.0
    cum = 0.0
    for i, c in enumerate(counts):
        if cum + c >= target and c > 0:
            return i * width + (target - cum) / c * width
        cum += c
    return max_value


@dataclass
class TensorStats:
    absmax: float
    count: int
    hist: list[float]
    channel_absmax: list[float] | None = None

    def percentile(self, q: float) -> float:
        return _hist_percentile(self.hist, self.absmax, q)


@dataclass
class CalibrationStats:
    tensors: dict[str, TensorStats]

    def alpha(self, name: str, method: str) -> float:
        """Clipping range: 'minmax' = max|x|, 'pXX.X' = percentile of |x|."""
        stats = self.tensors[name]
        if method == "minmax":
            return stats.absmax
        if method.startswith("p"):
            return stats.percentile(float(method[1:]))
        raise ValueError(f"unknown calibration method {method!r}")

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: asdict(v) for k, v in self.tensors.items()}))

    @classmethod
    def load(cls, path) -> "CalibrationStats":
        raw = json.loads(Path(path).read_text())
        return cls({k: TensorStats(**v) for k, v in raw.items()})


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(name.endswith(p) for p in patterns)


@torch.no_grad()
def calibrate(
    model: torch.nn.Module,
    inputs: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    n_bins: int = 4096,
    per_channel: tuple[str, ...] = (),
) -> CalibrationStats:
    """Two-pass statistics of every tap over (features, lengths) batches from `inputs()`."""
    absmax: dict[str, float] = {}
    counts: dict[str, int] = {}
    channel: dict[str, torch.Tensor] = {}

    def pass1(name, x):
        absmax[name] = max(absmax.get(name, 0.0), float(x.abs().max()))
        counts[name] = counts.get(name, 0) + x.numel()
        if _matches(name, per_channel):
            cmax = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0)
            channel[name] = torch.maximum(channel[name], cmax) if name in channel else cmax

    with collect_taps(model, pass1):
        for feats, lengths in inputs():
            model.forward_features(feats, lengths)

    hists = {name: AbsHistogram(value, n_bins) for name, value in absmax.items()}
    with collect_taps(model, lambda name, x: hists[name].update(x)):
        for feats, lengths in inputs():
            model.forward_features(feats, lengths)

    return CalibrationStats(
        {
            name: TensorStats(
                absmax=absmax[name],
                count=counts[name],
                hist=hists[name].counts.tolist(),
                channel_absmax=channel[name].tolist() if name in channel else None,
            )
            for name in absmax
        }
    )


def bn_scale_spread(model: torch.nn.Module) -> list[float]:
    """Per-layer max/median of the BatchNorm scale |gamma_c| / sqrt(var_c + eps) (Fig. 2a)."""
    spread = []
    for layer in model.encoder.layers:
        bn = layer.conv.batch_norm
        scale = (bn.weight.abs() / torch.sqrt(bn.running_var + bn.eps)).detach()
        spread.append(float(scale.max() / scale.median()))
    return spread


def max_over_percentile(stats: CalibrationStats, q: float = 99.9) -> dict[str, dict[str, float]]:
    """Ratio max|x| / p_q(|x|) per tensor, split into pre-encoder and encoder (Fig. 2b)."""
    groups: dict[str, dict[str, float]] = {"pre": {}, "encoder": {}}
    for name, t in stats.tensors.items():
        p = t.percentile(q)
        if p <= 0:
            continue
        groups["pre" if name.startswith("pre.") else "encoder"][name] = t.absmax / p
    return groups


def fp16_overflow(stats: CalibrationStats) -> list[str]:
    return [name for name, t in stats.tensors.items() if t.absmax > FP16_MAX]
