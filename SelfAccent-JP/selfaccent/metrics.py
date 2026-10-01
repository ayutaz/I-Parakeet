"""Automatic judges for reading and pitch accent.

The paper scores accent with screened native listeners. For a CPU-only toy
setting we replace the listener with a forced choice: the generated speech is
compared with oracle (HTS) renderings of the same sentence for every accent
type of the word, and the closest log-F0 contour (after DTW alignment on mel)
is taken as the perceived accent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from selfaccent.audio import MelConfig, MelExtractor, extract_f0, griffin_lim


@dataclass
class Utterance:
    mel: np.ndarray  # [frames, n_mels] log-mel
    logf0: np.ndarray  # [frames] log F0, NaN where unvoiced


def dtw_path(a: np.ndarray, b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Classic DTW (steps (1,0), (0,1), (1,1)) with Euclidean frame cost.

    The accumulated cost is computed one anti-diagonal at a time so that the
    inner loop is vectorized.
    """
    ta, tb = len(a), len(b)
    c = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    d = np.full((ta + 1, tb + 1), np.inf)
    d[0, 0] = 0.0
    for s in range(2, ta + tb + 1):
        i = np.arange(max(1, s - tb), min(ta, s - 1) + 1)
        j = s - i
        best = np.minimum(np.minimum(d[i - 1, j], d[i, j - 1]), d[i - 1, j - 1])
        d[i, j] = c[i - 1, j - 1] + best
    path = [(ta - 1, tb - 1)]
    i, j = ta, tb
    while (i, j) != (1, 1):
        moves = [(d[i - 1, j - 1], i - 1, j - 1), (d[i - 1, j], i - 1, j), (d[i, j - 1], i, j - 1)]
        _, i, j = min(moves, key=lambda m: m[0])
        path.append((i - 1, j - 1))
    return float(d[ta, tb]), path[::-1]


def _zscore(mel: np.ndarray) -> np.ndarray:
    return (mel - mel.mean(0)) / (mel.std(0) + 1e-5)


def _path(gen: Utterance, ref: Utterance) -> list[tuple[int, int]]:
    return dtw_path(_zscore(gen.mel), _zscore(ref.mel))[1]


def f0_distance(gen: Utterance, ref: Utterance, path: list[tuple[int, int]] | None = None) -> float:
    """Mean |Δ log F0| on DTW-aligned voiced frames, ignoring a global pitch offset."""
    path = path if path is not None else _path(gen, ref)
    g = np.array([gen.logf0[i] for i, _ in path])
    r = np.array([ref.logf0[j] for _, j in path])
    ok = np.isfinite(g) & np.isfinite(r)
    if ok.sum() < 3:
        return float("inf")
    g, r = g[ok], r[ok]
    return float(np.abs((g - g.mean()) - (r - r.mean())).mean())


def mel_distance(gen: Utterance, ref: Utterance) -> float:
    """Mean absolute log-mel difference along the DTW path (lower = closer reading)."""
    _, path = dtw_path(gen.mel, ref.mel)
    return float(np.mean([np.abs(gen.mel[i] - ref.mel[j]).mean() for i, j in path]))


def forced_choice(gen: Utterance, candidates: list[Utterance]) -> tuple[int, list[float]]:
    dists = [f0_distance(gen, c) for c in candidates]
    return int(np.argmin(dists)), dists


def utterance_from_wav(wav: np.ndarray, cfg: MelConfig = MelConfig(), extractor: MelExtractor | None = None) -> Utterance:
    ext = extractor or MelExtractor(cfg)
    mel = ext(torch.from_numpy(np.asarray(wav, dtype=np.float32))).T.numpy()
    f0 = extract_f0(wav, cfg)
    n = min(len(mel), len(f0))
    logf0 = np.where(f0[:n] > 0, np.log(np.maximum(f0[:n], 1e-5)), np.nan)
    return Utterance(mel[:n], logf0)


def utterance_from_mel(mel: torch.Tensor, cfg: MelConfig = MelConfig(), gl_iters: int = 32) -> Utterance:
    """Generated log-mel ([n_mels, frames]) -> Utterance, via Griffin-Lim for F0."""
    wav = griffin_lim(mel, cfg, n_iter=gl_iters)
    f0 = extract_f0(wav, cfg)
    m = mel.detach().T.numpy()
    n = min(len(m), len(f0))
    logf0 = np.where(f0[:n] > 0, np.log(np.maximum(f0[:n], 1e-5)), np.nan)
    return Utterance(m[:n], logf0)
