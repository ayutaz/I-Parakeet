"""Automatic judges for reading and pitch accent.

The paper scores accent with screened native listeners. For a CPU-only toy
setting we replace the listener with a forced choice: the generated speech is
compared with oracle (HTS) renderings of the same sentence for every accent
type of the word, and the candidate with the closest log-F0 contour is taken
as the perceived accent. Contours are compared only on frames where the
candidates differ from each other (the word, the following particle and the
downstep it causes), after DTW alignment on mel and removal of each
utterance's median pitch. Validation of the judge itself: docs/04_benchmark.md.
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


@dataclass
class ReferenceSet:
    """Oracle candidates mapped onto the frames of the first one.

    ``values[k, t]`` is the log-F0 of candidate k at frame t of candidate 0
    (DTW on mel), ``region`` marks frames where the candidates disagree after
    removing each one's median pitch. Only those frames carry information about
    the accent, so the judge compares contours there.
    """

    refs: list[Utterance]
    values: np.ndarray  # [K, T0]
    voiced: np.ndarray  # [T0]
    region: np.ndarray  # [T0]


def _project(src: Utterance, anchor: Utterance) -> np.ndarray:
    """Mean log-F0 of ``src`` frames aligned to each frame of ``anchor``."""
    _, path = dtw_path(_zscore(anchor.mel), _zscore(src.mel))
    acc = np.zeros(len(anchor.mel))
    cnt = np.zeros(len(anchor.mel))
    for i, j in path:
        if np.isfinite(src.logf0[j]):
            acc[i] += src.logf0[j]
            cnt[i] += 1
    out = np.full(len(anchor.mel), np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out


def build_reference_set(refs: list[Utterance], spread_threshold: float = 0.03) -> ReferenceSet:
    values = np.stack([_project(r, refs[0]) for r in refs])
    voiced = np.isfinite(values).all(0)
    centered = values - np.array([np.median(v[voiced]) if voiced.any() else 0.0 for v in values])[:, None]
    spread = np.where(voiced, np.nanmax(centered, 0) - np.nanmin(centered, 0), 0.0)
    region = (spread > spread_threshold) & voiced
    if region.sum() < 2:
        region = voiced.copy()
    return ReferenceSet(list(refs), values, voiced, region)


def reference_distances(gen: Utterance, refset: ReferenceSet) -> list[float]:
    g = _project(gen, refset.refs[0])
    ok = np.isfinite(g) & refset.voiced
    sel = ok & refset.region
    if sel.sum() < 2:
        return [float("inf")] * len(refset.refs)
    g = g - np.median(g[ok])
    out = []
    for v in refset.values:
        r = v - np.median(v[ok])
        out.append(float(np.abs(g[sel] - r[sel]).mean()))
    return out


def forced_choice(gen: Utterance, refset: ReferenceSet) -> tuple[int, list[float]]:
    """Index of the candidate whose pitch contour is closest to ``gen``."""
    dists = reference_distances(gen, refset)
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
