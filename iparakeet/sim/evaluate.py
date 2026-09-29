"""Evaluate one quantization recipe of the integer simulator on a manifest."""

import time

import torch

from iparakeet.analysis.range import CalibrationStats
from iparakeet.eval.manifest import ManifestEntry
from iparakeet.eval.rtf import RTFMeter
from iparakeet.eval.runner import iter_features
from iparakeet.eval.text import corpus_wer
from iparakeet.model.buckets import BucketSet, pad_features
from iparakeet.model.decoding import ctc_greedy_ids
from iparakeet.sim.int_parakeet import IntParakeet
from iparakeet.sim.recipe import Recipe


@torch.no_grad()
def evaluate_recipe(
    model,
    tokenizer,
    stats: CalibrationStats,
    recipe: Recipe,
    entries: list[ManifestEntry],
    max_frames: int = 3500,
    buckets: BucketSet | None = None,
    silence: torch.Tensor | None = None,
    device: str = "cpu",
) -> dict:
    """Integer-only transcription of every entry (batch 1); optional static-shape bucket padding."""
    sim = IntParakeet(model, stats, recipe, max_frames)
    hop = model.cfg.hop_length / model.cfg.sample_rate
    meter = RTFMeter()
    hyps = []
    for feats, _ in iter_features(model, entries, device):
        real = feats.shape[-1]
        if buckets is not None:
            feats = pad_features(feats[0].cpu(), buckets.padded_frames(real), silence)[None]
        start = time.perf_counter()
        acc = sim.forward_int(sim.quantize_input(feats.cpu()))
        ids = ctc_greedy_ids(acc[0], model.cfg.blank_id)
        meter.add(real * hop, time.perf_counter() - start, feats.shape[-1] * hop)
        hyps.append(tokenizer.decode(ids))
    wer = corpus_wer([e.text for e in entries], hyps)
    return {
        "recipe": recipe.name,
        "wer": wer.wer,
        "errors": wer.errors,
        "ref_words": wer.ref_words,
        "n_utts": wer.n_utts,
        "hyps": hyps,
        "seconds": meter.processing_seconds,
        "padding_ratio": meter.padding_ratio,
        "multipliers": sim.multiplier_report(),
    }
