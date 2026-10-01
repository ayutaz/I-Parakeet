"""Automatic benchmark: does the speech carry the prescribed reading and accent?

Conditions (cf. arXiv 2609.17234, which compares tags against plain kana):

* raw       frozen backbone, raw text (the word in kanji as usually written)
* kana      frozen backbone, the word written in katakana (no accent info)
* tag       backbone + adapter, the word as a tag with its dictionary accent
* tag_sweep backbone + adapter, the word as a tag with *every* accent type
            0..n; correct if the judge hears the requested type
            (controllability, not just reproducing the dictionary)

For each output the judge picks the closest oracle reference among all accent
types of the correct reading (see :mod:`selfaccent.metrics`). ``mel_dist`` is
the DTW log-mel distance to the reference with the requested accent, a proxy
for "right reading" (wrong or garbled readings are far from every reference).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from selfaccent.audio import MelConfig, MelExtractor
from selfaccent.data.toy_corpus import Sentence
from selfaccent.frontend import OracleRenderer, analyze
from selfaccent.metrics import ReferenceSet, build_reference_set, forced_choice, mel_distance, utterance_from_mel
from selfaccent.tags import AccentPhrase, make_tag, parse_tag_body


@dataclass(frozen=True)
class EvalItem:
    text: str
    word: str
    phrase: AccentPhrase  # dictionary reading and accent
    start: int
    end: int
    kind: str = ""

    @classmethod
    def from_sentence(cls, s: Sentence) -> "EvalItem":
        phrases = parse_tag_body(s.tag)
        if len(phrases) != 1:
            raise ValueError("benchmark items must be single accent phrases")
        return cls(s.text, s.word, phrases[0], s.start, s.end, s.kind)

    @property
    def n_morae(self) -> int:
        return len(self.phrase.morae)

    @property
    def accent(self) -> int:
        return self.phrase.accent

    def with_word(self, replacement: str) -> str:
        return self.text[: self.start] + replacement + self.text[self.end :]


def accent_class(phrase: AccentPhrase) -> str:
    n, k = len(phrase.morae), phrase.accent
    return "heiban" if k == 0 else "atamadaka" if k == 1 else "odaka" if k == n else "nakadaka"


def condition_inputs(item: EvalItem) -> dict[str, str]:
    out = {
        "raw": item.text,
        "kana": item.with_word(item.phrase.kana),
        "tag": item.with_word(make_tag([item.phrase])),
    }
    for k in range(item.n_morae + 1):
        out[f"tag_k{k}"] = item.with_word(make_tag([AccentPhrase(item.phrase.morae, k)]))
    return out


def oracle_wavs(oracle: OracleRenderer, item: EvalItem, **synth_kwargs) -> list[np.ndarray]:
    """Oracle waveforms of the sentence for accent types 0..n of the word."""
    idx = next(w.index for w in analyze(item.text) if w.start == item.start)
    out = []
    for k in range(item.n_morae + 1):
        labels = oracle.labels_with_override(item.text, idx, [AccentPhrase(item.phrase.morae, k)])
        out.append(oracle.synthesize_labels(labels, **synth_kwargs))
    return out


def build_references(oracle: OracleRenderer, item: EvalItem, cfg: MelConfig, gl_iters: int = 32) -> ReferenceSet:
    """Reference set for the judge.

    References take the same mel -> Griffin-Lim -> F0 path as generated
    speech, so that Griffin-Lim artifacts affect both sides equally.
    """
    ext = MelExtractor(cfg)
    refs = [utterance_from_mel(ext(torch.from_numpy(w)), cfg, gl_iters) for w in oracle_wavs(oracle, item)]
    return build_reference_set(refs)


def lenient_match(choice: int, target: int, n_morae: int) -> bool:
    """Heiban (0) and odaka (n) differ only on the following particle; merge them."""
    return choice == target or {choice, target} <= {0, n_morae}


def judge(mel: torch.Tensor, refset: ReferenceSet, target: int, n_morae: int, cfg: MelConfig, gl_iters: int) -> dict:
    gen = utterance_from_mel(mel, cfg, gl_iters)
    choice, dists = forced_choice(gen, refset)
    return {
        "target": target,
        "choice": choice,
        "correct": bool(choice == target),
        "correct_lenient": lenient_match(choice, target, n_morae),
        "f0_dists": [round(float(d), 5) for d in dists],
        "mel_dist": round(mel_distance(gen, refset.refs[target]), 5),
    }


def run_benchmark(
    backbones: dict[str, Any],
    items: list[EvalItem],
    seed: int = 0,
    gl_iters: int = 32,
    cfg: MelConfig = MelConfig(),
    conditions: tuple[str, ...] = ("raw", "kana", "tag", "tag_sweep"),
    progress=None,
) -> list[dict]:
    """``backbones`` maps "base" (frozen) and "adapted" (with adapter) to Backbone objects."""
    oracle = OracleRenderer(cfg.sample_rate)
    records = []
    for i, item in enumerate(items):
        refs = build_references(oracle, item, cfg, gl_iters)
        inputs = condition_inputs(item)
        base = {
            "item": i, "text": item.text, "word": item.word, "kind": item.kind,
            "n_morae": item.n_morae, "dict_accent": item.accent, "accent_class": accent_class(item.phrase),
        }
        jobs = []
        if "raw" in conditions:
            jobs.append(("raw", "base", inputs["raw"], item.accent))
        if "kana" in conditions:
            jobs.append(("kana", "base", inputs["kana"], item.accent))
        if "tag" in conditions:
            jobs.append(("tag", "adapted", inputs["tag"], item.accent))
        if "raw_adapted" in conditions:
            jobs.append(("raw_adapted", "adapted", inputs["raw"], item.accent))
        if "tag_sweep" in conditions:
            jobs.extend(("tag_sweep", "adapted", inputs[f"tag_k{k}"], k) for k in range(item.n_morae + 1))
        for cond, which, text, target in jobs:
            mel = backbones[which].synthesize([text], seeds=[seed + i])[0]
            records.append(
                {**base, "condition": cond, "input": text, **judge(mel, refs, target, item.n_morae, cfg, gl_iters)}
            )
        if progress:
            progress(i + 1, len(items))
    return records


def summarize(records: list[dict]) -> dict[str, dict]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)
    out = {}
    for cond, rs in by_cond.items():
        classes: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            key = r["accent_class"] if cond != "tag_sweep" else f"k{r['target']}"
            classes[key].append(r["correct"])
        out[cond] = {
            "n": len(rs),
            "accent_acc": float(np.mean([r["correct"] for r in rs])),
            "accent_acc_ci95": bootstrap_ci([r["correct"] for r in rs]),
            "accent_acc_lenient": float(np.mean([r["correct_lenient"] for r in rs])),
            "mel_dist": float(np.mean([r["mel_dist"] for r in rs])),
            "by_class": {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(classes.items())},
        }
    return out


def neutralizing_context(item: EvalItem) -> bool:
    """True if the word is followed by の, where heiban and odaka are realized alike."""
    return item.text[item.end : item.end + 1] == "の"


def validate_judge(
    items: list[EvalItem], speed: float = 1.1, half_tone: float = 1.0, gl_iters: int = 32, cfg: MelConfig = MelConfig()
) -> list[dict]:
    """Accuracy of the judge on oracle speech with a different tempo and pitch.

    For every item and accent type, the oracle speaks the sentence at
    ``speed`` / ``half_tone`` and the judge has to find the accent type among
    the unperturbed references. This bounds what the benchmark can resolve.
    """
    oracle = OracleRenderer(cfg.sample_rate)
    ext = MelExtractor(cfg)
    records = []
    for i, item in enumerate(items):
        refset = build_references(oracle, item, cfg, gl_iters)
        for k, wav in enumerate(oracle_wavs(oracle, item, speed=speed, half_tone=half_tone)):
            r = judge(ext(torch.from_numpy(wav)), refset, k, item.n_morae, cfg, gl_iters)
            records.append({"item": i, "text": item.text, "n_morae": item.n_morae, **r})
    return records


def bootstrap_ci(values: list[bool], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    stats = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
