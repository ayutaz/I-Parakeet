"""Text normalization and corpus WER (paper Sec. 4.1: Whisper English text normalizer)."""

from dataclasses import dataclass
from functools import lru_cache

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer


@lru_cache(maxsize=1)
def _normalizer() -> EnglishTextNormalizer:
    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    return _normalizer()(text)


@dataclass(frozen=True)
class WERResult:
    wer: float
    errors: int
    ref_words: int
    n_utts: int
    n_skipped: int


def corpus_wer(refs: list[str], hyps: list[str], apply_normalizer: bool = True) -> WERResult:
    """Corpus-level WER; references that normalize to empty strings are skipped."""
    if len(refs) != len(hyps):
        raise ValueError(f"got {len(refs)} references but {len(hyps)} hypotheses")
    kept_refs, kept_hyps = [], []
    for ref, hyp in zip(refs, hyps):
        if apply_normalizer:
            ref, hyp = normalize(ref), normalize(hyp)
        if not ref.strip():
            continue
        kept_refs.append(ref)
        kept_hyps.append(hyp)
    n_skipped = len(refs) - len(kept_refs)
    if not kept_refs:
        return WERResult(0.0, 0, 0, 0, n_skipped)
    out = jiwer.process_words(kept_refs, kept_hyps)
    errors = out.substitutions + out.deletions + out.insertions
    ref_words = out.substitutions + out.deletions + out.hits
    return WERResult(errors / ref_words, errors, ref_words, len(kept_refs), n_skipped)
