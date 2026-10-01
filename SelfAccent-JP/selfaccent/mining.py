"""Mining self-distillation pairs (arXiv 2609.17234, Sec. "self-distillation").

A pair holds a sentence with a *common* word that the frozen backbone already
reads correctly (teacher input) and the same sentence with that word replaced
by its kana + pitch-accent tag (student input). The backbone's own output for
the teacher input becomes the training target for the student input, so no
recordings are needed.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from selfaccent.frontend import Word, analyze, word_phrases
from selfaccent.tags import format_tag_body, make_tag, parse_tag_body

# Nodes that may follow a tagged word inside its accent phrase without changing
# the word's own accent (particles, auxiliaries, punctuation).
_FUNCTION_POS = ("助詞", "助動詞", "記号")


@dataclass(frozen=True)
class DistillPair:
    teacher_text: str
    student_text: str
    word: str | None = None
    tag: str | None = None  # tag body, None for identity pairs
    start: int | None = None
    end: int | None = None


def count_words(sentences: list[str]) -> Counter:
    counts: Counter = Counter()
    for s in sentences:
        counts.update(w.surface for w in analyze(s))
    return counts


def _standalone(words: list[Word], i: int) -> bool:
    """The word heads an accent phrase and is not the head of a compound."""
    if i + 1 < len(words):
        nxt = words[i + 1]
        if nxt.chain_flag == 1 and nxt.pos not in _FUNCTION_POS:
            return False
    return True


def candidate_words(text: str, pos_filter: tuple[str, ...] = ("名詞",)) -> list[tuple[Word, str]]:
    """Words of ``text`` that can be replaced by a tag, with their tag bodies."""
    words = analyze(text)
    out = []
    for i, w in enumerate(words):
        if w.pos not in pos_filter or w.start is None or not _standalone(words, i):
            continue
        phrases = word_phrases(w)
        if phrases is None:
            continue
        out.append((w, format_tag_body(phrases)))
    return out


def mine_pairs(
    sentences: list[str],
    *,
    word_counts: Counter | None = None,
    min_count: int = 2,
    max_per_sentence: int = 1,
    pos_filter: tuple[str, ...] = ("名詞",),
    exclude: set[str] | None = None,
    identity_ratio: float = 0.0,
    seed: int = 0,
) -> list[DistillPair]:
    """Build teacher/student pairs from raw sentences.

    ``word_counts`` defines what is "common" (by default counted on
    ``sentences``); words seen fewer than ``min_count`` times are skipped
    because the backbone is not trusted to read them. ``identity_ratio`` adds
    untagged pairs that keep the adapter from changing plain-text behaviour.
    """
    rng = random.Random(seed)
    counts = word_counts if word_counts is not None else count_words(sentences)
    exclude = exclude or set()
    pairs = []
    for text in sentences:
        cands = [
            (w, tag)
            for w, tag in candidate_words(text, pos_filter)
            if counts[w.surface] >= min_count and w.surface not in exclude
        ]
        rng.shuffle(cands)
        for w, tag in cands[:max_per_sentence]:
            student = text[: w.start] + make_tag(parse_tag_body(tag)) + text[w.end :]
            pairs.append(DistillPair(text, student, w.surface, tag, w.start, w.end))
        if identity_ratio > 0 and rng.random() < identity_ratio:
            pairs.append(DistillPair(text, text))
    return pairs


def tag_words(text: str, tags: dict[str, str]) -> str:
    """Replace the first occurrence of each word by its tag (for inference)."""
    for word, body in tags.items():
        pos = text.find(word)
        if pos < 0:
            raise KeyError(f"{word!r} not found in {text!r}")
        text = text[:pos] + make_tag(parse_tag_body(body)) + text[pos + len(word) :]
    return text


def save_pairs(pairs: list[DistillPair], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pairs(path: str | Path) -> list[DistillPair]:
    with open(path, encoding="utf-8") as f:
        return [DistillPair(**json.loads(line)) for line in f if line.strip()]
