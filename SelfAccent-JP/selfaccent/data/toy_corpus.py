"""Toy Japanese corpus: carrier sentences spoken by the HTS oracle.

Splits (all text is raw Japanese; audio is generated, so the corpus is free
and the reading/accent of every word is known):

* train          backbone training sentences (common words, kanji/kana as
                 usually written) plus katakana variants so that the backbone
                 learns to read katakana at all (needed for the plain-kana
                 baseline and as a realistic raw-text model).
* mine           unseen combinations of common words and templates; the text
                 source for self-distillation pairs (no audio needed).
* eval_common    held-out sentences with common words (sanity checks).
* eval_difficult difficult words never seen by the backbone.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from selfaccent.mining import candidate_words
from selfaccent.tags import parse_tag_body


@dataclass(frozen=True)
class Sentence:
    text: str
    word: str
    tag: str  # tag body with the dictionary reading / accent of ``word``
    template: int
    start: int
    end: int
    kind: str = "train"

    def to_dict(self) -> dict:
        return asdict(self)


def slot_sentences(words: list[str], templates: list[str]) -> list[Sentence]:
    """Fill every template with every word; keep only standalone accent-phrase heads."""
    out = []
    for w in words:
        for ti, tpl in enumerate(templates):
            text = tpl.format(w)
            pos = tpl.index("{}")
            for cand, tag in candidate_words(text):
                if cand.surface == w and cand.start == pos:
                    out.append(Sentence(text, w, tag, ti, pos, pos + len(w)))
                    break
    return out


def katakana_variant(s: Sentence) -> Sentence:
    kana = "".join(p.kana for p in parse_tag_body(s.tag))
    text = s.text[: s.start] + kana + s.text[s.end :]
    return replace(s, text=text, word=kana, end=s.start + len(kana), kind="kana_variant")


def unseen_words(words: list[str], texts: list[str]) -> list[str]:
    seen = set("".join(texts))
    return [w for w in words if not set(w) & seen]


def _is_katakana(word: str) -> bool:
    return all("ァ" <= c <= "ヺ" or c == "ー" for c in word)


def build_splits(
    common: list[str],
    difficult: list[str],
    templates: list[str],
    train_templates_per_word: int = 7,
    kana_variant_ratio: float = 0.2,
    eval_templates_per_word: int = 3,
    seed: int = 0,
) -> dict[str, list[Sentence]]:
    rng = random.Random(seed)
    by_word: dict[str, list[Sentence]] = {}
    for s in slot_sentences(common, templates):
        by_word.setdefault(s.word, []).append(s)
    train, mine, eval_common = [], [], []
    for w in common:
        sents = by_word.get(w, [])
        rng.shuffle(sents)
        if len(sents) < 3:
            train.extend(sents)
            continue
        n_train = min(train_templates_per_word, len(sents) - 2)
        train.extend(sents[:n_train])
        eval_common.append(replace(sents[n_train], kind="eval_common"))
        mine.extend(replace(s, kind="mine") for s in sents[n_train + 1 :])
    variants = [katakana_variant(s) for s in train if not _is_katakana(s.word) and rng.random() < kana_variant_ratio]
    train = train + variants

    hidden = unseen_words(difficult, [s.text for s in train])
    eval_difficult = []
    for w in hidden:
        sents = slot_sentences([w], templates)
        rng.shuffle(sents)
        eval_difficult.extend(replace(s, kind="eval_difficult") for s in sents[:eval_templates_per_word])
    return {"train": train, "mine": mine, "eval_common": eval_common, "eval_difficult": eval_difficult}


def save_sentences(sentences: list[Sentence], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_sentences(path: str | Path) -> list[Sentence]:
    with open(path, encoding="utf-8") as f:
        return [Sentence(**json.loads(line)) for line in f if line.strip()]
