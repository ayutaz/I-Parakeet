"""Japanese text analysis and an HTS "oracle" renderer built on pyopenjtalk.

The oracle is what makes a fully automatic toy benchmark possible: it can speak
any sentence with any reading / pitch accent for one word, so references exist
for every accent type of every word.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import pyopenjtalk  # prints an onnxruntime notice on import

from selfaccent.tags import AccentPhrase, split_morae

HTS_SAMPLE_RATE = 48000


def _clean_pron(pron: str) -> str:
    return pron.replace("’", "")


@dataclass(frozen=True)
class Word:
    index: int  # position in the NJD feature list
    surface: str
    pos: str
    pron: str
    mora_size: int
    acc: int
    chain_flag: int
    start: int | None  # character offsets in the input text (None if not found)
    end: int | None

    @property
    def heads_phrase(self) -> bool:
        return self.chain_flag != 1


def _frontend(text: str) -> list[dict]:
    return pyopenjtalk.run_frontend(text)


def analyze(text: str) -> list[Word]:
    words = []
    cursor = 0
    for i, node in enumerate(_frontend(text)):
        surface = node["string"]
        pos = text.find(surface, cursor) if surface else -1
        if pos >= 0:
            start, end = pos, pos + len(surface)
            cursor = end
        else:
            start = end = None
        words.append(
            Word(
                index=i,
                surface=surface,
                pos=node["pos"],
                pron=_clean_pron(node["pron"]),
                mora_size=int(node["mora_size"]),
                acc=int(node["acc"]),
                chain_flag=int(node["chain_flag"]),
                start=start,
                end=end,
            )
        )
    return words


def word_phrases(word: Word) -> list[AccentPhrase] | None:
    """Reading and accent of a word that heads its own accent phrase.

    Returns None when the word is chained into the previous phrase (its accent
    is then not its own) or when the reading is not plain katakana.
    """
    if not word.heads_phrase or word.mora_size == 0:
        return None
    try:
        morae = split_morae(word.pron)
    except ValueError:
        return None
    if len(morae) != word.mora_size or word.acc > len(morae):
        return None
    return [AccentPhrase(tuple(morae), word.acc)]


def _override(njd: list[dict], index: int, phrases: list[AccentPhrase]) -> list[dict]:
    njd = [dict(n) for n in njd]
    base = njd[index]
    nodes = []
    for k, phrase in enumerate(phrases):
        node = dict(base)
        node["string"] = base["string"] if k == 0 else ""
        node["pron"] = phrase.kana
        node["read"] = phrase.kana
        node["mora_size"] = len(phrase.morae)
        node["acc"] = phrase.accent
        if k > 0:
            node["chain_flag"] = 0
        nodes.append(node)
    return njd[:index] + nodes + njd[index + 1 :]


class OracleRenderer:
    """HTS (OpenJTalk mei voice) speech for arbitrary readings and accents."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        g = gcd(sample_rate, HTS_SAMPLE_RATE)
        self._up, self._down = sample_rate // g, HTS_SAMPLE_RATE // g

    def labels(self, text: str) -> list[str]:
        return pyopenjtalk.make_label(_frontend(text))

    def labels_with_override(self, text: str, index: int, phrases: list[AccentPhrase]) -> list[str]:
        return pyopenjtalk.make_label(_override(_frontend(text), index, phrases))

    def synthesize_labels(self, labels: list[str]) -> np.ndarray:
        x, sr = pyopenjtalk.synthesize(labels)
        assert sr == HTS_SAMPLE_RATE
        y = resample_poly(x / 32768.0, self._up, self._down)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def render(self, text: str) -> np.ndarray:
        return self.synthesize_labels(self.labels(text))

    def render_with_override(self, text: str, index: int, phrases: list[AccentPhrase]) -> np.ndarray:
        return self.synthesize_labels(self.labels_with_override(text, index, phrases))
