"""Pronunciation / pitch-accent tag notation.

A tag replaces a word in raw text with its katakana reading and Tokyo-dialect
pitch accent, using the notation of UtterTune (arXiv 2508.09767), which the
self-distillation paper (arXiv 2609.17234) builds on:

    <PHON_START>チ'ミ/モーリョー<PHON_END>

* ``'`` follows the accent nucleus (the last high mora before the pitch fall).
  A phrase without ``'`` is heiban (accent type 0).
* ``/`` separates accent phrases inside one tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PHON_START = "<PHON_START>"
PHON_END = "<PHON_END>"
ACCENT_MARK = "'"
PHRASE_SEP = "/"

# Small kana that merge with the preceding kana into one mora.
_SMALL_KANA = set("ァィゥェォャュョヮ")
_KATAKANA = set(chr(c) for c in range(ord("ァ"), ord("ヺ") + 1)) | {"ー"}

_TAG_RE = re.compile(re.escape(PHON_START) + "(.*?)" + re.escape(PHON_END), re.S)


class TagError(ValueError):
    pass


def hira_to_kata(text: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def split_morae(kana: str) -> list[str]:
    morae: list[str] = []
    for ch in kana:
        if ch not in _KATAKANA:
            raise TagError(f"not a katakana character: {ch!r} in {kana!r}")
        if ch in _SMALL_KANA:
            if not morae or morae[-1] in ("ー", "ッ", "ン"):
                raise TagError(f"small kana without a host mora in {kana!r}")
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


@dataclass(frozen=True)
class AccentPhrase:
    morae: tuple[str, ...]
    accent: int

    def __post_init__(self) -> None:
        if not self.morae:
            raise TagError("empty accent phrase")
        if not 0 <= self.accent <= len(self.morae):
            raise TagError(f"accent {self.accent} out of range for {len(self.morae)} morae")

    @classmethod
    def from_kana(cls, kana: str, accent: int) -> "AccentPhrase":
        return cls(tuple(split_morae(hira_to_kata(kana))), accent)

    @property
    def kana(self) -> str:
        return "".join(self.morae)

    def pitch(self, with_particle: bool = False) -> list[int]:
        """Tokyo-dialect high (1) / low (0) pattern per mora.

        With ``with_particle`` a following particle mora is appended, which is
        the only place where odaka (accent == n) differs from heiban.
        """
        n = len(self.morae) + (1 if with_particle else 0)
        k = self.accent
        out = []
        for i in range(1, n + 1):
            if k == 1:
                out.append(1 if i == 1 else 0)
            elif i == 1:
                out.append(0)
            else:
                out.append(1 if k == 0 or i <= k else 0)
        return out


def format_phrase(phrase: AccentPhrase) -> str:
    if phrase.accent == 0:
        return phrase.kana
    return "".join(phrase.morae[: phrase.accent]) + ACCENT_MARK + "".join(phrase.morae[phrase.accent :])


def format_tag_body(phrases: list[AccentPhrase]) -> str:
    if not phrases:
        raise TagError("a tag needs at least one accent phrase")
    return PHRASE_SEP.join(format_phrase(p) for p in phrases)


def parse_phrase(text: str) -> AccentPhrase:
    if text.count(ACCENT_MARK) > 1:
        raise TagError(f"more than one accent nucleus in {text!r}")
    if ACCENT_MARK in text:
        head, tail = text.split(ACCENT_MARK)
        if not head:
            raise TagError(f"accent mark before the first mora in {text!r}")
        head_morae = split_morae(head)
        morae = head_morae + split_morae(tail)
        return AccentPhrase(tuple(morae), len(head_morae))
    return AccentPhrase(tuple(split_morae(text)), 0)


def parse_tag_body(body: str) -> list[AccentPhrase]:
    if not body:
        raise TagError("empty tag")
    parts = body.split(PHRASE_SEP)
    if any(not p for p in parts):
        raise TagError(f"empty accent phrase in {body!r}")
    return [parse_phrase(p) for p in parts]


def make_tag(phrases: list[AccentPhrase]) -> str:
    return PHON_START + format_tag_body(phrases) + PHON_END


@dataclass(frozen=True)
class TagSpan:
    start: int
    end: int
    phrases: list[AccentPhrase]


def find_tags(text: str) -> list[TagSpan]:
    spans = [TagSpan(m.start(), m.end(), parse_tag_body(m.group(1))) for m in _TAG_RE.finditer(text)]
    rest = _TAG_RE.sub("", text)
    if PHON_START in rest or PHON_END in rest:
        raise TagError(f"unbalanced tag markers in {text!r}")
    return spans


def strip_tags(text: str) -> str:
    """Replace every tag by its bare katakana (the "plain kana" baseline)."""
    find_tags(text)  # validates
    return _TAG_RE.sub(lambda m: "".join(p.kana for p in parse_tag_body(m.group(1))), text)


def replace_span_with_tag(text: str, start: int, end: int, phrases: list[AccentPhrase]) -> str:
    return text[:start] + make_tag(phrases) + text[end:]
