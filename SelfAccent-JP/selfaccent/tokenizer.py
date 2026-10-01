"""Character tokenizer for raw-text backbones, aware of tag marker tokens."""

from __future__ import annotations

import re

from selfaccent.tags import PHON_END, PHON_START

_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
_MARKER_RE = re.compile("(" + re.escape(PHON_START) + "|" + re.escape(PHON_END) + ")")


def _segment(text: str) -> list[str]:
    out = []
    for part in _MARKER_RE.split(text):
        if part in (PHON_START, PHON_END):
            out.append(part)
        else:
            out.extend(part)
    return out


class CharTokenizer:
    def __init__(self, symbols: list[str]) -> None:
        if symbols[: len(_SPECIALS)] != _SPECIALS:
            raise ValueError("symbol list must start with the special tokens")
        self.symbols = list(symbols)
        self._index = {s: i for i, s in enumerate(self.symbols)}

    @classmethod
    def build(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({s for t in texts for s in _segment(t)})
        return cls(_SPECIALS + chars)

    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def __len__(self) -> int:
        return len(self.symbols)

    def id_of(self, symbol: str) -> int:
        return self._index[symbol]

    def encode(self, text: str, add_bos_eos: bool = True) -> list[int]:
        ids = [self._index.get(s, self.unk_id) for s in _segment(text)]
        return [self.bos_id, *ids, self.eos_id] if add_bos_eos else ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.symbols[i] for i in ids if i >= len(_SPECIALS))

    def unknown_symbols(self, text: str) -> set[str]:
        return {s for s in _segment(text) if s not in self._index}

    def extend(self, symbols: list[str]) -> list[int]:
        """Append unseen symbols at the end; return the id of every requested symbol."""
        for s in symbols:
            if s not in self._index:
                self._index[s] = len(self.symbols)
                self.symbols.append(s)
        return [self._index[s] for s in symbols]

    def to_dict(self) -> dict:
        return {"symbols": self.symbols}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["symbols"])
