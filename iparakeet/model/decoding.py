"""CTC greedy decoding and the SentencePiece tokenizer wrapper."""

import sentencepiece as spm
import torch


def ctc_greedy_ids(logits: torch.Tensor, blank: int) -> list[int]:
    """Argmax per frame, collapse repeats, drop blanks. logits: (T, V+1)."""
    best = logits.argmax(dim=-1).tolist()
    out, prev = [], None
    for token in best:
        if token != prev and token != blank:
            out.append(token)
        prev = token
    return out


class Tokenizer:
    def __init__(self, model_file: str) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Tokenizer":
        tok = cls.__new__(cls)
        tok.sp = spm.SentencePieceProcessor(model_proto=data)
        return tok

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)
