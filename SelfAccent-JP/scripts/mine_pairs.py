"""Mine teacher/student pairs for self-distillation.

    uv run python -m scripts.mine_pairs \
        --sentences data/toy/mine.jsonl --count-corpus data/toy/train.jsonl \
        --out exp/pairs.jsonl

``--sentences`` is the text to mine from (no audio needed). ``--count-corpus``
defines which words are "common" (seen at least ``--min-count`` times), i.e.
words the frozen backbone can be trusted to read; use the backbone's training
text when available, otherwise any large text in the same domain.
Both accept .jsonl (a "text" field per line) or plain text (one sentence per line).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from selfaccent.mining import count_words, mine_pairs, save_pairs


def read_texts(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if path.suffix == ".jsonl":
        return [json.loads(l)["text"] for l in lines]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", type=Path, required=True)
    ap.add_argument("--count-corpus", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--max-per-sentence", type=int, default=1)
    ap.add_argument("--identity-ratio", type=float, default=0.1)
    ap.add_argument("--exclude", type=Path, default=None, help="file with words never to use as teachers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sentences = read_texts(args.sentences)
    counts: Counter = count_words(read_texts(args.count_corpus) if args.count_corpus else sentences)
    exclude = set(args.exclude.read_text(encoding="utf-8").split()) if args.exclude else set()
    pairs = mine_pairs(
        sentences, word_counts=counts, min_count=args.min_count, max_per_sentence=args.max_per_sentence,
        exclude=exclude, identity_ratio=args.identity_ratio, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_pairs(pairs, args.out)
    tagged = [p for p in pairs if p.tag]
    print(
        f"{len(sentences)} sentences -> {len(tagged)} tagged pairs ({len({p.word for p in tagged})} words) "
        f"+ {len(pairs) - len(tagged)} identity pairs -> {args.out}"
    )


if __name__ == "__main__":
    main()
