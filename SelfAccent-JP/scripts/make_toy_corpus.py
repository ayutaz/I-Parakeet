"""Build the toy corpus: text splits, HTS (oracle) audio and log-mels.

    uv run python -m scripts.make_toy_corpus --out data/toy
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch

from selfaccent.audio import MelConfig, MelExtractor
from selfaccent.data.toy_corpus import build_splits, save_sentences
from selfaccent.data.wordlists import COMMON_WORDS, DIFFICULT_WORDS, TEMPLATES
from selfaccent.frontend import OracleRenderer
from selfaccent.tags import parse_tag_body


def accent_class(tag: str) -> str:
    p = parse_tag_body(tag)[0]
    n, k = len(p.morae), p.accent
    return "heiban" if k == 0 else "atamadaka" if k == 1 else "odaka" if k == n else "nakadaka"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/toy"))
    ap.add_argument("--train-templates-per-word", type=int, default=7)
    ap.add_argument("--kana-variant-ratio", type=float, default=0.2)
    ap.add_argument("--eval-templates-per-word", type=int, default=3)
    ap.add_argument("--save-wavs", type=int, default=10, help="also write the first N training wavs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    splits = build_splits(
        COMMON_WORDS, DIFFICULT_WORDS, TEMPLATES,
        train_templates_per_word=args.train_templates_per_word,
        kana_variant_ratio=args.kana_variant_ratio,
        eval_templates_per_word=args.eval_templates_per_word,
        seed=args.seed,
    )
    for name, sents in splits.items():
        save_sentences(sents, args.out / f"{name}.jsonl")

    cfg = MelConfig()
    oracle = OracleRenderer(cfg.sample_rate)
    ext = MelExtractor(cfg)
    wav_dir = args.out / "wavs"
    wav_dir.mkdir(exist_ok=True)
    mels, seconds = [], 0.0
    t0 = time.time()
    for i, s in enumerate(splits["train"]):
        wav = oracle.render(s.text)
        seconds += len(wav) / cfg.sample_rate
        mels.append(ext(torch.from_numpy(wav)).half())
        if i < args.save_wavs:
            sf.write(wav_dir / f"train_{i:04d}.wav", wav, cfg.sample_rate)
    torch.save(
        {"texts": [s.text for s in splits["train"]], "mels": mels, "mel_config": cfg.to_dict()},
        args.out / "train_mels.pt",
    )
    stats = {
        "sentences": {k: len(v) for k, v in splits.items()},
        "train_audio_minutes": round(seconds / 60, 2),
        "difficult_words_kept": sorted({s.word for s in splits["eval_difficult"]}),
        "difficult_accent_classes": Counter(
            accent_class(s.tag) for s in {s.word: s for s in splits["eval_difficult"]}.values()
        ),
        "common_accent_classes": Counter(accent_class(s.tag) for s in {s.word: s for s in splits["mine"]}.values()),
        "synthesis_seconds": round(time.time() - t0, 1),
    }
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
