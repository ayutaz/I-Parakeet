"""Synthesize speech, optionally with pronunciation / accent tags.

    # tags written inline
    uv run python -m scripts.infer --backbone exp/backbone/model.pt --adapter exp/adapter/adapter.pt \
        --text "昨日、<PHON_START>ソ'ゴ<PHON_END>について話しました。" --out out.wav

    # or: tag a word of plain text (repeatable)
    uv run python -m scripts.infer ... --text "齟齬について調べました。" --tag "齟齬=ソ'ゴ" --out out.wav

    # or: take the reading/accent from the OpenJTalk dictionary
    uv run python -m scripts.infer ... --text "齟齬について調べました。" --auto-tag 齟齬 --out out.wav

Tag notation: katakana reading, ' after the accent nucleus (none = heiban),
/ between accent phrases, e.g. チ'ミ/モーリョー. The waveform is made with
Griffin-Lim; --mel-out writes the log-mel (80 bins, 22.05 kHz, hop 256) for an
external HiFi-GAN-compatible vocoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from selfaccent.audio import MelConfig, griffin_lim
from selfaccent.backbones.tiny_matcha import TinyMatchaBackbone
from selfaccent.distill import load_adapter
from selfaccent.frontend import analyze, word_phrases
from selfaccent.mining import tag_words
from selfaccent.tags import find_tags, format_tag_body


def auto_tag(text: str, word: str) -> str:
    for w in analyze(text):
        if w.surface == word:
            phrases = word_phrases(w)
            if phrases:
                return format_tag_body(phrases)
    raise SystemExit(f"cannot derive a dictionary tag for {word!r} in {text!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tag", action="append", default=[], help="WORD=TAGBODY, e.g. 齟齬=ソ'ゴ")
    ap.add_argument("--auto-tag", action="append", default=[], help="tag WORD with its dictionary reading/accent")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mel-out", type=Path, default=None)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--gl-iters", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tags = dict(t.split("=", 1) for t in args.tag)
    tags.update({w: auto_tag(args.text, w) for w in args.auto_tag})
    text = tag_words(args.text, tags) if tags else args.text
    if find_tags(text) and args.adapter is None:
        print("warning: the input has tags but no --adapter was given", file=sys.stderr)

    backbone = TinyMatchaBackbone.from_checkpoint(
        args.backbone, n_steps=args.n_steps, temperature=args.temperature, length_scale=args.length_scale
    )
    if args.adapter:
        load_adapter(backbone, args.adapter)
    unknown = backbone.tokenizer.unknown_symbols(text)
    if unknown:
        print(f"warning: symbols unknown to the backbone: {''.join(sorted(unknown))}", file=sys.stderr)

    mel = backbone.synthesize([text], seeds=[args.seed])[0]
    cfg = MelConfig()
    wav = griffin_lim(mel, cfg, n_iter=args.gl_iters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, cfg.sample_rate)
    if args.mel_out:
        np.save(args.mel_out, mel.numpy())
    print(f"input: {text}\nwrote {args.out} ({len(wav) / cfg.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
