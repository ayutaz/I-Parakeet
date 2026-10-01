"""Train the toy raw-text backbone (TinyMatcha) on the toy corpus.

Only needed for the CPU toy experiment; with a real pretrained backbone
(CosyVoice2, Sarashina2.2-TTS, ...) skip this step.

    uv run python -m scripts.train_backbone --data data/toy --out exp/backbone
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from selfaccent.backbones.tiny_matcha import TinyMatcha, TinyMatchaBackbone, TinyMatchaConfig
from selfaccent.tokenizer import CharTokenizer
from selfaccent.training import TrainConfig, compute_mel_stats, train_backbone


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/toy"))
    ap.add_argument("--out", type=Path, default=Path("exp/backbone"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-channels", type=int, default=192)
    ap.add_argument("--dec-blocks", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.data / "train_mels.pt", weights_only=False)
    texts, mels = data["texts"], data["mels"]
    tokenizer = CharTokenizer.build(texts)
    mean, std = compute_mel_stats(mels)
    cfg = TinyMatchaConfig(
        n_vocab=len(tokenizer), n_mels=mels[0].shape[0], d_model=args.d_model, enc_layers=args.enc_layers,
        ffn_dim=4 * args.d_model, dec_channels=args.dec_channels, dec_blocks=args.dec_blocks,
    )
    torch.manual_seed(args.seed)
    backbone = TinyMatchaBackbone(TinyMatcha(cfg, mean, std), tokenizer)
    n_params = sum(p.numel() for p in backbone.model.parameters())
    print(f"vocab={len(tokenizer)} params={n_params / 1e6:.2f}M utterances={len(texts)}")

    extra = {"tokenizer": tokenizer.to_dict(), "mel_config": data["mel_config"]}
    log = open(args.out / "train_log.jsonl", "w")
    t0 = time.time()

    def log_fn(rec: dict) -> None:
        rec = {**rec, "elapsed_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(
            f"step {rec['step']:5d} ep {rec['epoch']:3d} loss {rec['loss']:.3f} prior {rec['prior']:.3f} "
            f"dur {rec['duration']:.3f} cfm {rec['cfm']:.3f} lr {rec['lr']:.2e} {rec['elapsed_s']:.0f}s",
            flush=True,
        )

    def ckpt_fn(step: int) -> None:
        backbone.model.save(args.out / "model.pt", extra={**extra, "step": step})

    tcfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, warmup_steps=args.warmup_steps, seed=args.seed
    )
    train_backbone(backbone, texts, mels, tcfg, log_fn, ckpt_fn, args.checkpoint_every)
    backbone.model.save(args.out / "model.pt", extra={**extra, "step": args.steps})
    (args.out / "config.json").write_text(json.dumps({"model": asdict(cfg), "train": asdict(tcfg)}, indent=1))
    print(f"saved {args.out / 'model.pt'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
