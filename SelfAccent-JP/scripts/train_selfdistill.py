"""Self-distill pronunciation / accent tags into a frozen backbone.

    uv run python -m scripts.train_selfdistill \
        --backbone exp/backbone/model.pt --pairs exp/pairs.jsonl --out exp/adapter

1. the frozen backbone speaks every teacher text (cached in --out)
2. tag tokens + LoRA are added; only they are trained, with the backbone's own
   loss on (student text, teacher output)
3. the adapter (LoRA + new embedding rows) is written to --out/adapter.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from selfaccent.backbones.tiny_matcha import TinyMatchaBackbone
from selfaccent.distill import (
    AdapterConfig,
    DistillConfig,
    generate_teacher_targets,
    prepare_student,
    save_adapter,
    train_student,
)
from selfaccent.metrics import dtw_path
from selfaccent.mining import load_pairs


def _pairs_key(pairs, args) -> str:
    h = hashlib.sha1()
    for p in pairs:
        h.update(p.teacher_text.encode())
    h.update(f"{args.seed}-{args.teacher_steps}-{args.teacher_temperature}".encode())
    return h.hexdigest()[:12]


def _dtw_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.T.numpy(), b.T.numpy()
    _, path = dtw_path(x, y)
    return float(np.mean([np.abs(x[i] - y[j]).mean() for i, j in path]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--teacher-steps", type=int, default=10, help="ODE steps for teacher synthesis")
    ap.add_argument("--teacher-temperature", type=float, default=0.667)
    ap.add_argument("--val-pairs", type=int, default=24, help="held-out tagged pairs for a fidelity check")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    rng = random.Random(args.seed)
    tagged_idx = [i for i, p in enumerate(pairs) if p.tag]
    val_idx = set(rng.sample(tagged_idx, min(args.val_pairs, len(tagged_idx) // 5)))
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [pairs[i] for i in sorted(val_idx)]

    backbone = TinyMatchaBackbone.from_checkpoint(
        args.backbone, n_steps=args.teacher_steps, temperature=args.teacher_temperature
    )
    for p in backbone.model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    cache = args.out / f"teacher_{_pairs_key(pairs, args)}.pt"
    if cache.exists():
        targets = torch.load(cache)
    else:
        targets = generate_teacher_targets(backbone, pairs, seed=args.seed)
        torch.save(targets, cache)
    by_text = {id(p): t for p, t in zip(pairs, targets)}
    train_targets = [by_text[id(p)] for p in train_pairs]
    val_targets = [by_text[id(p)] for p in val_pairs]
    print(f"teacher targets for {len(pairs)} pairs ({time.time() - t0:.0f}s, cache {cache.name})")

    acfg = AdapterConfig(r=args.r, alpha=args.alpha, dropout=args.dropout, targets=args.targets.split(","))
    trainable = prepare_student(backbone, acfg)
    n_train = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in backbone.model.parameters())
    print(f"trainable {n_train / 1e3:.1f}k / {n_all / 1e6:.2f}M params ({len(trainable)} tensors)")

    def fidelity() -> float:
        if not val_pairs:
            return float("nan")
        outs = backbone.synthesize([p.student_text for p in val_pairs], seeds=[args.seed + 10_000 + i for i in range(len(val_pairs))])
        return float(np.mean([_dtw_l1(o, t) for o, t in zip(outs, val_targets)]))

    before = fidelity()
    print(f"val student-vs-teacher mel L1 before training: {before:.3f}")
    log = open(args.out / "train_log.jsonl", "w")

    def log_fn(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {rec['step']:5d} loss {rec['loss']:.3f} prior {rec['prior']:.3f} dur {rec['duration']:.3f} "
              f"cfm {rec['cfm']:.3f} lr {rec['lr']:.2e} {time.time() - t0:.0f}s", flush=True)

    dcfg = DistillConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, warmup_steps=args.warmup_steps, seed=args.seed)
    train_student(backbone, train_pairs, train_targets, dcfg, log_fn)
    after = fidelity()
    print(f"val student-vs-teacher mel L1 after training: {after:.3f}")
    summary = {
        "pairs": len(pairs), "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
        "trainable_params": n_train, "val_mel_l1_before": before, "val_mel_l1_after": after,
        "adapter": asdict(acfg), "distill": asdict(dcfg), "seconds": round(time.time() - t0, 1),
    }
    save_adapter(backbone, acfg, args.out / "adapter.pt", extra=summary)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    size_kb = (args.out / "adapter.pt").stat().st_size / 1024
    print(f"saved {args.out / 'adapter.pt'} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    main()
