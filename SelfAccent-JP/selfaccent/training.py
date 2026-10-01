"""Training loop for the toy backbone (the paper uses existing pretrained ones)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from selfaccent.backbones.tiny_matcha import TinyMatchaBackbone


@dataclass
class TrainConfig:
    steps: int = 8000
    batch_size: int = 16
    lr: float = 5e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    seed: int = 0
    log_every: int = 100


def length_batches(lengths: list[int], batch_size: int, seed: int, bucket_size: int = 512) -> Iterator[list[int]]:
    """One epoch of batches; similar lengths are grouped to limit padding."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    for k in range(0, len(order), bucket_size):
        bucket = sorted(order[k : k + bucket_size], key=lambda i: lengths[i])
        batches.extend(bucket[j : j + batch_size] for j in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    yield from batches


def compute_mel_stats(mels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    cat = torch.cat([m.float() for m in mels], dim=1)
    return cat.mean(1), cat.std(1).clamp(min=1e-3)


def _lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def train_backbone(
    backbone: TinyMatchaBackbone,
    texts: list[str],
    mels: list[torch.Tensor],
    cfg: TrainConfig,
    log_fn: Callable[[dict], None] | None = None,
    checkpoint_fn: Callable[[int], None] | None = None,
    checkpoint_every: int = 0,
) -> list[dict]:
    torch.manual_seed(cfg.seed)
    model = backbone.model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.98), weight_decay=cfg.weight_decay)
    lengths = [m.shape[-1] for m in mels]
    history: list[dict] = []
    epoch = 0
    batches: Iterator[list[int]] = iter(())
    model.train()
    for step in range(cfg.steps):
        batch = next(batches, None)
        if batch is None:
            batches = length_batches(lengths, cfg.batch_size, cfg.seed + epoch)
            epoch += 1
            batch = next(batches)
        lr = _lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        losses = backbone.training_loss([texts[i] for i in batch], [mels[i].float() for i in batch])
        opt.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        rec = {"step": step, "epoch": epoch, "lr": lr, **{k: v.item() for k, v in losses.items()}}
        history.append(rec)
        if log_fn and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            log_fn(rec)
        if checkpoint_fn and checkpoint_every and (step + 1) % checkpoint_every == 0:
            checkpoint_fn(step + 1)
    model.eval()
    return history
