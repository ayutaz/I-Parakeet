"""Self-distillation of pronunciation / accent tags into a frozen backbone.

Procedure (arXiv 2609.17234, as far as the abstract describes it):

1. Mine pairs (teacher text with a common word, student text with that word
   replaced by its tag) -> :mod:`selfaccent.mining`.
2. Run the frozen backbone on the teacher text -> target (here: mel).
3. Add embedding rows for the tag symbols and LoRA on the backbone, then train
   only those with the backbone's own loss on (student text, target).

Because the target is the backbone's own rendition, no recordings are needed
and the adapter learns to make the tag sound exactly like the backbone's
correct reading of the word. At test time, any word (including unseen ones)
can be written as a tag.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from selfaccent.backbones.base import Backbone
from selfaccent.lora import adapter_state_dict, inject_lora, load_adapter_state_dict, mark_adapter_trainable
from selfaccent.mining import DistillPair
from selfaccent.tags import ACCENT_MARK, PHON_END, PHON_START, PHRASE_SEP

KATAKANA = [chr(c) for c in range(ord("ァ"), ord("ヺ") + 1)] + ["ー"]
TAG_SYMBOLS = [PHON_START, PHON_END, ACCENT_MARK, PHRASE_SEP] + KATAKANA


@dataclass
class AdapterConfig:
    # UtterTune defaults (configs/train/jsutjvs.yaml): r=16, alpha=64, dropout=0.05, q/k/v/o
    r: int = 16
    alpha: float = 64.0
    dropout: float = 0.05
    targets: list[str] | None = None


@dataclass
class DistillConfig:
    steps: int = 2000
    batch_size: int = 16
    lr: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 50


@dataclass
class _AdapterInfo:
    base_vocab_size: int
    base_vocab_sha1: str
    added_symbols: list[str] = field(default_factory=list)


def _vocab_sha1(symbols: list[str]) -> str:
    return hashlib.sha1("\n".join(symbols).encode("utf-8")).hexdigest()


def generate_teacher_targets(backbone: Backbone, pairs: list[DistillPair], seed: int = 0) -> list[Any]:
    """The frozen backbone's own output for every teacher text (seed + index per pair)."""
    return [backbone.synthesize([p.teacher_text], seeds=[seed + i])[0] for i, p in enumerate(pairs)]


def prepare_student(backbone: Backbone, cfg: AdapterConfig, tag_symbols: list[str] | None = None) -> list[str]:
    """Add tag tokens and LoRA, freeze the rest; return trainable parameter names."""
    symbols = list(TAG_SYMBOLS if tag_symbols is None else tag_symbols)
    base = list(backbone.tokenizer.symbols)
    backbone.add_tag_tokens(symbols)
    backbone.adapter_info = _AdapterInfo(len(base), _vocab_sha1(base), backbone.tokenizer.symbols[len(base) :])
    if hasattr(backbone, "prepare_for_adaptation"):
        backbone.prepare_for_adaptation()
    targets = cfg.targets or backbone.default_lora_targets()
    inject_lora(backbone.adapter_root(), targets, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
    return mark_adapter_trainable(backbone.model)


def _lr_at(step: int, cfg: DistillConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_student(
    backbone: Backbone,
    pairs: list[DistillPair],
    targets: list[Any],
    cfg: DistillConfig,
    log_fn: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Fit the adapter so that student texts reproduce the teacher targets."""
    if len(pairs) != len(targets):
        raise ValueError("one target per pair is required")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = [p for p in backbone.model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters; call prepare_student first")
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.98), weight_decay=cfg.weight_decay)
    order: list[int] = []
    history = []
    backbone.model.train()
    for step in range(cfg.steps):
        if len(order) < cfg.batch_size:
            fresh = list(range(len(pairs)))
            rng.shuffle(fresh)
            order.extend(fresh)
        idx, order = order[: cfg.batch_size], order[cfg.batch_size :]
        lr = _lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        losses = backbone.training_loss([pairs[i].student_text for i in idx], [targets[i] for i in idx])
        opt.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        record = {"step": step, "lr": lr, **{k: float(v) for k, v in losses.items()}}
        history.append(record)
        if log_fn and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            log_fn(record)
    backbone.model.eval()
    return history


def save_adapter(backbone: Backbone, cfg: AdapterConfig, path: str | Path, extra: dict | None = None) -> None:
    info: _AdapterInfo = backbone.adapter_info
    torch.save(
        {
            "adapter_config": asdict(cfg),
            "info": asdict(info),
            "state_dict": adapter_state_dict(backbone.model),
            "extra": extra or {},
        },
        path,
    )


def load_adapter(backbone: Backbone, path: str | Path) -> dict:
    """Attach a saved adapter to a freshly loaded backbone; returns ``extra``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    info = _AdapterInfo(**ckpt["info"])
    base = list(backbone.tokenizer.symbols)
    if len(base) != info.base_vocab_size or _vocab_sha1(base) != info.base_vocab_sha1:
        raise ValueError("adapter was trained for a different backbone vocabulary")
    prepare_student(backbone, AdapterConfig(**ckpt["adapter_config"]), tag_symbols=info.added_symbols)
    load_adapter_state_dict(backbone.model, ckpt["state_dict"])
    for p in backbone.model.parameters():
        p.requires_grad_(False)
    backbone.model.eval()
    return ckpt["extra"]
