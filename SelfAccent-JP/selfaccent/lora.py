"""Minimal LoRA and vocabulary-extension modules.

The adapter that self-distillation trains has two parts, as in UtterTune:
LoRA on selected linear layers, and embedding rows for the tag tokens that the
frozen backbone has never seen. Everything else stays frozen, so the adapter
is a few hundred kB and can be switched on and off.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)

    def merged(self) -> nn.Linear:
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scale * self.lora_B @ self.lora_A)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


class ExtendedEmbedding(nn.Module):
    """A frozen embedding table plus ``n_new`` trainable rows appended at the end."""

    def __init__(self, base: nn.Embedding, n_new: int, init: torch.Tensor | None = None) -> None:
        super().__init__()
        self.base = base
        self.n_base = base.num_embeddings
        if init is None:
            w = base.weight.detach()
            init = w.mean(0, keepdim=True) + w.std() * torch.randn(n_new, w.shape[1]) * 0.5
        self.extra = nn.Parameter(init.clone())

    @property
    def num_embeddings(self) -> int:
        return self.n_base + self.extra.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.base.embedding_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        is_new = ids >= self.n_base
        old = self.base(ids.clamp(max=self.n_base - 1))
        if not bool(is_new.any()):
            return old
        new = self.extra[(ids - self.n_base).clamp(min=0)]
        return torch.where(is_new.unsqueeze(-1), new, old)


def _matches(name: str, targets: list[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return any(leaf == t or name.endswith("." + t) or name == t for t in targets)


def inject_lora(model: nn.Module, targets: list[str], r: int, alpha: float, dropout: float = 0.0) -> list[str]:
    """Wrap every ``nn.Linear`` whose name ends with one of ``targets``."""
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name, targets):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced.append(name)
    return replaced


def _is_adapter_param(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in ("lora_A", "lora_B", "extra")


def mark_adapter_trainable(model: nn.Module) -> list[str]:
    names = []
    for name, p in model.named_parameters():
        p.requires_grad_(_is_adapter_param(name))
        if p.requires_grad:
            names.append(name)
    return names


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if _is_adapter_param(n)}


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = [n for n in params if _is_adapter_param(n) and n not in state]
    unexpected = [n for n in state if n not in params]
    if missing or unexpected:
        raise KeyError(f"adapter mismatch: missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t)


def merge_lora(model: nn.Module) -> None:
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module.merged())
