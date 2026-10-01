"""What self-distillation needs from a TTS backbone.

Any raw-text TTS can be adapted by implementing this protocol: the teacher
output type ("target") is whatever the backbone's native training loss consumes
(mel frames for flow/diffusion decoders, speech tokens for codec LMs). See
docs/03_architecture.md for how CosyVoice2-style LMs map onto it.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn

from selfaccent.tokenizer import CharTokenizer


class Backbone(Protocol):
    model: nn.Module
    tokenizer: CharTokenizer

    def synthesize(self, texts: list[str], seeds: list[int]) -> list[Any]:
        """Generate one target per text (deterministic for a given seed)."""

    def training_loss(self, texts: list[str], targets: list[Any]) -> dict[str, torch.Tensor]:
        """The backbone's own training loss for (text, target) pairs; must contain "loss"."""

    def add_tag_tokens(self, symbols: list[str]) -> int:
        """Add unseen symbols to the vocabulary with trainable embedding rows; return how many."""

    def adapter_root(self) -> nn.Module:
        """Sub-module in which LoRA layers are injected."""

    def default_lora_targets(self) -> list[str]:
        """Names of the linear layers to adapt."""
