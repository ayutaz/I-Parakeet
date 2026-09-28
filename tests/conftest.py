import pytest
import torch

from iparakeet.model.config import ParakeetConfig
from iparakeet.model.parakeet import ParakeetCTC


def tiny_config(**overrides) -> ParakeetConfig:
    params = dict(
        feat_in=80,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_expansion_factor=4,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=8,
        vocab_size=16,
        xscaling=False,
        pos_emb_max_len=512,
    )
    params.update(overrides)
    return ParakeetConfig(**params)


def randomize_(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Give every parameter and BN statistic a non-trivial random value."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (0.1 if p.dim() < 2 else p.shape[-1] ** -0.5))
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=gen) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=gen) + 0.5)
                m.weight.add_(1.0)
            if isinstance(m, torch.nn.LayerNorm):
                m.weight.add_(1.0)
    return model


@pytest.fixture
def tiny_cfg() -> ParakeetConfig:
    return tiny_config()


@pytest.fixture
def tiny_model(tiny_cfg) -> ParakeetCTC:
    torch.manual_seed(0)
    return randomize_(ParakeetCTC(tiny_cfg)).eval()
