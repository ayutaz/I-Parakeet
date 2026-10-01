import torch
from torch import nn

from selfaccent.lora import (
    ExtendedEmbedding,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
    load_adapter_state_dict,
    mark_adapter_trainable,
    merge_lora,
)


class Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.ffn = nn.Linear(d, d)

    def forward(self, x):
        return self.ffn(torch.relu(self.q_proj(x) + self.k_proj(x)))


class Toy(nn.Module):
    def __init__(self, vocab=10, d=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d), Block(d)])

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return x


def test_lora_linear_starts_as_identity_of_base():
    torch.manual_seed(0)
    base = nn.Linear(6, 4)
    lora = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 6)
    torch.testing.assert_close(lora(x), base(x))


def test_lora_linear_adds_scaled_low_rank_update():
    torch.manual_seed(0)
    base = nn.Linear(6, 4)
    lora = LoRALinear(base, r=2, alpha=4)
    nn.init.normal_(lora.lora_B)
    x = torch.randn(3, 6)
    expected = base(x) + 2.0 * (x @ lora.lora_A.T @ lora.lora_B.T)
    torch.testing.assert_close(lora(x), expected)


def test_inject_lora_replaces_only_target_modules():
    model = Toy()
    names = inject_lora(model, ["q_proj", "k_proj"], r=2, alpha=4)
    assert sorted(names) == sorted(
        ["blocks.0.q_proj", "blocks.0.k_proj", "blocks.1.q_proj", "blocks.1.k_proj"]
    )
    assert isinstance(model.blocks[0].q_proj, LoRALinear)
    assert isinstance(model.blocks[0].ffn, nn.Linear)


def test_mark_adapter_trainable_freezes_everything_else():
    model = Toy()
    inject_lora(model, ["q_proj"], r=2, alpha=4)
    model.emb = ExtendedEmbedding(model.emb, n_new=3)
    mark_adapter_trainable(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        "emb.extra",
        "blocks.0.q_proj.lora_A",
        "blocks.0.q_proj.lora_B",
        "blocks.1.q_proj.lora_A",
        "blocks.1.q_proj.lora_B",
    }


def test_extended_embedding_keeps_old_rows_and_trains_new_rows_only():
    torch.manual_seed(0)
    base = nn.Embedding(5, 4)
    emb = ExtendedEmbedding(base, n_new=2)
    old = base.weight.detach().clone()
    ids = torch.tensor([0, 4, 5, 6])
    torch.testing.assert_close(emb(ids)[:2], old[[0, 4]])
    assert emb.num_embeddings == 7
    mark_adapter_trainable(emb)
    opt = torch.optim.AdamW([p for p in emb.parameters() if p.requires_grad], lr=0.1, weight_decay=0.1)
    before = emb.extra.detach().clone()
    emb(ids).pow(2).sum().backward()
    opt.step()
    torch.testing.assert_close(base.weight.detach(), old)
    assert not torch.allclose(emb.extra.detach(), before)


def test_merge_lora_preserves_outputs_and_restores_linear():
    torch.manual_seed(0)
    model = Toy()
    inject_lora(model, ["q_proj", "k_proj"], r=2, alpha=4)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)
    ids = torch.tensor([[1, 2, 3]])
    before = model(ids)
    merge_lora(model)
    assert isinstance(model.blocks[0].q_proj, nn.Linear)
    torch.testing.assert_close(model(ids), before)


def test_adapter_state_dict_round_trip():
    torch.manual_seed(0)
    model = Toy()
    inject_lora(model, ["q_proj"], r=2, alpha=4)
    model.emb = ExtendedEmbedding(model.emb, n_new=2)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)
    nn.init.normal_(model.emb.extra)
    state = adapter_state_dict(model)
    assert set(state) == {
        "emb.extra",
        "blocks.0.q_proj.lora_A",
        "blocks.0.q_proj.lora_B",
        "blocks.1.q_proj.lora_A",
        "blocks.1.q_proj.lora_B",
    }

    torch.manual_seed(0)
    fresh = Toy()
    inject_lora(fresh, ["q_proj"], r=2, alpha=4)
    fresh.emb = ExtendedEmbedding(fresh.emb, n_new=2)
    load_adapter_state_dict(fresh, state)
    ids = torch.tensor([[1, 10, 11]])
    torch.testing.assert_close(fresh(ids), model(ids))


def test_lora_dropout_is_disabled_in_eval():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(4, 4), r=2, alpha=4, dropout=0.5)
    nn.init.normal_(lora.lora_B)
    lora.eval()
    x = torch.randn(2, 4)
    torch.testing.assert_close(lora(x), lora(x))
