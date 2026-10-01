import torch

from selfaccent.audio import MelConfig, MelExtractor
from selfaccent.benchmark import (
    EvalItem,
    build_references,
    condition_inputs,
    lenient_match,
    neutralizing_context,
    run_benchmark,
    summarize,
    validate_judge,
)
from selfaccent.data.toy_corpus import Sentence
from selfaccent.frontend import OracleRenderer, analyze
from selfaccent.tags import AccentPhrase, PHON_END, PHON_START, find_tags

ITEM = EvalItem.from_sentence(Sentence("昨日、橋を見ました。", "橋", "ハシ'", 0, 3, 4, "eval_difficult"))


def test_condition_inputs():
    inputs = condition_inputs(ITEM)
    assert inputs["raw"] == "昨日、橋を見ました。"
    assert inputs["kana"] == "昨日、ハシを見ました。"
    assert inputs["tag"] == f"昨日、{PHON_START}ハシ'{PHON_END}を見ました。"
    assert inputs["tag_k0"] == f"昨日、{PHON_START}ハシ{PHON_END}を見ました。"
    assert inputs["tag_k1"] == f"昨日、{PHON_START}ハ'シ{PHON_END}を見ました。"
    assert inputs["tag_k2"] == inputs["tag"]
    assert ITEM.n_morae == 2 and ITEM.accent == 2


class OracleBackbone:
    """Pretends to be a perfect TTS: speaks every input exactly as asked."""

    def __init__(self, item: EvalItem):
        self.cfg = MelConfig()
        self.oracle = OracleRenderer(self.cfg.sample_rate)
        self.ext = MelExtractor(self.cfg)
        self.item = item

    def synthesize(self, texts, seeds):
        out = []
        for text in texts:
            spans = find_tags(text)
            if spans:
                idx = next(w.index for w in analyze(self.item.text) if w.start == self.item.start)
                wav = self.oracle.render_with_override(self.item.text, idx, spans[0].phrases)
            else:
                wav = self.oracle.render(self.item.text)
            out.append(self.ext(torch.from_numpy(wav)))
        return out


def test_references_cover_every_accent_type():
    refset = build_references(OracleRenderer(22050), ITEM, MelConfig())
    assert len(refset.refs) == 3


def test_lenient_match_merges_heiban_and_odaka_only():
    assert lenient_match(0, 3, 3) and lenient_match(3, 0, 3)
    assert not lenient_match(1, 0, 3)
    assert lenient_match(2, 2, 3)


def test_neutralizing_context():
    assert neutralizing_context(EvalItem.from_sentence(Sentence("橋の写真を撮りました。", "橋", "ハシ'", 4, 0, 1)))
    assert not neutralizing_context(ITEM)


def test_validate_judge_on_unperturbed_oracle_is_perfect():
    records = validate_judge([ITEM], speed=1.0, half_tone=0.0)
    assert len(records) == 3
    assert all(r["correct"] for r in records)


def test_perfect_backbone_scores_full_accuracy():
    bb = OracleBackbone(ITEM)
    records = run_benchmark({"base": bb, "adapted": bb}, [ITEM], seed=0, gl_iters=32)
    conds = {r["condition"] for r in records}
    assert conds == {"raw", "kana", "tag", "tag_sweep"}
    for r in records:
        assert r["correct"], r
    summary = summarize(records)
    assert summary["tag"]["accent_acc"] == 1.0
    assert summary["tag_sweep"]["accent_acc"] == 1.0
    assert summary["tag_sweep"]["n"] == 3
    assert summary["tag"]["accent_acc_lenient"] == 1.0
