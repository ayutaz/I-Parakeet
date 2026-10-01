from selfaccent.data.toy_corpus import (
    Sentence,
    build_splits,
    katakana_variant,
    slot_sentences,
    unseen_words,
)
from selfaccent.tags import parse_tag_body


def test_slot_sentences_keep_only_standalone_words():
    out = slot_sentences(["橋", "桜"], ["{}を見ました。", "{}の写真を撮りました。"])
    texts = {s.text for s in out}
    assert "橋を見ました。" in texts
    assert "桜の写真を撮りました。" not in texts  # 桜 does not head its own phrase there
    s = next(s for s in out if s.text == "橋を見ました。")
    assert s.word == "橋" and s.tag == "ハシ'" and s.template == 0
    assert s.start == 0 and s.end == 1


def test_katakana_variant_rewrites_the_slot_word():
    s = Sentence("橋を見ました。", "橋", "ハシ'", 0, 0, 1)
    v = katakana_variant(s)
    assert v.text == "ハシを見ました。"
    assert v.word == "ハシ" and v.tag == "ハシ'" and v.end == 2


def test_unseen_words_filters_by_character():
    assert unseen_words(["齟齬", "海鼠", "跋扈"], ["海を見ました。"]) == ["齟齬", "跋扈"]


def test_splits_are_disjoint_and_hide_difficult_words():
    splits = build_splits(
        common=["橋", "箸", "雨", "コーヒー"],
        difficult=["齟齬", "跋扈", "海鼠"],
        templates=["{}を見ました。", "{}が好きです。", "彼は{}が苦手です。", "{}について調べました。"],
        train_templates_per_word=2,
        kana_variant_ratio=0.5,
        eval_templates_per_word=2,
        seed=0,
    )
    train = {s.text for s in splits["train"]}
    mine = {s.text for s in splits["mine"]}
    assert train and mine and not (train & mine)
    train_chars = set("".join(train))
    for s in splits["eval_difficult"]:
        assert not (set(s.word) & train_chars)
    assert {s.word for s in splits["eval_difficult"]} <= {"齟齬", "跋扈", "海鼠"}
    eval_common = {s.text for s in splits["eval_common"]}
    assert eval_common and not (eval_common & (train | mine))
    assert any(s.kind == "kana_variant" for s in splits["train"])
    for s in splits["eval_difficult"]:
        parse_tag_body(s.tag)  # valid tag
