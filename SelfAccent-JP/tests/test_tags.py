import pytest

from selfaccent.tags import (
    PHON_END,
    PHON_START,
    AccentPhrase,
    TagError,
    find_tags,
    format_tag_body,
    hira_to_kata,
    make_tag,
    parse_tag_body,
    replace_span_with_tag,
    split_morae,
    strip_tags,
)


def test_split_morae_attaches_small_kana_and_keeps_special_morae():
    assert split_morae("チミモーリョー") == ["チ", "ミ", "モ", "ー", "リョ", "ー"]
    assert split_morae("イッシューカン") == ["イ", "ッ", "シュ", "ー", "カ", "ン"]
    assert split_morae("ファイル") == ["ファ", "イ", "ル"]


def test_split_morae_rejects_non_katakana_and_orphan_small_kana():
    with pytest.raises(TagError):
        split_morae("ハシa")
    with pytest.raises(TagError):
        split_morae("ャマ")


def test_hira_to_kata_converts_only_hiragana():
    assert hira_to_kata("はし、ハシ") == "ハシ、ハシ"


@pytest.mark.parametrize(
    "accent, expected",
    [
        (0, [0, 1, 1, 1]),  # heiban: L H H H
        (1, [1, 0, 0, 0]),  # atamadaka: H L L L
        (2, [0, 1, 0, 0]),  # nakadaka
        (4, [0, 1, 1, 1]),  # odaka: same morae as heiban, fall comes on the particle
    ],
)
def test_pitch_pattern_follows_tokyo_rules(accent, expected):
    phrase = AccentPhrase(("サ", "ク", "ラ", "ン"), accent)
    assert phrase.pitch() == expected


def test_odaka_and_heiban_differ_on_following_particle():
    assert AccentPhrase(("ハ", "シ"), 2).pitch(with_particle=True) == [0, 1, 0]
    assert AccentPhrase(("ハ", "シ"), 0).pitch(with_particle=True) == [0, 1, 1]


def test_accent_phrase_validates_accent_range():
    with pytest.raises(TagError):
        AccentPhrase(("ハ", "シ"), 3)
    with pytest.raises(TagError):
        AccentPhrase((), 0)


def test_format_and_parse_round_trip_utter_tune_notation():
    body = "チ'ミ/モーリョー"
    phrases = parse_tag_body(body)
    assert phrases == [
        AccentPhrase(("チ", "ミ"), 1),
        AccentPhrase(("モ", "ー", "リョ", "ー"), 0),
    ]
    assert format_tag_body(phrases) == body


def test_parse_accent_after_last_mora_is_odaka():
    assert parse_tag_body("ハシ'") == [AccentPhrase(("ハ", "シ"), 2)]


@pytest.mark.parametrize("bad", ["", "'ハシ", "ハ'シ'", "ハシ//ハシ", "ハシ/", "橋"])
def test_parse_rejects_malformed_bodies(bad):
    with pytest.raises(TagError):
        parse_tag_body(bad)


def test_make_tag_wraps_body_in_special_tokens():
    tag = make_tag([AccentPhrase(("バ", "ッ", "コ"), 1)])
    assert tag == f"{PHON_START}バ'ッコ{PHON_END}"


def test_find_tags_returns_offsets_and_phrases():
    text = f"水を{PHON_START}マレ'ーシア{PHON_END}から買う。"
    spans = find_tags(text)
    assert len(spans) == 1
    span = spans[0]
    assert text[span.start : span.end] == f"{PHON_START}マレ'ーシア{PHON_END}"
    assert span.phrases == [AccentPhrase(("マ", "レ", "ー", "シ", "ア"), 2)]


def test_find_tags_rejects_unclosed_tag():
    with pytest.raises(TagError):
        find_tags(f"{PHON_START}ハシ を渡る")


def test_strip_tags_kana_mode_gives_plain_kana_baseline():
    text = f"{PHON_START}チ'ミ/モーリョー{PHON_END}が{PHON_START}バ'ッコ{PHON_END}する。"
    assert strip_tags(text) == "チミモーリョーがバッコする。"


def test_replace_span_with_tag():
    text = "魑魅魍魎が跋扈する。"
    out = replace_span_with_tag(text, 5, 7, [AccentPhrase(("バ", "ッ", "コ"), 1)])
    assert out == f"魑魅魍魎が{PHON_START}バ'ッコ{PHON_END}する。"
