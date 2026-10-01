import re

import numpy as np
import pytest

from selfaccent.frontend import OracleRenderer, analyze, word_phrases
from selfaccent.tags import AccentPhrase


def _phonemes(labels):
    return [l.split("-")[1].split("+")[0] for l in labels]


def _phrase_fields(labels):
    return sorted(set(re.search(r"/F:([^#]*)#", l).group(1) for l in labels[1:-1] if "/F:xx" not in l))


def test_analyze_gives_reading_accent_and_offsets():
    text = "昨日、橋を渡りました。"
    words = analyze(text)
    hashi = next(w for w in words if w.surface == "橋")
    assert hashi.pron == "ハシ"
    assert hashi.acc == 2
    assert hashi.mora_size == 2
    assert hashi.heads_phrase
    assert text[hashi.start : hashi.end] == "橋"
    wo = next(w for w in words if w.surface == "を")
    assert not wo.heads_phrase


def test_analyze_strips_devoicing_marks_from_pron():
    words = analyze("渡りました。")
    assert all("’" not in w.pron for w in words)
    assert any(w.pron == "マシ" for w in words)


@pytest.mark.parametrize("text, surface, expected", [
    ("橋を渡る。", "橋", AccentPhrase(("ハ", "シ"), 2)),
    ("端を歩く。", "端", AccentPhrase(("ハ", "シ"), 0)),
    ("箸で食べる。", "箸", AccentPhrase(("ハ", "シ"), 1)),
])
def test_word_phrases_follow_dictionary_accent(text, surface, expected):
    word = next(w for w in analyze(text) if w.surface == surface)
    assert word_phrases(word) == [expected]


def test_word_phrases_is_none_for_chained_words():
    wo = next(w for w in analyze("橋を渡る。") if w.surface == "を")
    assert word_phrases(wo) is None


def test_oracle_render_returns_normalized_audio():
    oracle = OracleRenderer(sample_rate=22050)
    wav = oracle.render("橋を渡る。")
    assert wav.dtype == np.float32
    assert 0.5 < len(wav) / 22050 < 3.0
    assert 0.05 < np.abs(wav).max() <= 1.0


def test_override_with_dictionary_accent_matches_plain_render():
    oracle = OracleRenderer(sample_rate=22050)
    text = "橋を渡る。"
    idx = next(w.index for w in analyze(text) if w.surface == "橋")
    a = oracle.render(text)
    b = oracle.render_with_override(text, idx, [AccentPhrase(("ハ", "シ"), 2)])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_override_changes_accent_phrase_labels():
    oracle = OracleRenderer(sample_rate=22050)
    text = "橋を渡る。"
    idx = next(w.index for w in analyze(text) if w.surface == "橋")
    heiban = oracle.labels_with_override(text, idx, [AccentPhrase(("ハ", "シ"), 0)])
    atamadaka = oracle.labels_with_override(text, idx, [AccentPhrase(("ハ", "シ"), 1)])
    # HTS encodes heiban as accent position == mora count of the phrase (ハシヲ = 3)
    assert "3_3" in _phrase_fields(heiban)
    assert "3_1" in _phrase_fields(atamadaka)


def test_override_can_replace_reading_with_several_phrases():
    oracle = OracleRenderer(sample_rate=22050)
    text = "橋を渡る。"
    idx = next(w.index for w in analyze(text) if w.surface == "橋")
    phrases = [AccentPhrase.from_kana("チミ", 1), AccentPhrase.from_kana("モーリョー", 0)]
    labels = oracle.labels_with_override(text, idx, phrases)
    ph = _phonemes(labels)
    assert ph[1:6] == ["ch", "i", "m", "i", "m"]
    assert "ry" in ph
    fields = _phrase_fields(labels)
    assert "2_1" in fields  # チ'ミ
    assert "5_5" in fields  # モーリョーヲ, heiban
