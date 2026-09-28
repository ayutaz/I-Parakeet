import pytest

from iparakeet.eval.text import corpus_wer, normalize


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("HELLO, WORLD!") == "hello world"


def test_normalize_uses_whisper_english_rules():
    # British spelling and spelled-out numbers are standardized (Whisper EnglishTextNormalizer)
    assert normalize("the colour of twenty five apples") == "the color of 25 apples"


def test_corpus_wer_counts_errors_over_all_reference_words():
    refs = ["a b c", "d e"]
    hyps = ["a x c", "d e f"]  # 1 substitution + 1 insertion over 5 words
    result = corpus_wer(refs, hyps)
    assert result.errors == 2
    assert result.ref_words == 5
    assert result.wer == pytest.approx(0.4)


def test_corpus_wer_normalizes_both_sides():
    result = corpus_wer(["Hello, World!"], ["hello world"])
    assert result.wer == 0.0


def test_corpus_wer_skips_references_that_normalize_to_empty():
    result = corpus_wer(["", "a b"], ["x", "a b"])
    assert result.n_skipped == 1
    assert result.n_utts == 1
    assert result.wer == 0.0


def test_corpus_wer_rejects_length_mismatch():
    with pytest.raises(ValueError):
        corpus_wer(["a"], ["a", "b"])
