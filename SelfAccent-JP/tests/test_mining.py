from collections import Counter

from selfaccent.mining import DistillPair, count_words, load_pairs, mine_pairs, save_pairs, tag_words
from selfaccent.tags import PHON_END, PHON_START


CORPUS = [
    "橋を渡る。",
    "古い橋が見える。",
    "橋の上で待つ。",
    "東京駅で会う。",
    "東京に行く。",
    "東京は大きい。",
    "魑魅魍魎が出る。",
]


def test_count_words_counts_surfaces():
    counts = count_words(CORPUS)
    assert counts["橋"] == 3
    assert counts["魑魅魍魎"] == 1


def test_frequent_word_becomes_tagged_student_text():
    pairs = mine_pairs(["橋を渡る。"], word_counts=Counter({"橋": 5}), min_count=2)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.teacher_text == "橋を渡る。"
    assert p.student_text == f"{PHON_START}ハシ'{PHON_END}を渡る。"
    assert (p.word, p.tag, p.start, p.end) == ("橋", "ハシ'", 0, 1)


def test_rare_words_and_non_nouns_are_not_mined():
    pairs = mine_pairs(CORPUS, min_count=2)
    words = {p.word for p in pairs}
    assert "魑魅魍魎" not in words
    assert words <= {"橋", "東京"}


def test_compound_heads_are_skipped():
    # 東京 inside 東京駅 carries the compound's accent phrase, not its own
    pairs = mine_pairs(CORPUS, min_count=2)
    assert all(p.teacher_text != "東京駅で会う。" for p in pairs)
    assert any(p.teacher_text == "東京に行く。" for p in pairs)


def test_max_per_sentence_and_determinism():
    text = ["橋と箸と端を見る。"]
    counts = Counter({"橋": 3, "箸": 3, "端": 3})
    a = mine_pairs(text, word_counts=counts, min_count=1, max_per_sentence=2, seed=1)
    b = mine_pairs(text, word_counts=counts, min_count=1, max_per_sentence=2, seed=1)
    assert len(a) == 2
    assert a == b


def test_identity_pairs_keep_text_unchanged():
    pairs = mine_pairs(["橋を渡る。"], word_counts=Counter({"橋": 5}), identity_ratio=1.0)
    identity = [p for p in pairs if p.tag is None]
    assert len(identity) == 1
    assert identity[0].student_text == identity[0].teacher_text


def test_exclude_list_is_respected():
    pairs = mine_pairs(["橋を渡る。"], word_counts=Counter({"橋": 5}), exclude={"橋"})
    assert pairs == []


def test_pairs_round_trip_jsonl(tmp_path):
    pairs = mine_pairs(CORPUS, min_count=2, identity_ratio=0.5)
    path = tmp_path / "pairs.jsonl"
    save_pairs(pairs, path)
    assert load_pairs(path) == pairs
    assert all(isinstance(p, DistillPair) for p in load_pairs(path))


def test_tag_words_replaces_requested_words():
    out = tag_words("魑魅魍魎が跋扈する。", {"魑魅魍魎": "チ'ミ/モーリョー", "跋扈": "バ'ッコ"})
    assert out == (
        f"{PHON_START}チ'ミ/モーリョー{PHON_END}が{PHON_START}バ'ッコ{PHON_END}する。"
    )
