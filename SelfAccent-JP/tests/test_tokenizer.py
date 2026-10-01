from selfaccent.tags import PHON_END, PHON_START
from selfaccent.tokenizer import CharTokenizer


def test_build_and_round_trip():
    tok = CharTokenizer.build(["橋を渡る。", "箸で食べる。"])
    ids = tok.encode("橋で食べる。")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "橋で食べる。"


def test_unknown_characters_map_to_unk():
    tok = CharTokenizer.build(["橋を渡る。"])
    ids = tok.encode("魑", add_bos_eos=False)
    assert ids == [tok.unk_id]
    assert tok.unknown_symbols("魑を渡る") == {"魑"}


def test_tag_markers_are_single_tokens_once_added():
    tok = CharTokenizer.build(["橋を渡る。"])
    n = len(tok)
    new_ids = tok.extend([PHON_START, PHON_END, "'", "/", "ハ", "シ"])
    assert new_ids == list(range(n, n + 6))
    ids = tok.encode(f"{PHON_START}ハシ'{PHON_END}を", add_bos_eos=False)
    assert ids == [new_ids[0], new_ids[4], new_ids[5], new_ids[2], new_ids[1], tok.id_of("を")]


def test_extend_skips_existing_symbols():
    tok = CharTokenizer.build(["ハシを渡る。"])
    n = len(tok)
    ids = tok.extend(["ハ", "'"])
    assert ids == [tok.id_of("ハ"), n]
    assert len(tok) == n + 1


def test_serialization_round_trip():
    tok = CharTokenizer.build(["橋を渡る。"])
    tok.extend([PHON_START])
    again = CharTokenizer.from_dict(tok.to_dict())
    assert again.symbols == tok.symbols
    assert again.encode(f"{PHON_START}橋") == tok.encode(f"{PHON_START}橋")


def test_build_is_deterministic():
    a = CharTokenizer.build(["あいう", "うえお"])
    b = CharTokenizer.build(["うえお", "あいう"])
    assert a.symbols == b.symbols
