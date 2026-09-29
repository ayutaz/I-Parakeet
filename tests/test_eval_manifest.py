import numpy as np
import soundfile as sf

from iparakeet.eval.manifest import (
    ManifestEntry,
    commonvoice_entries,
    librispeech_entries,
    read_manifest,
    write_manifest,
)


def _write_wav(path, seconds, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(seconds * sr), dtype=np.float32), sr)


def test_manifest_roundtrip(tmp_path):
    entries = [
        ManifestEntry(utt_id="u1", audio_filepath="/a/1.flac", duration=1.5, text="hello"),
        ManifestEntry(utt_id="u2", audio_filepath="/a/2.flac", duration=2.0, text="world"),
    ]
    path = tmp_path / "m.jsonl"
    write_manifest(path, entries)
    assert read_manifest(path) == entries


def test_librispeech_reader_pairs_transcripts_with_audio(tmp_path):
    chapter = tmp_path / "test-other" / "1688" / "142285"
    _write_wav(chapter / "1688-142285-0000.flac", 1.0)
    _write_wav(chapter / "1688-142285-0001.flac", 2.5)
    (chapter / "1688-142285.trans.txt").write_text(
        "1688-142285-0000 THERE'S IRON THEY SAY\n1688-142285-0001 IN ALL THE HEAVENS\n"
    )
    entries = librispeech_entries(tmp_path / "test-other")
    assert [e.utt_id for e in entries] == ["1688-142285-0000", "1688-142285-0001"]
    assert entries[0].text == "THERE'S IRON THEY SAY"
    assert entries[1].duration == 2.5
    assert entries[1].audio_filepath.endswith("1688-142285-0001.flac")


def test_commonvoice_reader_uses_tsv_sentence_and_clip_path(tmp_path):
    clips = tmp_path / "clips"
    _write_wav(clips / "common_voice_en_1.wav", 3.0)
    (tmp_path / "test.tsv").write_text(
        "client_id\tpath\tsentence\tup_votes\n" "abc\tcommon_voice_en_1.wav\tHello there.\t2\n"
    )
    entries = commonvoice_entries(tmp_path / "test.tsv", clips)
    assert len(entries) == 1
    assert entries[0].utt_id == "common_voice_en_1"
    assert entries[0].text == "Hello there."
    assert entries[0].duration == 3.0
