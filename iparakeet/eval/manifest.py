"""Dataset manifests (JSON lines) and readers for LibriSpeech and Common Voice."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    audio_filepath: str
    duration: float
    text: str


def write_manifest(path, entries: list[ManifestEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def read_manifest(path) -> list[ManifestEntry]:
    with Path(path).open(encoding="utf-8") as f:
        return [ManifestEntry(**json.loads(line)) for line in f if line.strip()]


def _duration(path: Path) -> float:
    return sf.info(str(path)).duration


def librispeech_entries(split_dir) -> list[ManifestEntry]:
    """Read a LibriSpeech split directory (<speaker>/<chapter>/*.trans.txt + *.flac)."""
    entries = []
    for trans in sorted(Path(split_dir).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            audio = trans.parent / f"{utt_id}.flac"
            entries.append(ManifestEntry(utt_id, str(audio), _duration(audio), text))
    return entries


def commonvoice_entries(tsv_path, clips_dir) -> list[ManifestEntry]:
    """Read a Common Voice split TSV (columns include `path` and `sentence`)."""
    entries = []
    with Path(tsv_path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            audio = Path(clips_dir) / row["path"]
            entries.append(ManifestEntry(audio.stem, str(audio), _duration(audio), row["sentence"]))
    return entries
