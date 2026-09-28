"""M1: download LibriSpeech / Parakeet-CTC-0.6B and build JSON-lines manifests.

    uv run python -m scripts.prepare_data --download --data-dir data \
        --splits dev-other test-clean test-other
    uv run python -m scripts.prepare_data --librispeech-root data/LibriSpeech --out data/manifests
    uv run python -m scripts.prepare_data --commonvoice-tsv cv/en/test.tsv --commonvoice-clips cv/en/clips \
        --out data/manifests

Common Voice must be downloaded manually (Mozilla terms of use); the paper does not state its version.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

from iparakeet.eval.manifest import commonvoice_entries, librispeech_entries, write_manifest

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
PARAKEET_URL = "https://huggingface.co/nvidia/parakeet-ctc-0.6b/resolve/main/parakeet-ctc-0.6b.nemo"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch LibriSpeech splits and the .nemo checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--librispeech-root", help="directory containing the split folders (default: <data-dir>/LibriSpeech)")
    parser.add_argument("--splits", nargs="+", default=["dev-other", "test-clean", "test-other"])
    parser.add_argument("--commonvoice-tsv")
    parser.add_argument("--commonvoice-clips")
    parser.add_argument("--out", default=None, help="manifest directory (default: <data-dir>/manifests)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "manifests"
    root = Path(args.librispeech_root) if args.librispeech_root else data_dir / "LibriSpeech"
    if args.download:
        download(PARAKEET_URL, data_dir / "parakeet-ctc-0.6b.nemo")
        for split in args.splits:
            archive = download(LIBRISPEECH_URL.format(split=split), data_dir / f"{split}.tar.gz")
            if not (root / split).exists():
                with tarfile.open(archive) as tar:
                    tar.extractall(data_dir, filter="data")
    for split in args.splits:
        if (root / split).exists():
            entries = librispeech_entries(root / split)
            write_manifest(out / f"{split}.jsonl", entries)
            print(f"{split}: {len(entries)} utterances, {sum(e.duration for e in entries) / 3600:.2f} h")
    if args.commonvoice_tsv:
        entries = commonvoice_entries(args.commonvoice_tsv, args.commonvoice_clips)
        write_manifest(out / "commonvoice-test.jsonl", entries)
        print(f"commonvoice-test: {len(entries)} utterances")


if __name__ == "__main__":
    main()
