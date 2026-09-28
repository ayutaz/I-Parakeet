"""M7: features padded to their bucket, written as qnn-net-run raw inputs + input lists + mapping.json.

    uv run python -m scripts.prepare_device_inputs --nemo data/parakeet-ctc-0.6b.nemo \
        --manifest data/manifests/test-other.jsonl --bucket-step 4 --out results/device_inputs
    adb push results/device_inputs /data/local/tmp/iparakeet/
"""

import argparse

from iparakeet.deploy.device import prepare_device_inputs
from iparakeet.eval.manifest import read_manifest
from iparakeet.model.buckets import BucketSet, uniform_buckets
from iparakeet.model.load_nemo import load_nemo


def load_model(path):
    return load_nemo(path).model


def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--buckets", type=float, nargs="+", default=None)
    parser.add_argument("--bucket-step", type=float, default=4.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    buckets = BucketSet(args.buckets) if args.buckets else uniform_buckets(3.0, 35.0, args.bucket_step)
    mapping = prepare_device_inputs(load_model(args.nemo).eval(), read_manifest(args.manifest), buckets, args.out)
    print(f"{len(mapping)} utterances in {len({m['bucket'] for m in mapping})} buckets -> {args.out}")
    return mapping


if __name__ == "__main__":
    main()
