"""M2: pick the uniform bucket step whose silence-padding overhead is closest to the paper's 23%.

    uv run python -m scripts.choose_buckets --manifest data/manifests/test-other.jsonl --target 0.23
"""

import argparse
import json

from iparakeet.eval.manifest import read_manifest
from iparakeet.model.buckets import search_uniform_step, uniform_buckets


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target", type=float, default=0.23)
    parser.add_argument("--lo", type=float, default=3.0)
    parser.add_argument("--hi", type=float, default=35.0)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.5, 1, 2, 3, 4, 5, 6, 8, 10, 16, 32])
    args = parser.parse_args(argv)
    durations = [e.duration for e in read_manifest(args.manifest)]
    step, ratio = search_uniform_step(durations, args.target, args.steps, args.lo, args.hi)
    result = {"step": step, "padding_ratio": ratio, "buckets": uniform_buckets(args.lo, args.hi, step).seconds}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
