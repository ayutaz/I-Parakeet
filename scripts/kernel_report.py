"""Accuracy report of the integer kernels (Swish variants, softmax range reductions).

    uv run python -m scripts.kernel_report --out results/kernels
"""

import argparse
import json
from pathlib import Path

from iparakeet.analysis.kernels import kernel_report_markdown, softmax_errors, swish_kernel_errors


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    swish = swish_kernel_errors(8, 8.0) + swish_kernel_errors(16, 60.0)
    softmax = softmax_errors((8.0, 16.0, 32.0), 8) + softmax_errors((32.0,), 16)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"swish": swish, "softmax": softmax}
    (out / "kernels.json").write_text(json.dumps(report, indent=2))
    md = kernel_report_markdown(swish, softmax)
    (out / "kernels.md").write_text(md + "\n")
    print(md)
    return report


if __name__ == "__main__":
    main()
