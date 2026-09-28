"""M6: export per-bucket ONNX graphs, QNN encodings and a QAIRT build script.

    uv run python -m scripts.export_npu --nemo data/parakeet-ctc-0.6b.nemo \
        --stats results/range/calibration_stats.json --recipe iparakeet_lut --bucket-step 4 --out results/npu

Checks written to summary.json: activations without an encoding (each would become an FP op) and
compute-op inputs that are not quantized in the QDQ graph; both must be 0 for an integer-only graph.
"""

import argparse
import json
from pathlib import Path

import onnx

from iparakeet.analysis.range import CalibrationStats
from iparakeet.deploy.encodings import build_encodings, missing_encodings
from iparakeet.deploy.onnx_graph import build_onnx
from iparakeet.deploy.qdq import insert_qdq, unquantized_activation_inputs
from iparakeet.deploy.qnn import build_script, write_htp_configs
from iparakeet.model.buckets import BucketSet, uniform_buckets
from iparakeet.model.load_nemo import load_nemo
from iparakeet.sim.int_parakeet import IntParakeet
from iparakeet.sim.recipe import RECIPES


def load_model(path):
    return load_nemo(path).model


def _save(model: onnx.ModelProto, path: Path) -> None:
    if model.ByteSize() > 1_500_000_000:
        onnx.save(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=path.name + ".data")
    else:
        onnx.save(model, path)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--recipe", default="iparakeet_lut", choices=sorted(RECIPES))
    parser.add_argument("--buckets", type=float, nargs="+", default=None, help="bucket lengths in seconds")
    parser.add_argument("--bucket-step", type=float, default=4.0, help="uniform 3-35 s buckets if --buckets is not given")
    parser.add_argument("--dsp-arch", default="v73")
    parser.add_argument("--qdq", action="store_true", help="also save QDQ graphs (<bucket>.qdq.onnx)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    model = load_model(args.nemo).eval()
    buckets = BucketSet(args.buckets) if args.buckets else uniform_buckets(3.0, 35.0, args.bucket_step)
    sim = IntParakeet(model, CalibrationStats.load(args.stats), RECIPES[args.recipe], buckets.frames[-1])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = unquantized = 0
    encodings = None
    for name, frames in zip(buckets.names, buckets.frames):
        graph = build_onnx(model, frames, max_frames=buckets.frames[-1])
        encodings = build_encodings(sim, graph)
        missing += len(missing_encodings(graph, encodings))
        qdq = insert_qdq(graph, encodings)
        unquantized += len(unquantized_activation_inputs(qdq))
        _save(graph, out / f"{name}.onnx")
        if args.qdq:
            _save(qdq, out / f"{name}.qdq.onnx")
    (out / "encodings.json").write_text(json.dumps(encodings, indent=1))
    (out / "qnn_build.sh").write_text(build_script(buckets.names))
    write_htp_configs(out, buckets.names, args.dsp_arch)
    summary = {"recipe": args.recipe, "buckets": buckets.seconds, "missing_encodings": missing,
               "unquantized_inputs": unquantized, "multipliers": sim.multiplier_report()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
