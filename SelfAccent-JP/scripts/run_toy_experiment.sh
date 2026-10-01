#!/usr/bin/env bash
# Full toy experiment on CPU (about 1.5 h on 4 cores):
#   corpus -> backbone -> pair mining -> self-distillation -> benchmark
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS_BACKBONE=${STEPS_BACKBONE:-8000}
STEPS_DISTILL=${STEPS_DISTILL:-1500}
THREADS=${THREADS:-3}

run() { echo "+ $*"; "$@"; }

[ -f data/toy/train_mels.pt ] || run uv run python -m scripts.make_toy_corpus --out data/toy

[ -f exp/backbone/model.pt ] || OMP_NUM_THREADS=$THREADS run uv run python -m scripts.train_backbone \
    --data data/toy --out exp/backbone --steps "$STEPS_BACKBONE" --threads "$THREADS"

run uv run python -m scripts.mine_pairs \
    --sentences data/toy/mine.jsonl --count-corpus data/toy/train.jsonl \
    --out exp/pairs.jsonl --min-count 3 --identity-ratio 0.1

OMP_NUM_THREADS=$THREADS run uv run python -m scripts.train_selfdistill \
    --backbone exp/backbone/model.pt --pairs exp/pairs.jsonl --out exp/adapter \
    --steps "$STEPS_DISTILL" --threads "$THREADS"

export OMP_NUM_THREADS=1
run uv run python -m scripts.benchmark validate-judge \
    --items data/toy/eval_difficult.jsonl --out results/judge_validation.json
run uv run python -m scripts.benchmark run \
    --backbone exp/backbone/model.pt --adapter exp/adapter/adapter.pt \
    --items data/toy/eval_difficult.jsonl --out results/benchmark_difficult.json
run uv run python -m scripts.benchmark run \
    --backbone exp/backbone/model.pt --adapter exp/adapter/adapter.pt \
    --items data/toy/eval_common.jsonl --conditions raw,tag,raw_adapted \
    --out results/benchmark_common.json
