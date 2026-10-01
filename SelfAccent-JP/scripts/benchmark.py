"""Benchmark reading / accent control, or validate the automatic judge.

    # 1) how reliable is the judge? (oracle speech at another tempo/pitch)
    uv run python -m scripts.benchmark validate-judge --items data/toy/eval_difficult.jsonl \
        --out results/judge_validation.json

    # 2) raw / kana / tag / tag_sweep on unseen difficult words
    uv run python -m scripts.benchmark run --backbone exp/backbone/model.pt \
        --adapter exp/adapter/adapter.pt --items data/toy/eval_difficult.jsonl \
        --out results/benchmark_difficult.json

Items followed by の are skipped by default (heiban and odaka are realized
alike there, so no listener or judge can tell them apart).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from selfaccent.backbones.tiny_matcha import TinyMatchaBackbone
from selfaccent.benchmark import (
    EvalItem,
    bootstrap_ci,
    neutralizing_context,
    run_benchmark,
    summarize,
    validate_judge,
)
from selfaccent.data.toy_corpus import load_sentences
from selfaccent.distill import load_adapter

CONDITION_LABELS = {
    "raw": "raw text (frozen backbone)",
    "kana": "plain kana (frozen backbone)",
    "tag": "tag, dictionary accent (+adapter)",
    "tag_sweep": "tag, every accent type (+adapter)",
    "raw_adapted": "raw text (+adapter, drift check)",
}


def load_items(path: Path, max_items: int, keep_neutralizing: bool, seed: int) -> list[EvalItem]:
    items = [EvalItem.from_sentence(s) for s in load_sentences(path)]
    if not keep_neutralizing:
        items = [it for it in items if not neutralizing_context(it)]
    if max_items and len(items) > max_items:
        rng = np.random.default_rng(seed)
        items = [items[i] for i in sorted(rng.choice(len(items), max_items, replace=False))]
    return items


def markdown_table(summary: dict) -> str:
    lines = [
        "| condition | n | accent acc (strict) | 95% CI | accent acc (heiban=odaka) | mel dist |",
        "|---|---|---|---|---|---|",
    ]
    for cond in ["raw", "kana", "tag", "tag_sweep", "raw_adapted"]:
        if cond not in summary:
            continue
        s = summary[cond]
        lo, hi = s["accent_acc_ci95"]
        lines.append(
            f"| {CONDITION_LABELS.get(cond, cond)} | {s['n']} | {s['accent_acc']:.3f} | [{lo:.2f}, {hi:.2f}] "
            f"| {s['accent_acc_lenient']:.3f} | {s['mel_dist']:.3f} |"
        )
    return "\n".join(lines)


def cmd_validate(args) -> None:
    items = load_items(args.items, args.max_items, args.keep_neutralizing, args.seed)
    t0 = time.time()
    records = validate_judge(items, speed=args.speed, half_tone=args.half_tone, gl_iters=args.gl_iters)
    strict = [r["correct"] for r in records]
    lenient = [r["correct_lenient"] for r in records]
    by_n: dict = {}
    for r in records:
        by_n.setdefault(r["n_morae"], []).append(r["correct"])
    out = {
        "items": len(items), "judgements": len(records), "speed": args.speed, "half_tone": args.half_tone,
        "accuracy": float(np.mean(strict)), "accuracy_ci95": bootstrap_ci(strict),
        "accuracy_lenient": float(np.mean(lenient)),
        "by_n_morae": {str(k): {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(by_n.items())},
        "chance": float(np.mean([1 / (r["n_morae"] + 1) for r in records])),
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": out, "records": records}, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_run(args) -> None:
    items = load_items(args.items, args.max_items, args.keep_neutralizing, args.seed)
    kw = dict(n_steps=args.n_steps, temperature=args.temperature)
    base = TinyMatchaBackbone.from_checkpoint(args.backbone, **kw)
    adapted = TinyMatchaBackbone.from_checkpoint(args.backbone, **kw)
    if args.adapter:
        load_adapter(adapted, args.adapter)
    conditions = tuple(args.conditions.split(","))
    t0 = time.time()

    def progress(i: int, n: int) -> None:
        if i % 10 == 0 or i == n:
            print(f"{i}/{n} items ({time.time() - t0:.0f}s)", flush=True)

    records = run_benchmark(
        {"base": base, "adapted": adapted}, items, seed=args.seed, gl_iters=args.gl_iters,
        conditions=conditions, progress=progress,
    )
    summary = summarize(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) for k, v in vars(args).items() if k != "func"}
    args.out.write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, ensure_ascii=False, indent=1))
    table = markdown_table(summary)
    args.out.with_suffix(".md").write_text(table + "\n")
    print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    for name, fn in (("run", cmd_run), ("validate-judge", cmd_validate)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--items", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--max-items", type=int, default=0)
        p.add_argument("--keep-neutralizing", action="store_true", help="also evaluate words followed by の")
        p.add_argument("--gl-iters", type=int, default=32)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int, default=1)
    run = sub.choices["run"]
    run.add_argument("--backbone", type=Path, required=True)
    run.add_argument("--adapter", type=Path, default=None)
    run.add_argument("--conditions", default="raw,kana,tag,tag_sweep")
    run.add_argument("--n-steps", type=int, default=10)
    run.add_argument("--temperature", type=float, default=0.667)
    val = sub.choices["validate-judge"]
    val.add_argument("--speed", type=float, default=1.1)
    val.add_argument("--half-tone", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
