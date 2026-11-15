"""Generate (or reuse) the answers the crossover analysis needs.

    uv run python scripts/dial_generate.py --dataset hotpotqa \
        --slice results/en_rag/slice --cache results/crossover/cache/hotpotqa \
        --retriever e5-base --k 5 --levels 0,0.2,0.4,0.6,0.8,1.0 \
        --generator qwen2.5-1.5b --max-tokens 64

`--levels natural` means the retriever's own top k (p = 0) and nothing else.
Every answer lands in `<cache>/gen_<generator>.jsonl`; a prompt that is already
there is never generated again, so the script is resumable and a killed run
costs one answer.

An answer can depend on the prompt before it. With `--stage`, each answer names its
stage, and a resumed stage first regenerates its last answer and discards the copy, so that the next prompt
follows its real predecessor. `--rank-cache` reads the rankings from another directory than the answers,
and `--variant` names a Czech prompt variant, for the dev set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def build_prompts(ds, run: dict, dataset: str, k: int, levels: list[float], qids: list[str], variant: str | None = None):
    from measurerag.crossover.dial import dial_context, dial_uniforms, gold_set
    from measurerag.rag.generator import build_prompt

    by_id = {d.doc_id: d for d in ds.docs}
    out = []
    for p in levels:
        for qid in qids:
            gold = gold_set(ds.qrels.get(qid, {}))
            u = dial_uniforms(dataset, qid, gold)
            ctx = dial_context(run[qid], gold, k, p, u)
            passages = [by_id[d].indexable() for d in ctx if d in by_id]
            out.append((p, qid, build_prompt(ds.queries[qid], passages, language=ds.language, variant=variant)))
    return out


def main() -> int:
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rank-cache", default=None, help="where the rankings are, if not in --cache")
    ap.add_argument("--variant", default=None, help="a Czech prompt variant; default: the frozen one")
    ap.add_argument("--stage", default=None, help="names this run in the cache, so that it can resume in order")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="cuda offloads every layer")
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--levels", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N question ids (sorted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    levels = [0.0] if args.levels == "natural" else [float(x) for x in args.levels.split(",")]
    ds = Dataset.load_slice(args.slice)
    run = load_run(Path(args.rank_cache or args.cache), args.retriever)["run"]
    qids = sorted(q for q in ds.answers if q in run)
    if args.limit:
        qids = qids[: args.limit]

    cache = GenCache(Path(args.cache) / f"gen_{args.generator}.jsonl")
    prompts = build_prompts(ds, run, args.dataset, args.k, levels, qids, args.variant)
    todo: dict[str, str] = {}
    for _, _, prompt in prompts:
        key = gen_key(args.generator, args.max_tokens, args.seed, prompt)
        if key not in cache and key not in todo:
            todo[key] = prompt
    n_distinct = len({gen_key(args.generator, args.max_tokens, args.seed, pr) for _, _, pr in prompts})
    print(
        f"[{args.dataset} {args.retriever} k={args.k} {args.generator} mt={args.max_tokens}] "
        f"{len(prompts)} (level, question) pairs, {n_distinct} distinct prompts, "
        f"{n_distinct - len(todo)} cached, {len(todo)} to generate",
        flush=True,
    )
    if args.dry_run or not todo:
        return 0

    from measurerag.openmp import preflight
    from measurerag.rag.generator import LlamaGenerator

    preflight()
    gen = LlamaGenerator(
        args.generator, device=args.device, seed=args.seed, max_tokens=args.max_tokens,
        n_ctx=4096, n_threads=args.n_threads,
    )
    threads = getattr(gen.llm, "n_threads", args.n_threads)
    last = cache.last_of_stage.get(args.stage) if args.stage else None
    if last is not None:
        by_key = {gen_key(args.generator, args.max_tokens, args.seed, pr): pr for _, _, pr in prompts}
        if last["key"] in by_key:
            gen.generate(by_key[last["key"]])
            print(f"  resumed: regenerated this stage's last answer first, and discarded the copy", flush=True)
    t0 = time.perf_counter()
    for i, (key, prompt) in enumerate(todo.items(), 1):
        text, info = gen.generate(prompt)
        cache.put({
            "key": key, "generator": args.generator, "max_tokens": args.max_tokens,
            "seed": args.seed, "n_threads": threads, "device": args.device, "text": text,
            "completion_tokens": info["completion_tokens"], "prompt_tokens": info["prompt_tokens"],
            "seconds": round(info["seconds"], 3),
            **({"stage": args.stage} if args.stage else {}),
        })
        if i % 25 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(todo)}  {el / i:.2f} s/answer  eta {(len(todo) - i) * el / i / 60:.0f} min", flush=True)
    gen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
