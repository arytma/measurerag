"""The bench and the dial's cache: check that they agree, then reuse the bench's other answers.

    .venv/bin/python scripts/sqad32_backfill.py --agree --from results/cz_rag --slice <slice> \
        --cache <cache> [--report <file>]
    .venv/bin/python scripts/sqad32_backfill.py --from results/cz_rag --slice <slice> \
        --cache <cache> --retrievers bm25,retromae-cs [--report <file>]

An answer depends on the prompt and, a little, on the prompt before it in the same process
. The dial's level 0 and the bench walk the questions in the same order, so on
their shared retriever they must give the same strings. `--agree` checks that on every question of
the set, with no generation, and exits 2 on any difference. It replaces a spot check that regenerated
ten answers in random order: that check could fail on the order alone.

Copying is for the natural rows of later question sets only, and only after the set's dial has run:
the dial starts from an empty cache. An answer that may have been cut by the dial's cap of 64 tokens
is left out, and is generated when a later set needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CAP = 64
SEED = 20260101
K = 3
BASE = "qwen2.5-1.5b"


def agreement(rows: Iterable[tuple[str, str, int]], cap: int = CAP) -> dict:
    """Rows of (bench answer, dial answer, the bench answer's tokens). A long answer may be cut by
    the dial's cap, so there the dial's answer only has to start the bench's."""
    out = {"questions": 0, "identical": 0, "cut_by_the_cap": 0, "different": 0}
    for bench, dial, tokens in rows:
        out["questions"] += 1
        if bench == dial:
            out["identical"] += 1
        elif tokens >= cap - 1 and bench.startswith(dial):
            out["cut_by_the_cap"] += 1
        else:
            out["different"] += 1
    return out


def main() -> int:
    from llama_cpp import Llama

    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.openmp import preflight
    from measurerag.rag.generator import build_prompt, resolve_model_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="bench results directory")
    ap.add_argument("--slice", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rank-cache", default=None, help="where the rankings are, if not in --cache")
    ap.add_argument("--agree", action="store_true", help="compare, copy nothing")
    ap.add_argument("--agree-retriever", default="e5-small")
    ap.add_argument("--generator", default=BASE, help="the bench's smaller generator whose answers are compared")
    ap.add_argument("--retrievers", default="bm25,retromae-cs", help="what to copy")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    preflight()
    ds = Dataset.load_slice(args.slice)
    by_id = {d.doc_id: d for d in ds.docs}
    tok = Llama(model_path=str(resolve_model_path(args.generator)), vocab_only=True, verbose=False)
    cache = GenCache(Path(args.cache) / f"gen_{args.generator}.jsonl")

    def walk(rid: str):
        run = load_run(Path(args.rank_cache or args.cache), rid)["run"]
        preds = json.loads((Path(args.src) / f"predictions_{rid}+{args.generator}.json").read_text(encoding="utf-8"))
        for q in sorted(ds.answers):
            if q in preds:
                prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in run[q][:K] if d in by_id], language=ds.language)
                yield q, gen_key(args.generator, CAP, SEED, prompt), preds[q], len(tok.tokenize(preds[q].encode("utf-8"), add_bos=False))

    if args.agree:
        rows, missing = [], 0
        for _, key, answer, n in walk(args.agree_retriever):
            row = cache.get(key)
            if row is None:
                missing += 1
            else:
                rows.append((answer, row["text"], n))
        out: dict = {**agreement(rows), "retriever": args.agree_retriever, "missing_from_the_dial": missing, "generations": 0}
        print(out, flush=True)
        failed = bool(out["different"] or missing or not out["questions"])
        if failed:
            print("THE BENCH AND THE DIAL DISAGREE", flush=True)
    else:
        copied = capped = present = 0
        per: dict[str, int] = {}
        for rid in args.retrievers.split(","):
            for _, key, answer, n in walk(rid):
                if n >= CAP - 1:      # may have been cut by a cap of 64; generate it instead
                    capped += 1
                elif key in cache:
                    present += 1
                else:
                    cache.put({"key": key, "generator": args.generator, "max_tokens": CAP, "seed": SEED, "n_threads": None,
                               "text": answer, "completion_tokens": n, "prompt_tokens": None, "seconds": None,
                               "copied_from": f"bench predictions_{rid}"})
                    copied += 1
                    per[rid] = per.get(rid, 0) + 1
        print(f"copied {copied} {per}, already present {present}, too long to copy {capped}", flush=True)
        out = {"copied": copied, "copied_per_retriever": per, "already_present": present, "too_long_to_copy": capped, "generations": 0}
        failed = False
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
