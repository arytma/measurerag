"""Is max_tokens = 64 the same experiment as run 1's 256?

Compares answers generated in this run at 64 tokens, through the dial's generation
path, with run 1's harness predictions at 256, on the first N questions of
the same slice with the same retriever (e5-small, k = 5). Identical strings
confirm two things at once: the token cap never bit, and the dial's generation
path reproduces the harness (same prompt bytes, same ranking, same decoding).

The specification's rule: if EM or F1 differs by more than 0.005, keep 256 and cut the grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.metrics import exact_match, token_f1
    from measurerag.rag.generator import build_prompt

    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="results/en_rag/slice")
    ap.add_argument("--cache", default="results/crossover/cache/hotpotqa")
    ap.add_argument("--run-1", default="results/en_rag")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/crossover/max_tokens_check.json")
    ap.add_argument("--retriever", default="e5-small")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--generators", default="qwen2.5-1.5b,qwen2.5-3b")
    args = ap.parse_args()

    ds = Dataset.load_slice(args.slice)
    run = load_run(Path(args.cache), args.retriever)["run"]
    by_id = {d.doc_id: d for d in ds.docs}
    qids = sorted(q for q in ds.answers if q in run)[: args.n]
    report: dict = {"n_questions": len(qids), "retriever": args.retriever, "k": args.k, "tolerance": 0.005, "generators": {}}
    ok = True
    for gen in args.generators.split(","):
        old = json.loads((Path(args.run_1) / f"predictions_{args.retriever}+{gen}.json").read_text(encoding="utf-8"))
        cache = GenCache(Path(args.cache) / f"gen_{gen}.jsonl")
        new, missing, differ, tokens = {}, 0, [], []
        for q in qids:
            prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in run[q][: args.k]], language=ds.language)
            row = cache.get(gen_key(gen, 64, 20260101, prompt))
            if row is None:
                missing += 1
                continue
            new[q] = row["text"]
            tokens.append(row["completion_tokens"])
            if row["text"] != old[q]:
                differ.append({"qid": q, "night_one_256": old[q], "tonight_64": row["text"]})
        if missing:
            print(f"{gen}: {missing} answers not generated yet", file=sys.stderr)
            return 2
        em_old = sum(exact_match(old[q], ds.answers[q]) for q in qids) / len(qids)
        em_new = sum(exact_match(new[q], ds.answers[q]) for q in qids) / len(qids)
        f1_old = sum(token_f1(old[q], ds.answers[q]) for q in qids) / len(qids)
        f1_new = sum(token_f1(new[q], ds.answers[q]) for q in qids) / len(qids)
        within = abs(em_old - em_new) <= 0.005 and abs(f1_old - f1_new) <= 0.005
        ok &= within
        report["generators"][gen] = {
            "em_256_night_one": em_old, "em_64_tonight": em_new,
            "f1_256_night_one": f1_old, "f1_64_tonight": f1_new,
            "identical_strings": len(qids) - len(differ), "differing_strings": differ,
            "max_completion_tokens_at_64": max(tokens), "hit_cap": sum(t >= 64 for t in tokens),
            "within_tolerance": within,
        }
    report["verdict"] = "use 64" if ok else "keep 256"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({g: {k: v for k, v in r.items() if k != "differing_strings"} for g, r in report["generators"].items()}, indent=2))
    print("verdict:", report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
