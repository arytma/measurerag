"""The fixed Czech prompt on 20 yes/no questions: how often does the model still answer in English?

A check made before the first v0.3.0 cut, kept as its record. The complete run measures compliance on every
set instead.

    .venv/bin/python scripts/czech_prompt_smoke.py --out <file>

Takes the first 20 ano or ne questions of the released 200, e5-small's top 3, both generators.
Answers land in a cache of their own, so that they cannot change the dial's order. Prints counts and answer classes, never a question or a passage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

K, MT, SEED = 3, 64, 20260101
N = 20


def classify(text: str) -> str:
    from measurerag.metrics import normalize_answer

    a = normalize_answer(text)
    if a in ("ano", "ne"):
        return "ano or ne"
    if a in ("yes", "no"):
        return "yes or no in English"
    if a == "unanswerable":
        return "unanswerable"
    first = a.split()[0] if a.split() else ""
    if first in ("ano", "ne"):
        return "ano or ne, then more words"
    if first in ("yes", "no"):
        return "yes or no, then more words"
    return "something else"


def main() -> int:
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.metrics import exact_match, normalize_answer
    from measurerag.openmp import preflight
    from measurerag.rag.generator import LlamaGenerator, build_prompt

    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="results/cz_rag/slice")
    ap.add_argument("--cache", default="results/crossover/cache/v030_copy", help="reads the rankings")
    ap.add_argument("--answers", default="results/crossover/cache/v030_smoke", help="its own answers: never the dial's cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preflight()
    ds = Dataset.load_slice(REPO / args.slice)
    if ds.language != "cs":
        raise SystemExit("this smoke test is for a Czech slice")
    by_id = {d.doc_id: d for d in ds.docs}
    run = load_run(REPO / args.cache, "e5-small")["run"]
    qids = [q for q in sorted(ds.answers) if any(normalize_answer(a) in ("ano", "ne") for a in ds.answers[q])][:N]
    report: dict = {"questions": len(qids), "yes_no_questions_in_the_set": sum(any(normalize_answer(a) in ("ano", "ne") for a in ds.answers[q]) for q in ds.answers), "generators": {}}
    for g in ("qwen2.5-1.5b", "qwen2.5-3b"):
        cache = GenCache(REPO / args.answers / f"gen_{g}.jsonl")
        gen = LlamaGenerator(g, device="cpu", seed=SEED, max_tokens=MT, n_ctx=4096)
        classes: dict[str, int] = {}
        em = made = 0
        for q in qids:
            prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in run[q][:K] if d in by_id], language=ds.language)
            key = gen_key(g, MT, SEED, prompt)
            row = cache.get(key)
            if row is None:
                text, info = gen.generate(prompt)
                made += 1
                row = {"key": key, "generator": g, "max_tokens": MT, "seed": SEED, "n_threads": getattr(gen.llm, "n_threads", None), "text": text,
                       "completion_tokens": info["completion_tokens"], "prompt_tokens": info["prompt_tokens"], "seconds": round(info["seconds"], 3)}
                cache.put(row)
            c = classify(row["text"])
            classes[c] = classes.get(c, 0) + 1
            em += exact_match(row["text"], ds.answers[q])
        gen.close()
        report["generators"][g] = {"classes": dict(sorted(classes.items())), "em": em / len(qids), "generations": made}
        print(g, report["generators"][g], flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
