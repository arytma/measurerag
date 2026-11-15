"""The same question and the same model, on the passages of each SQAD version.

    .venv/bin/python scripts/czech_same_questions.py --plan   # count, generate nothing
    .venv/bin/python scripts/czech_same_questions.py          # generate what is missing, write the records

The two question sets of 200 differ, so a gap in exact match between them can come from the questions,
from the passages, or from which 200 were drawn. Some questions sit in the test split of both versions.
Of those, the ones already answered in one set are answered here in the other corpus: e5-small's top 3,
Qwen2.5 1.5B, the dial's cap and seed, the fixed prompt. Only the passages change.

Writes results/czech/records/same_questions.jsonl.gz, one row per question and corpus, ids and scores
only, with the id of the same question in the other version. A separate cache, so that nothing here can
touch a dial's answers. Prints no question and no passage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from measurerag.czech_store import load_answers, load_sets, write_jsonl_gz  # noqa: E402

K, MT, SEED = 3, 64, 20260101
BASE = "qwen2.5-1.5b"
CONFIG = f"e5-small+{BASE}"
MAX_GENERATIONS = 3200
OUT = REPO / "results/czech/first_cut/same_questions_rewritten.jsonl.gz"   # the first cut's run; what ships is split by source under answers/
SIDES = {
    "copy": {"slice": "results/crossover/slices/czech30_testall", "cache": "results/crossover/cache/v030_copy", "set": "released"},
    "sqad32": {"slice": "results/crossover/slices/czech32_testall", "cache": "results/crossover/cache/v030_32", "set": "sqad32_full"},
}


def pairs_in_a_set(questions: dict[str, dict[str, str]], in_set: dict[str, set[str]]) -> list[tuple[str, str]]:
    """(copy id, SQAD 3.2 id) for a question text in both test splits and in at least one of the two sets."""
    by_text = {side: {t.strip(): q for q, t in qs.items()} for side, qs in questions.items()}
    out = []
    for text in set(by_text["copy"]) & set(by_text["sqad32"]):
        a, b = by_text["copy"][text], by_text["sqad32"][text]
        if a in in_set["copy"] or b in in_set["sqad32"]:
            out.append((a, b))
    return sorted(out)


def main() -> int:
    from czech_records import scores
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.metrics import exact_match
    from measurerag.openmp import preflight
    from measurerag.rag.generator import LlamaGenerator, build_prompt

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--answers", default="results/crossover/cache/v030_same", help="its own answers: never a dial's cache")
    ap.add_argument("--compute", default="results/czech/compute.json")
    args = ap.parse_args()
    os.chdir(REPO)

    records = REPO / "results/czech/records"
    meta = load_sets(records)["sets"]
    data = {side: Dataset.load_slice(REPO / spec["slice"]) for side, spec in SIDES.items()}
    in_set = {side: set(meta[spec["set"]]["questions"]) for side, spec in SIDES.items()}
    pairs = pairs_in_a_set({s: d.queries for s, d in data.items()}, in_set)
    answered = {side: {r["q"]: r for r in load_answers(records, spec["set"]) if r["config"] == CONFIG and r["kind"] == "dial" and float(r["p"]) == 0.0} for side, spec in SIDES.items()}
    by_id = {side: {d.doc_id: d for d in ds.docs} for side, ds in data.items()}
    run = {side: load_run(REPO / spec["cache"], "e5-small")["run"] for side, spec in SIDES.items()}
    todo = {"sqad32": sorted(b for a, b in pairs if b not in answered["sqad32"]), "copy": sorted(a for a, b in pairs if a not in answered["copy"])}
    planned = sum(len(v) for v in todo.values())
    used = json.loads((REPO / args.compute).read_text(encoding="utf-8"))["generations"]
    print(f"{len(pairs)} questions; {planned} to generate {({k: len(v) for k, v in todo.items()})}; {used} generations used of {MAX_GENERATIONS}", flush=True)
    if args.plan:
        return 0
    if used + planned > MAX_GENERATIONS:
        print("this would pass the generation cap: nothing generated")
        return 1

    preflight()
    cache = GenCache(REPO / args.answers / f"gen_{BASE}.jsonl")
    gen = LlamaGenerator(BASE, device="cpu", seed=SEED, max_tokens=MT, n_ctx=4096)
    made, t0 = 0, time.time()

    def prompt_of(side: str, q: str) -> tuple[str, list[str]]:
        ctx = list(run[side][q][:K])
        return build_prompt(data[side].queries[q], [by_id[side][d].indexable() for d in ctx if d in by_id[side]], language=data[side].language), ctx

    for side in ("sqad32", "copy"):
        for q in todo[side]:
            prompt, _ = prompt_of(side, q)
            key = gen_key(BASE, MT, SEED, prompt)
            if key not in cache:
                text, info = gen.generate(prompt)
                made += 1
                cache.put({"key": key, "generator": BASE, "max_tokens": MT, "seed": SEED, "n_threads": getattr(gen.llm, "n_threads", None), "text": text,
                           "completion_tokens": info["completion_tokens"], "prompt_tokens": info["prompt_tokens"], "seconds": round(info["seconds"], 3)})
    gen.close()
    seconds = round(time.time() - t0, 1)

    rows = []
    for a, b in pairs:
        for side, q, other_side, other in (("copy", a, "sqad32", b), ("sqad32", b, "copy", a)):
            prompt, ctx = prompt_of(side, q)
            if q in answered[side]:
                done = answered[side][q]
                if done["context"] != ctx:
                    raise SystemExit(f"{side} {q}: the set's top 3 is not this script's top 3")
                answer, source = done["answer"], SIDES[side]["set"]
            else:
                answer, source = cache.get(gen_key(BASE, MT, SEED, prompt))["text"], "generated_here"
            rows.append({"set": "same_questions", "q": q, "corpus": side, "other_q": other, "from": source, "config": CONFIG, "kind": "same_question", "p": 0.0,
                         "max_tokens": MT, "seed": SEED, "context": ctx, "answer": answer, **scores(answer, data[side].answers[q]),
                         "em_other_key": int(exact_match(answer, data[other_side].answers[other]))})
    write_jsonl_gz(OUT, rows)
    report = {"questions": len(pairs), "generations": made, "seconds": seconds, "config": CONFIG, "max_tokens": MT, "seed": SEED, "generations_before": used}
    (REPO / "results/czech/same_questions_compute.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
