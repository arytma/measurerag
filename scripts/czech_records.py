"""Export the Czech per-question records that ship with the release. Ids, never SQAD text.

    .venv/bin/python scripts/czech_records.py            # every set; fails if an answer is missing
    .venv/bin/python scripts/czech_records.py --partial  # what exists so far (the queue runs this)

Reads the private caches (slices, rankings, generation cache, bench predictions) and writes deterministic
gzip files of JSON lines:

    answers/<source>/answers_<set>.jsonl.gz       one row per (configuration, dial level, question)
    results/czech/records/rankings_<pop>.jsonl.gz each question's gold id and every retriever's top 10 ids
    results/czech/records/sets.json               each set's question ids and hashes

A row of answers_ holds the set, question id, configuration, seed, the document ids shown to the
generator in rank order, the generated answer, EM, F1 and the labelled secondary metrics. The gold
answers are not shipped: EM and F1 are stored, and `scripts/rescore_czech.py --verify-scores`
recomputes them against a copy of the data the reader downloads. The answers are data under their
source's licence, so they stay apart from the code, one folder per source (answers/README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from measurerag.czech_store import FIRST_CUT, write_jsonl_gz  # noqa: E402

BASE, BIG = "qwen2.5-1.5b", "qwen2.5-3b"
E2, E4 = "gemma-4-e2b", "gemma-4-e4b"
M3, M8 = "ministral-3-3b", "ministral-3-8b"
Q2, Q4 = "qwen3.5-2b", "qwen3.5-4b"
N2, N4 = "gemma-3n-e2b", "gemma-3n-e4b"          # v0.4.2
# Every Czech set is answered by every family; the smaller of each also runs the natural top 3 and the bench
#
GENS = (BASE, BIG, E2, E4, M3, M8, Q2, Q4, N2, N4)
SMALL = (BASE, E2, M3, Q2, N2)
SEED, DIAL_MT, BENCH_MT, K = 20260101, 64, 256, 3
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TOP = 10
C = "results/crossover/cache"
SL = "results/crossover/slices"
COMPLETE = f"{C}/complete"
RANK_COPY, RANK_32 = f"{C}/v030_copy", f"{C}/v030_32"
V020_BENCH = ".local/data/v020_outputs/cz_rag"
ANSWERS = REPO / "answers"

# set key -> what it is. `tag` seeds the dial's draws, as in the released protocol. Each set has its own answers.
SETS = {
    "sqad32_debiased": {"label": "SQAD 3.2, debiased at 0.138", "slice": f"{SL}/complete_sqad32_debiased", "cache": f"{COMPLETE}/sqad32_debiased", "rank_cache": RANK_32, "tag": "czech32", "language": "cs", "ranking": "sqad32_test", "source": "sqad32"},
    "sqad32_full": {"label": "SQAD 3.2, full", "slice": f"{SL}/complete_sqad32_full", "cache": f"{COMPLETE}/sqad32_full", "rank_cache": RANK_32, "tag": "czech32", "bench": "results/cz_rag_32", "language": "cs", "ranking": "sqad32_test", "source": "sqad32"},
    "released": {"label": "SQAD 3.0 copy, the released 200", "slice": "results/cz_rag/slice", "cache": f"{COMPLETE}/released", "rank_cache": RANK_COPY, "tag": "czech", "bench": "results/cz_rag", "language": "cs", "ranking": "copy_test", "source": "sqad30_copy"},
    "copy_debiased": {"label": "SQAD 3.0 copy, debiased at 0.138", "slice": f"{SL}/complete_copy_debiased", "cache": f"{COMPLETE}/copy_debiased", "rank_cache": RANK_COPY, "tag": "czech", "language": "cs", "ranking": "copy_test", "source": "sqad30_copy"},
    "extra010_sqad32": {"label": "SQAD 3.2, debiased at 0.10 (sensitivity row)", "slice": f"{SL}/complete_extra010_sqad32", "cache": f"{COMPLETE}/extra010_sqad32", "rank_cache": RANK_32, "tag": "czech32", "language": "cs", "ranking": "sqad32_test", "source": "sqad32"},
    "extra010_copy": {"label": "SQAD 3.0 copy, debiased at 0.10 (sensitivity row)", "slice": f"{SL}/complete_extra010_copy", "cache": f"{COMPLETE}/extra010_copy", "rank_cache": RANK_COPY, "tag": "czech", "language": "cs", "ranking": "copy_test", "source": "sqad30_copy"},
    # the answers v0.2.0 reported: the old prompt, on the released 200, for the defect's measured size
    "released_v020": {"label": "SQAD 3.0 copy, the released 200, v0.2.0's prompt", "slice": "results/cz_rag/slice", "cache": f"{C}/czech", "rank_cache": f"{C}/czech", "tag": "czech",
                      "bench": V020_BENCH, "language": "en", "ranking": "copy_test", "source": "sqad30_copy", "generators": (BASE, BIG)},
}
# retrieval only: no generation, not re-run. Rankings from the SQAD 3.2 experiment.
RANKINGS = {
    "copy_test": {"label": "SQAD 3.0 copy, whole test split", "slice": f"{SL}/czech30_testall", "cache": RANK_COPY},
    "sqad32_test": {"label": "SQAD 3.2, whole test split", "slice": f"{SL}/czech32_testall", "cache": RANK_32},
}
CROSS_SIDES = {"copy": {"slice": f"{SL}/complete_cross_copy", "cache": f"{COMPLETE}/cross_copy", "source": "sqad30_copy"},
               "sqad32": {"slice": f"{SL}/complete_cross_32", "cache": f"{COMPLETE}/cross_32", "source": "sqad32"}}
RETRIEVERS = ("bm25", "bm25-nodiacritics", "e5-small", "retromae-cs")
OUT = REPO / "results/czech/records"


def scores(answer: str, gold: list[str]) -> dict:
    from measurerag.czech_sensitivity import folded_f1, prefix_match, yes_no_in_czech
    from measurerag.metrics import exact_match, token_f1

    em = exact_match(answer, gold)
    return {
        "em": int(em),
        "f1": round(token_f1(answer, gold), 12),
        "f1_folded": round(folded_f1(answer, gold), 12),
        "prefix5": int(max(em, prefix_match(answer, gold))),
        "em_yes_no_mapped": int(exact_match(yes_no_in_czech(answer), gold)),
    }


def export_set(key: str, spec: dict, ds, partial: bool, other_key: dict[str, list[str]] | None = None) -> tuple[list[dict], dict]:
    from measurerag.crossover.dial import dial_context, dial_uniforms, gold_set
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.nq import contains_answer
    from measurerag.metrics import exact_match, normalize_answer
    from measurerag.rag.generator import build_prompt

    by_id = {d.doc_id: d for d in ds.docs}
    language = spec["language"]
    rank_cache = REPO / spec.get("rank_cache", spec["cache"])
    qids = sorted(q for q in ds.answers if gold_set(ds.qrels.get(q, {})))
    gens = spec.get("generators", GENS)
    small = [g for g in SMALL if g in gens]
    caches = {g: GenCache(REPO / spec["cache"] / f"gen_{g}.jsonl") for g in gens}
    rows: list[dict] = []
    missing = 0

    def row(q, cfg, kind, p, mt, ctx, answer, leak=None):
        r = {"set": key, "q": q, "config": cfg, "kind": kind, "p": p, "max_tokens": mt, "seed": SEED, "context": list(ctx),
             "answer": answer, **scores(answer, ds.answers[q])}
        if leak is not None:
            r["leak"] = leak
        if other_key is not None:
            alt = other_key.get(q)
            r["em_other_key"] = None if not alt else int(exact_match(answer, alt))
        return r

    # ---- the dial: e5-small, both generators, every level --------------------------------------
    run = load_run(rank_cache, "e5-small")["run"]
    for p in LEVELS:
        for q in qids:
            gold = gold_set(ds.qrels[q])
            ctx = dial_context(run[q], gold, K, p, dial_uniforms(spec["tag"], q, gold))
            prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language=language)
            # the released rule searches for the gold as it stands; a gold ano or ne then counts as a word
            # in a passage, so the span-only flag ships beside it
            shown = [by_id[d].indexable()[:1200] for d in ctx if d in by_id and d not in gold]
            spans = [a for a in ds.answers[q] if normalize_answer(a) not in ("yes", "no", "ano", "ne")]
            leak_released = int(any(contains_answer(s, ds.answers[q]) for s in shown))
            leak_span = int(bool(spans) and any(contains_answer(s, spans) for s in shown))
            for g in gens:
                cached = caches[g].get(gen_key(g, DIAL_MT, SEED, prompt))
                if cached is None:
                    missing += 1
                    continue
                r = row(q, f"e5-small+{g}", "dial", p, DIAL_MT, ctx, cached["text"], leak=leak_released)
                r["leak_span_only"] = leak_span
                rows.append(r)

    # ---- the other retrievers with 1.5B, at their own top 3 ------------------------------------
    for rid in ("bm25", "retromae-cs"):
        if spec.get("bench"):
            break                    # a set with a bench has these rows from the bench, below
        rr = rank_cache / f"retrieval_{rid}.json"
        if not rr.exists():
            continue
        rn = load_run(rank_cache, rid)["run"]
        for q in qids:
            ctx = [d for d in rn[q][:K]]
            prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language=language)
            for g in small:
                cached = caches[g].get(gen_key(g, DIAL_MT, SEED, prompt))
                if cached is None:
                    missing += 1
                    continue
                rows.append(row(q, f"{rid}+{g}", "natural", 0.0, DIAL_MT, ctx, cached["text"]))

    # ---- the bench, at its own token cap --------------------------------------------------------
    if spec.get("bench"):
        for rid in ("bm25", "e5-small", "retromae-cs"):
            # v0.2.0's cache has no retromae-cs ranking: the corpus is the same, so the copy's ranking stands in
            rc = rank_cache if (rank_cache / f"retrieval_{rid}.json").exists() else REPO / RANK_COPY
            rn = load_run(rc, rid)["run"]
            for g in small:
                pf = REPO / spec["bench"] / f"predictions_{rid}+{g}.json"
                if not pf.exists():
                    missing += len(qids)
                    continue
                preds = json.loads(pf.read_text(encoding="utf-8"))
                for q in qids:
                    if q in preds:
                        rows.append(row(q, f"{rid}+{g}", "bench", 0.0, BENCH_MT, rn[q][:K], preds[q]))
                    else:
                        missing += 1
    if missing and not partial:
        raise SystemExit(f"{key}: {missing} answers are missing")
    rows.sort(key=lambda r: (r["q"], r["config"], r["kind"], r["p"]))
    meta = {"label": spec["label"], "n_questions": len(qids), "questions": qids, "tag": spec["tag"], "ranking": spec["ranking"], "language": language,
            "corpus_sha256": ds.corpus_hash(), "queries_sha256": ds.query_hash(), "n_docs": len(ds.docs),
            "rows": len(rows), "missing": missing, "complete": missing == 0}
    return rows, meta


def export_cross_version(partial: bool) -> dict:
    """One question on each version's passages, e5-small's top 3, both generators, each version's own gold.
    A row also says whether the answer matches the other version's gold. One file per source."""
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.czech_labels import answer_type
    from measurerag.datasets.base import Dataset
    from measurerag.metrics import exact_match
    from measurerag.rag.generator import build_prompt

    pairs = json.loads((REPO / "results/czech/plan/questions.json").read_text(encoding="utf-8"))["cross_version"]
    data = {side: Dataset.load_slice(REPO / spec["slice"]) for side, spec in CROSS_SIDES.items()}
    meta: dict = {"pairs": len(pairs), "sides": {}}
    for side, spec in CROSS_SIDES.items():
        ds, other = data[side], data["sqad32" if side == "copy" else "copy"]
        ranking = REPO / spec["cache"] / "retrieval_e5-small.json"
        if not ranking.exists():
            if not partial:
                raise SystemExit(f"{ranking}: missing")
            continue
        run = load_run(REPO / spec["cache"], "e5-small")["run"]
        by_id = {d.doc_id: d for d in ds.docs}
        caches = {g: GenCache(REPO / spec["cache"] / f"gen_{g}.jsonl") for g in GENS}
        rows, missing = [], 0
        for a, b in pairs:
            q, o = (a, b) if side == "copy" else (b, a)
            ctx = list(run[q][:K])
            prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language="cs")
            for g in GENS:
                got = caches[g].get(gen_key(g, DIAL_MT, SEED, prompt))
                if got is None:
                    missing += 1
                    continue
                rows.append({"set": "cross_version", "q": q, "corpus": side, "other_q": o, "config": f"e5-small+{g}", "kind": "cross_version", "p": 0.0, "max_tokens": DIAL_MT, "seed": SEED,
                             "context": ctx, "answer": got["text"], "type": answer_type(ds.answers[q]), "gold_in_top3": int(bool(set(ctx) & {d for d, g in ds.qrels[q].items() if g > 0})),
                             **scores(got["text"], ds.answers[q]), "em_other_key": int(exact_match(got["text"], other.answers[o]))})
        if missing and not partial:
            raise SystemExit(f"cross_version {side}: {missing} answers are missing")
        rows.sort(key=lambda r: (r["q"], r["config"]))
        size = write_jsonl_gz(ANSWERS / spec["source"] / "answers_cross_version.jsonl.gz", rows)
        meta["sides"][side] = {"corpus_sha256": ds.corpus_hash(), "n_docs": len(ds.docs), "rows": len(rows), "missing": missing, "bytes": size}
    return meta


def export_rankings(key: str, spec: dict) -> dict:
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset

    ds = Dataset.load_slice(REPO / spec["slice"])
    rows: list[dict] = []
    runs = {}
    for rid in RETRIEVERS:
        p = REPO / spec["cache"] / f"retrieval_{rid}.json"
        if p.exists():
            runs[rid] = load_run(REPO / spec["cache"], rid)["run"]
    for q in sorted(ds.queries):
        r = {"q": q, "gold": sorted(d for d, g in ds.qrels[q].items() if g > 0)}
        for rid, run in runs.items():
            r[rid] = run[q][:TOP]
        rows.append(r)
    size = write_jsonl_gz(OUT / f"rankings_{key}.jsonl.gz", rows)
    return {"label": spec["label"], "n_questions": len(rows), "retrievers": sorted(runs), "top": TOP, "corpus_sha256": ds.corpus_hash(), "n_docs": len(ds.docs), "bytes": size}


def other_key_map(key: str, ds) -> dict[str, list[str]]:
    """Each question's gold in the other SQAD version, joined on the question's text. Empty if it has none."""
    from measurerag.datasets.sqad32 import ARCHIVE, gold_answers, load_parsed

    if key in ("sqad32_full", "sqad32_debiased", "extra010_sqad32"):
        import glob

        import pandas as pd

        by_q: dict[str, list[str]] = {}
        for n in ("train", "validation", "test"):
            df = pd.read_parquet(glob.glob(str(REPO / f"data/hf/datasets--Pehy--cs_sqad-3.0/snapshots/*/data/{n}-*.parquet"))[0])
            for r in df.itertuples(index=False):
                texts = [str(t) for t in r.answers["text"]]
                if texts:
                    by_q.setdefault(str(r.question), texts)
        return {q: by_q[ds.queries[q]] for q in ds.queries if ds.queries[q] in by_q}
    recs, _ = load_parsed(ARCHIVE.parent / "parsed")
    by_q = {}
    for r in recs:
        g = gold_answers(r)
        if r.get("question") and g:
            by_q.setdefault(r["question"], g)
    return {q: by_q[ds.queries[q]] for q in ds.queries if ds.queries[q] in by_q}


def signature(spec: dict) -> list[list]:
    """The sizes of the files a set is exported from: unchanged sizes, nothing new to export."""
    gens = spec.get("generators", GENS)
    paths = [REPO / spec["cache"] / f"gen_{g}.jsonl" for g in gens]
    if spec.get("bench"):
        paths += [REPO / spec["bench"] / f"predictions_{r}+{g}.json" for r in ("bm25", "e5-small", "retromae-cs") for g in SMALL if g in gens]
    return [[p.name, p.stat().st_size] for p in paths if p.exists()]


def main() -> int:
    from measurerag.datasets.base import Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--sets", default=None, help="comma-separated set keys, or cross_version")
    args = ap.parse_args()
    os.chdir(REPO)
    OUT.mkdir(parents=True, exist_ok=True)
    # a partial export leaves the shipped sets.json alone until the run is whole
    meta_path = REPO / "results/czech/complete/sets_partial.json" if args.partial else OUT / "sets.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"sets": {}, "rankings": {}}
    want = args.sets.split(",") if args.sets else [*SETS, "cross_version"]
    if not args.partial:
        if meta.get("first_cut"):                                  # the first cut's description moves out, and stays checkable
            keep = REPO / FIRST_CUT / "sets.json"
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
        meta = {"sets": {}, "rankings": {}}                        # a final export names exactly the sets it wrote
    state_path = REPO / "logs/complete/records_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if (args.partial and state_path.exists()) else {}
    for key in want:
        if key == "cross_version":
            meta["cross_version"] = export_cross_version(args.partial)
            print("cross_version", meta["cross_version"], flush=True)
            continue
        spec = SETS[key]
        sig = signature(spec)
        if args.partial and (not sig or state.get(key) == sig):
            continue         # nothing new for this set: the queue calls this after every stage
        ds = Dataset.load_slice(REPO / spec["slice"])
        rows, m = export_set(key, spec, ds, args.partial, None if args.partial else other_key_map(key, ds))
        m["bytes"] = write_jsonl_gz(ANSWERS / spec["source"] / f"answers_{key}.jsonl.gz", rows)
        m["source"] = spec["source"]
        meta["sets"][key] = m
        state[key] = sig
        print(key, "rows", m["rows"], "missing", m["missing"], "bytes", m["bytes"], flush=True)
    if args.partial:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    else:
        for key, spec in RANKINGS.items():
            meta["rankings"][key] = export_rankings(key, spec)
            print("rankings", key, meta["rankings"][key]["n_questions"], "bytes", meta["rankings"][key]["bytes"], flush=True)
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
