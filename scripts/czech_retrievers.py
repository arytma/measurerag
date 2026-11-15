"""v0.4.0: more retrievers on SQAD's two test splits, retrieval only.

    .venv/bin/python scripts/czech_retrievers.py --reproduce   # e5-small encoded again on the Mac, against the shipped records
    .venv/bin/python scripts/czech_retrievers.py --records     # the new rankings' top 10, from the caches, into the records
    .venv/bin/python scripts/czech_retrievers.py               # the readings, from the records alone
    .venv/bin/python scripts/czech_retrievers.py --check       # the same, compared byte for byte with what is in the tree

The rankings come from scripts/retrieval_cache.py, one cache per retriever, in results/crossover/cache/v040_copy (the copy's
843 test questions, slice czech30_testall) and v040_32 (SQAD 3.2's 4,771, czech32_testall). Those caches are private, like
every ranking cache. What ships:

    results/czech/records/rankings_v040_<pop>.jsonl.gz   each question's top 10 ids from every new retriever; the gold and
                                                          the old retrievers stay in rankings_<pop>.jsonl.gz, unchanged
    results/czech/v040/reproduce_e5small.json             e5-small encoded again, against the shipped top 10
    results/czech/v040/compute.json                       each ranking's wall time and build record on the Mac, from its cache
    results/czech/v040/retrievers.json                    the readings, recomputed from the two records files

The readings use rescore_czech.py's retrieval metrics and the same resamples as its large-sample rows (the tags
"czech-v030", <pop>, "all"), so e5-small's and BM25's rows here equal the shipped ones and every difference is paired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from measurerag.crossover.stats import bootstrap_counts  # noqa: E402
from measurerag.czech_store import load_rankings, read_jsonl_gz, write_jsonl_gz  # noqa: E402
from measurerag.metrics import gold_in_context, ndcg_at_k, recall_at_k  # noqa: E402
import crossover  # noqa: E402
from rescore_czech import N_BOOT, mean_ci  # noqa: E402

TOP = 10
POPULATIONS = {"copy_test": {"label": "SQAD 3.0 copy, whole test split", "cache": "results/crossover/cache/v040_copy"},
               "sqad32_test": {"label": "SQAD 3.2, whole test split", "cache": "results/crossover/cache/v040_32"}}
OLD = ("bm25", "bm25-nodiacritics", "e5-small", "retromae-cs")
NEW = ("e5-base", "granite-107m", "granite-278m", "embeddinggemma", "simcse-cs-en", "bm25-lemma")
REFERENCE, LEXICAL = "e5-small", "bm25-nodiacritics"
# What is read: each new retriever against e5-small and against BM25 without diacritics, and four pairs inside a kind,
# the last added after the rankings were read
PAIRS = tuple((n, REFERENCE) for n in NEW) + tuple((n, LEXICAL) for n in NEW) + (
    ("e5-base", "e5-small"), ("granite-278m", "granite-107m"), ("bm25-lemma", "bm25-nodiacritics"),
    # lemmas against plain BM25, which keeps diacritics as BM25 over lemmas does, so that one factor changes; added after
    # the rankings were read (BM25 over lemmas minus BM25 without diacritics changes two); no rule reads it
    ("bm25-lemma", "bm25"))
# shipped differences, repeated as references beside the new ones (the README's lead and Phase B's anchor)
REFERENCE_PAIRS = (("e5-small", "bm25-nodiacritics"), ("e5-small", "bm25"))
# The rule for an end-to-end run: the anchor is e5-small's own g3 lead over BM25 on SQAD 3.2's test split
RULE_POP, RULE_ANCHOR = "sqad32_test", ("e5-small", "bm25")
RECORDS = Path("results/czech/records")
OUT = Path("results/czech/v040")
# The cost, timed: one timed row per retriever with the budget bench on SQAD 3.2's corpus, under the hygiene
# (scripts/czech_timed_queue.py); only rows the hygiene accepted are read
TIMED_RESULTS = Path("results/cz_v040_timed/results.json")
TIMED_BATCHES = OUT / "timed" / "batches.jsonl"
PROBE_BASELINE = Path("results/crossover/calibration/baseline.json")
COST_FIELDS = ("peak_rss_gb", "index_bytes", "model_bytes", "retrieval_ms_p50", "retrieval_ms_p95")


def new_records_path(root: Path, pop: str) -> Path:
    return root / RECORDS / f"rankings_v040_{pop}.jsonl.gz"


def cached(root: Path, pop: str, rid: str) -> dict | None:
    from measurerag.crossover.retrieval import cache_path

    p = cache_path(root / POPULATIONS[pop]["cache"], rid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cached_run(root: Path, pop: str, rid: str) -> dict | None:
    c = cached(root, pop, rid)
    return c["run"] if c else None


# ---- e5-small, encoded again ------------------------------------------------------------------------------

def reproduce(root: Path) -> dict:
    out: dict = {"note": "e5-small encoded again on the Mac by scripts/retrieval_cache.py, against the shipped top 10"}
    for pop in POPULATIONS:
        shipped = load_rankings(root, pop)
        run = cached_run(root, pop, REFERENCE)
        if run is None:
            raise SystemExit(f"no e5-small cache for {pop} in {POPULATIONS[pop]['cache']}")
        qids = sorted(shipped)
        if sorted(run) != qids:
            raise SystemExit(f"{pop}: the re-encoded ranking covers other questions than the records")
        differs = [q for q in qids if list(run[q][:TOP]) != shipped[q][REFERENCE]]
        out[pop] = {"questions": len(qids), "top_10_differs": len(differs),
                    "top_10_same_ids_other_order": sum(set(run[q][:TOP]) == set(shipped[q][REFERENCE]) for q in differs),
                    "top_3_differs": sum(list(run[q][:3]) != shipped[q][REFERENCE][:3] for q in qids),
                    "differing_questions": differs}
    return out


# ---- the records ------------------------------------------------------------------------------------------

def export_records(root: Path) -> dict:
    wrote, compute = {}, {}
    for pop in POPULATIONS:
        qids = sorted(load_rankings(root, pop))
        runs = {rid: run for rid in NEW if (run := cached_run(root, pop, rid)) is not None}
        for rid, run in runs.items():
            if sorted(run) != qids:
                raise SystemExit(f"{pop} {rid}: the ranking covers other questions than the records")
        rows = [{"q": q, **{rid: list(runs[rid][q][:TOP]) for rid in runs}} for q in qids]
        write_jsonl_gz(new_records_path(root, pop), rows)
        wrote[pop] = sorted(runs)
        compute[pop] = {}
        for rid in (REFERENCE, *NEW):
            c = cached(root, pop, rid)
            if c:
                compute[pop][rid] = {"wall_seconds": c["wall_seconds"], "build": c["build"], "declaration": c["declaration"],
                                     "corpus_sha256": c["corpus_sha256"], "queries_sha256": c["queries_sha256"], "run_sha256": c["run_sha256"]}
    (root / OUT).mkdir(parents=True, exist_ok=True)
    (root / OUT / "compute.json").write_text(dump({"machine": "the Mac (Apple M4), CPU", "rankings": compute}), encoding="utf-8")
    return wrote


def load_all(root: Path, pop: str) -> dict:
    rank = load_rankings(root, pop)
    p = new_records_path(root, pop)
    if p.exists():
        for r in read_jsonl_gz(p):
            rank[r["q"]].update({k: v for k, v in r.items() if k != "q"})
    return rank


# ---- the readings -----------------------------------------------------------------------------------------

def per_question(rank: dict, qids: list[str], rid: str) -> dict:
    rel = {q: {g: 1 for g in rank[q]["gold"]} for q in qids}
    return {"ndcg10": np.array([ndcg_at_k(rank[q][rid], rel[q], 10) for q in qids]),
            "recall10": np.array([recall_at_k(rank[q][rid], rel[q], 10) for q in qids]),
            "g3": np.array([gold_in_context(rank[q][rid], rel[q], 3) for q in qids])}


def readings(root: Path) -> dict:
    out: dict = {"n_boot": N_BOOT, "metrics": ["ndcg10", "recall10", "g3"], "populations": {}}
    for pop, meta in POPULATIONS.items():
        rank = load_all(root, pop)
        qids = sorted(rank)
        counts = bootstrap_counts(len(qids), N_BOOT, "czech-v030", pop, "all").astype(float)
        have = [r for r in (*OLD, *NEW) if r in rank[qids[0]]]
        arrs = {rid: per_question(rank, qids, rid) for rid in have}
        row: dict = {"label": meta["label"], "n": len(qids), "retrievers": have, "means": {}, "differences": {}}
        for rid in have:
            row["means"][rid] = {m: mean_ci(arrs[rid][m], counts) for m in out["metrics"]}
        for hi, lo in (*PAIRS, *REFERENCE_PAIRS):
            if hi in arrs and lo in arrs:
                row["differences"][f"{hi} - {lo}"] = {m: mean_ci(arrs[hi][m] - arrs[lo][m], counts) for m in out["metrics"]}
        out["populations"][pop] = row
    out["phase_b_rule"] = phase_b_rule(out["populations"][RULE_POP])
    c = cost(root)
    if c:
        out["cost"] = c
    return crossover.clean(out)        # rounded as results/czech/analysis.json is


def cost(root: Path) -> dict | None:
    """The price of each retriever and of each difference read: peak resident memory, index and model bytes,
    retrieval latency p50 and p95 (one query at a time, encoding included), from the accepted timed rows alone."""
    if not (root / TIMED_RESULTS).exists() or not (root / TIMED_BATCHES).exists():
        return None
    res = json.loads((root / TIMED_RESULTS).read_text(encoding="utf-8"))
    batches = [json.loads(x) for x in (root / TIMED_BATCHES).read_text(encoding="utf-8").splitlines() if x.strip()]
    accepted: dict = {}
    for b in batches:                   # in record order: a later "superseded" row takes an accepted row back
        if b.get("superseded"):
            accepted.pop(b["stage"], None)
        elif b.get("accepted"):
            accepted[b["stage"]] = b
    base = json.loads((root / PROBE_BASELINE).read_text(encoding="utf-8"))
    rows = {r["config_id"]: r for r in res["results"] if r.get("status") == "ok" and r["config_id"] in accepted}
    out: dict = {"dataset": "SQAD 3.2, its corpus and the 1,995 questions of SQAD 3.2 full",
                 "probe_baseline_macos": base["macos"], "retrievers": {}, "differences": {}}
    for rid in (*OLD, *NEW):
        if rid not in rows:
            continue
        r, b = rows[rid], accepted[rid]
        out["retrievers"][rid] = {**{k: r["measurement"][k] for k in COST_FIELDS}, "n_queries": r["n_queries"],
                                  "macos": r["machine"]["os_version"], "chip": r["machine"]["chip"], "attempt": b["attempt"],
                                  "pre_probe_drift": b["pre_probe"]["drift"], "contended_samples": b["contention"]["n_contended"],
                                  "samples": b["contention"]["n_samples"]}
    have = out["retrievers"]
    for hi, lo in (*PAIRS, *REFERENCE_PAIRS):
        if hi in have and lo in have:
            out["differences"][f"{hi} - {lo}"] = {"peak_rss_gb": have[hi]["peak_rss_gb"] - have[lo]["peak_rss_gb"],
                                                  "retrieval_ms_p95": have[hi]["retrieval_ms_p95"] - have[lo]["retrieval_ms_p95"]}
    out["complete"] = all(rid in have for rid in (*OLD, *NEW))
    return out


def phase_b_rule(row: dict) -> dict:
    """The rule fixed before the readings: the new retriever with the largest g3 gain over e5-small on SQAD 3.2's test split goes end to end
    only if that gain is established and at least e5-small's own g3 lead over BM25 there."""
    anchor = row["differences"][" - ".join(RULE_ANCHOR)]["g3"]
    gains = {n: row["differences"][f"{n} - {REFERENCE}"]["g3"] for n in NEW if f"{n} - {REFERENCE}" in row["differences"]}
    rule = {"anchor": {"difference": " - ".join(RULE_ANCHOR), **anchor}, "retrievers_read": sorted(gains),
            "complete": len(gains) == len(NEW)}
    if gains:
        best = max(gains, key=lambda n: (gains[n]["point"], n))
        g = gains[best]
        rule.update({"largest_gain": {"retriever": best, **g},
                     "runs": bool(rule["complete"] and g["established"] and g["point"] > 0 and g["point"] >= anchor["point"])})
    return rule


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args()
    root = Path(args.root)
    if args.reproduce:
        rep = reproduce(root)
        (root / OUT).mkdir(parents=True, exist_ok=True)
        (root / OUT / "reproduce_e5small.json").write_text(dump(rep), encoding="utf-8")
        for pop in POPULATIONS:
            r = rep[pop]
            print(f"czech_retrievers: {pop}: e5-small's top 10 differs on {r['top_10_differs']} of {r['questions']} questions "
                  f"({r['top_10_same_ids_other_order']} of them the same ids in another order); top 3 differs on {r['top_3_differs']}")
        return 0 if all(rep[p]["top_10_differs"] == 0 for p in POPULATIONS) else 1
    if args.records:
        for pop, rids in export_records(root).items():
            print(f"czech_retrievers: {pop}: wrote the top 10 of {', '.join(rids) or 'no new retriever yet'}")
        return 0
    text = dump(readings(root))
    path = root / OUT / "retrievers.json"
    if args.check:
        same = path.exists() and path.read_text(encoding="utf-8") == text
        print(f"czech_retrievers: {'OK: the readings recompute byte for byte' if same else 'DIFFERS: ' + str(path)}")
        return 0 if same else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"czech_retrievers: wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
