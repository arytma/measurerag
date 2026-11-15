"""Build the question sets for the SQAD 3.2 experiment. No model runs.

    .venv/bin/python scripts/sqad32_slices.py

Writes, under results/crossover/slices/ (git-ignored):
    czech32_full      3a: SQAD 3.2, 200 test questions by the loader's seeded rule
    czech32_debiased  3b: 3a's unflagged questions, topped up to 200
    czech32_testall   the whole 3.2 test split, for one ranking per retriever
    czech30_debiased  3c: the released 200's unflagged questions, topped up to 200
    czech30_testall   the copy of 3.0's whole test split, the same use
    czech32_filtered  retrieval only: unflagged test questions, flagged windows out of the corpus
    czech30_archive   retrieval only: the real 3.0 archive, reconstructed windows, 2,000 questions
and results/sqad32/slices.json with the counts of each rule.

The corpus does not change between a full set and its debiased set, only the
questions, so a question's prompts are the same in both and its answers are reused.
The Hugging Face copy of 3.0 dropped its filtered records from the corpus too; the
czech32_filtered slice measures what that difference does to retrieval.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from measurerag.datasets.base import Dataset, Doc  # noqa: E402
from measurerag.datasets.sqad import load as load_30  # noqa: E402
from measurerag.datasets.sqad32 import (  # noqa: E402
    ARCHIVE,
    ARCHIVE_30,
    answer_position,
    contexts_and_rows,
    debias_flags,
    doc_id,
    load as load_32,
    load_parsed,
    opens_article,
    pair_position,
)
from measurerag.seeds import rng  # noqa: E402

N = 200
N_ARCHIVE = 2000
OUT = REPO / "results/crossover/slices"
RELEASED = REPO / "results/cz_rag/slice"


def restrict(ds: Dataset, qids: list[str], name: str, note: str, docs: list[Doc] | None = None) -> Dataset:
    return Dataset(
        name=name, language=ds.language, docs=docs if docs is not None else ds.docs,
        queries={q: ds.queries[q] for q in qids},
        qrels={q: ds.qrels[q] for q in qids},
        answers={q: ds.answers[q] for q in qids},
        license=ds.license, source=ds.source, notes=[*ds.notes, note],
    )


def flags_for(ds: Dataset, threshold: float, starts: dict[str, int] | None = None, selections: dict[str, str] | None = None) -> dict[str, dict]:
    """`starts` is the copy of 3.0's sentence offset; `selections` the archives' answer sentence."""
    by_id = {d.doc_id: d.text for d in ds.docs}
    out = {}
    for q in ds.queries:
        ctx = by_id[next(iter(ds.qrels[q]))]
        if selections is not None:
            pos = pair_position(ctx, ds.answers[q], selections.get(q, ""))
        else:
            pos = answer_position(ctx, ds.answers[q], (starts or {}).get(q, 0))
        out[q] = debias_flags(ds.queries[q], ctx, pos, threshold)
    return out


def counts(flags: dict[str, dict], qids, opens: dict[str, bool] | None = None) -> dict:
    qids = list(qids)
    out = {
        "questions": len(qids),
        "flag_early": sum(flags[q]["early"] for q in qids),
        "flag_heavy": sum(flags[q]["heavy"] for q in qids),
        "flag_both": sum(flags[q]["early"] and flags[q]["heavy"] for q in qids),
        "flag_any": sum(flags[q]["flagged"] for q in qids),
    }
    if opens is not None:
        out["answer_sentence_opens_article"] = sum(bool(opens.get(q)) for q in qids)
    return out


def debiased(base: list[str], pool: list[str], flags: dict[str, dict], tag: str) -> tuple[list[str], list[str]]:
    """Base's unflagged questions, then seeded unflagged questions from the pool up to N."""
    kept = [q for q in sorted(base) if not flags[q]["flagged"]]
    rest = sorted(q for q in pool if q not in set(base) and not flags[q]["flagged"])
    order = rng("sqad32", tag, "topup").permutation(len(rest)).tolist()
    return kept, [rest[i] for i in order[: max(0, N - len(kept))]]


def main() -> int:
    os.chdir(REPO)
    bias = json.loads((REPO / "results/sqad32/bias.json").read_text(encoding="utf-8"))
    threshold = float(bias["threshold_full"])
    report: dict = {"threshold": threshold, "n_target": N}
    labels: dict[str, dict] = {}

    # ---- SQAD 3.2 -------------------------------------------------------------
    full32 = load_32(max_queries=None)
    a = load_32(max_queries=N)
    assert a.corpus_hash() == full32.corpus_hash()
    recs32, arts32 = load_parsed(ARCHIVE.parent / "parsed")
    open32 = {r["id"]: opens_article(arts32[r["article"]], r.get("selection", "")) for r in recs32 if r["id"] in full32.queries}
    sel32 = {r["id"]: r.get("selection", "") for r in recs32}
    f32 = flags_for(full32, threshold, selections=sel32)
    kept, top = debiased(list(a.queries), list(full32.queries), f32, "3b")
    report["sqad32"] = {
        "corpus_sha256": full32.corpus_hash(), "n_docs": len(full32.docs),
        "test_split": counts(f32, full32.queries, open32),
        "sample_3a": counts(f32, a.queries, open32),
        "kept_from_3a": len(kept), "topped_up": len(top),
        "sample_3b_answer_sentence_opens_article": sum(bool(open32.get(q)) for q in kept + top),
    }
    a.save(OUT / "czech32_full")
    b = restrict(full32, kept + top, "sqad-3.2-test-debiased", "Debiased by this kit's own two rules and thresholds.")
    b.save(OUT / "czech32_debiased")
    full32.save(OUT / "czech32_testall")

    # retrieval only: the flagged windows leave the corpus as well
    all_rows = contexts_and_rows(recs32, arts32)
    keep_docs: dict[str, Doc] = {}
    for x in all_rows:
        fl = debias_flags(x["question"], x["context"], pair_position(x["context"], x["answers"], x["selection"]), threshold)
        if not fl["flagged"]:
            d = doc_id(x["context"])
            keep_docs.setdefault(d, Doc(doc_id=d, text=x["context"], title=""))
    docs_f = sorted(keep_docs.values(), key=lambda d: d.doc_id)
    q_f = [q for q in sorted(full32.queries) if not f32[q]["flagged"]]
    restrict(full32, q_f, "sqad-3.2-test-filtered", "Retrieval only: flagged pairs removed from questions and corpus.", docs_f).save(OUT / "czech32_filtered")
    report["sqad32_filtered"] = {"n_docs": len(docs_f), "questions": len(q_f)}

    # ---- SQAD 3.0, the copy behind the released numbers ---------------------------
    full30 = load_30(cache_dir=REPO / "data", max_queries=None)
    rel = Dataset.load_slice(RELEASED)
    assert rel.corpus_hash() == full30.corpus_hash(), "the released corpus is not the copy's"
    assert set(rel.queries) <= set(full30.queries)
    import glob

    import pandas as pd

    # the copy's answer_start is the answer sentence's offset, not the answer's
    df = pd.read_parquet(glob.glob(str(REPO / "data/hf/datasets--Pehy--cs_sqad-3.0/snapshots/*/data/test-*.parquet"))[0])
    starts = {str(r.id): (int(r.answers["answer_start"][0]) if len(r.answers["answer_start"]) else 0) for r in df.itertuples(index=False)}
    recs30, arts30 = load_parsed(ARCHIVE_30.parent / "parsed")
    open30 = {r["id"]: opens_article(arts30[r["article"]], r.get("selection", "")) for r in recs30 if r["id"] in full30.queries}
    f30 = flags_for(full30, threshold, starts)
    kept30, top30 = debiased(list(rel.queries), list(full30.queries), f30, "3c")
    report["sqad30_copy"] = {
        "corpus_sha256": full30.corpus_hash(), "n_docs": len(full30.docs),
        "test_split": counts(f30, full30.queries, open30),
        "released_200": counts(f30, rel.queries, open30),
        "kept_from_released": len(kept30), "topped_up": len(top30),
        "sample_3c_answer_sentence_opens_article": sum(bool(open30.get(q)) for q in kept30 + top30),
    }
    restrict(full30, kept30 + top30, "sqad-3.0-test-debiased",
             "Debiased by this kit's own two rules and thresholds.").save(OUT / "czech30_debiased")
    restrict(full30, sorted(full30.queries), "sqad-3.0-test", "The copy's whole test split, for rankings.").save(OUT / "czech30_testall")

    # ---- retrieval only: the real 3.0 archive, windows, no filter --------------------
    rows30 = contexts_and_rows(recs30, arts30)
    docs30: dict[str, Doc] = {}
    for x in rows30:
        d = doc_id(x["context"])
        docs30.setdefault(d, Doc(doc_id=d, text=x["context"], title=""))
    order = rng("sqad32", "archive30", "queries").permutation(len(rows30)).tolist()[:N_ARCHIVE]
    chosen = [rows30[i] for i in sorted(order)]
    arch = Dataset(
        name="sqad-3.0-archive", language="cs", docs=sorted(docs30.values(), key=lambda d: d.doc_id),
        queries={x["id"]: x["question"] for x in chosen}, qrels={x["id"]: {doc_id(x["context"]): 1} for x in chosen},
        answers={x["id"]: x["answers"] for x in chosen},
        license="LGPL-3.0 (SQAD 3.0 on LINDAT, http://hdl.handle.net/11234/1-3069)", source="http://hdl.handle.net/11234/1-3069",
        notes=["Retrieval only. Every archive record, reconstructed windows, seeded sample of questions."],
    )
    arch.save(OUT / "czech30_archive")
    f3a = flags_for(arch, threshold, selections={r["id"]: r.get("selection", "") for r in recs30})
    open3a = {x["id"]: x["opens_article"] for x in chosen}
    report["sqad30_archive"] = {"n_docs": len(arch.docs), "sample": counts(f3a, arch.queries, open3a)}

    # per-question labels, for the analysis's subgroups: ids and booleans only
    labels = {
        "sqad32_test": {q: {"flagged": f32[q]["flagged"], "early": f32[q]["early"], "heavy": f32[q]["heavy"], "opens": bool(open32.get(q))} for q in full32.queries},
        "sqad30_copy_test": {q: {"flagged": f30[q]["flagged"], "early": f30[q]["early"], "heavy": f30[q]["heavy"], "opens": bool(open30.get(q))} for q in full30.queries},
        "sqad30_archive": {q: {"flagged": f3a[q]["flagged"], "early": f3a[q]["early"], "heavy": f3a[q]["heavy"], "opens": bool(open3a.get(q))} for q in arch.queries},
    }
    (REPO / "results/sqad32").mkdir(parents=True, exist_ok=True)
    (REPO / "results/sqad32/slices.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    labels_path = OUT / "sqad32_labels.json"
    labels_path.write_text(json.dumps(labels, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
