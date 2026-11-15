"""Describe the Czech question sets with no generation: why might exact match differ between them?

    .venv/bin/python scripts/czech_diagnostics.py

Needs the data: the pinned Hugging Face copy of SQAD 3.0 and the two LINDAT archives (MD5 checked).
Writes per-question labels, ids and categories only, which ship with the release, so the tables in
results/czech/analysis.json can be recomputed from the tree by scripts/rescore_czech.py:
    results/czech/records/labels_copy_test.jsonl.gz     every test question of the SQAD 3.0 copy
    results/czech/records/labels_sqad32_test.jsonl.gz   every test question of SQAD 3.2
and aggregate counts that join the two versions on the question's text, which need the data to
recompute: results/czech/question_join.json.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

THRESHOLD_STATED = 0.13793103448275862   # the 90th percentile of the full SQAD 3.0 archive
THRESHOLD_SENSITIVITY = 0.10             # the 90th percentile of the copy
K = 3
GEN_CHARS = 1200                          # the budget per passage the generator sees
RETRIEVERS = ("bm25", "bm25-nodiacritics", "e5-small", "retromae-cs")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def labels_for(ds, positions: dict, meta: dict, rankings: dict) -> list[dict]:
    from measurerag.czech_labels import answer_tokens, answer_type, spaced_punct
    from measurerag.datasets.nq import contains_answer
    from measurerag.datasets.sqad32 import debias_flags

    by_id = {d.doc_id: d.text for d in ds.docs}
    idx = {d.doc_id: d for d in ds.docs}
    rows = []
    for q in sorted(ds.queries):
        gold = next(iter(ds.qrels[q]))
        ctx = by_id[gold]
        ans = ds.answers[q]
        pos = positions.get(q)
        stated = debias_flags(ds.queries[q], ctx, pos, THRESHOLD_STATED)
        sens = debias_flags(ds.queries[q], ctx, pos, THRESHOLD_SENSITIVITY)
        kind = answer_type(ans)
        norm, raw = {}, {}
        for rid, run in rankings.items():
            shown = [idx[d].indexable()[:GEN_CHARS] for d in run[q][:K] if d in idx]
            norm[rid] = int(any(contains_answer(s, ans) for s in shown)) if kind != "yes/no" else None
            raw[rid] = int(any(any(a in s for a in ans) for s in shown)) if kind != "yes/no" else None
        rows.append({
            "q": q, "type": kind, "answer_tokens": answer_tokens(ans), "answer_chars": len(ans[0]) if ans else 0,
            "answer_start": pos, "overlap": round(stated["overlap"], 6), "gold_chars": len(ctx),
            "early": bool(stated["early"]), "heavy_0138": bool(stated["heavy"]), "heavy_010": bool(sens["heavy"]),
            "flagged_0138": bool(stated["flagged"]), "flagged_010": bool(sens["flagged"]),
            "spaced_punct": spaced_punct(ans[0]) if ans else False,
            "sqad_a_type": meta.get(q, {}).get("a_type"), "sqad_q_type": meta.get(q, {}).get("q_type"),
            "opens_article": meta.get(q, {}).get("opens"),
            "answer_in_top3_normalised": norm, "answer_in_top3_verbatim": raw,
        })
    return rows


def main() -> int:
    import pandas as pd
    from measurerag.crossover.retrieval import load_run
    from measurerag.czech_labels import answer_tokens, spaced_punct
    from measurerag.datasets.base import Dataset
    from measurerag.datasets.sqad32 import ARCHIVE, ARCHIVE_30, ARCHIVE_30_MD5, ARCHIVE_MD5, answer_position, load_parsed, opens_article, pair_position
    from measurerag.metrics import normalize_answer
    from measurerag.czech_store import write_jsonl_gz

    os.chdir(REPO)
    for path, want in ((ARCHIVE, ARCHIVE_MD5), (ARCHIVE_30, ARCHIVE_30_MD5)):
        got = md5(path)
        if got != want:
            raise SystemExit(f"{path}: MD5 {got}, expected {want}")

    SL = REPO / "results/crossover/slices"
    copy = Dataset.load_slice(SL / "czech30_testall")
    s32 = Dataset.load_slice(SL / "czech32_testall")
    copy_rank = {r: load_run(REPO / "results/crossover/cache/v030_copy", r)["run"] for r in RETRIEVERS}
    s32_rank = {r: load_run(REPO / "results/crossover/cache/v030_32", r)["run"] for r in RETRIEVERS}

    recs30, arts30 = load_parsed(ARCHIVE_30.parent / "parsed")
    recs32, arts32 = load_parsed(ARCHIVE.parent / "parsed")
    by30 = {r["id"]: r for r in recs30}
    by32 = {r["id"]: r for r in recs32}

    # the copy's answer_start is the answer sentence's offset, not the answer's
    frames = {n: pd.read_parquet(glob.glob(str(REPO / f"data/hf/datasets--Pehy--cs_sqad-3.0/snapshots/*/data/{n}-*.parquet"))[0]) for n in ("train", "validation", "test")}
    starts = {str(r.id): (int(r.answers["answer_start"][0]) if len(r.answers["answer_start"]) else 0) for r in frames["test"].itertuples(index=False)}
    docs_copy = {d.doc_id: d.text for d in copy.docs}
    docs_32 = {d.doc_id: d.text for d in s32.docs}
    pos_copy = {q: answer_position(docs_copy[next(iter(copy.qrels[q]))], copy.answers[q], starts.get(q, 0)) for q in copy.queries}
    pos_32 = {q: pair_position(docs_32[next(iter(s32.qrels[q]))], s32.answers[q], by32[q].get("selection", "")) for q in s32.queries}

    def meta(by, arts, ids):
        out = {}
        for q in ids:
            r = by.get(q)
            if r:
                out[q] = {"a_type": r.get("a_type"), "q_type": r.get("q_type"),
                          "opens": opens_article(arts[r["article"]], r.get("selection", "")) if r.get("article") in arts else None}
        return out

    out_dir = REPO / "results/czech/records"
    labels_copy = labels_for(copy, pos_copy, meta(by30, arts30, copy.queries), copy_rank)
    labels_32 = labels_for(s32, pos_32, meta(by32, arts32, s32.queries), s32_rank)

    # the same question in the other version's test split, by id; and where a SQAD 3.2 question comes from
    in_copy = {str(q) for df in frames.values() for q in df["question"]}
    in_archive = {r["question"] for r in recs30 if r.get("question")}
    id_30 = {t: q for q, t in copy.queries.items()}
    id_32 = {t: q for q, t in s32.queries.items()}
    for row in labels_copy:
        row["same_question"] = id_32.get(copy.queries[row["q"]])
    for row in labels_32:
        text = s32.queries[row["q"]]
        row["same_question"] = id_30.get(text)
        row["origin"] = "in the copy" if text in in_copy else "dropped from the copy" if text in in_archive else "new in SQAD 3.2"
    write_jsonl_gz(out_dir / "labels_copy_test.jsonl.gz", labels_copy)
    write_jsonl_gz(out_dir / "labels_sqad32_test.jsonl.gz", labels_32)

    # ---- the two versions joined on the question's text; aggregate counts only -----------------
    copy_rows = []
    for n, df in frames.items():
        for r in df.itertuples(index=False):
            texts = [str(t) for t in r.answers["text"]]
            copy_rows.append({"id": str(r.id), "split": n, "question": str(r.question), "answers": texts})
    from measurerag.datasets.sqad32 import gold_answers

    rows32: dict[str, list[dict]] = {}
    for r in recs32:
        if r.get("question"):
            rows32.setdefault(r["question"], []).append({"id": r["id"], "split": r.get("split"), "answers": gold_answers(r)})
    joined = same = diff = yes_no = copy_spaced = s32_spaced = 0
    both_spaced = only_copy = only_32 = 0
    longer_copy = longer_32 = 0
    split_pairs: dict[str, int] = {}
    for x in copy_rows:
        match = rows32.get(x["question"])
        if not match or not x["answers"]:
            continue
        m = next((y for y in match if y["answers"]), None)
        if m is None:
            continue
        joined += 1
        split_pairs[f"copy {x['split']}, SQAD 3.2 {m['split']}"] = split_pairs.get(f"copy {x['split']}, SQAD 3.2 {m['split']}", 0) + 1
        a, b = x["answers"][0], m["answers"][0]
        na, nb = normalize_answer(a), normalize_answer(b)
        if na in ("ano", "ne"):
            yes_no += 1
        if na == nb:
            same += 1
        else:
            diff += 1
            if answer_tokens([a]) > answer_tokens([b]):
                longer_copy += 1
            elif answer_tokens([b]) > answer_tokens([a]):
                longer_32 += 1
        sa, sb = spaced_punct(a), spaced_punct(b)
        copy_spaced += sa
        s32_spaced += sb
        both_spaced += sa and sb
        only_copy += sa and not sb
        only_32 += sb and not sa
    report = {
        "note": "SQAD record ids are not stable between versions, so these join on the question's text. Aggregate counts only.",
        "sqad30_archive_md5": ARCHIVE_30_MD5, "sqad32_archive_md5": ARCHIVE_MD5,
        "copy_rows": len(copy_rows), "sqad32_records_with_a_question": sum(len(v) for v in rows32.values()),
        "joined_on_question_text": joined, "joined_by_split": dict(sorted(split_pairs.items())),
        "gold_same_after_normalising": same, "gold_differs_after_normalising": diff,
        "where_it_differs_the_copy_is_longer": longer_copy, "where_it_differs_sqad32_is_longer": longer_32,
        "yes_no_gold": yes_no,
        "gold_with_spaced_punctuation": {"copy": copy_spaced, "sqad32": s32_spaced, "both": both_spaced, "only_copy": only_copy, "only_sqad32": only_32},
    }
    (REPO / "results/czech/question_join.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
