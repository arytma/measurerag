"""Recompute what the documents say about the SQAD 3.0 copy, from the archive you download yourself.

    # 1. download sqad_v3.tar.xz from http://hdl.handle.net/11234/1-3069 (LINDAT)
    .venv/bin/python scripts/sqad_copy_vs_archive.py --archive sqad_v3.tar.xz          # print, and compare with the tree
    .venv/bin/python scripts/sqad_copy_vs_archive.py --archive sqad_v3.tar.xz --write  # rewrite results/czech_checks.json

The archive's MD5 is checked first, against the value recorded in
src/measurerag/datasets/sqad32.py and docs/DATASETS_AND_LICENCES.md. The copy is read through the
pinned revision (or from `--copy-dir`, a folder holding its train, validation and test parquet files).
Recomputes the record counts, the filter the copy applies and the two overlap thresholds. Nothing of
SQAD is written: counts only. Exit 1 if an MD5 or a number differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

NOTE = ("Counts from comparing the Hugging Face copy with the LINDAT archive of SQAD 3.0. No SQAD text. "
        "scripts/sqad_copy_vs_archive.py recomputes every number here from the archive, whose MD5 it checks, and the pinned copy. "
        "The thresholds are the 90th percentile of the question-context overlap: on the archive with its windows reconstructed, and on the copy as it stands.")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_rows(copy_dir: str | None, cache_dir: str | None) -> list[dict]:
    import pandas as pd

    from measurerag.datasets.sqad import MIRROR, REVISION, SPLIT_FILES

    rows = []
    for split, rel in SPLIT_FILES.items():
        if copy_dir:
            path = Path(copy_dir) / Path(rel).name
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(MIRROR, rel, repo_type="dataset", revision=REVISION, cache_dir=cache_dir)
        for r in pd.read_parquet(path).itertuples(index=False):
            rows.append({"id": str(r.id), "split": split, "question": str(r.question), "context": str(r.context)})
    return rows


def compare(records: list[dict], articles: dict[str, str], copy: list[dict]) -> dict:
    """Which archive records did the copy keep, by where their answer sentence sits?"""
    from measurerag.datasets.sqad32 import contexts_and_rows, jaccard, squeeze

    by_id = {r["id"]: r for r in records}
    kept = {x["id"] for x in copy if x["id"] in by_id and squeeze(by_id[x["id"]].get("question", "")) == squeeze(x["question"])}
    tab = {"answer_sentence_opens_the_article": {"kept": 0, "dropped": 0}, "answer_sentence_comes_later": {"kept": 0, "dropped": 0}, "answer_sentence_not_located": {"kept": 0, "dropped": 0}}
    for r in records:
        art = squeeze(articles.get(r.get("article"), ""))
        sel = squeeze(r.get("selection", ""))
        i = art.find(sel) if sel else -1
        name = "answer_sentence_not_located" if i < 0 else "answer_sentence_opens_the_article" if i == 0 else "answer_sentence_comes_later"
        tab[name]["kept" if r["id"] in kept else "dropped"] += 1

    windows = contexts_and_rows(records, articles)
    p90 = lambda values: statistics.quantiles(values, n=10, method="inclusive")[-1]     # noqa: E731
    return {
        "copy_rows": len(copy), "copy_rows_with_same_id_and_question": len(kept), "archive_records": len(records),
        "filter": tab,
        "threshold_archive_p90": round(p90([jaccard(x["question"], x["context"], fold=True) for x in windows]), 6),
        "threshold_copy_p90": round(p90([jaccard(x["question"], x["context"], fold=True) for x in copy]), 6),
    }


# What LINDAT's page for the item says, read on 10 Nov 2026 ("New version consits of 13477 records", and "Size 13477"). It is not
# recomputed: the archive itself holds the number this script counts, which the documents give first.
LINDAT_STATED = {"records_stated_on_lindat": 13477, "stated_read": "2026-11-10"}


def as_checks(c: dict, repo_id: str, revision: str, md5_value: str) -> dict:
    return {
        "measured": "2026-11-08", "note": NOTE,
        "copy": {"repo_id": repo_id, "revision": revision, "rows": c["copy_rows"], "rows_with_same_id_and_question_in_the_archive": c["copy_rows_with_same_id_and_question"],
                 "licence_governing": "LGPL-3.0 (LINDAT, http://hdl.handle.net/11234/1-3069)"},
        "archive": {"records": c["archive_records"], "handle": "http://hdl.handle.net/11234/1-3069", "md5": md5_value, **LINDAT_STATED},
        "filter_read_from_the_comparison": c["filter"],
        "thresholds": {"archive_p90": c["threshold_archive_p90"], "copy_p90": c["threshold_copy_p90"]},
    }


def main(argv: list[str] | None = None) -> int:
    from measurerag.datasets.sqad import MIRROR, REVISION
    from measurerag.datasets.sqad32 import ARCHIVE_30_MD5, load_parsed, read_archive

    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="the SQAD 3.0 archive from LINDAT, sqad_v3.tar.xz")
    ap.add_argument("--copy-dir", default=None, help="a folder with the copy's parquet files; default: the pinned revision on Hugging Face")
    ap.add_argument("--cache-dir", default=str(REPO / "data" / "hf"))
    ap.add_argument("--expect-md5", default=ARCHIVE_30_MD5)
    ap.add_argument("--checks", default=str(REPO / "results/czech_checks.json"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    got = md5(Path(args.archive))
    if got != args.expect_md5:
        print(f"sqad_copy_vs_archive: MD5 {got}, expected {args.expect_md5}: this is not the archive the documents describe")
        return 1
    print("sqad_copy_vs_archive: archive MD5 ok:", got)
    with tempfile.TemporaryDirectory() as tmp:
        read_archive(args.archive, tmp)
        records, articles = load_parsed(tmp)
    # the same order on every machine: the tar's own order must not decide a tie
    records = sorted(records, key=lambda r: r["id"])
    out = as_checks(compare(records, articles, copy_rows(args.copy_dir, args.cache_dir)), MIRROR, REVISION, got)
    text = json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    print(text, end="")
    path = Path(args.checks)
    if args.write:
        path.write_text(text, encoding="utf-8")
        print("sqad_copy_vs_archive: wrote", path)
        return 0
    if path.read_text(encoding="utf-8") != text:
        print("sqad_copy_vs_archive: DIFFERS from", path)
        return 1
    print("sqad_copy_vs_archive: OK, identical to", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
