"""The stated debiasing rule for v0.3.0: the 90th percentile of the full SQAD 3.0 archive.

    .venv/bin/python scripts/sqad32_archive_rule.py

The 200-question debiased sets (3b, 3c) were built with the copy's 90th percentile, which
stays as a sensitivity row. This script runs no model. It writes:
    results/sqad32/archive_rule.json   the threshold; flag counts under both thresholds;
                                       how many generations a full 200-question set would need
    results/crossover/slices/sqad32_labels_archive_rule.json   per-question flags, for the analysis
    results/crossover/slices/czech32_archive_rule, czech30_archive_rule   the sets for v0.3.0
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd  # noqa: E402
from measurerag.datasets.base import Dataset  # noqa: E402
from measurerag.datasets.sqad import load as load_30  # noqa: E402
from measurerag.datasets.sqad32 import ARCHIVE, ARCHIVE_30, answer_position, contexts_and_rows, debias_flags, jaccard, load as load_32, load_parsed, pair_position  # noqa: E402
from sqad32_slices import OUT, RELEASED, N, counts, debiased, flags_for, restrict  # noqa: E402

PY = ".venv/bin/python"
B, G = "qwen2.5-1.5b", "qwen2.5-3b"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def rows_counts(rows: list[dict], threshold: float) -> dict:
    fl = [debias_flags(x["question"], x["context"], x["position"], threshold) for x in rows]
    return {"pairs": len(rows), "flag_early": sum(f["early"] for f in fl), "flag_heavy": sum(f["heavy"] for f in fl),
            "flag_both": sum(f["early"] and f["heavy"] for f in fl), "flag_any": sum(f["flagged"] for f in fl), "flag_any_share": round(sum(f["flagged"] for f in fl) / len(rows), 4)}


def planned(slice_: Path, cache: str, tag: str, retriever: str, gen: str, levels: str) -> int:
    o = subprocess.run([PY, "scripts/dial_generate.py", "--dataset", tag, "--slice", str(slice_.relative_to(REPO)), "--cache", cache, "--retriever", retriever, "--k", "3",
                        "--levels", levels, "--generator", gen, "--dry-run"], cwd=REPO, env=ENV, capture_output=True, text=True)
    m = re.search(r"(\d+) to generate", o.stdout)
    if not m:
        raise SystemExit(f"dry run failed: {o.stdout} {o.stderr[-400:]}")
    return int(m.group(1))


def main() -> int:
    os.chdir(REPO)
    copy_threshold = float(json.loads((REPO / "results/sqad32/bias.json").read_text(encoding="utf-8"))["threshold_full"])

    recs30, arts30 = load_parsed(ARCHIVE_30.parent / "parsed")
    rows30 = contexts_and_rows(recs30, arts30)
    for x in rows30:
        x["position"] = pair_position(x["context"], x["answers"], x["selection"])
    threshold = st.quantiles([jaccard(x["question"], x["context"], fold=True) for x in rows30], n=10, method="inclusive")[-1]

    recs32, arts32 = load_parsed(ARCHIVE.parent / "parsed")
    rows32 = contexts_and_rows(recs32, arts32)
    for x in rows32:
        x["position"] = pair_position(x["context"], x["answers"], x["selection"])
    sel32 = {r["id"]: r.get("selection", "") for r in recs32}
    sel30 = {r["id"]: r.get("selection", "") for r in recs30}

    copy_rows = []
    for name in ("train", "validation", "test"):
        df = pd.read_parquet(glob.glob(str(REPO / f"data/hf/datasets--Pehy--cs_sqad-3.0/snapshots/*/data/{name}-*.parquet"))[0])
        for r in df.itertuples(index=False):
            starts = [int(s) for s in r.answers["answer_start"]]
            texts = [str(t) for t in r.answers["text"]]
            copy_rows.append({"id": str(r.id), "split": name, "question": str(r.question), "context": str(r.context),
                              "position": answer_position(str(r.context), texts, starts[0] if starts else 0)})

    full32 = load_32(max_queries=None)
    a = load_32(max_queries=N)
    full30 = load_30(cache_dir=REPO / "data", max_queries=None)
    rel = Dataset.load_slice(RELEASED)
    arch = Dataset.load_slice(OUT / "czech30_archive")
    df = pd.read_parquet(glob.glob(str(REPO / "data/hf/datasets--Pehy--cs_sqad-3.0/snapshots/*/data/test-*.parquet"))[0])
    starts = {str(r.id): (int(r.answers["answer_start"][0]) if len(r.answers["answer_start"]) else 0) for r in df.itertuples(index=False)}

    report: dict = {"threshold_archive_p90": threshold, "threshold_copy_p90": copy_threshold, "flags": {}}
    both = {}
    for tname, t in (("archive_rule", threshold), ("copy_rule", copy_threshold)):
        f32, f30, fa = flags_for(full32, t, selections=sel32), flags_for(full30, t, starts), flags_for(arch, t, selections=sel30)
        both[tname] = (f32, f30, fa)
        report["flags"][tname] = {
            "threshold": t,
            "sqad30_archive_all": rows_counts(rows30, t),
            "sqad30_copy_all": rows_counts(copy_rows, t),
            "sqad32_all": rows_counts(rows32, t),
            "sqad32_test_split": counts(f32, full32.queries),
            "sqad30_copy_test_split": counts(f30, full30.queries),
            "sqad30_archive_sample": counts(fa, arch.queries),
            "released_200": counts(f30, rel.queries),
            "sample_3a": counts(f32, a.queries),
        }
    f32, f30, fa = both["archive_rule"]

    # the sets v0.3.0 would run: nothing is generated here; only count what is missing
    kept, top = debiased(list(a.queries), list(full32.queries), f32, "v0.3.0-archive-rule")
    kept30, top30 = debiased(list(rel.queries), list(full30.queries), f30, "v0.3.0-archive-rule-copy")
    s32, s30 = OUT / "czech32_archive_rule", OUT / "czech30_archive_rule"
    restrict(full32, kept + top, "sqad-3.2-test-archive-rule", "The stated rule for v0.3.0: the 90th percentile of the full SQAD 3.0 archive.").save(s32)
    restrict(full30, kept30 + top30, "sqad-3.0-test-archive-rule", "The stated rule for v0.3.0: the 90th percentile of the full SQAD 3.0 archive.").save(s30)
    c32, c30 = "results/crossover/cache/czech32", "results/crossover/cache/czech30d"
    levels = "0,0.2,0.4,0.6,0.8,1.0"
    need32 = {"dial_1.5b": planned(s32, c32, "czech32", "e5-small", B, levels), "dial_3b": planned(s32, c32, "czech32", "e5-small", G, levels),
              "bm25_1.5b": planned(s32, c32, "czech32", "bm25", B, "natural"), "retromae_1.5b": planned(s32, c32, "czech32", "retromae-cs", B, "natural")}
    need30 = {"dial_1.5b": planned(s30, c30, "czech", "e5-small", B, levels), "dial_3b": planned(s30, c30, "czech", "e5-small", G, levels),
              "bm25_1.5b": planned(s30, c30, "czech", "bm25", B, "natural"), "retromae_1.5b": planned(s30, c30, "czech", "retromae-cs", B, "natural")}
    report["full_200_question_sets"] = {
        "sqad32": {"kept_from_3a": len(kept), "topped_up": len(top), "new_generations_needed": need32, "total": sum(need32.values())},
        "sqad30_copy": {"kept_from_released": len(kept30), "topped_up": len(top30), "new_generations_needed": need30, "total": sum(need30.values())},
        "note": "Not run. Counted with the dial's dry run against the answers already cached.",
    }

    labels = {
        "sqad32_test": {q: {"flagged_archive_rule": f32[q]["flagged"]} for q in full32.queries},
        "sqad30_copy_test": {q: {"flagged_archive_rule": f30[q]["flagged"]} for q in full30.queries},
        "sqad30_archive": {q: {"flagged_archive_rule": fa[q]["flagged"]} for q in arch.queries},
    }
    (OUT / "sqad32_labels_archive_rule.json").write_text(json.dumps(labels, sort_keys=True) + "\n", encoding="utf-8")
    (REPO / "results/sqad32/archive_rule.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("threshold_archive_p90", "threshold_copy_p90", "full_200_question_sets")}, indent=1))
    for tname in ("archive_rule", "copy_rule"):
        print(tname, {k: (v.get("flag_any"), v.get("questions", v.get("pairs"))) for k, v in report["flags"][tname].items() if isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
