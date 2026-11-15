"""Cache one top-100 ranking per retriever for a slice.

    uv run python scripts/retrieval_cache.py --slice results/en_rag/slice \
        --cache results/crossover/cache/hotpotqa --retriever e5-base:dense:e5-base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse(spec: str) -> dict:
    rid, kind, *rest = spec.split(":")
    cfg = {"id": rid, "kind": kind}
    if rid == "bm25-nodiacritics":
        cfg["strip_diacritics"] = True
    if rid == "bm25-lemma":          # BM25 over Czech lemmas
        cfg["lemmatize"] = "cs"
    if kind == "dense":
        cfg["model"] = rest[0]
    if kind == "tfidf-svd":
        cfg["n_components"] = int(rest[0]) if rest else 256
    return cfg


def main() -> int:
    from measurerag.crossover.retrieval import build_run, cache_path
    from measurerag.datasets.base import Dataset
    from measurerag.openmp import preflight
    from measurerag.seeds import seed_everything

    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--retriever", required=True, help="id:kind[:model]")
    ap.add_argument("--reuse-index", default=None)
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args()

    preflight()
    seed_everything(args.seed)
    rcfg = parse(args.retriever)
    out = cache_path(Path(args.cache), rcfg["id"])
    if out.exists():
        print(f"{out} exists; not recomputing")
        return 0
    ds = Dataset.load_slice(args.slice)
    res = build_run(
        ds, rcfg, Path(args.cache) / "work" / rcfg["id"],
        reuse_index=Path(args.reuse_index) if args.reuse_index else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    print(rcfg["id"], json.dumps(res["metrics"][5]), res["wall_seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
