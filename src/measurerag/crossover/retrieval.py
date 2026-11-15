"""One ranking per retriever x dataset, computed once and cached.

Generation is the expensive step and is repeated across the dial's levels and
both generators; retrieval is not, so it is written to disk with the scores and
reused. The ranking is produced exactly as the budget harness produces it -
`search_one`, one query at a time, query threads pinned to 1 - so that a dial
context at p = 0 is byte-for-byte the context the harness handed the generator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..datasets.base import Dataset
from ..hashing import sha256_json
from ..metrics import evaluate_run


def cache_path(cache_dir: Path, retriever_id: str) -> Path:
    return Path(cache_dir) / f"retrieval_{retriever_id}.json"


def load_run(cache_dir: Path, retriever_id: str) -> dict:
    return json.loads(cache_path(cache_dir, retriever_id).read_text(encoding="utf-8"))


def build_run(
    ds: Dataset,
    rcfg: dict,
    workdir: Path,
    top_k: int = 100,
    build_threads: int | None = None,
    reuse_index: Path | None = None,
) -> dict:
    """Rank every query in `ds`. Returns {"run": {qid: [doc ids]}, "scores": ..., ...}."""
    import os

    from ..retrievers import build_retriever, describe_retriever

    def set_threads(n: int) -> None:
        try:
            import torch

            torch.set_num_threads(max(1, n))
        except Exception:
            pass

    rcfg = dict(rcfg)
    kind = rcfg.pop("kind")
    rid = rcfg.pop("id", kind)
    rcfg.setdefault("device", "cpu")
    retriever = build_retriever(kind, **rcfg)
    t0 = time.perf_counter()
    if reuse_index is not None and kind == "dense":
        # the harness already embedded this exact slice; load its flat index
        import faiss

        retriever._doc_ids = [d.doc_id for d in ds.docs]
        retriever._index = faiss.read_index(str(reuse_index))
        retriever._load_encoder()
        build = {"reused_index": str(reuse_index), "n_docs": retriever._index.ntotal}
        if retriever._index.ntotal != len(ds.docs):
            raise RuntimeError(f"{reuse_index}: {retriever._index.ntotal} vectors, slice has {len(ds.docs)}")
    else:
        set_threads(build_threads or os.cpu_count() or 1)
        stats = retriever.build(ds.docs, Path(workdir))
        build = {"build_seconds": stats.build_seconds, "index_bytes": stats.index_bytes, **stats.extra}
    set_threads(1)

    run: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for qid in sorted(ds.queries):
        hits = retriever.search_one(ds.queries[qid], top_k)
        run[qid] = [d for d, _ in hits]
        scores[qid] = [round(float(s), 6) for _, s in hits]
    retriever.close()

    return {
        "retriever_id": rid,
        "retriever": {"kind": kind, **rcfg},
        "declaration": describe_retriever(kind, **rcfg),
        "dataset": ds.name,
        "corpus_sha256": ds.corpus_hash(),
        "queries_sha256": ds.query_hash(),
        "top_k": top_k,
        "build": build,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "metrics": {k: evaluate_run(run, ds.qrels, ks=[10, 100], context_k=k) for k in (3, 5, 10)},
        "run": run,
        "scores": scores,
        "run_sha256": sha256_json(run),
    }
