"""BEIR datasets, pulled from the Hugging Face mirrors.

The canonical BEIR zips live on a TU Darmstadt web server, which is outside the
network allowlist this kit runs under (Hugging Face, GitHub, PyPI only), so the
`BeIR/*` mirrors are used instead. They carry the same corpora, queries and
qrels.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import Dataset, DatasetUnavailable, Doc

# Pinned to the commits every result used: each was the repository's main on 14 Nov 2026, the build's cache was
# made from it, and no commit came after 2026-04-09, before the first build. A later main cannot change a slice.
SUBSETS: dict[str, dict] = {
    "scifact": {
        "repo": "BeIR/scifact",
        "qrels_repo": "BeIR/scifact-qrels",
        "revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
        "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
        "split": "test",
        "language": "en",
        "license": "CC BY-NC 2.0 (SciFact, Wadden et al. 2020; the allenai/scifact card)",
    },
    "nfcorpus": {
        "repo": "BeIR/nfcorpus",
        "qrels_repo": "BeIR/nfcorpus-qrels",
        "revision": "b5026a0e96e8a7ac4f95f482a596389289d46269",
        "qrels_revision": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
        "split": "test",
        "language": "en",
        "license": "CC BY-SA 4.0 (NFCorpus, Boteva et al. 2016)",
    },
    "fiqa": {
        "repo": "BeIR/fiqa",
        "qrels_repo": "BeIR/fiqa-qrels",
        "revision": "979c07a7cb5ccc6ca009792241fa1250b98055dd",
        "qrels_revision": "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
        "split": "test",
        "language": "en",
        "license": "see FiQA-2018 task terms (Maia et al. 2018)",
    },
    "hotpotqa": {
        "repo": "BeIR/hotpotqa",
        "qrels_repo": "BeIR/hotpotqa-qrels",
        "revision": "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b",
        "qrels_revision": "b15429e9244c8ec966985d7778427c3b1543b314",
        "split": "test",
        "language": "en",
        "license": "CC BY-SA 4.0 (HotpotQA, Yang et al. 2018)",
    },
}


def load(
    subset: str = "scifact",
    cache_dir: str | Path | None = None,
    split: str | None = None,
    **_,
) -> Dataset:
    if subset not in SUBSETS:
        raise KeyError(f"unknown BEIR subset {subset!r}; known: {', '.join(SUBSETS)}")
    meta = SUBSETS[subset]
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        from datasets import load_dataset as hf_load
    except Exception as exc:  # pragma: no cover
        raise DatasetUnavailable(f"the `datasets` package is required: {exc}") from exc

    hf_cache = str(Path(cache_dir) / "hf") if cache_dir else None

    try:
        corpus = hf_load(meta["repo"], "corpus", split="corpus", revision=meta["revision"], cache_dir=hf_cache)
        queries = hf_load(meta["repo"], "queries", split="queries", revision=meta["revision"], cache_dir=hf_cache)
        qrels_split = split or meta["split"]
        qrels = hf_load(meta["qrels_repo"], split=qrels_split, revision=meta["qrels_revision"], cache_dir=hf_cache)
    except Exception as exc:
        raise DatasetUnavailable(f"{meta['repo']}: {exc}") from exc

    rel: dict[str, dict[str, int]] = {}
    for row in qrels:
        qid = str(row["query-id"])
        rel.setdefault(qid, {})[str(row["corpus-id"])] = int(row["score"])

    # BEIR ships every query in one file and judges only the evaluation split
    q = {str(r["_id"]): r["text"] for r in queries if str(r["_id"]) in rel}

    docs = [
        Doc(doc_id=str(r["_id"]), text=r.get("text") or "", title=r.get("title") or "")
        for r in corpus
    ]

    return Dataset(
        name=f"beir-{subset}",
        language=meta["language"],
        docs=docs,
        queries=q,
        qrels={k: v for k, v in rel.items() if k in q},
        license=meta["license"],
        source=f"https://huggingface.co/datasets/{meta['repo']} (split {qrels_split})",
        notes=[
            "BEIR mirror on Hugging Face; the canonical zips are outside this "
            "kit's network allowlist."
        ],
    )
