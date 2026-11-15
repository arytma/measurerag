"""HotpotQA for the end-to-end stage: BEIR corpus + distractor-setting answers.

Retrieval gold and the corpus come from the BEIR mirror; the short answers come
from the original HotpotQA distractor validation split, joined on the question
id, which BEIR preserves. That gives one dataset carrying both nDCG and EM/F1,
so "the retriever found it but the generator fumbled it" is visible as a gap
between gold-in-context and EM rather than a mystery.

The BEIR corpus is 5.2M paragraphs. It is streamed and capped: every gold
paragraph for a sampled question is kept, the rest is a seeded reservoir.
Licence: CC BY-SA 4.0.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import Dataset, DatasetUnavailable, Doc
from ..seeds import rng

BEIR_REPO = "BeIR/hotpotqa"
QRELS_REPO = "BeIR/hotpotqa-qrels"
ANSWER_REPO = "hotpotqa/hotpot_qa"
ANSWER_FILE = "distractor/validation-00000-of-00001.parquet"
CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
# Pinned to the commits every result used: each was the repository's main on 14 Nov 2026, the build's cache was
# made from it, and no commit came after 2026-04-09, before the first build. A later main cannot change a slice.
REVISIONS = {BEIR_REPO: "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b", QRELS_REPO: "b15429e9244c8ec966985d7778427c3b1543b314",
             ANSWER_REPO: "1908d6afbbead072334abe2965f91bd2709910ab"}


def _pairs(df, key_col: str, value_col: str, wrap: bool = False) -> dict:
    """Two columns of a DataFrame as a dict, addressed by name.

    Not `itertuples`: BEIR's query id column is literally named `_id`, which is
    not a valid Python identifier, so pandas silently renames it to a *position*
    - `_1` with the index included, `_0` without. The original code asked for
    `_1` while passing `index=False`, so it could never have worked; nothing had
    reached this loader to find out. Addressing columns by name cannot drift
    with a pandas version or a keyword argument.
    """
    if key_col not in df.columns or value_col not in df.columns:
        raise DatasetUnavailable(
            f"HotpotQA: expected columns {key_col!r} and {value_col!r}, "
            f"got {list(df.columns)}"
        )
    out = {}
    for key, value in zip(df[key_col], df[value_col]):
        if value is None or (wrap and not value):
            continue
        out[str(key)] = [str(value)] if wrap else str(value)
    return out


def load(
    cache_dir: str | Path | None = None,
    split: str = "test",
    max_docs: int | None = 100_000,
    max_queries: int | None = 500,
    **_,
) -> Dataset:
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    hf_cache = str(Path(cache_dir) / "hf") if cache_dir else None

    def grab(repo: str, filename: str) -> str:
        return hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo], cache_dir=hf_cache)

    try:
        from datasets import load_dataset as hf_load

        qrels_rows = hf_load(QRELS_REPO, split=split, revision=REVISIONS[QRELS_REPO], cache_dir=hf_cache)
        queries_df = pd.read_parquet(grab(BEIR_REPO, QUERIES_FILE))
        answers_df = pd.read_parquet(grab(ANSWER_REPO, ANSWER_FILE))
        corpus_path = grab(BEIR_REPO, CORPUS_FILE)
    except Exception as exc:
        raise DatasetUnavailable(f"HotpotQA: {exc}") from exc

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    answers = _pairs(answers_df, "id", "answer", wrap=True)
    query_text = _pairs(queries_df, "_id", "text")

    # only questions we can both judge and score end-to-end
    usable = sorted(q for q in qrels if q in answers and q in query_text)
    if max_queries is not None and len(usable) > max_queries:
        idx = rng("hotpotqa", split, "queries").choice(
            len(usable), size=max_queries, replace=False
        )
        usable = [usable[i] for i in sorted(idx.tolist())]

    keep_q = set(usable)
    gold_ids = {d for q in keep_q for d in qrels[q]}

    # ---- stream the 5.2M-paragraph corpus, keep gold, sample the rest -------
    r = rng("hotpotqa", "corpus", max_docs or 0)
    gold_docs: list[Doc] = []
    reservoir: list[Doc] = []
    capacity = None if max_docs is None else max(0, max_docs - len(gold_ids))
    seen_other = 0

    pf = pq.ParquetFile(corpus_path)
    for batch in pf.iter_batches(batch_size=20_000, columns=["_id", "title", "text"]):
        ids = batch.column("_id").to_pylist()
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        for doc_id, title, text in zip(ids, titles, texts):
            doc_id = str(doc_id)
            doc = Doc(doc_id=doc_id, text=text or "", title=title or "")
            if doc_id in gold_ids:
                gold_docs.append(doc)
                continue
            if capacity is None:
                reservoir.append(doc)
                continue
            if capacity == 0:
                seen_other += 1
                continue
            if len(reservoir) < capacity:
                reservoir.append(doc)
            else:
                j = int(r.integers(0, seen_other + 1))
                if j < capacity:
                    reservoir[j] = doc
            seen_other += 1

    docs = sorted(gold_docs + reservoir, key=lambda d: d.doc_id)
    present = {d.doc_id for d in docs}

    return Dataset(
        name="hotpotqa-rag",
        language="en",
        docs=docs,
        queries={q: query_text[q] for q in usable},
        qrels={q: {d: g for d, g in qrels[q].items() if d in present} for q in usable},
        answers={q: answers[q] for q in usable},
        license="CC BY-SA 4.0 (HotpotQA, Yang et al. 2018; BEIR mirror)",
        source=f"{BEIR_REPO} corpus + {ANSWER_REPO} distractor/validation answers",
        notes=[
            "Answers come from the distractor split; retrieval gold comes from "
            "BEIR qrels. They join on the HotpotQA question id.",
            "Corpus capped by a seeded reservoir sample with every gold "
            "paragraph forced in, so recall is bounded by the retriever, not by "
            "the sampling.",
            "Multi-hop: a question needs two paragraphs, so gold-in-context@k "
            "here means 'at least one', which is a floor, not the full evidence.",
        ],
    )
