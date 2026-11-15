"""DaReCzech - loader written against the published schema, NOT fetched.

The specification is explicit: build the loader, mark it [FILL], do not download. Seznam
distributes DaReCzech under a request procedure (the paper is Kocian et al.,
ECIR 2022, "Siamese BERT-based Model for Web Search Relevance Ranking"), and it
has been requested by email.

[FILL: needs the dataset from Seznam, requested by email]

Published schema (TSV, per the dataset card):
    query, url, doc, title, bte, label
where `label` is a relevance score in [0, 1] derived from editorial annotation.
The splits shipped are train-big, train-small, dev and test.
"""

from __future__ import annotations

from pathlib import Path

from .base import Dataset, DatasetBlocked, Doc
from ..seeds import rng

FILL_MESSAGE = (
    "[FILL: needs the dataset from Seznam, requested by email] DaReCzech is not "
    "fetched by this kit. Obtain it from Seznam, then set "
    "dataset.options.data_dir to the directory holding the TSV splits."
)

# DaReCzech labels are continuous in [0, 1]; nDCG needs integer gains. Three
# levels keep the mapping legible and match the usual editorial scale.
GRADE_THRESHOLDS = ((0.75, 3), (0.5, 2), (0.25, 1))


def grade(label: float) -> int:
    for threshold, g in GRADE_THRESHOLDS:
        if label >= threshold:
            return g
    return 0


def load(
    cache_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    split: str = "test",
    max_docs: int | None = 100_000,
    max_queries: int | None = 2_000,
    **_,
) -> Dataset:
    if not data_dir:
        raise DatasetBlocked(FILL_MESSAGE)
    root = Path(data_dir)
    candidates = sorted(root.glob(f"*{split}*.tsv")) + sorted(root.glob(f"*{split}*.tsv.gz"))
    if not candidates:
        raise DatasetBlocked(f"{FILL_MESSAGE} (no *{split}*.tsv under {root})")

    import pandas as pd

    df = pd.concat(
        [pd.read_csv(p, sep="\t", quoting=3, dtype=str) for p in candidates], ignore_index=True
    )
    required = {"query", "url", "title", "bte", "label"}
    missing = required - set(df.columns)
    if missing:
        raise DatasetBlocked(
            f"DaReCzech schema mismatch: missing columns {sorted(missing)}. "
            "[VERIFY] the shipped column names against this loader."
        )

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    docs: dict[str, Doc] = {}
    query_ids: dict[str, str] = {}

    for row in df.itertuples(index=False):
        q_text = str(row.query)
        qid = query_ids.setdefault(q_text, f"q{len(query_ids):06d}")
        queries[qid] = q_text
        doc_id = str(row.url)
        docs.setdefault(
            doc_id,
            Doc(doc_id=doc_id, title=str(row.title or ""), text=str(row.bte or "")),
        )
        g = grade(float(row.label))
        if g > 0:
            qrels.setdefault(qid, {})[doc_id] = g

    if max_queries is not None and len(queries) > max_queries:
        keep = sorted(q for q in queries if q in qrels)
        idx = rng("dareczech", split, "queries").choice(
            len(keep), size=min(max_queries, len(keep)), replace=False
        )
        chosen = {keep[i] for i in sorted(idx.tolist())}
        queries = {q: queries[q] for q in chosen}
        qrels = {q: v for q, v in qrels.items() if q in chosen}

    gold = {d for v in qrels.values() for d in v}
    if max_docs is not None and len(docs) > max_docs:
        others = sorted(d for d in docs if d not in gold)
        room = max(0, max_docs - len(gold))
        idx = rng("dareczech", "docs").choice(
            len(others), size=min(room, len(others)), replace=False
        )
        keep = gold | {others[i] for i in idx.tolist()}
        docs = {k: v for k, v in docs.items() if k in keep}

    return Dataset(
        name=f"dareczech-{split}",
        language="cs",
        docs=sorted(docs.values(), key=lambda d: d.doc_id),
        queries={q: t for q, t in queries.items() if q in qrels},
        qrels=qrels,
        license="see the DaReCzech terms from Seznam (research use after request)",
        source=str(root),
        notes=[
            "Loader written against the published schema and never executed "
            "against the real files; [VERIFY] on first contact with the data.",
            f"Continuous labels mapped to integer grades at {GRADE_THRESHOLDS}.",
        ],
    )
