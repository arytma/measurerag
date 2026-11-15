"""CWRCzech - the specification's PRIMARY Czech set. BLOCKED, by design not by accident.

Seznam distributes CWRCzech only after you read the disclaimer at
https://github.com/seznam/CWRCzech/blob/main/disclaimer.md and request access by
email at the address given there; they then send a private link. There is no public
artefact to download. This kit does not work around such a gate, so the loader
raises DatasetBlocked with the instructions and the run falls back to CsFEVER.

The loader below is written against the published schema (README of
seznam/CWRCzech, SIGIR 2024 paper), so the day the parquet files arrive it works
without edits: point it at the directory.

    measurerag run -c configs/cz_small.yaml   # after setting, in the config:
    #   dataset: {name: cwrczech, options: {data_dir: /path/to/cwrczech}}

Published columns: requestId, query, url, title, bte, rank, clicks, dwellTime.
"""

from __future__ import annotations

from pathlib import Path

from .base import Dataset, DatasetBlocked, Doc
from ..seeds import rng

REQUEST_URL = "https://github.com/seznam/CWRCzech"

BLOCKED_MESSAGE = (
    "CWRCzech is not publicly downloadable. Read the disclaimer at "
    f"{REQUEST_URL}/blob/main/disclaimer.md and request access by email at the "
    "address given there; Seznam replies "
    "with a private link to the parquet files. Then set "
    "dataset.options.data_dir in the config to the directory holding them. "
    "This kit will not attempt to bypass the request gate."
)

# The manually annotated test set is the evaluation half; the click corpus is
# the collection the specification asks to slice to 100k documents.
ANNOTATED_GLOB = "*test*.parquet"
CLICKS_GLOB = "*train*.parquet"


def _relevance_from_clicks(clicks: int, dwell: float | None) -> int:
    """Click signal -> graded relevance, the usual three-level reading.

    A click with a long dwell is strong evidence, a click alone is weak, no
    click is nothing. [VERIFY] against the grades Seznam ships with the
    annotated set once it is in hand - if the annotated file carries an explicit
    relevance column, that column wins and this function is only used for the
    click corpus.
    """
    if clicks <= 0:
        return 0
    if dwell is not None and dwell >= 30:
        return 2
    return 1


def load(
    cache_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    max_docs: int | None = 100_000,
    max_queries: int | None = 2_000,
    **_,
) -> Dataset:
    if not data_dir:
        raise DatasetBlocked(BLOCKED_MESSAGE)
    root = Path(data_dir)
    if not root.exists():
        raise DatasetBlocked(f"{BLOCKED_MESSAGE} (data_dir {root} does not exist)")

    import pandas as pd

    annotated = sorted(root.glob(ANNOTATED_GLOB))
    clicks = sorted(root.glob(CLICKS_GLOB))
    if not annotated:
        raise DatasetBlocked(
            f"no file matching {ANNOTATED_GLOB} under {root}. "
            "[FILL] confirm the filenames Seznam ships and adjust ANNOTATED_GLOB."
        )

    eval_df = pd.concat([pd.read_parquet(p) for p in annotated], ignore_index=True)

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    docs: dict[str, Doc] = {}

    def add_row(row, judged: bool) -> None:
        doc_id = str(row["url"])
        if doc_id not in docs:
            docs[doc_id] = Doc(
                doc_id=doc_id,
                title=str(row.get("title") or ""),
                text=str(row.get("bte") or ""),
            )
        if not judged:
            return
        qid = str(row["requestId"])
        queries[qid] = str(row["query"])
        if "relevance" in row.index:  # explicit annotation wins over clicks
            grade = int(row["relevance"])
        else:
            dwell = row.get("dwellTime")
            grade = _relevance_from_clicks(
                int(row.get("clicks") or 0),
                None if dwell is None or pd.isna(dwell) else float(dwell),
            )
        if grade > 0:
            qrels.setdefault(qid, {})[doc_id] = grade

    for _, row in eval_df.iterrows():
        add_row(row, judged=True)

    if max_queries is not None and len(queries) > max_queries:
        keep = sorted(queries)
        idx = rng("cwrczech", "queries").choice(len(keep), size=max_queries, replace=False)
        chosen = {keep[i] for i in sorted(idx.tolist())}
        queries = {q: queries[q] for q in chosen}
        qrels = {q: v for q, v in qrels.items() if q in chosen}

    gold = {d for v in qrels.values() for d in v}
    for path in clicks:
        if max_docs is not None and len(docs) >= max_docs:
            break
        for _, row in pd.read_parquet(path).iterrows():
            add_row(row, judged=False)
            if max_docs is not None and len(docs) >= max_docs + len(gold):
                break

    # keep every gold document, then trim the rest to the cap
    if max_docs is not None and len(docs) > max_docs:
        others = sorted(d for d in docs if d not in gold)
        room = max(0, max_docs - len(gold))
        idx = rng("cwrczech", "docs").choice(len(others), size=min(room, len(others)), replace=False)
        keep = gold | {others[i] for i in idx.tolist()}
        docs = {k: v for k, v in docs.items() if k in keep}

    queries = {q: t for q, t in queries.items() if q in qrels}
    return Dataset(
        name="cwrczech",
        language="cs",
        docs=sorted(docs.values(), key=lambda d: d.doc_id),
        queries=queries,
        qrels=qrels,
        license="see the CWRCzech disclaimer; research use after request",
        source=f"{REQUEST_URL} (private link, obtained by email)",
        notes=[
            "Documents are title + body-text-extract (bte, <=230 chars), which "
            "is what Seznam ships - not full page text.",
            "[VERIFY] the relevance grading against the annotated file's own "
            "columns; _relevance_from_clicks is a click-signal approximation.",
        ],
    )
