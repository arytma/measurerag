"""CsFEVER v2 - the Czech retrieval FALLBACK.

The specification's primary Czech set is CWRCzech, which is gated behind an email
request (see cwrczech.py). CsFEVER is the declared fallback: Czech claims paired
with the Czech Wikipedia pages that support or refute them, which is a genuine
claim-to-evidence retrieval task over a real Czech corpus.

Two things to be honest about when citing numbers from it:
  * The unit of retrieval is a Wikipedia *page*, not a passage, so absolute
    scores are not comparable with a passage-level benchmark.
  * It is an entity-heavy task. A claim usually names its evidence page, which
    flatters lexical matching. That is a property of the data, not a bug, but it
    means BM25 will look stronger here than on a paraphrase-heavy corpus.

Licence: CC BY-SA 3.0 (inherited from Czech Wikipedia).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .base import Dataset, DatasetUnavailable, Doc
from ..seeds import rng

REPO = "ctu-aic/csfever_v2"
CORPUS_FILE = "wiki_pages/wiki_pages.jsonl"
# Pinned to the commits every result used: each was the repository's main on 14 Nov 2026, the build's cache was
# made from it, and no commit came after 2026-04-09, before the first build. A later main cannot change a slice.
REVISION = "20507f3f46f53d937cc61be466f5f22f85448288"


def _download(filename: str, cache_dir) -> Path:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return Path(
        hf_hub_download(
            REPO,
            filename,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=str(Path(cache_dir) / "hf") if cache_dir else None,
        )
    )


def load(
    cache_dir: str | Path | None = None,
    split: str = "test",
    variant: str = "original",
    max_docs: int | None = 100_000,
    max_queries: int | None = 2_000,
    max_doc_chars: int = 2_000,
    **_,
) -> Dataset:
    """Stream the corpus so the full 1.1 GB of Wikipedia never lands in RAM.

    Capping happens *here* rather than afterwards because materialising 500k
    pages first would cost several GB on a machine whose whole premise is that
    memory is scarce. Gold pages are kept unconditionally; the rest of the slice
    is a seeded reservoir sample, so the same cap always yields the same corpus.
    """
    try:
        claims_path = _download(f"{variant}/{split}.jsonl", cache_dir)
        corpus_path = _download(CORPUS_FILE, cache_dir)
    except Exception as exc:
        raise DatasetUnavailable(f"{REPO}: {exc}") from exc

    # ---- claims: FEVER repeats one claim per evidence set, so union them ----
    claims: dict[str, str] = {}
    gold_titles: dict[str, set[str]] = {}
    with open(claims_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("label") == "NOT ENOUGH INFO":
                continue  # no evidence page exists; nothing to retrieve
            titles = {
                ev[0]
                for ev in (row.get("evidence") or [])
                if isinstance(ev, (list, tuple)) and ev and ev[0]
            }
            if not titles:
                continue
            qid = str(row["id"])
            claims[qid] = row["claim"]
            gold_titles.setdefault(qid, set()).update(titles)

    # cap the queries before the corpus, so the gold set we must keep is small
    if max_queries is not None and len(claims) > max_queries:
        keep = sorted(claims)
        idx = rng("csfever", variant, split, "queries").choice(
            len(keep), size=max_queries, replace=False
        )
        chosen = {keep[i] for i in sorted(idx.tolist())}
        claims = {q: claims[q] for q in chosen}
        gold_titles = {q: gold_titles[q] for q in chosen}

    wanted: set[str] = set().union(*gold_titles.values()) if gold_titles else set()

    # ---- corpus: one streaming pass, gold kept, remainder reservoir-sampled --
    r = rng("csfever", "corpus", max_docs or 0)
    gold_docs: dict[str, Doc] = {}
    reservoir: list[Doc] = []
    capacity = None if max_docs is None else max(0, max_docs - len(wanted))
    seen_other = 0
    title_to_id: dict[str, str] = {}

    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            title = row.get("title") or ""
            doc_id = str(row.get("id"))
            text = (row.get("text") or "")[:max_doc_chars]
            doc = Doc(doc_id=doc_id, text=text, title=title)

            if title in wanted and title not in title_to_id:
                title_to_id[title] = doc_id
                gold_docs[doc_id] = doc
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

    docs = sorted(
        list(gold_docs.values()) + reservoir, key=lambda d: int(d.doc_id) if d.doc_id.isdigit() else 0
    )

    qrels: dict[str, dict[str, int]] = {}
    unresolved = 0
    for qid, titles in gold_titles.items():
        hits = {title_to_id[t]: 1 for t in titles if t in title_to_id}
        if hits:
            qrels[qid] = hits
        else:
            unresolved += 1

    # a claim whose every evidence page is missing from the dump has no gold;
    # keeping it would score every retriever 0 for a defect in the data
    queries = {q: c for q, c in claims.items() if q in qrels}

    return Dataset(
        name="csfever-v2",
        language="cs",
        docs=docs,
        queries=queries,
        qrels=qrels,
        license="CC BY-SA 3.0 (Czech Wikipedia, via ctu-aic/csfever_v2)",
        source=f"https://huggingface.co/datasets/{REPO} ({variant}/{split})",
        notes=[
            "FALLBACK for the Czech retrieval stage: CWRCzech is gated behind an "
            "email request to Seznam and was not worked around.",
            "Retrieval unit is a Wikipedia page truncated to "
            f"{max_doc_chars} characters, not a passage.",
            "Entity-heavy task: claims usually name their evidence page, which "
            "flatters lexical matching. Read BM25's score here with that in mind.",
            f"{unresolved} claims dropped: their evidence pages are absent from the dump.",
        ],
    )
