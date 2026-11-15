"""SQAD 3.0 - Czech question answering, for the Czech end-to-end stage.

The canonical home is LINDAT. LINDAT is outside this kit's network allowlist
(Hugging Face, GitHub, PyPI), so the Hugging Face copy `Pehy/cs_sqad-3.0` is used
(a filtered copy), which ships part of SQAD 3.0 converted to the SQuAD JSON shape. It is
pinned to one revision, so a later change to the copy cannot change these numbers.

Licence, checked on LINDAT on 8 Nov 2026: SQAD 3.0 (http://hdl.handle.net/11234/1-3069)
is under the GNU Lesser General Public License 3.0 (LGPL-3.0). The mirror's card
says `mit`; a mirror cannot relicense someone else's corpus, so the upstream
licence governs. No SQAD text is included in this repository.

The retrieval task is built the standard way for a reading-comprehension set:
every distinct context paragraph across all splits becomes a document, and a
test question's gold document is the paragraph it was written from.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .base import Dataset, DatasetUnavailable, Doc
from ..seeds import rng

MIRROR = "Pehy/cs_sqad-3.0"
REVISION = "f9e71384a8f7285412360cecabb7a852f49adbb9"
UPSTREAM = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3069"
LICENCE = "LGPL-3.0 (SQAD 3.0 on LINDAT, http://hdl.handle.net/11234/1-3069; the upstream licence governs, the HF mirror's card says `mit`)"
LICENCE_NOTE = ("Licence checked on LINDAT on 8 Nov 2026: SQAD 3.0 is LGPL-3.0 "
                "(http://hdl.handle.net/11234/1-3069). The mirror's card says `mit`; the upstream licence governs.")
SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}


def _doc_id(text: str) -> str:
    """Contexts have no id in the SQuAD shape, so hash them - stable across runs
    and across machines, and it deduplicates repeated paragraphs for free."""
    return "ctx-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load(
    cache_dir: str | Path | None = None,
    split: str = "test",
    corpus_splits: tuple[str, ...] = ("train", "validation", "test"),
    max_docs: int | None = 100_000,
    max_queries: int | None = 2_000,
    **_,
) -> Dataset:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    hf_cache = str(Path(cache_dir) / "hf") if cache_dir else None

    try:
        frames = {
            name: pd.read_parquet(
                hf_hub_download(MIRROR, SPLIT_FILES[name], repo_type="dataset", revision=REVISION, cache_dir=hf_cache)
            )
            for name in corpus_splits
        }
    except Exception as exc:
        raise DatasetUnavailable(f"{MIRROR}: {exc}") from exc

    # every distinct paragraph in the whole set is a candidate document, so the
    # test questions are not searching a corpus of only their own answers
    docs: dict[str, Doc] = {}
    for df in frames.values():
        for ctx in df["context"].tolist():
            ctx = str(ctx)
            did = _doc_id(ctx)
            if did not in docs:
                docs[did] = Doc(doc_id=did, text=ctx, title="")

    eval_df = frames[split] if split in frames else None
    if eval_df is None:
        raise DatasetUnavailable(f"split {split!r} not among {sorted(SPLIT_FILES)}")

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    answers: dict[str, list[str]] = {}
    for row in eval_df.itertuples(index=False):
        qid = str(row.id)
        texts = [str(t) for t in (row.answers.get("text") if hasattr(row.answers, "get") else row.answers["text"])]
        if not texts:
            continue
        queries[qid] = str(row.question)
        qrels[qid] = {_doc_id(str(row.context)): 1}
        answers[qid] = texts

    if max_queries is not None and len(queries) > max_queries:
        keep = sorted(queries)
        idx = rng("sqad", split, "queries").choice(len(keep), size=max_queries, replace=False)
        chosen = {keep[i] for i in sorted(idx.tolist())}
        queries = {q: queries[q] for q in chosen}
        qrels = {q: qrels[q] for q in chosen}
        answers = {q: answers[q] for q in chosen}

    return Dataset(
        name=f"sqad-3.0-{split}",
        language="cs",
        docs=sorted(docs.values(), key=lambda d: d.doc_id),
        queries=queries,
        qrels=qrels,
        answers=answers,
        license=LICENCE,
        source=f"https://huggingface.co/datasets/{MIRROR} (mirror of {UPSTREAM})",
        notes=[
            "Czech QA. Corpus is every distinct context paragraph across all "
            "three splits; gold is the paragraph the question was written from.",
            LICENCE_NOTE,
            "Answers are short spans, so EM is a meaningful metric; a generator "
            "that writes a sentence around the span will lose EM and keep F1.",
        ],
    )
