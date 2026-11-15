"""Dataset contract shared by every loader.

A loader returns a `Dataset`. Everything downstream - retrievers, the budget
harness, the RAG stage - only ever sees this shape, which is what makes adding
a new corpus a ~60 line job (see README, "Adding a dataset").
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..hashing import sha256_records
from ..seeds import stable_sample


class DatasetBlocked(RuntimeError):
    """Raised when a dataset cannot be obtained without a login, a form or a
    key. We never work around such a gate; we log BLOCKED and fall back."""


class DatasetUnavailable(RuntimeError):
    """Raised when a download failed for an ordinary reason (offline, 404)."""


@dataclass
class Doc:
    doc_id: str
    text: str
    title: str = ""

    def indexable(self) -> str:
        return f"{self.title}\n{self.text}".strip() if self.title else self.text


@dataclass
class Dataset:
    """A retrieval task, optionally with gold answers for the RAG stage."""

    name: str
    language: str  # "en" | "cs"
    docs: list[Doc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    answers: dict[str, list[str]] = field(default_factory=dict)
    license: str = "unknown"
    source: str = ""
    notes: list[str] = field(default_factory=list)
    caps_applied: dict[str, Any] = field(default_factory=dict)

    # -- fingerprints -----------------------------------------------------
    def corpus_hash(self) -> str:
        return sha256_records(
            ({"id": d.doc_id, "title": d.title, "text": d.text} for d in self.docs)
        )

    def query_hash(self) -> str:
        return sha256_records(
            ({"id": q, "text": self.queries[q]} for q in sorted(self.queries))
        )

    def qrels_hash(self) -> str:
        return sha256_records(
            (
                {"q": q, "rels": dict(sorted(self.qrels[q].items()))}
                for q in sorted(self.qrels)
            )
        )

    def fingerprint(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language,
            "n_docs": len(self.docs),
            "n_queries": len(self.queries),
            "n_qrels": sum(len(v) for v in self.qrels.values()),
            "n_answers": len(self.answers),
            "license": self.license,
            "source": self.source,
            "caps_applied": self.caps_applied,
            "corpus_sha256": self.corpus_hash(),
            "queries_sha256": self.query_hash(),
            "qrels_sha256": self.qrels_hash(),
            "notes": list(self.notes),
        }


    # -- materialised slices ---------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Write the exact slice used, so a child process (or a second run, or
        a reviewer) sees byte-for-byte the same data without re-downloading."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            d / "corpus.jsonl",
            ({"doc_id": x.doc_id, "title": x.title, "text": x.text} for x in self.docs),
        )
        write_jsonl(
            d / "queries.jsonl",
            (
                {"query_id": q, "text": self.queries[q], "answers": self.answers.get(q, [])}
                for q in sorted(self.queries)
            ),
        )
        write_jsonl(
            d / "qrels.jsonl",
            (
                {"query_id": q, "doc_id": doc, "relevance": rel}
                for q in sorted(self.qrels)
                for doc, rel in sorted(self.qrels[q].items())
            ),
        )
        meta = {
            "name": self.name,
            "language": self.language,
            "license": self.license,
            "source": self.source,
            "notes": self.notes,
            "caps_applied": self.caps_applied,
            "fingerprint": self.fingerprint(),
        }
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return d

    @staticmethod
    def load_slice(directory: str | Path) -> "Dataset":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        docs = [
            Doc(doc_id=r["doc_id"], text=r["text"], title=r.get("title", ""))
            for r in read_jsonl(d / "corpus.jsonl")
        ]
        queries: dict[str, str] = {}
        answers: dict[str, list[str]] = {}
        for r in read_jsonl(d / "queries.jsonl"):
            queries[r["query_id"]] = r["text"]
            if r.get("answers"):
                answers[r["query_id"]] = list(r["answers"])
        qrels: dict[str, dict[str, int]] = {}
        for r in read_jsonl(d / "qrels.jsonl"):
            qrels.setdefault(r["query_id"], {})[r["doc_id"]] = int(r["relevance"])
        ds = Dataset(
            name=meta["name"],
            language=meta["language"],
            docs=docs,
            queries=queries,
            qrels=qrels,
            answers=answers,
            license=meta.get("license", "unknown"),
            source=meta.get("source", ""),
            notes=list(meta.get("notes", [])),
        )
        ds.caps_applied = meta.get("caps_applied", {})
        return ds

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]

    @property
    def texts(self) -> list[str]:
        return [d.indexable() for d in self.docs]


# --------------------------------------------------------------------------
# capping - this run's scale limits, applied deterministically and recorded
# --------------------------------------------------------------------------
def apply_caps(
    ds: Dataset, max_docs: int | None, max_queries: int | None, keep_gold: bool = True
) -> Dataset:
    """Cap a dataset to a seeded slice.

    `keep_gold` guarantees every judged-relevant document for a surviving query
    stays in the corpus. Without it a cap silently destroys the ceiling: recall
    would be bounded by the sampling, not by the retriever, and the whole table
    would be measuring the wrong thing.
    """
    caps: dict[str, Any] = {}

    if max_queries is not None and len(ds.queries) > max_queries:
        qids = stable_sample(sorted(ds.queries), max_queries, ds.name, "queries")
        caps["max_queries"] = max_queries
        ds.queries = {q: ds.queries[q] for q in qids}
        ds.qrels = {q: v for q, v in ds.qrels.items() if q in ds.queries}
        ds.answers = {q: v for q, v in ds.answers.items() if q in ds.queries}

    if max_docs is not None and len(ds.docs) > max_docs:
        caps["max_docs"] = max_docs
        gold_ids: set[str] = set()
        if keep_gold:
            for q in ds.queries:
                gold_ids.update(d for d, g in ds.qrels.get(q, {}).items() if g > 0)

        by_id = {d.doc_id: d for d in ds.docs}
        gold_ids = {d for d in gold_ids if d in by_id}
        budget = max_docs - len(gold_ids)
        caps["gold_docs_forced"] = len(gold_ids)
        if budget <= 0:
            # more gold than the cap: keep the gold, the cap is advisory here
            kept = sorted(gold_ids)
            caps["cap_overridden_by_gold"] = True
        else:
            pool = [d.doc_id for d in ds.docs if d.doc_id not in gold_ids]
            kept = sorted(gold_ids) + stable_sample(pool, budget, ds.name, "docs")
        keep = set(kept)
        ds.docs = [d for d in ds.docs if d.doc_id in keep]
        # qrels pointing at dropped documents would inflate the denominator of
        # recall for a document that is not even in the corpus.
        ds.qrels = {
            q: {d: g for d, g in rels.items() if d in keep} for q, rels in ds.qrels.items()
        }

    ds.caps_applied = caps
    return ds


# --------------------------------------------------------------------------
# small io helpers
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def cache_root(override: str | None = None) -> Path:
    """Everything downloaded lives under the repo by default.

    Keeping it in-repo (and git-ignored) means a user can see exactly how
    much disk the kit is using and delete it in one command.
    """
    root = Path(override or os.environ.get("MEASURERAG_CACHE", "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root
