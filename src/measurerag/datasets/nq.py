"""Natural Questions for the end-to-end stage: BEIR nq corpus + nq-open answers.

The crossover replication set. BEIR's nq ships a Wikipedia passage corpus and
qrels but no short answers; nq-open ships short answers but no passages. Both
come from the NQ development set, so they join on the question text.

Gold passages are mapped, not given, and the rule is deliberately strict
(`GOLD_RULE`): a passage is gold for a question only if BEIR judges it relevant
AND its body text contains one of the nq-open answers as a whole-token span
after SQuAD normalisation. Containment alone would count every passage that
mentions "1990" as gold for a question whose answer is 1990; BEIR relevance
alone would count passages that are about the right page but do not carry the
answer. A question with no passage satisfying both is dropped, and the loader
records how many were (`coverage`), because a mapping rule that keeps a third
of the questions is itself a result.

Licence: BEIR nq CC BY-SA 4.0 as tagged; NQ-open CC BY-SA 3.0.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .base import Dataset, DatasetUnavailable, Doc
from ..metrics import normalize_answer
from ..seeds import rng

BEIR_REPO = "BeIR/nq"
QRELS_REPO = "BeIR/nq-qrels"
ANSWER_REPO = "google-research-datasets/nq_open"
# Pinned to the commits every result used: each was the repository's main on 14 Nov 2026, the build's cache was
# made from it, and no commit came after 2026-04-09, before the first build. A later main cannot change a slice.
REVISIONS = {BEIR_REPO: "b7253e6c379163d024ddb1d6948152a91a2e3b46", QRELS_REPO: "519acd4e48bb3e5da22b2b888ce36c614f4f2bc9",
             ANSWER_REPO: "5dd9790a83002ad084ddeb7c420dc716852c6f28"}
ANSWER_FILE = "nq_open/validation-00000-of-00001.parquet"
CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
QUERIES_FILE = "queries/queries-00000-of-00001.parquet"

GOLD_RULE = (
    "gold(q) = { d : BEIR nq qrels mark d relevant to q, and normalize(d.text) "
    "contains normalize(a) as a whole-token span for some nq-open answer a }; "
    "normalize = SQuAD answer normalisation (lowercase, strip punctuation and "
    "English articles, collapse whitespace); passage title not searched; "
    "questions joined BEIR<->nq-open on normalize_question(text) and kept only "
    "if gold(q) is non-empty"
)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().rstrip("?").strip())


def contains_answer(passage: str, answers: list[str]) -> bool:
    hay = f" {normalize_answer(passage)} "
    for a in answers:
        na = normalize_answer(a)
        if na and f" {na} " in hay:
            return True
    return False


def load(
    cache_dir: str | Path | None = None,
    max_docs: int | None = 100_000,
    max_queries: int | None = 300,
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
        qrels_df = pd.read_csv(grab(QRELS_REPO, "test.tsv"), sep="\t", dtype=str)
        queries_df = pd.read_parquet(grab(BEIR_REPO, QUERIES_FILE))
        answers_df = pd.read_parquet(grab(ANSWER_REPO, ANSWER_FILE))
        corpus_path = grab(BEIR_REPO, CORPUS_FILE)
    except Exception as exc:
        raise DatasetUnavailable(f"NQ: {exc}") from exc

    qrels: dict[str, dict[str, int]] = {}
    for q, d, s in zip(qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"]):
        if int(s) > 0:
            qrels.setdefault(str(q), {})[str(d)] = int(s)

    by_text: dict[str, list[str]] = {}
    for question, answer in zip(answers_df["question"], answers_df["answer"]):
        by_text.setdefault(normalize_question(question), [str(a) for a in answer])

    query_text: dict[str, str] = {}
    answers: dict[str, list[str]] = {}
    for qid, text in zip(queries_df["_id"], queries_df["text"]):
        qid = str(qid)
        if qid in qrels and normalize_question(text) in by_text:
            query_text[qid] = str(text)
            answers[qid] = by_text[normalize_question(text)]

    # ---- pass 1: read every qrels passage, map gold by containment ----------
    judged = {d for q in answers for d in qrels[q]}
    judged_text: dict[str, Doc] = {}
    pf = pq.ParquetFile(corpus_path)
    for batch in pf.iter_batches(batch_size=50_000, columns=["_id", "title", "text"]):
        for doc_id, title, text in zip(
            batch.column("_id").to_pylist(), batch.column("title").to_pylist(), batch.column("text").to_pylist()
        ):
            if str(doc_id) in judged:
                judged_text[str(doc_id)] = Doc(doc_id=str(doc_id), text=text or "", title=title or "")

    mapped: dict[str, dict[str, int]] = {}
    for q in sorted(answers):
        gold = {
            d: 1 for d in qrels[q] if d in judged_text and contains_answer(judged_text[d].text, answers[q])
        }
        if gold:
            mapped[q] = gold
    coverage = {
        "beir_test_queries_with_qrels": len(qrels),
        "joined_to_nq_open": len(answers),
        "with_mapped_gold": len(mapped),
        "coverage_of_joined": round(len(mapped) / len(answers), 4) if answers else 0.0,
        "judged_passages_of_joined": sum(len(qrels[q]) for q in answers),
        "gold_passages_of_mapped": sum(len(v) for v in mapped.values()),
    }

    usable = sorted(mapped)
    if max_queries is not None and len(usable) > max_queries:
        idx = rng("nq", "test", "queries").choice(len(usable), size=max_queries, replace=False)
        usable = [usable[i] for i in sorted(idx.tolist())]
    keep_q = set(usable)
    # every judged passage of a kept question stays in the corpus - the gold ones
    # and the relevant-but-answerless ones - so the corpus a retriever searches
    # does not depend on the mapping rule
    forced = {d for q in keep_q for d in qrels[q] if d in judged_text}

    # ---- pass 2: seeded reservoir over everything else ---------------------
    r = rng("nq", "corpus", max_docs or 0)
    capacity = None if max_docs is None else max(0, max_docs - len(forced))
    reservoir: list[Doc] = []
    seen_other = 0
    for batch in pf.iter_batches(batch_size=50_000, columns=["_id", "title", "text"]):
        for doc_id, title, text in zip(
            batch.column("_id").to_pylist(), batch.column("title").to_pylist(), batch.column("text").to_pylist()
        ):
            doc_id = str(doc_id)
            if doc_id in forced:
                continue
            doc = Doc(doc_id=doc_id, text=text or "", title=title or "")
            if capacity is None:
                reservoir.append(doc)
            elif len(reservoir) < capacity:
                reservoir.append(doc)
            else:
                j = int(r.integers(0, seen_other + 1))
                if j < capacity:
                    reservoir[j] = doc
            seen_other += 1

    docs = sorted([judged_text[d] for d in forced] + reservoir, key=lambda d: d.doc_id)

    return Dataset(
        name="nq-rag",
        language="en",
        docs=docs,
        queries={q: query_text[q] for q in usable},
        qrels={q: mapped[q] for q in usable},
        answers={q: answers[q] for q in usable},
        license="CC BY-SA 4.0 (BEIR nq); CC BY-SA 3.0 (NQ-open answers)",
        source=f"{BEIR_REPO} corpus + {QRELS_REPO} test qrels + {ANSWER_REPO} validation answers",
        notes=[
            "Gold passages are MAPPED, not annotated: " + GOLD_RULE,
            "Mapping coverage: " + ", ".join(f"{k}={v}" for k, v in coverage.items()),
            "Corpus capped by a seeded reservoir sample with every judged passage "
            "of a kept question forced in.",
        ],
    )
