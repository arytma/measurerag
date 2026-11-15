"""Retrieval and end-to-end metrics.

Definitions are the ones in docs/METRICS.md; if you change one, change both.
No LLM-as-judge anywhere - it is not deterministic and this kit promises
byte-identical reruns.
"""

from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter
from typing import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]  # query_id -> doc_id -> graded relevance


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    """Graded nDCG@k with the standard linear-gain formulation.

    Gains are the raw relevance grades (BEIR qrels are 0/1 or 0/1/2). The ideal
    ranking is every judged document sorted by grade, truncated to k.
    """
    if not rel:
        return 0.0
    gains = [float(rel.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    """Fraction of the *relevant* documents (grade > 0) present in the top k.

    Unjudged documents count as non-relevant, the usual BEIR assumption. It
    flatters nothing here because every retriever is scored the same way.
    """
    positives = {d for d, g in rel.items() if g > 0}
    if not positives:
        return 0.0
    return len(positives.intersection(ranked[:k])) / len(positives)


def gold_in_context(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    """1.0 if at least one relevant document made it into the k passages the
    generator will actually see. This is the bridge metric: it separates
    'the retriever failed' from 'the generator failed'."""
    positives = {d for d, g in rel.items() if g > 0}
    return 1.0 if positives.intersection(ranked[:k]) else 0.0


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Qrels,
    ks: Sequence[int] = (10, 100),
    context_k: int | None = None,
) -> dict[str, float]:
    """Macro-average over queries that have at least one judgement.

    Queries with no qrels entry are skipped, not scored as zero - scoring them
    as zero would punish a retriever for a gap in the annotation.
    """
    scored = [q for q in run if qrels.get(q)]
    out: dict[str, float] = {}
    if not scored:
        for k in ks:
            out[f"ndcg@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
        out["n_queries"] = 0
        return out

    for k in ks:
        out[f"ndcg@{k}"] = sum(ndcg_at_k(run[q], qrels[q], k) for q in scored) / len(scored)
        out[f"recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in scored) / len(scored)
    if context_k is not None:
        out[f"gold_in_context@{context_k}"] = sum(
            gold_in_context(run[q], qrels[q], context_k) for q in scored
        ) / len(scored)
    out["n_queries"] = float(len(scored))
    return out


# --------------------------------------------------------------------------
# end-to-end answer quality
# --------------------------------------------------------------------------
_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA normalisation, widened for Czech.

    Lowercase, strip punctuation, collapse whitespace, drop English articles.
    Diacritics are *kept*: in Czech, stripping them merges distinct words, and
    a model that answers without them has answered differently.
    """
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return 1.0 if any(p == normalize_answer(g) for g in golds) else 0.0


def _f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, golds: Sequence[str]) -> float:
    """Best F1 against any acceptable gold answer."""
    return max((_f1(pred, g) for g in golds), default=0.0)


def evaluate_answers(
    preds: Mapping[str, str], golds: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    keys = [q for q in preds if q in golds]
    if not keys:
        return {"em": 0.0, "f1": 0.0, "n_answered": 0.0}
    return {
        "em": sum(exact_match(preds[q], golds[q]) for q in keys) / len(keys),
        "f1": sum(token_f1(preds[q], golds[q]) for q in keys) / len(keys),
        "n_answered": float(len(keys)),
    }


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile.

    Deliberately not numpy's linear interpolation: nearest-rank returns an
    observation that actually happened, which is what a latency ceiling is
    about, and it is exactly reproducible.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": (sum(values) / len(values)) if values else 0.0,
        "n": float(len(values)),
    }
