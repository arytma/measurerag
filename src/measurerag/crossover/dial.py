"""The controlled dial: move gold-in-context while corpus, k and prompt stay fixed.

Take one retriever's ranking. In its top k, replace each gold passage, with
probability p, by the next-ranked non-gold passage from below rank k. Nothing
else changes: the same question, the same k passages' worth of text, the same
template, the same generator. So an EM difference between two dial levels is
caused by the gold that was taken away and by nothing else a retriever change
would drag along with it (passage length, topical drift, a different k).

Common random numbers. Each gold passage of a question gets one uniform draw u,
seeded by the question id alone, and it is removed at every level with p > u.
The levels are therefore nested - whatever is gone at p = 0.4 is also gone at
p = 0.6 - which is what makes the difference between two adjacent levels a
paired, low-variance comparison rather than two independent coin tosses, and
it is why p = 0 is exactly the retriever's own top k and p = 1 contains no gold.

Replacements are assigned in the order the passages drop out (ascending u, then
rank), so a passage that has been swapped keeps the same substitute at every
higher p. A level's context therefore never reshuffles passages that did not
change.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..seeds import rng

DIAL_LEVELS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def gold_set(rel: Mapping[str, int]) -> set[str]:
    return {d for d, g in rel.items() if g > 0}


def gold_counts(context: Sequence[str], gold: set[str]) -> tuple[int, int]:
    """(any gold in context, all gold in context) as 0/1.

    A question with no gold at all counts as neither: it cannot be "all gold"
    vacuously, because that would inflate the all-gold rate with questions the
    annotation simply does not cover.
    """
    if not gold:
        return 0, 0
    present = gold.intersection(context)
    return int(bool(present)), int(present == gold)


def dial_uniforms(dataset: str, qid: str, gold: set[str], seed_tag: str = "dial") -> dict[str, float]:
    """One u in [0, 1) per gold passage, from the question id alone.

    Drawn over every gold passage in sorted id order, not only the ones a
    retriever happened to find, so the draw does not depend on the retriever.
    """
    ordered = sorted(gold)
    draws = rng(seed_tag, dataset, qid).random(len(ordered))
    return {d: float(u) for d, u in zip(ordered, draws)}


def dial_context(
    ranked: Sequence[str],
    gold: set[str],
    k: int,
    p: float,
    uniforms: Mapping[str, float],
) -> list[str]:
    """The k passage ids handed to the generator at dial level p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    top = list(ranked[:k])
    golds_in = [d for d in top if d in gold]
    if not golds_in or p == 0.0:
        return top
    rank = {d: i for i, d in enumerate(top)}
    drop_order = sorted(golds_in, key=lambda d: (uniforms[d], rank[d]))
    spares = [d for d in ranked[k:] if d not in gold]
    if len(spares) < len(drop_order):
        raise ValueError(
            f"ranking too short to replace {len(drop_order)} gold passages "
            f"(only {len(spares)} non-gold passages below rank {k})"
        )
    substitute = {d: spares[i] for i, d in enumerate(drop_order)}
    # u < p, so p = 1.0 removes every gold passage (u is drawn from [0, 1))
    removed = {d for d in drop_order if uniforms[d] < p}
    return [substitute[d] if d in removed else d for d in top]
