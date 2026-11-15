"""Sensitivity checks for Czech exact match. These are not headline metrics.

`metrics.exact_match` stays the metric. It keeps diacritics and does not
lemmatise, so an inflected form of the right answer scores 0. The checks here ask
how much that costs, and they are reported beside EM, labelled as checks.

    folded_f1        token F1 after lowercasing and stripping diacritics
    prefix_match     same token count, and each token agrees in its first 5 characters
    lemma_match      same token count, and each token's lemmas intersect the gold's
    yes_no_variant   the English word for a gold `ano` or `ne`
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .metrics import normalize_answer


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def folded_f1(pred: str, golds: Sequence[str]) -> float:
    def one(p: str, g: str) -> float:
        a, b = strip_diacritics(normalize_answer(p)).split(), strip_diacritics(normalize_answer(g)).split()
        if not a or not b:
            return float(a == b)
        same = sum((Counter(a) & Counter(b)).values())
        if same == 0:
            return 0.0
        pr, rc = same / len(a), same / len(b)
        return 2 * pr * rc / (pr + rc)

    return max((one(pred, g) for g in golds), default=0.0)


def prefix_match(pred: str, golds: Sequence[str], n: int = 5) -> float:
    p = normalize_answer(pred).split()
    for g in golds:
        t = normalize_answer(g).split()
        if p and len(p) == len(t) and all(a[:n] == b[:n] for a, b in zip(p, t)):
            return 1.0
    return 0.0


def load_forms(path: str | Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["form"]] = set(d["lemmas"])
    return out


PLACEHOLDER_LEMMAS = frozenset({"[number]", "#num#", "[url]"})


def lemma_match(pred: str, golds: Sequence[str], forms: Mapping[str, set[str]]) -> float:
    """1 if every token could be the same lemma as its gold token.

    Forms come from the vertical files of SQAD 3.2, so this can only recognise
    words the archive contains. A word it does not know stands for itself. Those
    files give every number and every URL the same placeholder lemma, which says
    nothing about whether two tokens agree, so a placeholder never counts: 1998 and
    2001 are different answers.
    """
    def lem(t: str) -> set[str]:
        return {x for x in forms.get(t, set()) if x not in PLACEHOLDER_LEMMAS and any(c.isalpha() for c in x)} | {t}

    p = normalize_answer(pred).split()
    for g in golds:
        t = normalize_answer(g).split()
        if p and len(p) == len(t) and all(lem(a) & lem(b) for a, b in zip(p, t)):
            return 1.0
    return 0.0


DECLINED = ("unanswerable", "nelze odpovědět")
YES_NO_CLASSES = ("ano or ne", "yes or no in English", "unanswerable", "something else")


def yes_no_class(pred: str) -> str:
    """What a whole answer to a yes/no question is, after exact match's normalisation."""
    w = normalize_answer(pred)
    if w in ("ano", "ne"):
        return "ano or ne"
    if w in ("yes", "no"):
        return "yes or no in English"
    return "unanswerable" if w in DECLINED else "something else"


def compliance(counts: Mapping[str, int]) -> float | None:
    """Czech yes/no answers among those that did not decline."""
    answered = sum(counts.values()) - counts.get("unanswerable", 0)
    return counts.get("ano or ne", 0) / answered if answered else None


def choose_variant(table: Mapping[str, Mapping[str, float | None]], threshold: float = 0.95) -> dict:
    """The prompt rule, fixed before the dev set was read. `table` is variant -> generator -> compliance."""
    def low(v: str) -> float:
        return min((c or 0.0) for c in table[v].values())

    def mean(v: str) -> float:
        return sum((c or 0.0) for c in table[v].values()) / len(table[v])

    for v in ("v2", "v3"):
        if low(v) >= threshold:
            return {"chosen": v, "passed": True, "shortfall": 0.0}
    order = {"v2": 0, "v3": 1, "v1": 2}
    best = sorted(order, key=lambda v: (-low(v), -mean(v), order[v]))[0]
    return {"chosen": best, "passed": False, "shortfall": threshold - low(best)}


def choose_after_amendment(compliance_v4: Mapping[str, float | None], span_em_v4: Mapping[str, float], span_em_v1: Mapping[str, float],
                           threshold: float = 0.95, span_margin: float = 0.05) -> dict:
    """The amended prompt rule. Each mapping is generator -> value. V4 if it passes, else V1, and no third outcome."""
    lowest = min((c or 0.0) for c in compliance_v4.values())
    below = {g: round(span_em_v1[g] - span_em_v4[g], 9) for g in span_em_v1}   # rounded: exactly 0.05 below is allowed
    compliance_passed, span_passed = lowest >= threshold, all(d <= span_margin for d in below.values())
    passed = compliance_passed and span_passed
    return {"chosen": "v4" if passed else "v1", "passed": passed, "compliance_passed": compliance_passed, "span_passed": span_passed,
            "lowest_compliance": lowest, "span_em_below_v1": below, "primary_metric": "em" if passed else "em_yes_no_mapped"}


def yes_no_in_czech(pred: str) -> str:
    """A whole answer of yes or no, in Czech. Nothing inside a longer answer changes."""
    return {"yes": "ano", "no": "ne"}.get(normalize_answer(pred), pred)


def yes_no_variant(pred: str, golds: Sequence[str]) -> float:
    gold = {normalize_answer(g) for g in golds}
    p = normalize_answer(pred)
    return float(("ano" in gold and p == "yes") or ("ne" in gold and p == "no"))
