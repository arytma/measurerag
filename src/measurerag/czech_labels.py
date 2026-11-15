"""Simple labels for a gold answer, for the Czech diagnostics. Rules only, no model.

    answer_type    yes/no, number, date, single word, multi-word name, other
    length_band    the gold answer's length in tokens, as a band
    spaced_punct   does the gold string keep tokeniser spacing around punctuation?

These describe the question sets. They are not metrics, and exact match does not use them.
"""

from __future__ import annotations

import re
from typing import Sequence

from .metrics import normalize_answer

TYPES = ("yes/no", "number", "date", "single word", "multi-word name", "other")
BANDS = ("1", "2", "3", "4-5", "6+")

# Czech month names, stems: they cover the declined forms (ledna, leden, lednu, ...)
_MONTHS = ("ledn", "leden", "únor", "březn", "březen", "dubn", "duben", "květn", "květen", "červn", "červen",
           "červenc", "srpn", "srpen", "září", "říjn", "říjen", "listopad", "prosinc", "prosinec")
_NUMBER = re.compile(r"^[+\-]?\d[\d\s.,:/\-]*%?$")
_DAY_MONTH_YEAR = re.compile(r"^\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{2,4}$")
_YEAR = re.compile(r"^(?:rok[au]?\s+)?(1\d{3}|20\d{2})$")
_CONNECTORS = {"a", "z", "ze", "u", "v", "ve", "na", "nad", "pod", "de", "von", "van", "the", "of", "la", "le", "do", "di", "da", "i"}
_SPACED = re.compile(r"\s[.,;:!?)\]\-/%]|[(\[\-/]\s")


def _is_date(a: str) -> bool:
    low = a.lower().strip()
    if _DAY_MONTH_YEAR.match(low) or _YEAR.match(low):
        return True
    words = re.findall(r"[^\W\d_]+", low)
    has_digit = any(c.isdigit() for c in low)
    return has_digit and any(w.startswith(_MONTHS) for w in words)


def answer_type(answers: Sequence[str]) -> str:
    """The type of the first gold answer."""
    if not answers:
        return "other"
    a = answers[0].strip()
    if normalize_answer(a) in ("ano", "ne", "yes", "no"):
        return "yes/no"
    if _is_date(a):
        return "date"
    if _NUMBER.match(a):
        return "number"
    words = a.split()
    if len(words) == 1:
        return "single word"
    capitals = [w for w in words if w.lower() not in _CONNECTORS]
    if capitals and all(w[0].isupper() or w[0].isdigit() for w in capitals):
        return "multi-word name"
    return "other"


def answer_tokens(answers: Sequence[str]) -> int:
    """Tokens of the first gold answer, after the normalisation exact match uses."""
    return len(normalize_answer(answers[0]).split()) if answers else 0


def length_band(tokens: int) -> str:
    if tokens <= 1:
        return "1"
    if tokens == 2:
        return "2"
    if tokens == 3:
        return "3"
    return "4-5" if tokens <= 5 else "6+"


def spaced_punct(answer: str) -> bool:
    """True for a gold string like `U . S . A .` or `Jean - Paul`, which keeps a tokeniser's spaces."""
    return bool(_SPACED.search(answer))
