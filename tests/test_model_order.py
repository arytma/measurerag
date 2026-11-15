"""The order in which the documents name models: the United States first, then France, then the Czech Republic, then
China (Gemma 3n, Gemma 4, Ministral 3, Qwen2.5, Qwen3.5; for encoders, the US ones, then Seznam's, then bge-m3).

The prose of every public document but docs/CROSSOVER.md follows it, sentence by sentence; the README's family tables
are put in it at render time (scripts/fill.py), and the stats file keeps the order and direction they were computed in.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fill  # noqa: E402

DOCS = ["README.md", "README.cs.md", "CHANGELOG.md", "THIRD_PARTY.md", "NOTICE", "answers/README.md", "docs/BUDGETS.md",
        "docs/DATASETS_AND_LICENCES.md", "docs/LIMITATIONS.md", "docs/METHOD.md", "docs/METRICS.md", "docs/REPRODUCE.md"]
RANKS = [
    (re.compile(r"\bQwen[\w.\-]*|\bbge-m3\b|\bBAAI\b|\bAlibaba\b"), 3),
    (re.compile(r"\bretromae[\w\-]*|\bsimcse[\w\-]*|\bSeznam\b"), 2),
    (re.compile(r"\b[Mm]inistral[\w.\-]*|\bMistral\b"), 1),
    (re.compile(r"\bGemma\b|\bgemma[\w.\-]*|\bEmbeddingGemma\b|\b[Gg]ranite[\w.\-]*|\bmultilingual-e5[\w\-]*|\be5-(?:small|base)\b|\bGoogle\b|\bIBM\b"), 0),
]
CONFIG_ID = re.compile(r"\b[\w.\-]+\+[\w.\-]+\b")          # a configuration's id, retriever+generator, names no order


def out_of_order(text: str) -> str | None:
    text = CONFIG_ID.sub(" ", text)
    found = sorted((m.start(), rank, m.group(0)) for rx, rank in RANKS for m in rx.finditer(text))
    highest = -1
    for _, rank, name in found:
        if rank < highest:
            return name
        highest = max(highest, rank)
    return None


def units(text: str):
    """Each table row, and each sentence of the prose, without code or comments."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"^(```+|~~~+).*?^\1", " ", text, flags=re.S | re.M)
    text = re.sub(r"`[^`\n]*`", " ", text)
    yield from (line for line in text.split("\n") if line.lstrip().startswith("|"))
    prose = re.sub(r"^\s*\|.*$", "", text, flags=re.M)
    yield from re.split(r"(?<=[.!?;])\s+|\n\s*\n|\n\s*[*-] |\*\* ", prose)


@pytest.mark.parametrize("rel", DOCS)
def test_every_document_names_its_models_in_order(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    bad = [(name, " ".join(u.split())[:160]) for u in units(text) if (name := out_of_order(u))]
    assert not bad, bad


def numbers(table: str) -> Counter:
    return Counter(n.lstrip("+-") for n in re.findall(r"[+\-]?\d+\.\d+", table))


def tokens() -> dict:
    return json.loads((REPO / "results/stats.json").read_text(encoding="utf-8"))["tokens"]


def test_family_columns_groups_each_familys_models_and_moves_whole_columns():
    table = tokens()["cz.table_families_levels_mapped"]
    out = fill.family_columns(table)
    head = fill._cells(out.split("\n")[0])
    assert [fill._family(c) for c in head[2:]] == [f for f in fill.FAMILY_ORDER for _ in range(2)]
    # the same columns, cell for cell: each row is the original row permuted
    for a, b in zip(table.split("\n"), out.split("\n")):
        assert sorted(fill._cells(a)) == sorted(fill._cells(b))


def test_western_first_writes_a_later_countrys_family_second_and_negates_its_values():
    table = tokens()["cz.table_families_differences_mapped"]
    out = fill.western_first(table)
    assert len(out.split("\n")) == len(table.split("\n"))
    assert numbers(out) == numbers(table)                 # the same values, some read the other way round
    for line in out.split("\n")[2:]:
        a, b = (fill._family(x) for x in fill._cells(line)[1].split(" minus "))
        assert fill.COUNTRY[a] <= fill.COUNTRY[b], line
    # a flipped row: Qwen3.5 minus Gemma 4 on SQAD 3.2 debiased, -0.025 [-0.044, -0.007] for the smaller models
    orig = next(x for x in table.split("\n") if x.startswith("| SQAD 3.2, debiased") and "Qwen3.5" in x.split("|")[2].split("minus")[0] and "Gemma 4" in x.split("|")[2])
    flipped = next(x for x in out.split("\n") if x.startswith("| SQAD 3.2, debiased") and "Gemma 4" in x.split("|")[2].split("minus")[0] and "Qwen3.5" in x.split("|")[2])
    assert fill._cell_other_way(fill._cells(orig)[2]) == fill._cells(flipped)[2]
    # a row within one country keeps its direction
    assert any(x.split("|")[2].strip().startswith("Qwen3.5") and "minus Qwen2.5" in x for x in out.split("\n"))
