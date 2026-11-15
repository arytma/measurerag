"""Every difference the READMEs show says which model's value is subtracted from which.

docs/CROSSOVER.md and results/stats.json keep each difference in the direction it was computed. The READMEs read some
of them the other way round at render time (scripts/fill.py: |western_first for a table, |neg for a value), so that a
sentence or a row names its models in the documents' order; a value's sign can then differ from CROSSOVER's. A reader
comparing the two sees why only if the README names the subtraction, "A minus B", in the row or in its column header,
and only if the row is CROSSOVER's row, as it stands or read the other way round with every value negated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
READMES = ["README.md", "README.cs.md"]
SIGNED = re.compile(r"(?<![\w.,])[+\-−]\d")          # a signed value, "+0.089" or "+0,089"; not "Qwen2.5-3B"
MINUS = re.compile(r"\bminus\b")                          # the word is the same in Czech
NUMBER = r"[+\-−]\d+\.\d+"
DIFFERENCE = re.compile(rf"(\S+) minus (\S+): ({NUMBER}) \[({NUMBER}), ({NUMBER})\]((?: \(not established\))?)")


def tables(text: str) -> list[list[list[str]]]:
    """Each Markdown table outside code fences, as rows of cells (the header first, the separator left out)."""
    text = re.sub(r"^(```+|~~~+).*?^\1", "", text, flags=re.S | re.M)
    out, current = [], []
    for line in text.split("\n") + [""]:
        if line.lstrip().startswith("|"):
            current.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif current:
            out.append([r for r in current if not all(re.fullmatch(r":?-+:?", c) for c in r)])
            current = []
    return out


def unnamed_differences(table: list[list[str]]) -> list[str]:
    """The signed cells of a table whose subtraction neither the cell, its column header nor a label cell of its row
    (one with no signed value) names."""
    head, bad = table[0], []
    for row in table[1:]:
        labels = " ".join(c for c in row if not SIGNED.search(c))
        for i, cell in enumerate(row):
            if SIGNED.search(cell) and not (MINUS.search(cell) or MINUS.search(head[i]) or MINUS.search(labels)):
                bad.append(f"{head[i]}: {cell} (row {row[0]})")
    return bad


@pytest.mark.parametrize("rel", READMES)
def test_every_signed_value_in_a_readme_table_says_what_it_subtracts(rel):
    found = [t for t in tables((REPO / rel).read_text(encoding="utf-8")) if any(SIGNED.search(c) for r in t[1:] for c in r)]
    if rel == "README.md":     # the families' differences and their steps; README.cs.md has none and states them in prose
        assert len(found) >= 2, "the README's two difference tables are gone: update this test"
    for table in found:
        assert not unnamed_differences(table), unnamed_differences(table)


def test_the_rule_finds_a_difference_that_names_no_subtraction():
    table = [["set", "families", "smaller models [95%]"], ["SQAD 3.2", "Gemma 4 and Qwen3.5", "+0.025 [+0.007, +0.044]"]]
    assert unnamed_differences(table) == ["smaller models [95%]: +0.025 [+0.007, +0.044] (row SQAD 3.2)"]
    table[1][1] = "Gemma 4 minus Qwen3.5"
    assert unnamed_differences(table) == []
    assert not SIGNED.search("Qwen2.5-3B-Instruct, bge-m3, 1.5B, 0.138")


def family_differences(rel: str) -> dict[tuple[str, str, str], list[str]]:
    """The table whose header has a "families" column: (set, first family, second family) -> its value cells."""
    found = [t for t in tables((REPO / rel).read_text(encoding="utf-8")) if "families" in t[0]]
    assert len(found) == 1, f"{rel}: {len(found)} tables of differences between families"
    head, *rows = found[0]
    col, out = head.index("families"), {}
    for row in rows:
        a, b = re.fullmatch(r"(.+) minus (.+)", row[col]).groups()
        out[(row[0], a, b)] = row[col + 1:]
    return out


def read_other_way(cell: str) -> str:
    """CROSSOVER's "x minus y: d [lo, hi]" as "y minus x: -d [-hi, -lo]", written here independently of fill.py."""
    x, y, d, lo, hi, mark = DIFFERENCE.fullmatch(cell).groups()

    def neg(v: str) -> str:
        return "+" + v[1:] if v[0] in "-−" or float(v) == 0 else "-" + v[1:]

    return f"{y} minus {x}: {neg(d)} [{neg(hi)}, {neg(lo)}]{mark}"


def test_the_readmes_family_differences_are_crossovers_rows_one_way_or_the_other():
    readme, crossover = family_differences("README.md"), family_differences("docs/CROSSOVER.md")
    assert len(readme) == len(crossover)
    reversed_rows = 0
    for (s, a, b), cells in readme.items():
        if (s, a, b) in crossover:
            assert cells == crossover[(s, a, b)], (s, a, b)
        else:
            assert (s, b, a) in crossover, f"{s}: {a} minus {b} is in neither direction in CROSSOVER"
            assert cells == [read_other_way(c) for c in crossover[(s, b, a)]], (s, a, b)
            reversed_rows += 1
    assert reversed_rows, "no row is read the other way round: the sign rule is not exercised"


def blocks(text: str) -> list[str]:
    """Each paragraph and each list item of a template."""
    return re.split(r"\n\s*\n|\n(?=\s*[*-] )", text)


@pytest.mark.parametrize("rel", [f"templates/{r}" for r in READMES])
def test_every_value_read_the_other_way_round_sits_beside_its_subtraction(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    flipped = [b for b in blocks(text) if "|neg" in b]
    assert flipped, f"{rel} reads no value the other way round: drop it from this test"
    assert all(MINUS.search(re.sub(r"\{\{[^}]*\}\}", " ", b)) for b in flipped), [
        " ".join(b.split())[:200] for b in flipped if not MINUS.search(re.sub(r"\{\{[^}]*\}\}", " ", b))]
