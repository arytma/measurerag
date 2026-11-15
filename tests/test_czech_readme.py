"""The Czech summary, README.cs.md: Czech number marks from the same tokens, read back by the checks, and the
language line at the top of both READMEs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fill  # noqa: E402
from check_docs import ANY_NUMBER, czech_documents, from_czech_numbers, norm, traced_numbers, untraced  # noqa: E402

NBSP = "\u00a0"


@pytest.mark.parametrize("value, czech", [
    ("0.8399", "0,8399"),
    ("4,771", f"4{NBSP}771"),
    ("100,000", f"100{NBSP}000"),
    ("+0.039 [+0.016, +0.061]", "+0,039 [+0,016; +0,061]"),
    ("+0.126 to +0.170", "+0,126 až +0,170"),
    ("-0.038 to -0.013", "-0,038 až -0,013"),
    ("2.34", "2,34"),
    ("0", "0"),
])
def test_a_token_takes_czech_number_marks(value, czech):
    assert fill.czech_number(value) == czech
    # and the checks read the Czech number back as the number the stats file holds
    assert [norm(n) for n in ANY_NUMBER.findall(from_czech_numbers(czech))] == [norm(n) for n in ANY_NUMBER.findall(value)]


def test_a_token_with_english_words_cannot_be_written_in_czech():
    stats = {"a": "SQAD 3.2 full and the copy debiased", "b": "below zero on all four", "c": "+0.1 to +0.2"}
    text, missing = fill.fill("{{a|cs}} {{b|cs}} {{c|cs}} {{c|xx}}", stats)
    assert text == ""
    assert {m.split(" ")[0] for m in missing} == {"a|cs", "b|cs", "c|xx"}


def test_the_checks_read_czech_numbers_back_and_catch_a_wrong_one():
    traced = traced_numbers({"x": "0.8399", "n": "4,771"})
    assert untraced(f"nDCG@10 0,8399 na 4{NBSP}771 otázkách.", traced, 10, czech=True) == []
    bad = untraced(f"nDCG@10 0,8398 na 4{NBSP}772 otázkách.", traced, 10, czech=True)
    assert len(bad) == 2


def test_the_czech_summary_is_a_document_under_the_number_rule():
    cfg = fill.docs_config(REPO)
    assert "README.cs.md" in cfg["documents"] and "README.cs.md" in czech_documents(cfg)
    assert "README.cs.md" not in (cfg.get("number_rule_exempt") or {})
    template = (REPO / "templates/README.cs.md").read_text(encoding="utf-8")
    tokens = fill.TOKEN.findall(template)
    stats = json.loads((REPO / "results/stats.json").read_text(encoding="utf-8"))["tokens"]
    # every number a token gives it is written with Czech marks, except the copy's repository name in code
    assert all(t.endswith("|cs") or t == "czech_checks.copy_repo" for t in tokens), tokens
    assert all(fill.token_name(t) in stats for t in tokens)


def test_both_readmes_start_with_the_language_line():
    en = (REPO / "README.md").read_text(encoding="utf-8").split("\n")
    cs = (REPO / "README.cs.md").read_text(encoding="utf-8").split("\n")
    assert en[0].startswith("<!-- Generated from templates/README.md") and en[1] == "English | [Čeština](README.cs.md)"
    assert cs[0].startswith("<!-- Generated from templates/README.cs.md") and cs[1] == "[English](README.md) | Čeština"


@pytest.mark.parametrize("value, other_way", [
    ("-0.038 to -0.013", "+0.013 to +0.038"),
    ("+0.000 to +0.061", "-0.061 to +0.000"),
    ("-0.025 [-0.044, -0.007]", "+0.025 [+0.007, +0.044]"),
    ("+0.011 [-0.006, +0.029] (not established)", "-0.011 [-0.029, +0.006] (not established)"),
    ("-0.012", "+0.012"),
])
def test_a_difference_reads_the_other_way_round(value, other_way):
    """{{token|neg}} lets a sentence name its models in the order it lists them: A minus B written as B minus A."""
    assert fill.negated(value) == other_way
    assert fill.negated(other_way) == value.replace("-0.000", "+0.000")
    if "established" not in value:                     # the Czech marks refuse English words, as designed
        assert fill.token_value("t|neg|cs", {"t": value}) == fill.czech_number(other_way)


def test_neg_refuses_anything_but_a_difference():
    with pytest.raises(ValueError):
        fill.negated("below zero on all four")
