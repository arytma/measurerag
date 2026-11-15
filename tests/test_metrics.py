"""Metrics against hand-computed values - if these drift, every number does."""

from __future__ import annotations

import math

import pytest

from measurerag.metrics import (
    evaluate_answers,
    evaluate_run,
    exact_match,
    gold_in_context,
    latency_summary,
    ndcg_at_k,
    normalize_answer,
    percentile,
    recall_at_k,
    token_f1,
)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 1}, 10) == pytest.approx(1.0)


def test_ndcg_uses_log2_discount():
    # one relevant document in position 2 -> 1/log2(3) over an ideal of 1.0
    assert ndcg_at_k(["x", "a"], {"a": 1}, 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_respects_graded_relevance():
    # ideal puts grade 2 first; we put grade 1 first
    got = ndcg_at_k(["a", "b"], {"a": 1, "b": 2}, 10)
    ideal = 2 + 1 / math.log2(3)
    assert got == pytest.approx((1 + 2 / math.log2(3)) / ideal)


def test_ndcg_truncates_at_k():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_ndcg_with_no_judgements_is_zero():
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_recall_counts_only_positive_grades():
    rel = {"a": 1, "b": 0, "c": 2}
    assert recall_at_k(["a", "b"], rel, 10) == pytest.approx(0.5)
    assert recall_at_k(["a", "c"], rel, 10) == pytest.approx(1.0)


def test_recall_truncates_at_k():
    assert recall_at_k(["x", "a"], {"a": 1}, 1) == 0.0


def test_gold_in_context_is_binary():
    assert gold_in_context(["x", "a"], {"a": 1}, 5) == 1.0
    assert gold_in_context(["x", "a"], {"a": 1}, 1) == 0.0


def test_evaluate_run_skips_unjudged_queries_instead_of_zeroing_them():
    run = {"q1": ["a"], "q2": ["z"]}
    qrels = {"q1": {"a": 1}}  # q2 has no judgement at all
    out = evaluate_run(run, qrels, ks=[10])
    assert out["n_queries"] == 1
    assert out["ndcg@10"] == pytest.approx(1.0)


def test_evaluate_run_reports_gold_in_context_at_the_generator_k():
    run = {"q1": ["x", "x", "x", "x", "x", "a"]}
    qrels = {"q1": {"a": 1}}
    out = evaluate_run(run, qrels, ks=[10], context_k=5)
    assert out["gold_in_context@5"] == 0.0
    assert out["recall@10"] == 1.0


# -- answers --------------------------------------------------------------
def test_normalisation_keeps_czech_diacritics():
    # stripping them would merge distinct words and score a wrong answer right
    assert normalize_answer("Přírodní") == "přírodní"
    assert normalize_answer("Přírodní") != normalize_answer("Prirodni")


def test_normalisation_drops_english_articles_and_punctuation():
    assert normalize_answer("The  cat's, hat.") == "cats hat"


def test_exact_match_against_any_gold():
    assert exact_match("Praha", ["Brno", "praha"]) == 1.0
    assert exact_match("Ostrava", ["Brno", "Praha"]) == 0.0


def test_token_f1_is_partial_credit():
    assert token_f1("Praha je mesto", ["Praha"]) == pytest.approx(0.5)
    assert token_f1("Praha", ["Praha"]) == 1.0
    assert token_f1("Brno", ["Praha"]) == 0.0


def test_evaluate_answers_only_scores_questions_with_gold():
    out = evaluate_answers({"a": "x", "b": "y"}, {"a": ["x"]})
    assert out["n_answered"] == 1
    assert out["em"] == 1.0


# -- latency --------------------------------------------------------------
def test_percentile_is_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile(values, 100) == 10


def test_percentile_returns_an_observed_value():
    # nearest-rank never interpolates, so the answer is always in the input
    values = [1.0, 2.0, 100.0]
    assert percentile(values, 95) in values


def test_latency_summary_on_empty_input_is_zeros():
    assert latency_summary([]) == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0.0}
