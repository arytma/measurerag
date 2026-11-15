"""Result flattening, the budget table and the curves."""

from __future__ import annotations

import csv

from measurerag import report


def result(**kw) -> dict:
    base = {
        "config_id": "bm25",
        "retriever_id": "bm25",
        "retriever_name": "bm25",
        "dataset": "toy",
        "language": "en",
        "device": "cpu",
        "status": "ok",
        "n_docs": 200,
        "n_queries": 20,
        "metrics": {"ndcg@10": 0.5, "recall@10": 0.6, "recall@100": 0.9,
                    "gold_in_context@5": 0.4},
        "measurement": {
            "peak_rss_gb": 1.0, "index_bytes": 1024**2, "model_bytes": 0,
            "retrieval_ms_p50": 1.0, "retrieval_ms_p95": 2.0,
        },
        "build": {"build_seconds": 0.5},
        "verdicts": [
            {"budget": "ram4g_ret200ms", "fits": True, "reasons": []},
            {"budget": "ram8g_ret200ms", "fits": True, "reasons": []},
        ],
    }
    base.update(kw)
    return base


BUDGETS = [
    {"name": "ram4g_ret200ms", "ram_gb": 4, "retrieval_ms": 200, "e2e_s": None},
    {"name": "ram8g_ret200ms", "ram_gb": 8, "retrieval_ms": 200, "e2e_s": None},
]


def test_flatten_puts_quality_and_cost_in_one_row():
    row = report.flatten(result())
    assert row["ndcg@10"] == 0.5
    assert row["peak_rss_gb"] == 1.0
    assert row["index_mb"] == 1.0
    assert row["fits:ram4g_ret200ms"] == "FITS"


def test_flatten_carries_gold_in_context_through():
    assert report.flatten(result())["gold_in_context@5"] == 0.4


def test_flatten_keeps_the_error_for_a_failed_configuration():
    row = report.flatten({"config_id": "x", "status": "failed", "error": "boom"})
    assert row["error"] == "boom"


def test_csv_is_written_with_a_stable_header(tmp_path):
    path = report.write_csv([report.flatten(result())], tmp_path / "r.csv")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["config_id"] == "bm25"
    assert rows[0]["fits:ram4g_ret200ms"] == "FITS"


def test_csv_unions_columns_across_ragged_rows(tmp_path):
    rows = [report.flatten(result()), report.flatten({"config_id": "y", "status": "failed",
                                                      "error": "e"})]
    path = report.write_csv(rows, tmp_path / "r.csv")
    header = open(path, encoding="utf-8").readline()
    assert "ndcg@10" in header and "error" in header


def test_table_labels_rows_by_config_id_not_by_retriever_kind():
    # two BM25 variants in one run must not both print as "bm25"
    results = [result(config_id="bm25"), result(config_id="bm25-nodiacritics")]
    table = report.budget_table(results, BUDGETS)
    assert "bm25-nodiacritics" in table


def test_table_sorts_by_quality_descending():
    results = [
        result(config_id="low", metrics={"ndcg@10": 0.1}),
        result(config_id="high", metrics={"ndcg@10": 0.9}),
    ]
    lines = report.budget_table(results, BUDGETS).splitlines()
    body = [ln for ln in lines if ln.startswith(("low", "high"))]
    assert body[0].startswith("high")


def test_table_brackets_a_score_that_fits_no_budget():
    r = result(verdicts=[{"budget": "ram4g_ret200ms", "fits": False, "reasons": ["too fat"]},
                         {"budget": "ram8g_ret200ms", "fits": False, "reasons": ["too fat"]}])
    table = report.budget_table([r], BUDGETS)
    assert "[0.5000]" in table


def test_table_does_not_bracket_a_score_that_fits_something():
    assert "[0.5000]" not in report.budget_table([result()], BUDGETS)


def test_table_lists_failed_configurations_separately():
    results = [result(), {"config_id": "boom", "status": "failed", "error": "ImportError: x"}]
    table = report.budget_table(results, BUDGETS)
    assert "Failed configurations:" in table and "ImportError" in table


def test_table_survives_a_run_where_everything_failed():
    table = report.budget_table([{"config_id": "a", "status": "failed", "error": "x"}], BUDGETS)
    assert "(no configuration completed)" in table


def test_table_explains_the_budgets_it_used():
    table = report.budget_table([result()], BUDGETS)
    assert "RAM <= 4 GB, retrieval p95 <= 200 ms" in table


def test_curves_are_written(tmp_path):
    a = report.curve_quality_vs_memory([result()], BUDGETS, tmp_path / "m.png")
    b = report.curve_quality_vs_latency([result()], BUDGETS, tmp_path / "l.png")
    assert a.exists() and a.stat().st_size > 0
    assert b.exists() and b.stat().st_size > 0


def test_curves_return_none_when_nothing_succeeded(tmp_path):
    assert report.curve_quality_vs_memory([], BUDGETS, tmp_path / "m.png") is None


# -- the generator throughput column, named honestly ----------------------
def _rag_row(**over):
    from measurerag.report import flatten

    base = {
        "config_id": "bm25+qwen2.5-1.5b",
        "rag_status": "ok",
        "build": {},
        "metrics": {},
        "measurement": {"gen_tokens_per_s": 0.7203, "e2e_s_p50": 5.82, "e2e_s_p95": 8.8},
        "answers": {"em": 0.29, "f1": 0.3742, "n_answered": 200.0},
        "generator": {"key": "qwen2.5-1.5b", "completion_tokens": 863},
    }
    base.update(over)
    return flatten(base)


def test_the_throughput_column_says_it_includes_prefill():
    """0.72 "tokens/s" is prefill-dominated, not a broken laptop."""
    row = _rag_row()
    assert "gen_tokens_per_s" not in row, "the bare name invites the wrong reading"
    assert row["gen_tok_per_s_incl_prefill"] == 0.72


def test_tokens_per_answer_makes_the_short_answer_task_visible():
    assert _rag_row()["tokens_per_answer"] == 4.32


def test_tokens_per_answer_survives_zero_answered():
    row = _rag_row(answers={"em": 0.0, "f1": 0.0, "n_answered": 0})
    assert row["tokens_per_answer"] is None


def test_a_retrieval_only_row_has_no_generator_columns():
    from measurerag.report import flatten

    row = flatten({"config_id": "bm25", "build": {}, "metrics": {}, "measurement": {}})
    for k in ("em", "f1", "tokens_per_answer", "gen_tok_per_s_incl_prefill"):
        assert k not in row
