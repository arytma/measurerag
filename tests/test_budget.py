"""Budget accounting - the verdicts are the product, so they get tested hard."""

from __future__ import annotations

from measurerag.budget import Budget, Measurement, budget_grid, check, check_all


def m(**kw) -> Measurement:
    base = dict(peak_rss_gb=1.0, retrieval_ms_p50=10.0, retrieval_ms_p95=20.0)
    base.update(kw)
    return Measurement(**base)


B = Budget(name="b", ram_gb=4.0, retrieval_ms=200.0)


def test_comfortable_configuration_fits():
    v = check(m(), B)
    assert v.fits and v.reasons == []


def test_ram_over_budget_fails_with_a_reason():
    v = check(m(peak_rss_gb=4.5), B)
    assert not v.fits
    assert "peak RSS" in v.reasons[0]


def test_exactly_at_the_ceiling_fits():
    # the ceiling is inclusive; a 4.00 GB configuration is a 4 GB configuration
    assert check(m(peak_rss_gb=4.0, retrieval_ms_p95=200.0), B).fits


def test_latency_over_budget_fails():
    v = check(m(retrieval_ms_p95=200.1), B)
    assert not v.fits and "retrieval p95" in v.reasons[0]


def test_both_violations_are_reported_not_just_the_first():
    v = check(m(peak_rss_gb=9.0, retrieval_ms_p95=900.0), B)
    assert len(v.reasons) == 2


def test_p50_never_decides_a_verdict():
    # a fast median cannot rescue a slow tail
    assert not check(m(retrieval_ms_p50=1.0, retrieval_ms_p95=500.0), B).fits


def test_retrieval_only_configuration_cannot_fit_an_e2e_budget():
    rag_budget = Budget(name="e", ram_gb=16.0, retrieval_ms=500.0, e2e_s=8.0)
    v = check(m(), rag_budget)
    assert not v.fits
    assert "no generator" in v.reasons[0]


def test_e2e_budget_compares_p95():
    rag_budget = Budget(name="e", ram_gb=16.0, retrieval_ms=500.0, e2e_s=3.0)
    assert check(m(e2e_s_p50=0.5, e2e_s_p95=2.9), rag_budget).fits
    assert not check(m(e2e_s_p50=0.5, e2e_s_p95=3.1), rag_budget).fits


def test_grid_is_the_cross_product():
    assert len(budget_grid([4, 8, 16], [200, 500])) == 6
    assert len(budget_grid([4, 8], [200, 500], [3, 8])) == 8


def test_grid_names_are_unique_and_stable():
    names = [b.name for b in budget_grid([4, 8, 16], [200, 500])]
    assert len(set(names)) == len(names)
    assert names[0] == "ram4g_ret200ms"


def test_check_all_returns_one_verdict_per_budget():
    budgets = budget_grid([4, 8, 16], [200, 500])
    verdicts = check_all(m(peak_rss_gb=6.0), budgets)
    assert len(verdicts) == 6
    # 6 GB fits the 8 and 16 GB budgets, not the 4 GB ones
    assert [v.fits for v in verdicts] == [False, False, True, True, True, True]


def test_budget_roundtrips_through_dict():
    b = Budget.from_dict({"name": "x", "ram_gb": 8, "retrieval_ms": 200, "e2e_s": 3})
    assert b.as_dict() == {"name": "x", "ram_gb": 8.0, "retrieval_ms": 200.0, "e2e_s": 3.0}
