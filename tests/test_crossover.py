"""Run 2: the dial, gold-in-context counting, the root and its bootstrap, fill."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from measurerag.crossover.dial import DIAL_LEVELS, dial_context, dial_uniforms, gold_counts
from measurerag.crossover.stats import (
    bootstrap_counts,
    interval,
    isotonic_root,
    linear_root,
    pava_decreasing,
    resampled_means,
    root_with_interval,
)

REPO = Path(__file__).resolve().parents[1]

RANKED = [f"d{i}" for i in range(20)]


# -- the dial ----------------------------------------------------------------
@pytest.mark.parametrize("gold", [{"d1"}, {"d0", "d3"}, {"d2", "d17"}])
def test_p_zero_is_the_unchanged_top_k(gold):
    u = dial_uniforms("t", "q1", gold)
    assert dial_context(RANKED, gold, 5, 0.0, u) == RANKED[:5]


@pytest.mark.parametrize("gold", [{"d1"}, {"d0", "d3"}, {"d0", "d1", "d2", "d3", "d4"}])
def test_p_one_leaves_no_gold(gold):
    u = dial_uniforms("t", "q1", gold)
    ctx = dial_context(RANKED, gold, 5, 1.0, u)
    assert not gold.intersection(ctx)
    assert len(ctx) == 5 and len(set(ctx)) == 5


def test_replacements_are_the_next_ranked_non_gold_in_place():
    gold = {"d1", "d6"}  # d6 is below k, so it can never be a replacement
    u = {"d1": 0.1, "d6": 0.5}
    assert dial_context(RANKED, gold, 5, 1.0, u) == ["d0", "d5", "d2", "d3", "d4"]


def test_levels_are_nested_and_substitutes_stable():
    gold = {"d0", "d3"}
    u = {"d0": 0.7, "d3": 0.3}  # d3 drops first, so it takes the first spare
    ctx = {p: dial_context(RANKED, gold, 5, p, u) for p in DIAL_LEVELS}
    assert ctx[0.2] == RANKED[:5]
    assert ctx[0.4] == ["d0", "d1", "d2", "d5", "d4"]
    assert ctx[0.6] == ctx[0.4]
    assert ctx[0.8] == ["d6", "d1", "d2", "d5", "d4"]
    assert ctx[1.0] == ctx[0.8]


def test_uniforms_depend_on_the_question_not_the_ranking():
    a = dial_uniforms("hotpotqa", "q7", {"x", "y"})
    b = dial_uniforms("hotpotqa", "q7", {"y", "x"})
    assert a == b and all(0.0 <= v < 1.0 for v in a.values())
    assert a != dial_uniforms("hotpotqa", "q8", {"x", "y"})


def test_no_gold_in_top_k_is_unchanged_at_any_level():
    gold = {"d12"}
    u = dial_uniforms("t", "q", gold)
    for p in DIAL_LEVELS:
        assert dial_context(RANKED, gold, 5, p, u) == RANKED[:5]


# -- gold in context ---------------------------------------------------------
def test_gold_counting_one_gold_passage():
    assert gold_counts(["a", "b"], {"a"}) == (1, 1)
    assert gold_counts(["b", "c"], {"a"}) == (0, 0)


def test_gold_counting_two_gold_passages():
    assert gold_counts(["a", "b", "c"], {"a", "c"}) == (1, 1)
    assert gold_counts(["a", "b"], {"a", "c"}) == (1, 0)
    assert gold_counts(["b"], {"a", "c"}) == (0, 0)


def test_gold_counting_no_gold_is_never_all_gold():
    assert gold_counts(["a"], set()) == (0, 0)


# -- fits, roots, bootstrap --------------------------------------------------
def test_pava_pools_violators():
    x, fit = pava_decreasing([0, 1, 2, 3], [3.0, 1.0, 2.0, 0.0])
    assert x.tolist() == [0, 1, 2, 3]
    assert fit.tolist() == [3.0, 1.5, 1.5, 0.0]


def test_roots_on_an_exact_line():
    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    y = [0.3 - 0.5 * xi for xi in x]  # root at 0.6
    assert linear_root(x, y) == pytest.approx(0.6)
    assert isotonic_root(x, y) == pytest.approx(0.6)


def test_no_root_in_range_is_nan():
    assert np.isnan(linear_root([0, 1], [1.0, 0.5]))
    assert np.isnan(isotonic_root([0, 1], [1.0, 0.5]))


def test_bootstrap_counts_are_seeded_and_complete():
    a = bootstrap_counts(50, 200, "fixture")
    b = bootstrap_counts(50, 200, "fixture")
    assert np.array_equal(a, b)
    assert (a.sum(axis=1) == 50).all()


def test_bootstrap_recovers_a_known_root():
    """Simulated questions whose curves have a known root, g = 0.5.

    Each of 5 operating points has g in {0.1, 0.3, 0.5, 0.7, 0.9}; the per-point
    difference Delta_ret - Delta_gen is 0.4 * (0.5 - g). Outcomes are +1/0/-1 per
    question with that mean, so the bootstrap root interval must contain 0.5.
    """
    n = 2000
    g = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    target = 0.4 * (0.5 - g)
    gen = np.random.default_rng(0)
    # represent D per question as the difference of two 0/1 outcomes
    a = np.zeros((5, n), dtype=np.int64)
    c = np.zeros((5, n), dtype=np.int64)
    for i, t in enumerate(target):
        base = gen.random(n) < 0.3
        a[i] = base
        c[i] = base
        m = int(round(abs(t) * n))
        idx = gen.choice(n, size=m, replace=False)
        if t > 0:
            a[i, idx] = 1
            c[i, idx] = 0
        else:
            a[i, idx] = 0
            c[i, idx] = 1
    counts = bootstrap_counts(n, 2000, "known-root")
    d_boot = resampled_means(a, counts) - resampled_means(c, counts)
    d_point = a.mean(axis=1) - c.mean(axis=1)
    x_boot = np.tile(g, (counts.shape[0], 1))
    for method in ("linear", "isotonic"):
        s = root_with_interval(g, d_point, x_boot, d_boot, method)
        assert s.point == pytest.approx(0.5, abs=0.02)
        assert s.lo <= 0.5 <= s.hi
        assert s.hi - s.lo < 0.2


def test_censored_resamples_are_counted():
    g = np.array([0.2, 0.8])
    x_boot = np.tile(g, (3, 1))
    y_boot = np.array([[1.0, -1.0], [1.0, 0.5], [-0.5, -1.0]])
    s = root_with_interval(g, np.array([1.0, -1.0]), x_boot, y_boot, "linear")
    assert (s.n_no_root, s.no_root_above, s.no_root_below) == (2, 1, 1)


def test_interval_ignores_nan():
    assert interval(np.array([np.nan, 1.0, 2.0, 3.0])) == (1.0, 3.0)


# -- fill ----------------------------------------------------------------------
def _fill(tmp_path, template: str, stats: dict):
    t = tmp_path / "T.template.md"
    s = tmp_path / "stats.json"
    o = tmp_path / "T.md"
    t.write_text(template, encoding="utf-8")
    s.write_text(json.dumps(stats), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fill.py"), "--template", str(t), "--stats", str(s), "--out", str(o)],
        capture_output=True, text=True,
    ), o


def test_fill_replaces_tokens(tmp_path):
    proc, out = _fill(tmp_path, "g* = {{g_star}} [{{lo}}, {{hi}}]\n", {"g_star": "0.6", "lo": "0.5", "hi": "0.7"})
    assert proc.returncode == 0, proc.stderr
    assert out.read_text(encoding="utf-8") == "g* = 0.6 [0.5, 0.7]\n"


def test_fill_fails_on_a_missing_token(tmp_path):
    proc, out = _fill(tmp_path, "g* = {{g_star}} and {{nowhere}}\n", {"g_star": "0.6"})
    assert proc.returncode != 0
    assert "nowhere" in proc.stderr
    assert not out.exists()


def test_break_even_gain_on_a_line():
    from measurerag.crossover.stats import break_even_gain

    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.1, 0.2])  # slope 0.2
    assert break_even_gain(x, y, 0.25, 0.05) == pytest.approx(0.25)
    assert np.isnan(break_even_gain(x, y, 0.9, 0.05))  # only 0.1 of headroom
    assert break_even_gain(x, y, 0.3, 0.0) == 0.0


def test_interval_bounds_are_symmetric_order_statistics():
    from measurerag.crossover.stats import interval

    # 1..10000: the 95% bounds are the 250th and the 9,750th order statistics
    assert interval(np.arange(1, 10001, dtype=float)) == (250.0, 9750.0)
