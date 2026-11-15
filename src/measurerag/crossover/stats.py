"""Paired bootstrap over questions, and the root of a difference curve.

Everything is resampled jointly: one resample draws one multiset of questions
and every condition - every dial level, every generator, every natural-sweep
configuration - is re-scored on that same multiset. That is what makes a
difference between two conditions a *paired* estimate.

Resamples are drawn as integer count vectors from a seeded generator, and EM is
0/1, so every resampled mean is an exact integer ratio: no BLAS reduction order
or thread count can move a digit, which is what lets crossover.json be
byte-identical across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..seeds import rng


def bootstrap_counts(n: int, n_boot: int, *tags: object) -> np.ndarray:
    """(n_boot, n) int64 matrix; row b says how often each question is drawn."""
    g = rng("bootstrap", *tags)
    return g.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.int64)


def resampled_means(outcomes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """outcomes (n_cond, n) of 0/1 ints -> (n_boot, n_cond) means."""
    n = counts.shape[1]
    sums = counts @ outcomes.astype(np.int64).T  # exact integer arithmetic
    return sums / float(n)


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    # rounded: (1 - 0.95) / 2 is 0.025000000000000022 in floating point, which
    # made inverted_cdf take the 251st of 10,000 order statistics for the lower
    # bound but the 9,750th for the upper
    lo = round((1.0 - level) / 2.0, 12)
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(s, lo, method="inverted_cdf")), float(np.quantile(s, 1 - lo, method="inverted_cdf")))


def established(lo: float, hi: float) -> bool:
    """An effect is established when its 95% interval excludes zero."""
    return bool(lo > 0 or hi < 0)


# --------------------------------------------------------------------------
# fits and roots
# --------------------------------------------------------------------------
def _sorted_xy(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.lexsort((y, x))
    return x[order], y[order]


def pava_decreasing(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares monotone non-increasing fit (pool adjacent violators).

    Written out rather than taken from scikit-learn so the fit, and therefore
    the bootstrap, is a fixed sequence of float operations with no library
    version in the loop. Ties in x are pooled first.
    """
    xs, ys = _sorted_xy(x, y)
    ux, inv = np.unique(xs, return_inverse=True)
    w = np.bincount(inv).astype(float)
    v = np.bincount(inv, weights=ys) / w
    blocks: list[list[float]] = []  # [value, weight, n_points]
    for vi, wi in zip(v, w):
        blocks.append([vi, wi, 1])
        # decreasing: a later block may not exceed an earlier one
        while len(blocks) > 1 and blocks[-1][0] > blocks[-2][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    fitted = np.concatenate([np.full(int(n), val) for val, _, n in blocks])
    return ux, fitted


def piecewise_root(x: np.ndarray, y: np.ndarray) -> float:
    """First zero crossing of the linear interpolation through (x, y), x ascending.

    Returns NaN when the curve never reaches zero on [x_min, x_max]. A flat
    stretch that sits exactly on zero returns its midpoint.
    """
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(x[0]) if y[0] == 0 else float("nan")
    for i in range(x.size - 1):
        y0, y1 = y[i], y[i + 1]
        if y0 == 0 and y1 == 0:
            return float((x[i] + x[i + 1]) / 2)
        if y0 == 0:
            return float(x[i])
        if (y0 > 0) != (y1 > 0) and y1 != 0:
            return float(x[i] + (x[i + 1] - x[i]) * y0 / (y0 - y1))
        if y1 == 0:
            return float(x[i + 1])
    return float("nan")


def isotonic_root(x: Sequence[float], y: Sequence[float]) -> float:
    ux, fitted = pava_decreasing(x, y)
    return piecewise_root(ux, fitted)


def linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Ordinary least squares y = a + b x, closed form."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    xm, ym = xs.mean(), ys.mean()
    sxx = float(((xs - xm) ** 2).sum())
    if sxx == 0:
        return float(ym), 0.0
    b = float(((xs - xm) * (ys - ym)).sum() / sxx)
    return float(ym - b * xm), b


def linear_root(x: Sequence[float], y: Sequence[float]) -> float:
    """Root of the OLS line, NaN if it lies outside the observed range of x."""
    a, b = linear_fit(x, y)
    xs = np.asarray(x, dtype=float)
    if b == 0:
        return float("nan")
    r = -a / b
    return float(r) if xs.min() <= r <= xs.max() else float("nan")


@dataclass
class RootSummary:
    point: float
    lo: float
    hi: float
    n_boot: int
    n_no_root: int          # resamples in which the fitted curve never crossed zero
    no_root_above: int      # of those, curve positive throughout (retriever wins everywhere)
    no_root_below: int      # of those, curve negative throughout (generator wins everywhere)

    def as_dict(self) -> dict:
        return {
            "point": self.point, "lo": self.lo, "hi": self.hi, "n_boot": self.n_boot,
            "n_no_root": self.n_no_root, "no_root_above": self.no_root_above,
            "no_root_below": self.no_root_below,
            "interval_censored": bool(self.n_no_root > 0.025 * self.n_boot),
        }


def root_with_interval(
    x_point: np.ndarray, y_point: np.ndarray, x_boot: np.ndarray, y_boot: np.ndarray, method: str
) -> RootSummary:
    """Root on the point estimate plus a percentile interval over resamples.

    A resample whose curve never crosses zero is censored at the edge of the
    observed range it fell off (all positive -> the top of the g range, all
    negative -> the bottom) rather than dropped, because dropping it would
    narrow the interval exactly when the data least support a crossing. How many
    were censored is reported; if more than 2.5% were, the interval is flagged.
    """
    f = isotonic_root if method == "isotonic" else linear_root
    point = f(x_point, y_point)
    roots = np.empty(x_boot.shape[0])
    above = below = 0
    for b in range(x_boot.shape[0]):
        r = f(x_boot[b], y_boot[b])
        if np.isnan(r):
            if np.mean(y_boot[b]) > 0:
                above += 1
                r = float(np.max(x_boot[b]))
            else:
                below += 1
                r = float(np.min(x_boot[b]))
        roots[b] = r
    lo, hi = interval(roots)
    return RootSummary(point, lo, hi, int(x_boot.shape[0]), above + below, above, below)


def break_even_gain(x: np.ndarray, y: np.ndarray, x0: float, target: float) -> float:
    """Smallest dx >= 0 with y(x0 + dx) - y(x0) >= target, on the linear
    interpolation through (x, y) with x ascending.

    This is the crossover stated as a step size rather than a place: at
    operating point x0, how much gold-in-context must a retriever upgrade add
    before its EM gain matches a fixed rival gain `target`? NaN when no point up
    to max(x) gets there - the headroom above x0 is not enough.
    """
    if target <= 0:
        return 0.0
    y0 = float(np.interp(x0, x, y))
    need = y0 + target
    xs = [x0] + [float(v) for v in x if v > x0]
    ys = [y0] + [float(v) for v, u in zip(y, x) if u > x0]
    for i in range(len(xs) - 1):
        if ys[i + 1] >= need:
            if ys[i + 1] == ys[i]:
                return xs[i] - x0
            return xs[i] + (xs[i + 1] - xs[i]) * (need - ys[i]) / (ys[i + 1] - ys[i]) - x0
    return float("nan")
