"""Small statistics helpers for the evaluation harness.

The point of these is thesis-grade rigour: instead of reporting a single number
from a single run (which an LLM-backed pipeline does NOT reproduce exactly even at
temperature 0), every headline metric is reported with a 95% confidence interval,
and the single-vs-multi-agent comparison is backed by a significance test.

Everything degrades gracefully: scipy gives exact critical values / tests when it
is installed (it is, via scikit-learn), but a normal-approximation fallback keeps
the harness runnable without it. Only the standard library + numpy are required.
"""
from __future__ import annotations

import math
import random
import statistics

try:  # exact critical values / tests when available
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - fallback path
    _scipy_stats = None


def _z(conf: float) -> float:
    """Two-sided normal critical value for a confidence level."""
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.ppf(1 - (1 - conf) / 2))
    return 1.959963984540054 if abs(conf - 0.95) < 1e-9 else 1.959963984540054


def _t(conf: float, df: int) -> float:
    """Two-sided Student-t critical value (falls back to normal)."""
    if df <= 0:
        return float("nan")
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(1 - (1 - conf) / 2, df))
    return _z(conf)


def mean_ci(values: list[float], conf: float = 0.95) -> tuple[float, float]:
    """Return (mean, half_width) for a t-based confidence interval.

    half_width is nan when there are fewer than two samples (no spread to
    estimate). Report the interval as mean ± half_width.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    m = statistics.mean(values)
    if n == 1:
        return (m, float("nan"))
    s = statistics.stdev(values)  # sample standard deviation
    half = _t(conf, n - 1) * s / math.sqrt(n)
    return (m, half)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Preferred over the normal-approximation interval because it stays inside
    [0, 1] and behaves well for the small n and near-100% rates this suite hits.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = _z(conf)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(b: int, c: int) -> tuple[int, float]:
    """Exact McNemar test on the two discordant cell counts.

    For paired pass/fail outcomes of two systems on the same questions, `b` is the
    number where system A passed and B failed, `c` the reverse. Concordant pairs
    carry no information. Returns (n_discordant, p_value); a small p means the two
    systems differ significantly. Uses the exact binomial test (right for the
    small counts here); falls back to a hand-rolled exact p without scipy.
    """
    n = b + c
    if n == 0:
        return (0, 1.0)
    if _scipy_stats is not None:
        p = float(_scipy_stats.binomtest(min(b, c), n, 0.5).pvalue)
        return (n, p)
    # Exact two-sided binomial p-value at p=0.5 (symmetric): 2 * lower tail.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return (n, min(1.0, 2 * tail))


def bootstrap_diff_ci(
    a: list[float],
    b: list[float],
    conf: float = 0.95,
    iters: int = 10000,
    seed: int = 0,
    paired: bool = False,
) -> tuple[float, float, float]:
    """Bootstrap CI for the difference of means (a - b).

    Returns (point_estimate, lo, hi). When `paired` (same questions in both runs,
    aligned by index) the per-pair differences are resampled, which is the right
    model for comparing two configs on the identical question set.
    """
    if not a or not b:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    lo_q, hi_q = (1 - conf) / 2, 1 - (1 - conf) / 2

    if paired:
        diffs = [x - y for x, y in zip(a, b)]
        point = statistics.mean(diffs)
        n = len(diffs)
        samples = [
            statistics.mean(rng.choice(diffs) for _ in range(n)) for _ in range(iters)
        ]
    else:
        point = statistics.mean(a) - statistics.mean(b)
        na, nb = len(a), len(b)
        samples = [
            statistics.mean(rng.choice(a) for _ in range(na))
            - statistics.mean(rng.choice(b) for _ in range(nb))
            for _ in range(iters)
        ]
    samples.sort()
    lo = samples[max(0, int(lo_q * iters) - 1)]
    hi = samples[min(iters - 1, int(hi_q * iters))]
    return (point, lo, hi)
