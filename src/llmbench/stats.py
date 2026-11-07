"""Statistics for paired comparison of two systems on the same task set.

Everything here is standard-library only. The functions are cross-checked
against SciPy in ``tests/test_stats.py``.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _log_pmf_half(k: int, n: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            - n * math.log(2))


def _cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2)."""
    return min(1.0, sum(math.exp(_log_pmf_half(i, n)) for i in range(k + 1)))


def paired_counts(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, int, int]:
    """Return (both_correct, a_only, b_only, both_wrong) for 0/1 vectors."""
    if len(a) != len(b):
        raise ValueError("paired vectors must have equal length")
    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n00 = len(a) - n11 - n10 - n01
    return n11, n10, n01, n00


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test from the discordant counts.

    ``b`` = items only system A got right, ``c`` = items only system B got
    right. Under H0 the discordant pairs split 50/50 (exact binomial).
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * _cdf_half(min(b, c), n))


def paired_bootstrap_diff(a: Sequence[int], b: Sequence[int], reps: int = 5000,
                          alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for accuracy(A) - accuracy(B) on paired items.

    Returns (point_estimate, lo, hi).
    """
    n = len(a)
    if n == 0:
        return (float("nan"),) * 3
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        means.append(sum(rng.choices(d, k=n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * reps)]
    hi = means[min(reps - 1, int((1 - alpha / 2) * reps))]
    return sum(d) / n, lo, hi


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def cohen_kappa(r1: Sequence, r2: Sequence) -> float:
    """Cohen's kappa for two raters over categorical labels."""
    if len(r1) != len(r2) or not r1:
        raise ValueError("rating vectors must be non-empty and equal length")
    n = len(r1)
    cats = sorted(set(r1) | set(r2), key=str)
    po = sum(1 for x, y in zip(r1, r2) if x == y) / n
    pe = sum((sum(1 for x in r1 if x == c) / n) * (sum(1 for y in r2 if y == c) / n)
             for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_by_rater: Sequence[Sequence]) -> float:
    """Fleiss' kappa. ``ratings_by_rater[r][i]`` = rater r's label for item i."""
    raters = len(ratings_by_rater)
    if raters < 2:
        raise ValueError("need at least two raters")
    n_items = len(ratings_by_rater[0])
    if any(len(r) != n_items for r in ratings_by_rater) or n_items == 0:
        raise ValueError("all raters must rate the same non-empty item list")
    cats = sorted({x for r in ratings_by_rater for x in r}, key=str)
    counts = [[sum(1 for r in ratings_by_rater if r[i] == c) for c in cats]
              for i in range(n_items)]
    p_i = [(sum(x * x for x in row) - raters) / (raters * (raters - 1)) for row in counts]
    p_bar = sum(p_i) / n_items
    p_j = [sum(row[j] for row in counts) / (n_items * raters) for j in range(len(cats))]
    pe = sum(p * p for p in p_j)
    if pe == 1.0:
        return 1.0
    return (p_bar - pe) / (1 - pe)


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def percentile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]
