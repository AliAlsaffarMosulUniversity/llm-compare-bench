import math
import random

import pytest

from llmbench import stats


def test_wilson_matches_known_value():
    lo, hi = stats.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_mcnemar_exact_known_cases():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(5, 5) == 1.0
    assert stats.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_mcnemar_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for b, c in [(3, 9), (10, 20), (50, 31), (0, 7), (120, 90)]:
        ref = scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue
        assert stats.mcnemar_exact(b, c) == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_holm_monotone_and_bounded():
    p = [0.01, 0.04, 0.03, 0.5]
    adj = stats.holm(p)
    assert adj[0] == pytest.approx(0.04)        # 4 * 0.01
    assert adj[2] == pytest.approx(0.09)        # 3 * 0.03
    assert adj[1] == pytest.approx(0.09)        # max(running, 2*0.04)
    assert all(a >= b for a, b in zip(adj, p))
    assert max(adj) <= 1.0


def test_bootstrap_ci_contains_point_and_zero_for_identical():
    a = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    d, lo, hi = stats.paired_bootstrap_diff(a, a, reps=500)
    assert d == lo == hi == 0
    b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    d, lo, hi = stats.paired_bootstrap_diff(a, b, reps=2000, seed=3)
    assert lo <= d <= hi
    assert d == pytest.approx(-0.3)


def test_bootstrap_coverage_is_reasonable():
    rng = random.Random(11)
    covered, trials, n = 0, 150, 120
    for _ in range(trials):
        a = [int(rng.random() < 0.8) for _ in range(n)]
        b = [int(rng.random() < 0.7) for _ in range(n)]
        _, lo, hi = stats.paired_bootstrap_diff(a, b, reps=400, seed=rng.randint(0, 10**6))
        covered += lo <= 0.10 <= hi
    assert 0.85 <= covered / trials <= 1.0


def test_cohen_kappa_extremes():
    assert stats.cohen_kappa([1, 2, 1, 2], [1, 2, 1, 2]) == 1.0
    assert stats.cohen_kappa([1, 1, 2, 2], [2, 2, 1, 1]) == -1.0
    # textbook example: 20 yes/no items, po=0.7, pe=0.5 -> 0.4
    r1 = ["y"] * 10 + ["n"] * 10
    r2 = ["y"] * 7 + ["n"] * 3 + ["y"] * 3 + ["n"] * 7
    assert stats.cohen_kappa(r1, r2) == pytest.approx(0.4)


def test_fleiss_perfect_and_known():
    same = [[1, 2, 0, 1], [1, 2, 0, 1], [1, 2, 0, 1]]
    assert stats.fleiss_kappa(same) == pytest.approx(1.0)
    # Wikipedia worked example (10 subjects, 14 raters, 5 categories) -> 0.210
    rows = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    raters = [[] for _ in range(14)]
    for row in rows:
        labels = [c for c, k in enumerate(row, 1) for _ in range(k)]
        for r, lab in zip(raters, labels):
            r.append(lab)
    assert stats.fleiss_kappa(raters) == pytest.approx(0.210, abs=1e-3)
