"""Monte Carlo study of the paired exact McNemar test.

Used to answer a practical design question: how many tasks are needed before a
given accuracy gap between two systems is reliably detectable, and does the
test keep its nominal false-positive rate?

Paired outcomes are drawn from a 2x2 joint distribution built from the two
marginal accuracies and a correlation ``rho`` between the systems' per-item
correctness (items that are hard for one system tend to be hard for the other).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .stats import mcnemar_exact


def joint_probs(p_a: float, p_b: float, rho: float) -> tuple[float, float, float, float]:
    """Return (p11, p10, p01, p00) for correctness of A and B on one item."""
    s = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    p11 = p_a * p_b + rho * s
    p10 = p_a - p11
    p01 = p_b - p11
    p00 = 1 - p11 - p10 - p01
    cells = (p11, p10, p01, p00)
    if min(cells) < -1e-12:
        raise ValueError(f"rho={rho} is infeasible for p_a={p_a}, p_b={p_b}")
    return tuple(max(0.0, c) for c in cells)  # type: ignore[return-value]


def _binomial(rng: random.Random, n: int, p: float) -> int:
    if hasattr(rng, "binomialvariate"):
        return rng.binomialvariate(n, p)
    return sum(1 for _ in range(n) if rng.random() < p)


@dataclass
class PowerResult:
    n: int
    p_a: float
    p_b: float
    rho: float
    reps: int
    rejection_rate: float   # type-I error if p_a == p_b, power otherwise
    mean_discordant: float


def simulate_rejection_rate(n: int, p_a: float, p_b: float, rho: float = 0.5,
                            reps: int = 2000, alpha: float = 0.05,
                            seed: int = 0) -> PowerResult:
    """Fraction of simulated experiments in which exact McNemar rejects H0."""
    _, p10, p01, _ = joint_probs(p_a, p_b, rho)
    q = p10 + p01
    share = p10 / q if q > 0 else 0.5
    rng = random.Random(seed)
    rejected = 0
    disc_total = 0
    for _ in range(reps):
        d = _binomial(rng, n, q)
        b = _binomial(rng, d, share)
        c = d - b
        disc_total += d
        if mcnemar_exact(b, c) < alpha:
            rejected += 1
    return PowerResult(n, p_a, p_b, rho, reps, rejected / reps, disc_total / reps)
