import pytest

from llmbench import power


def test_joint_probs_valid_and_marginals():
    p11, p10, p01, p00 = power.joint_probs(0.8, 0.7, 0.5)
    assert p11 + p10 == pytest.approx(0.8)
    assert p11 + p01 == pytest.approx(0.7)
    assert p11 + p10 + p01 + p00 == pytest.approx(1.0)


def test_infeasible_rho_rejected():
    with pytest.raises(ValueError):
        power.joint_probs(0.9, 0.1, 0.99)


def test_type_i_error_not_inflated():
    r = power.simulate_rejection_rate(200, 0.8, 0.8, rho=0.5, reps=3000, seed=5)
    assert r.rejection_rate <= 0.06       # exact test is conservative


def test_power_increases_with_n_and_effect():
    small = power.simulate_rejection_rate(100, 0.8, 0.7, reps=1500, seed=1).rejection_rate
    large = power.simulate_rejection_rate(400, 0.8, 0.7, reps=1500, seed=1).rejection_rate
    tiny_effect = power.simulate_rejection_rate(400, 0.8, 0.78, reps=1500, seed=1).rejection_rate
    assert large > small
    assert large > tiny_effect
