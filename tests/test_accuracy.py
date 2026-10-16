import math

import pytest

from aib_analysis.math.accuracy import (
    base_rate_geometric_mean_probability,
    binary_probability_on_outcome,
    field_probability_multiplier,
    geometric_mean_probability,
    geometric_mean_probability_ci,
    paired_probability_ratio,
    yes_rate,
)


def test_binary_probability_on_outcome() -> None:
    assert binary_probability_on_outcome(0.8, True) == 0.8
    assert binary_probability_on_outcome(0.8, False) == pytest.approx(0.2)


def test_geometric_mean_probability_with_and_without_weights() -> None:
    assert geometric_mean_probability([0.5, 0.5]) == pytest.approx(0.5)
    assert geometric_mean_probability([0.25, 1.0]) == pytest.approx(0.5)
    assert geometric_mean_probability([0.25, 1.0], [1.0, 3.0]) == pytest.approx(0.25 ** 0.25)


@pytest.mark.parametrize(
    ("probabilities", "weights"),
    [([], None), ([0.0], None), ([0.5], [1.0, 1.0]), ([1.2], None)],
)
def test_geometric_mean_probability_rejects_bad_input(probabilities: list[float], weights: list[float] | None) -> None:
    with pytest.raises(ValueError):
        geometric_mean_probability(probabilities, weights)


def test_geometric_mean_probability_ci_contains_the_estimate() -> None:
    probabilities = [0.9, 0.7, 0.6, 0.8, 0.3, 0.75]
    lower, upper = geometric_mean_probability_ci(probabilities)
    assert lower < geometric_mean_probability(probabilities) < upper
    assert geometric_mean_probability_ci([0.5]) is None


def test_paired_probability_ratio() -> None:
    ratio, ci = paired_probability_ratio([0.6, 0.8, 0.5], [0.3, 0.4, 0.25])
    assert ratio == pytest.approx(2.0)
    assert ci == pytest.approx((2.0, 2.0))
    with pytest.raises(ValueError):
        paired_probability_ratio([0.5], [0.5, 0.5])


def test_base_rate_forecaster() -> None:
    resolutions = [True, False, False, False]
    assert yes_rate(resolutions) == 0.25
    expected = math.exp((math.log(0.25) + 3 * math.log(0.75)) / 4)
    assert base_rate_geometric_mean_probability(resolutions) == pytest.approx(expected)
    assert base_rate_geometric_mean_probability(resolutions) > 0.5


def test_field_probability_multiplier_undoes_numeric_halving() -> None:
    binary_peer = 100 * math.log(1.2)
    numeric_peer = 100 * math.log(1.2) / 2
    assert field_probability_multiplier([binary_peer, numeric_peer], [1.0, 1.0], [False, True]) == pytest.approx(1.2)
    weighted_peer = 0.5 * 100 * math.log(1.5)
    assert field_probability_multiplier([weighted_peer], [0.5], [False]) == pytest.approx(1.5)
