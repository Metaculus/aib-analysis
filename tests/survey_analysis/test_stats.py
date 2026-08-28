"""Statistics correctness for survey_analysis.stats.

The report leans entirely on these numbers, so each test pins a specific
guarantee: the right test is chosen per variable kind, the coefficient matches
scipy to full precision, degenerate inputs return "insufficient" instead of NaN,
and the plain-language direction/significance helpers use the documented cutoffs.
Runs without env vars or local data.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats as sps

from aib_analysis.survey_analysis import stats
from aib_analysis.survey_analysis.stats import CorrelationResult

from tests.survey_analysis.conftest import make_feature


# --------------------------------------------------------------------------- #
# correlate_with_score: method selection + numeric agreement with scipy
# --------------------------------------------------------------------------- #
def test_binary_uses_pearson_and_equals_pointbiserial():
    # A 0/1 trait vs a continuous score: the docstring claims Pearson r equals
    # the point-biserial r. Verify against scipy's dedicated function.
    xs = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    ys = [1.0, 4.0, 0.5, 3.0, 2.0, 5.0, 1.5, 6.0, 4.5, 0.0]
    features = [
        make_feature(variables={"frontier": float(x)}, score=y) for x, y in zip(xs, ys)
    ]
    result = stats.correlate_with_score(features, "frontier")
    r_pb, p_pb = sps.pointbiserialr(xs, ys)

    assert result.method == "Pearson"
    assert result.n == 10
    assert result.coefficient == pytest.approx(r_pb, abs=1e-12)
    assert result.p_value == pytest.approx(p_pb, abs=1e-12)


def test_ordinal_uses_spearman_and_matches_scipy():
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    ys = [0.0, 0.2, 0.1, 0.9, 0.8, 1.5, 2.0, 1.9]  # monotone-ish, with ties broken
    features = [
        make_feature(variables={"hours_mid": float(x)}, score=y) for x, y in zip(xs, ys)
    ]
    result = stats.correlate_with_score(features, "hours_mid")
    r_sp, p_sp = sps.spearmanr(xs, ys)

    assert result.method == "Spearman"
    assert result.coefficient == pytest.approx(r_sp, abs=1e-12)
    assert result.p_value == pytest.approx(p_sp, abs=1e-12)


def test_perfect_positive_linear_recovers_r_one():
    features = [
        make_feature(variables={"n_research_sources": float(i)}, score=2.0 * i + 3.0)
        for i in range(10)
    ]
    # n_research_sources is a "count" -> Spearman; a strictly increasing score
    # gives a perfect rank correlation of +1.
    result = stats.correlate_with_score(features, "n_research_sources")
    assert result.coefficient == pytest.approx(1.0, abs=1e-12)


def test_perfect_negative_recovers_r_minus_one():
    features = [
        make_feature(variables={"hours_mid": float(i)}, score=-1.0 * i)
        for i in range(10)
    ]
    result = stats.correlate_with_score(features, "hours_mid")
    assert result.coefficient == pytest.approx(-1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Pairing / missing-data handling
# --------------------------------------------------------------------------- #
def test_none_values_are_dropped_pairwise():
    features = [
        make_feature(variables={"hours_mid": 1.0}, score=1.0),
        make_feature(variables={"hours_mid": None}, score=5.0),   # value missing
        make_feature(variables={"hours_mid": 2.0}, score=None),   # score missing
        make_feature(variables={"hours_mid": 3.0}, score=3.0),
    ] + [make_feature(variables={"hours_mid": float(i)}, score=float(i)) for i in range(6)]
    result = stats.correlate_with_score(features, "hours_mid")
    # Only the fully-paired rows count: 2 explicit + 6 filler = 8.
    assert result.n == 8


def test_too_few_observations_returns_insufficient():
    features = [
        make_feature(variables={"hours_mid": float(i)}, score=float(i)) for i in range(7)
    ]
    result = stats.correlate_with_score(features, "hours_mid")
    assert result.coefficient is None and result.p_value is None
    assert result.method == "(insufficient data)"
    assert result.n == 7  # below MIN_N_FOR_CORRELATION (8)


def test_constant_predictor_returns_insufficient_not_nan():
    # All-equal x makes Pearson/Spearman undefined (NaN). Must be caught up front.
    features = [make_feature(variables={"frontier": 1.0}, score=float(i)) for i in range(12)]
    result = stats.correlate_with_score(features, "frontier")
    assert result.coefficient is None
    assert result.method == "(insufficient data)"
    assert result.n == 12


def test_nan_coefficient_is_normalized_to_none():
    # Constant score -> scipy returns NaN r; the code should surface None, never NaN.
    features = [
        make_feature(variables={"hours_mid": float(i)}, score=5.0) for i in range(12)
    ]
    result = stats.correlate_with_score(features, "hours_mid")
    assert result.coefficient is None or not math.isnan(result.coefficient)


# --------------------------------------------------------------------------- #
# CorrelationResult presentation helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "coef, expected_fragment",
    [
        (0.05, "no clear relationship"),
        (-0.05, "no clear relationship"),
        (0.2, "a weak link to higher"),
        (-0.2, "a weak link to lower"),
        (0.4, "a moderate link to higher"),
        (0.7, "a strong link to higher"),
        (-0.7, "a strong link to lower"),
    ],
)
def test_direction_phrase_cutoffs(coef, expected_fragment):
    result = CorrelationResult("k", "L", "Pearson", coef, 0.01, 30)
    assert expected_fragment in result.direction_phrase()


def test_direction_phrase_boundaries_are_inclusive_lower():
    # 0.1 is "weak" (>= 0.1), 0.3 is "moderate", 0.5 is "strong".
    assert "weak" in CorrelationResult("k", "L", "Pearson", 0.1, 0.01, 30).direction_phrase()
    assert "moderate" in CorrelationResult("k", "L", "Pearson", 0.3, 0.01, 30).direction_phrase()
    assert "strong" in CorrelationResult("k", "L", "Pearson", 0.5, 0.01, 30).direction_phrase()


def test_direction_phrase_none_coefficient():
    assert CorrelationResult("k", "L", "m", None, None, 0).direction_phrase() == "no estimate"
