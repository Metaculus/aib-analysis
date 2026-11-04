"""Holm correction, one-sided p-values, and verdicts for the precommitted hypotheses.

Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import precommitted, report
from aib_analysis.survey_analysis.precommitted import (
    PRECOMMITTED_HYPOTHESES,
    PRECOMMITTED_KEYS,
    holm_adjust,
    one_sided_p,
)
from aib_analysis.survey_analysis.stats import CorrelationResult
from tests.survey_analysis.conftest import make_feature


def _result(coefficient: float | None, p_value: float | None) -> CorrelationResult:
    return CorrelationResult(
        key="k", label="k", method="Pearson", coefficient=coefficient, p_value=p_value, n=40
    )


def test_holm_matches_hand_computed_example():
    # p.adjust(c(0.01, 0.04, 0.03), "holm") == c(0.03, 0.06, 0.06)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_keeps_order_monotone():
    # A later step can never come out smaller than an earlier one.
    assert holm_adjust([0.5, 0.001, 0.4]) == pytest.approx([0.8, 0.003, 0.8])


def test_holm_is_capped_at_one():
    assert holm_adjust([0.6, 0.001, 0.7]) == pytest.approx([1.0, 0.003, 1.0])


def test_one_sided_p_halves_when_direction_matches():
    assert one_sided_p(_result(0.3, 0.04), predicted_sign=1) == pytest.approx(0.02)
    assert one_sided_p(_result(-0.3, 0.04), predicted_sign=-1) == pytest.approx(0.02)


def test_one_sided_p_is_large_when_direction_is_opposite():
    assert one_sided_p(_result(-0.3, 0.04), predicted_sign=1) == pytest.approx(0.98)


def test_one_sided_p_missing_when_untestable():
    assert one_sided_p(_result(None, None), predicted_sign=1) is None


def _pool_tracking_every_hypothesis(size: int = 30) -> list:
    pool = []
    for index in range(size):
        variables = {
            hypothesis.key: float(index % 2 if hypothesis.predicted_sign > 0 else 1 - index % 2)
            for hypothesis in PRECOMMITTED_HYPOTHESES
        }
        variables["uses_capping"] = float(index % 3 == 0)
        pool.append(make_feature(variables=variables, score=float(index % 2) + 0.01 * index))
    return pool


def test_feature_stats_gives_precommitted_features_one_sided_p_and_holm_q():
    pool = _pool_tracking_every_hypothesis()
    results, q_by_key = report._feature_stats(pool)
    for hypothesis_result in precommitted.evaluate_precommitted(pool):
        key = hypothesis_result.hypothesis.key
        assert results[key].p_value == pytest.approx(hypothesis_result.one_sided_p)
        assert q_by_key[key] == pytest.approx(hypothesis_result.holm_p)


def test_exploratory_false_discovery_family_excludes_precommitted_features():
    pool = _pool_tracking_every_hypothesis()
    results, q_by_key = report._feature_stats(pool)
    exploratory = [key for key, res in results.items() if key not in PRECOMMITTED_KEYS and res.p_value is not None]
    assert exploratory == ["uses_capping"]
    # With one exploratory feature, the Benjamini-Hochberg q equals its own p.
    assert q_by_key["uses_capping"] == pytest.approx(results["uses_capping"].p_value)


def test_untestable_hypothesis_counts_as_p_one_in_the_family():
    keys = [hypothesis.key for hypothesis in PRECOMMITTED_HYPOTHESES]
    pool = [make_feature(variables={keys[0]: float(index % 2)}, score=float(index % 2)) for index in range(20)]
    results = precommitted.evaluate_precommitted(pool)
    assert results[0].correlation.coefficient == pytest.approx(1.0)
    assert all(result.one_sided_p is None for result in results[1:])
    # Only one testable hypothesis, but Holm still multiplies by the full family size.
    assert results[0].holm_p == pytest.approx(min(1.0, 3 * results[0].one_sided_p))
