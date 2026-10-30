"""Holm correction, one-sided p-values, and verdicts for the precommitted hypotheses.

Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import precommitted
from aib_analysis.survey_analysis.precommitted import (
    PRECOMMITTED_HYPOTHESES,
    HypothesisResult,
    PrecommittedHypothesis,
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


def _hypothesis(sign: int) -> PrecommittedHypothesis:
    return PrecommittedHypothesis(key="k", statement="s", predicted_sign=sign, prior_evidence="e")


def test_verdicts():
    supported = HypothesisResult(_hypothesis(1), _result(0.4, 0.01), 0.005, 0.015)
    opposite = HypothesisResult(_hypothesis(1), _result(-0.4, 0.01), 0.995, 1.0)
    weak = HypothesisResult(_hypothesis(1), _result(0.1, 0.5), 0.25, 0.5)
    untestable = HypothesisResult(_hypothesis(1), _result(None, None), None, None)
    assert supported.verdict == "Supported"
    assert opposite.verdict.startswith("Not supported (the estimate pointed the opposite way")
    assert weak.verdict == "Not supported"
    assert untestable.verdict.startswith("Not testable")


def test_untestable_hypothesis_counts_as_p_one_in_the_family():
    keys = [hypothesis.key for hypothesis in PRECOMMITTED_HYPOTHESES]
    pool = [make_feature(variables={keys[0]: float(index % 2)}, score=float(index % 2)) for index in range(20)]
    results = precommitted.evaluate_precommitted(pool)
    assert results[0].correlation.coefficient == pytest.approx(1.0)
    assert all(result.one_sided_p is None for result in results[1:])
    # Only one testable hypothesis, but Holm still multiplies by the full family size.
    assert results[0].holm_p == pytest.approx(min(1.0, 3 * results[0].one_sided_p))
