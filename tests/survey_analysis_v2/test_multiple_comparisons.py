"""Benjamini-Hochberg q-values and the evidence-summary wiring.

The report now tests many features at once and controls the false-discovery rate
with BH-adjusted q-values, calling a result "significant" only when q < 0.05.
Because that correction is the whole defense against spurious findings, these
tests pin the math (against hand-computed references and the known BH
properties) and the report's use of it. Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis_v2 import config, report
from aib_analysis.survey_analysis_v2.stats import CorrelationResult


# --------------------------------------------------------------------------- #
# _benjamini_hochberg: numeric correctness
# --------------------------------------------------------------------------- #
def _bh_reference(pvalues: list[float]) -> list[float]:
    """Independent step-up reference (R's p.adjust(method='BH')), input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [1.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, pvalues[i] * m / (rank + 1))
        out[i] = min(running, 1.0)
    return out


def test_bh_matches_hand_computed_example():
    # p.adjust(c(0.001, 0.5, 0.7, 0.9), "BH") == c(0.004, 0.9, 0.9, 0.9)
    q = report._benjamini_hochberg([0.001, 0.5, 0.7, 0.9])
    assert q == pytest.approx([0.004, 0.9, 0.9, 0.9])


def test_bh_all_equal_pvalues():
    # Equal p across m tests -> every q equals p (m/rank cancels the smallest).
    q = report._benjamini_hochberg([0.05, 0.05, 0.05, 0.05, 0.05])
    assert q == pytest.approx([0.05] * 5)


@pytest.mark.parametrize(
    "pvalues",
    [
        [0.04, 0.01, 0.03, 0.005, 0.02],  # unsorted
        [0.2, 0.8, 0.01, 0.5, 0.9, 0.001, 0.3],
        [0.9, 0.9, 0.9],
        [0.5],
    ],
)
def test_bh_matches_reference_and_preserves_input_order(pvalues):
    q = report._benjamini_hochberg(pvalues)
    assert q == pytest.approx(_bh_reference(pvalues))
    assert len(q) == len(pvalues)


def test_bh_never_below_raw_p_and_capped_at_one():
    pvalues = [0.001, 0.02, 0.4, 0.6, 0.99]
    q = report._benjamini_hochberg(pvalues)
    for p_value, q_value in zip(pvalues, q):
        assert q_value >= p_value - 1e-12  # adjustment only inflates
        assert q_value <= 1.0


def test_bh_monotone_in_p_rank():
    pvalues = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5, 0.9]
    q = report._benjamini_hochberg(pvalues)
    paired = sorted(zip(pvalues, q))
    assert all(paired[i][1] <= paired[i + 1][1] + 1e-12 for i in range(len(paired) - 1))


def test_bh_empty():
    assert report._benjamini_hochberg([]) == []


# --------------------------------------------------------------------------- #
# Significance now keys off q, not raw p
# --------------------------------------------------------------------------- #
def test_correlation_sentence_significant_only_when_q_below_threshold():
    # Same tiny raw p; significance flips entirely on the q-value.
    result = CorrelationResult("frontier", "Frontier", "Pearson", 0.4, 0.001, 40)

    sig = report._correlation_sentence(result, q_value=0.02)
    assert "significant after false-discovery correction" in sig
    assert "q = 0.020" in sig

    not_sig = report._correlation_sentence(result, q_value=0.30)
    assert "significant" not in not_sig
    assert "q = 0.300" in not_sig


def test_correlation_sentence_threshold_is_config_driven():
    result = CorrelationResult("frontier", "Frontier", "Pearson", 0.4, 0.001, 40)
    just_under = report._correlation_sentence(result, q_value=config.EVIDENCE_SIGNIFICANT_Q - 1e-6)
    at_threshold = report._correlation_sentence(result, q_value=config.EVIDENCE_SIGNIFICANT_Q)
    assert "significant after false-discovery correction" in just_under
    assert "significant" not in at_threshold  # strict <


def test_correlation_sentence_handles_missing_q():
    result = CorrelationResult("frontier", "Frontier", "Pearson", 0.4, 0.001, 40)
    sentence = report._correlation_sentence(result, q_value=None)
    assert "q = n/a" in sentence
    assert "significant" not in sentence


def test_fmt_pq_small_values():
    assert report._fmt_pq(None) == "n/a"
    assert report._fmt_pq(0.0004) == "<0.001"
    assert report._fmt_pq(0.0123) == "0.012"


# --------------------------------------------------------------------------- #
# _feature_stats: one shared q across the whole tested family
# --------------------------------------------------------------------------- #
def test_all_feature_keys_are_unique_and_resolvable():
    from aib_analysis.survey_analysis_v2.features import variable_spec

    pairs = report._all_feature_keys()
    keys = [key for key, _label in pairs]
    assert len(keys) == len(set(keys)), "duplicate feature key in evidence family"
    for key, _label in pairs:
        variable_spec(key)  # raises if a key cannot be correlated


def test_feature_stats_assigns_q_only_to_estimable_features():
    from tests.survey_analysis_v2.conftest import make_feature

    # A pool where every feature is estimable; q map should cover the same keys
    # that produced a p-value, and every q sits in [0, 1].
    pool = [
        make_feature(
            variables={key: float(i % 2) for key, _ in report._all_feature_keys()},
            score=float(i),
        )
        for i in range(30)
    ]
    results, q_by_key = report._feature_stats(pool)
    estimable = {k for k, r in results.items() if r.p_value is not None}
    assert set(q_by_key) == estimable
    assert all(0.0 <= q <= 1.0 for q in q_by_key.values())
