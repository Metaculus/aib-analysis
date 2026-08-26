"""Report-side helpers: score labeling, numeric binning, formatting.

These shape what the reader sees. The binning helper in particular has fiddly
edge behavior (equal-label merges, small-sample fallback to 2 bins) that is easy
to break silently. Runs without env vars or local data.
"""

from __future__ import annotations

from aib_analysis.survey_analysis_v2 import plots, report

from tests.survey_analysis_v2.conftest import make_cell, make_feature


# --------------------------------------------------------------------------- #
# _fmt_p / _fmt_num
# --------------------------------------------------------------------------- #
def test_fmt_p_buckets():
    assert report._fmt_p(None) == "n/a"
    assert report._fmt_p(0.0005) == "p < 0.001"
    assert report._fmt_p(0.0123) == "p = 0.012"


def test_fmt_num_integer_vs_decimal():
    assert report._fmt_num(4.0) == "4"
    assert report._fmt_num(3.5) == "3.5"


def test_correlation_sentence_shows_q_and_significance_from_q():
    from aib_analysis.survey_analysis_v2.stats import CorrelationResult

    # One flowing sentence; significance is based on the q-value, and q is shown.
    sig = CorrelationResult("k", "Trait", "Pearson", 0.4, 0.01, 30)
    sentence = report._correlation_sentence(sig, q_value=0.02)
    assert sentence.startswith('"Trait" shows ')
    assert "q = 0.020" in sentence
    assert "significant after false-discovery correction" in sentence
    # One flowing clause: the label leads straight into "shows ..." with the stats
    # in a trailing parenthetical, not the old "label: stats. This shows ..." form.
    assert ". This shows" not in sentence
    assert sentence.rstrip().endswith(").")

    ns = CorrelationResult("k", "Trait", "Pearson", 0.4, 0.01, 30)
    sentence_ns = report._correlation_sentence(ns, q_value=0.30)
    assert "q = 0.300" in sentence_ns
    assert "significant" not in sentence_ns

    none = CorrelationResult("k", "Trait", "(insufficient data)", None, None, 3, note="too few")
    assert "too few observations" in report._correlation_sentence(none, q_value=None)


# --------------------------------------------------------------------------- #
# _labeled_scores_binary
# --------------------------------------------------------------------------- #
def test_labeled_scores_binary_splits_by_value():
    features = [
        make_feature(variables={"frontier": 1.0}, score=5.0),
        make_feature(variables={"frontier": 1.0}, score=7.0),
        make_feature(variables={"frontier": 0.0}, score=1.0),
        make_feature(variables={"frontier": None}, score=9.0),  # excluded
    ]
    labeled = report._labeled_scores_binary(features, "frontier")
    as_dict = dict(labeled)
    assert as_dict["Yes"] == [5.0, 7.0]
    assert as_dict["No"] == [1.0]


# --------------------------------------------------------------------------- #
# _labeled_scores_numeric: quantile binning
# --------------------------------------------------------------------------- #
def test_numeric_binning_produces_ordered_nonempty_bins():
    # 12 distinct values -> 3 bins of 4, labeled by their value range.
    features = [
        make_feature(variables={"n_research_sources": float(i)}, score=float(i))
        for i in range(12)
    ]
    labeled = report._labeled_scores_numeric(features, "n_research_sources", n_bins=3)
    assert len(labeled) == 3
    assert all(scores for _label, scores in labeled)
    # Every score assigned exactly once.
    assigned = [s for _l, scores in labeled for s in scores]
    assert sorted(assigned) == [float(i) for i in range(12)]


def test_numeric_binning_small_sample_falls_back_to_two_bins():
    # Below 2*MIN_CELL_SIZE total, n_bins collapses to 2.
    n = 2 * plots.MIN_CELL_SIZE - 1
    features = [
        make_feature(variables={"n_research_sources": float(i)}, score=float(i))
        for i in range(n)
    ]
    labeled = report._labeled_scores_numeric(features, "n_research_sources", n_bins=3)
    assert len(labeled) <= 2


def test_numeric_binning_merges_equal_labels():
    # All identical values -> a single merged bin (labels would otherwise repeat).
    features = [
        make_feature(variables={"n_research_sources": 2.0}, score=float(i))
        for i in range(12)
    ]
    labeled = report._labeled_scores_numeric(features, "n_research_sources", n_bins=3)
    assert len(labeled) == 1
    assert labeled[0][0] == "2"
    assert len(labeled[0][1]) == 12


# --------------------------------------------------------------------------- #
# _labeled_scores_ordinal
# --------------------------------------------------------------------------- #
def test_labeled_scores_ordinal_groups_by_bucket_in_order():
    features = [
        make_feature(cells={"llm_calls": make_cell(matched=["1"])}, score=1.0),
        make_feature(cells={"llm_calls": make_cell(matched=["2-5"])}, score=2.0),
        make_feature(cells={"llm_calls": make_cell(matched=["2-5"])}, score=4.0),
    ]
    labeled = report._labeled_scores_ordinal(features, "llm_calls")
    labels = [label for label, _ in labeled]
    # Preserves full ordinal order (even empty buckets), first bucket first.
    assert labels == report.config.ORDINAL_ORDER["llm_calls"]
    as_dict = dict(labeled)
    assert as_dict["1"] == [1.0]
    assert as_dict["2-5"] == [2.0, 4.0]


# --------------------------------------------------------------------------- #
# _group_counts
# --------------------------------------------------------------------------- #
def test_group_counts_counts_each_membership():
    features = [
        make_feature(groups=["winner", "top_10"]),
        make_feature(groups=["winner"]),
        make_feature(groups=["non_winner"]),
    ]
    counts = report._group_counts(features)
    assert counts["winner"] == 2
    assert counts["top_10"] == 1
    assert counts["non_winner"] == 1
