"""Config self-consistency guards.

config.py is the single source of truth: the report, the charts, and the two
review docs are all generated from it. A silent drift here (an ordinal option
with no midpoint, a boolean whose match substring matches nothing, a correlation
key with no spec) produces a wrong-but-plausible report, never an error. These
tests turn each such drift into a failure. Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import config
from aib_analysis.survey_analysis.features import NUMERIC_VARIABLE_SPECS, variable_spec


# --------------------------------------------------------------------------- #
# Column accounting: the review doc claims "every one of the 24 columns"
# --------------------------------------------------------------------------- #
def test_column_count_matches_review_doc_claim():
    assert len(config.COLUMNS) == 24


def test_every_column_is_charted_excluded_or_used_for_joining():
    charted = {spec.slug for spec in config.QUESTION_SPECS}
    excluded = set(config.EXCLUDED_COLUMNS)
    # Remaining columns are the ones used only for joining / feature derivation.
    join_or_feature = {"timestamp", "confirm", "bot_name"}  # subset of excluded already
    accounted = charted | excluded
    for slug in config.COLUMNS:
        assert slug in accounted or slug in join_or_feature, f"{slug} unaccounted for"


def test_charted_and_excluded_do_not_overlap():
    charted = {spec.slug for spec in config.QUESTION_SPECS}
    assert charted.isdisjoint(config.EXCLUDED_COLUMNS)


# --------------------------------------------------------------------------- #
# Midpoint maps must line up 1:1 with the ordinal vocab and be monotone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "slug, midpoint_map",
    [
        ("hours", config.HOURS_MIDPOINT),
        ("llm_calls", config.LLM_CALLS_MIDPOINT),
        ("cost_per_q", config.COST_MIDPOINT),
        ("iterations", config.ITERATIONS_MIDPOINT),
    ],
)
def test_midpoint_keys_match_vocab_exactly(slug, midpoint_map):
    # Any option without a midpoint would be silently dropped from correlations
    # (unless declared unordered); any stray midpoint key would be dead config.
    unordered = set(config.UNORDERED_OPTIONS.get(slug, []))
    assert set(midpoint_map) == set(config.SINGLE_SELECT_VOCAB[slug]) - unordered


@pytest.mark.parametrize(
    "slug, midpoint_map",
    [
        ("hours", config.HOURS_MIDPOINT),
        ("llm_calls", config.LLM_CALLS_MIDPOINT),
        ("cost_per_q", config.COST_MIDPOINT),
        ("iterations", config.ITERATIONS_MIDPOINT),
    ],
)
def test_midpoints_are_non_decreasing_along_ordinal_order(slug, midpoint_map):
    # The ordinal order and the midpoint magnitudes must agree, otherwise the
    # chart axis and the Spearman input rank respondents differently.
    ordered = [midpoint_map[opt] for opt in config.ORDINAL_ORDER[slug]]
    assert ordered == sorted(ordered), f"{slug} midpoints not monotone: {ordered}"


# --------------------------------------------------------------------------- #
# Ordinal ordering must be a permutation of the single-select vocab
# --------------------------------------------------------------------------- #
def test_ordinal_order_matches_single_select_vocab():
    for slug, order in config.ORDINAL_ORDER.items():
        unordered = set(config.UNORDERED_OPTIONS.get(slug, []))
        assert set(order) == set(config.SINGLE_SELECT_VOCAB[slug]) - unordered, slug


def test_unordered_options_are_real_options_of_ordinal_questions():
    for slug, options in config.UNORDERED_OPTIONS.items():
        assert slug in config.ORDINAL_ORDER, slug
        assert set(options) <= set(config.SINGLE_SELECT_VOCAB[slug]), slug


# --------------------------------------------------------------------------- #
# Boolean features must reference real columns and real substrings
# --------------------------------------------------------------------------- #
def test_boolean_feature_substring_exists_in_its_column_vocab():
    for feature in config.BOOLEAN_FEATURES:
        vocab = (
            config.MULTISELECT_VOCAB.get(feature.column_slug)
            or config.SINGLE_SELECT_VOCAB.get(feature.column_slug)
        )
        assert vocab is not None, f"{feature.key}: unknown column {feature.column_slug}"
        hit = any(feature.match_substring.lower() in opt.lower() for opt in vocab)
        assert hit, f"{feature.key}: substring {feature.match_substring!r} matches no option"


def test_boolean_feature_keys_unique():
    keys = [f.key for f in config.BOOLEAN_FEATURES]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# Correlation keys must all resolve to a spec
# --------------------------------------------------------------------------- #
def test_every_correlation_key_resolves():
    for spec in config.QUESTION_SPECS:
        for key in spec.correlations:
            resolved = variable_spec(key)  # raises KeyError if unknown
            assert resolved.kind in {"binary", "count", "ordinal", "continuous"}


def test_every_question_spec_slug_is_a_real_column_or_explained():
    for spec in config.QUESTION_SPECS:
        if spec.kind == "derived":
            assert spec.slug in config.DERIVED_SPEC_SOURCES, spec.slug
        else:
            assert spec.slug in config.COLUMNS, spec.slug


def test_charted_correlations_cover_every_measured_feature():
    # The evidence-summary bullets and the in-body correlation charts must be the
    # same set: every measured feature is charted under exactly one question, and
    # no question charts a feature that is not a measured feature. This keeps the
    # summary list and the body one-to-one so neither can silently gain an entry
    # the other lacks.
    measured = {f.key for f in config.BOOLEAN_FEATURES}
    measured |= {s.key for s in NUMERIC_VARIABLE_SPECS}
    charted = [key for spec in config.QUESTION_SPECS for key in spec.correlations]

    assert len(charted) == len(set(charted)), "a feature is charted under more than one question"
    assert set(charted) == measured, (
        f"charted-only: {set(charted) - measured}; bullet-only: {measured - set(charted)}"
    )


def test_numeric_variable_specs_have_known_kinds():
    for spec in NUMERIC_VARIABLE_SPECS:
        assert spec.kind in {"binary", "count", "ordinal", "continuous"}


# --------------------------------------------------------------------------- #
# Group config
# --------------------------------------------------------------------------- #
def test_group_labels_and_colors_cover_chart_groups():
    for group in config.CHART_GROUP_ORDER:
        assert group in config.GROUP_LABELS
        assert group in config.GROUP_COLORS


def test_chart_group_order_is_group_order_plus_everyone():
    assert config.CHART_GROUP_ORDER == ["everyone"] + config.GROUP_ORDER
