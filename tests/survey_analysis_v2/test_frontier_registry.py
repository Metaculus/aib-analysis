"""Frontier classification and model-registry integrity.

Frontier status is the report's single most consequential derived label, so this
locks down the rule (high-power AND released after the cutoff, strict), the
substring matcher that turns free text into registry hits, and the registry's own
internal consistency. Runs without env vars or local data.
"""

from __future__ import annotations

import re
from datetime import timedelta

from aib_analysis.survey_analysis_v2 import config, parsing
from aib_analysis.survey_analysis_v2.config import MODEL_REGISTRY, FRONTIER_RELEASE_CUTOFF


# --------------------------------------------------------------------------- #
# The frontier rule itself
# --------------------------------------------------------------------------- #
def test_frontier_requires_both_high_power_and_recent():
    after = FRONTIER_RELEASE_CUTOFF + timedelta(days=1)
    before = FRONTIER_RELEASE_CUTOFF - timedelta(days=1)
    assert config._m("x", high_power=True, release_date=after).is_frontier is True
    assert config._m("x", high_power=True, release_date=before).is_frontier is False
    assert config._m("x", high_power=False, release_date=after).is_frontier is False
    assert config._m("x", high_power=False, release_date=before).is_frontier is False


def test_frontier_cutoff_is_strict_exclusive():
    on_cutoff = config._m("x", high_power=True, release_date=FRONTIER_RELEASE_CUTOFF)
    assert on_cutoff.released_after_cutoff is False  # strictly greater-than
    assert on_cutoff.is_frontier is False


def test_unknown_release_date_never_frontier():
    assert config._m("x", high_power=True, release_date=None).is_frontier is False


def test_registry_frontier_matches_rule_exactly():
    # Every registry row's is_frontier must equal (high_power AND date>cutoff).
    for model in MODEL_REGISTRY:
        expected = (
            model.high_power
            and model.release_date is not None
            and model.release_date > FRONTIER_RELEASE_CUTOFF
        )
        assert model.is_frontier == expected, model.display


# --------------------------------------------------------------------------- #
# Registry-wide invariants that keep matching correct
# --------------------------------------------------------------------------- #
def test_no_duplicate_normalized_keys():
    keys = [m.normalized_key for m in MODEL_REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate normalized_key breaks maximal-match logic"


def test_no_empty_normalized_keys():
    assert all(m.normalized_key for m in MODEL_REGISTRY)


def test_small_tier_markers_are_never_high_power():
    # Whole-word markers only: "mini" inside "geMINI" must not trip this.
    marker = re.compile(r"\b(mini|nano|flash|fast|haiku|lite|small|free)\b")
    for model in MODEL_REGISTRY:
        if marker.search(model.display.lower()):
            assert model.high_power is False, f"{model.display} carries a small-tier marker"


# --------------------------------------------------------------------------- #
# Token matching: longest-match / maximal containment
# --------------------------------------------------------------------------- #
def test_base_model_not_matched_inside_more_specific_sibling():
    # "GPT-5.4" must not also register a bare "GPT-5" hit.
    hits = [m.display for m in parsing.match_model_token("GPT-5.4")]
    assert hits == ["GPT-5.4"]


def test_dotted_version_does_not_match_shorter_base():
    # "GPT-5.5" contains "gpt5" textually, but must resolve to only GPT-5.5.
    hits = [m.display for m in parsing.match_model_token("GPT-5.5")]
    assert hits == ["GPT-5.5"]


def test_bare_base_model_still_matches_itself():
    hits = [m.display for m in parsing.match_model_token("GPT-5")]
    assert hits == ["GPT-5"]


def test_mini_variant_matches_mini_not_base():
    hits = [m.display for m in parsing.match_model_token("GPT-5.4 mini")]
    assert hits == ["GPT-5.4 mini"]


def test_unknown_token_matches_nothing():
    assert parsing.match_model_token("SomeBrandNewModelX") == []


def test_empty_token_matches_nothing():
    assert parsing.match_model_token("") == []
    assert parsing.match_model_token("   ") == []


# --------------------------------------------------------------------------- #
# is_frontier_final on realistic cell text
# --------------------------------------------------------------------------- #
def test_frontier_final_any_frontier_flips_true():
    assert parsing.is_frontier_final("GPT-4o, GPT-5.4") is True
    assert parsing.is_frontier_final("Claude Sonnet 4.5, gpt-5.1") is True


def test_frontier_final_all_non_frontier_stays_false():
    assert parsing.is_frontier_final("GPT-4o") is False
    assert parsing.is_frontier_final("Claude Sonnet 4.6") is False
    assert parsing.is_frontier_final("GPT-5") is False  # high power but pre-cutoff


def test_frontier_final_blank_is_false():
    assert parsing.is_frontier_final("") is False
    assert parsing.is_frontier_final("   ") is False


def test_frontier_final_ignored_artifact_is_false():
    # Pure artifact tokens carry no model and cannot be frontier.
    assert parsing.is_frontier_final("openrouter/free") is False
    assert parsing.is_frontier_final("Mutiple Models") is False


def test_kimi_generic_left_unknown_and_not_frontier():
    # Config note: generic 'Kimi' has an unspecified version -> not counted frontier.
    matched, ignored, unmatched = parsing.classify_models("Chinese SOTAs (GLM, Kimi, etc)")
    assert any(m.display == "Kimi" for m in matched)
    assert parsing.is_frontier_final("Chinese SOTAs (GLM, Kimi, etc)") is False
