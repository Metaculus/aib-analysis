"""Additional parsing edge cases beyond test_parsing.py.

Focus on the tricky spots: options that are substrings of other options,
duplicate selections, case/whitespace robustness, the "o3" short-key collision
risk, and team-size oddities. Runs without env vars or local data.
"""

from __future__ import annotations

from aib_analysis.survey_analysis_v2 import config, parsing


# --------------------------------------------------------------------------- #
# Multi-select: overlapping option strings
# --------------------------------------------------------------------------- #
def test_asknews_deepnews_not_double_counted_as_other_asknews():
    # "AskNews DeepNews" and "Other AskNews" share the word "AskNews". Longest-
    # first removal must consume "AskNews DeepNews" without leaving "AskNews" to
    # spuriously match "Other AskNews".
    matched, other = parsing.parse_multiselect(
        "AskNews DeepNews", config.MULTISELECT_VOCAB["research"]
    )
    assert matched == ["AskNews DeepNews"]
    assert "Other AskNews" not in matched
    assert other == []


def test_both_asknews_options_when_both_present():
    matched, other = parsing.parse_multiselect(
        "AskNews DeepNews, Other AskNews", config.MULTISELECT_VOCAB["research"]
    )
    assert set(matched) == {"AskNews DeepNews", "Other AskNews"}
    assert other == []


def test_duplicate_selection_counted_once():
    matched, _other = parsing.parse_multiselect("Exa, Exa", config.MULTISELECT_VOCAB["research"])
    assert matched == ["Exa"]


def test_multiselect_case_insensitive():
    matched, _other = parsing.parse_multiselect("exa, TAVILY", config.MULTISELECT_VOCAB["research"])
    assert set(matched) == {"Exa", "Tavily"}


def test_multiselect_only_write_ins():
    matched, other = parsing.parse_multiselect(
        "my scraper, some other thing", config.MULTISELECT_VOCAB["research"]
    )
    assert matched == []
    assert other == ["my scraper", "some other thing"]


def test_multiselect_whitespace_and_trailing_punctuation_stripped():
    matched, other = parsing.parse_multiselect(
        "  Exa ,  ; ", config.MULTISELECT_VOCAB["research"]
    )
    assert matched == ["Exa"]
    assert other == []  # empty/punctuation-only leftovers dropped


# --------------------------------------------------------------------------- #
# Single-select
# --------------------------------------------------------------------------- #
def test_single_select_is_exact_not_substring():
    # A value that merely contains an option is NOT a match (kept as Other).
    canonical, other = parsing.parse_single_select(
        "about the same for both, mostly", config.SINGLE_SELECT_VOCAB["research_vs_reasoning"]
    )
    assert canonical is None
    assert other == "about the same for both, mostly"


def test_single_select_case_insensitive_exact():
    canonical, other = parsing.parse_single_select(
        "STRONG RESEARCH LEAN", config.SINGLE_SELECT_VOCAB["research_vs_reasoning"]
    )
    assert canonical == "Strong research lean" and other is None


# --------------------------------------------------------------------------- #
# Model tokenization / the "o3" collision risk
# --------------------------------------------------------------------------- #
def test_o3_matches_only_as_its_own_token():
    matched, _ignored, _unmatched = parsing.classify_models("o3, Cohere")
    displays = [m.display for m in matched]
    assert "o3" in displays and "Cohere" in displays


def test_o3_not_spuriously_found_inside_other_model_names():
    # "GPT-5.4" / "Claude Opus 4.6" contain no standalone o3; ensure o3 doesn't leak in.
    matched, _i, _u = parsing.classify_models("GPT-5.4, Claude Opus 4.6")
    assert "o3" not in [m.display for m in matched]


def test_tokenize_models_splits_on_comma_and_semicolon():
    assert parsing.tokenize_models("GPT-5.4; Claude Opus 4.8, o3") == [
        "GPT-5.4",
        "Claude Opus 4.8",
        "o3",
    ]


def test_classify_models_separates_ignored_and_unmatched():
    matched, ignored, unmatched = parsing.classify_models("GPT-5.4, Mutiple Models, ZzzUnknown")
    assert [m.display for m in matched] == ["GPT-5.4"]
    assert ignored == ["Mutiple Models"]
    assert unmatched == ["ZzzUnknown"]


# --------------------------------------------------------------------------- #
# Team size
# --------------------------------------------------------------------------- #
def test_team_size_number_with_note_is_other_only():
    # A number with extra prose is treated purely as an Other write-in, not as
    # both a counted size and an Other bar (avoids double counting in charts).
    size, other = parsing.parse_team_size("3 (two part time)")
    assert size is None and other == "3 (two part time)"


def test_team_size_pure_text_unparseable():
    size, other = parsing.parse_team_size("a couple")
    assert size is None and other == "a couple"


def test_team_size_zero_is_kept_as_is():
    # "0" is a digit -> parsed literally (does not become solo=1).
    assert parsing.parse_team_size("0") == (0, None)


def test_count_research_sources_ignores_write_ins():
    raw = "Exa, Tavily, my homemade tool"
    assert parsing.count_research_sources(raw, config.RESEARCH_SOURCE_OPTIONS) == 2


# --------------------------------------------------------------------------- #
# feature_present: matches parsed options, immune to write-in collisions
# --------------------------------------------------------------------------- #
def test_feature_present_matches_within_a_selected_option():
    # "base rates" is a substring of a full strategy option -> present.
    matched = ["Explicitly calculate/estimate base rates in a rigorous way"]
    assert parsing.feature_present(matched, "base rates") is True


def test_feature_present_true_for_short_substring_of_real_option():
    assert parsing.feature_present(["Exa"], "exa") is True


def test_feature_present_ignores_write_in_only_text():
    # The substring "exa" living in a write-in ("hexagon scraper") must NOT set
    # the flag, because only parsed canonical options are considered.
    assert parsing.feature_present([], "exa") is False
    assert parsing.feature_present(["Tavily"], "exa") is False


def test_feature_present_case_insensitive():
    assert parsing.feature_present(["ANTHROPIC WEB SEARCH"], "web search") is True
