"""Unit tests for survey analysis v2 parsing. Run without any env vars."""

from __future__ import annotations

from aib_analysis.survey_analysis import config, parsing
from aib_analysis.survey_analysis.loading import _alnum_key, _match_prize, PrizeOwner


def test_multiselect_matches_options_with_embedded_commas():
    raw = (
        "Static web scraping (Only HTML, possibly converted to markdown), "
        "OpenAI web search, Exa"
    )
    matched, other = parsing.parse_multiselect(raw, config.MULTISELECT_VOCAB["research"])
    assert "Static web scraping (Only HTML, possibly converted to markdown)" in matched
    assert "OpenAI web search" in matched
    assert "Exa" in matched
    assert other == []


def test_multiselect_captures_other_write_in():
    raw = "Exa, Tavily, my custom scraper thing"
    matched, other = parsing.parse_multiselect(raw, config.MULTISELECT_VOCAB["research"])
    assert set(matched) == {"Exa", "Tavily"}
    assert other == ["my custom scraper thing"]


def test_multiselect_returns_in_vocab_order():
    raw = "Tavily, Exa"
    matched, _other = parsing.parse_multiselect(raw, config.MULTISELECT_VOCAB["research"])
    assert matched == ["Exa", "Tavily"]  # vocab order, not input order


def test_multiselect_blank():
    matched, other = parsing.parse_multiselect("", config.MULTISELECT_VOCAB["research"])
    assert matched == [] and other == []


def test_single_select_exact_and_other():
    canonical, other = parsing.parse_single_select("1-2", config.SINGLE_SELECT_VOCAB["iterations"])
    assert canonical == "1-2" and other is None

    canonical, other = parsing.parse_single_select(
        "Hard to define, maybe 10?", config.SINGLE_SELECT_VOCAB["iterations"]
    )
    assert canonical is None
    assert other == "Hard to define, maybe 10?"


def test_bucket_midpoint():
    assert parsing.bucket_to_midpoint("$1-2.99", config.COST_MIDPOINT) == 2.0
    assert parsing.bucket_to_midpoint(None, config.COST_MIDPOINT) is None
    assert parsing.bucket_to_midpoint("unknown", config.COST_MIDPOINT) is None


def test_team_size():
    assert parsing.parse_team_size("1") == (1, None)
    assert parsing.parse_team_size("12") == (12, None)
    assert parsing.parse_team_size("Not in a team") == (1, None)
    assert parsing.parse_team_size("") == (None, None)


def test_team_size_messy_is_other_only_not_double_counted():
    # A number with extra prose is treated purely as an Other write-in, never as
    # both a counted size and an Other bar.
    size, other = parsing.parse_team_size("3 (two of us part time)")
    assert size is None
    assert other == "3 (two of us part time)"


def test_multiselect_short_option_not_matched_inside_a_word():
    # "Exa" must not match inside "hexagon"; "Perplexity" is a real match.
    matched, other = parsing.parse_multiselect(
        "Perplexity, hexagon-search tool", config.MULTISELECT_VOCAB["research"]
    )
    assert "Exa" not in matched
    assert "Perplexity" in matched
    assert any("hexagon" in token for token in other)


def test_multiselect_short_option_still_matches_as_real_segment():
    matched, _other = parsing.parse_multiselect(
        "Exa, Tavily", config.MULTISELECT_VOCAB["research"]
    )
    assert set(matched) == {"Exa", "Tavily"}


def test_frontier_high_power_recent_is_frontier():
    assert parsing.is_frontier_final("GPT-5.4") is True
    assert parsing.is_frontier_final("GPT-5.5, Claude Opus 4.8") is True


def test_frontier_mini_variant_not_frontier():
    assert parsing.is_frontier_final("GPT-5.4 mini") is False


def test_frontier_old_model_not_frontier():
    assert parsing.is_frontier_final("GPT-4o") is False
    assert parsing.is_frontier_final("Claude Sonnet 4.5") is False


def test_frontier_mixed_final_counts_if_any_frontier():
    # A frontier model anywhere in the final cell flips it to frontier.
    assert parsing.is_frontier_final("GPT-4o, GPT-5.4") is True


def test_classify_models_mini_token_not_matched_to_base():
    matched, _ignored, _unmatched = parsing.classify_models("GPT-5.4 mini")
    displays = [m.display for m in matched]
    assert displays == ["GPT-5.4 mini"]
    assert "GPT-5.4" not in displays


def test_classify_models_ignored_token():
    _matched, ignored, unmatched = parsing.classify_models("Mutiple Models")
    assert ignored == ["Mutiple Models"]
    assert unmatched == []


def test_count_research_sources():
    raw = "AskNews DeepNews, Exa, Tavily, Perplexity"
    assert parsing.count_research_sources(raw, config.RESEARCH_SOURCE_OPTIONS) == 4


def test_prize_tokenized_match_against_bot_list():
    owners = [
        PrizeOwner(
            owner_username="someone",
            bot_usernames=["5cast-v1", "5cast-v2"],
            winner_count=1,
            aib_prize=100.0,
            total_prize=100.0,
        )
    ]
    owner, kind = _match_prize("5cast-v1", owners)
    assert owner is not None and owner.winner_count == 1
    assert kind == "exact"


def test_alnum_key_strips_punctuation_and_emoji():
    assert _alnum_key("MWG Bot") == "mwgbot"
    assert _alnum_key("Preseen-Atlas") == "preseenatlas"
