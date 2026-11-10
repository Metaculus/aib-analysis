"""End-to-end feature derivation from synthetic respondents.

build_features is where raw answers become the booleans, numeric variables, and
frontier flag that every downstream number depends on. These build Respondents
in memory (no CSVs, no leaderboard) and assert the derived features, including
the documented treatment of blank answers. Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import config
from aib_analysis.survey_analysis.features import build_features
from aib_analysis.survey_analysis.loading import Respondent


def _respondent(**answers: str) -> Respondent:
    record = {slug: "" for slug in config.COLUMNS}
    record.update(answers)
    record.setdefault("bot_name", "synthbot")
    return Respondent(bot_name=record["bot_name"], answers=record)


def test_full_answer_derives_expected_features():
    respondent = _respondent(
        final_model="GPT-5.4, Claude Opus 4.8",
        research="AskNews DeepNews, Exa, Tavily",
        strategies="Explicitly calculate/estimate base rates in a rigorous way",
        hours="16-40hrs",
        llm_calls="2-5",
        cost_per_q="$1-2.99",
        team_size="3",
        research_vs_reasoning="Strong reasoning lean",
    )
    feature = build_features([respondent])[0]

    assert feature.frontier is True
    assert feature.variables["frontier"] == 1.0
    assert feature.booleans["uses_asknews"] is True
    assert feature.booleans["uses_exa"] is True
    assert feature.booleans["uses_base_rates"] is True
    assert feature.variables["n_research_sources"] == 3.0
    assert feature.variables["hours_mid"] == config.HOURS_MIDPOINT["16-40hrs"]
    assert feature.variables["llm_calls_mid"] == config.LLM_CALLS_MIDPOINT["2-5"]
    assert feature.variables["cost_mid"] == config.COST_MIDPOINT["$1-2.99"]
    assert feature.variables["team_size"] == 3.0
    # research_vs_reasoning ordinal index: "Strong reasoning lean" is last (=4).
    assert feature.variables["research_vs_reasoning_ord"] == 4.0


def test_blank_answers_yield_false_booleans_and_none_numerics():
    # DOCUMENTED BEHAVIOR: a skipped column is item non-response. The raw booleans
    # dict still reads False, but the correlation *variable* is None (missing), so
    # a blank answer is dropped pairwise instead of being counted as a real "no".
    feature = build_features([_respondent()])[0]

    assert all(value is False for value in feature.booleans.values())
    assert feature.variables["uses_base_rates"] is None  # blank source -> None (excluded)
    assert feature.frontier is False
    assert feature.variables["frontier"] is None  # blank final_model -> None (excluded)
    # Numerics from skipped ordinal/count columns are None (dropped pairwise).
    assert feature.variables["n_research_sources"] is None
    assert feature.variables["moderate_research_sources"] is None
    assert feature.variables["hours_mid"] is None
    assert feature.variables["llm_calls_mid"] is None
    assert feature.variables["team_size"] is None  # blank team_size answer -> None


def test_not_in_a_team_becomes_solo_one():
    feature = build_features([_respondent(team_size="Not in a team")])[0]
    assert feature.variables["team_size"] == 1.0


def test_research_write_in_no_rule_covers_stops_the_run():
    with pytest.raises(ValueError, match="my custom homemade scraper"):
        build_features([_respondent(research="Exa, my custom homemade scraper")])


def test_not_counted_research_write_in_stays_other_and_adds_nothing():
    feature = build_features([_respondent(research="Exa, my own weird system")])[0]
    assert feature.cells["research"].matched == ["Exa"]
    assert feature.cells["research"].other == ["my own weird system"]
    assert feature.variables["n_research_sources"] == 1.0


def test_write_in_naming_a_form_option_ticks_it_once():
    static = "Static web scraping (Only HTML, possibly converted to markdown)"
    feature = build_features(
        [_respondent(research=f"Exa, Exa (neural news search), {static}, Wayback Machine snapshots, Firecrawl")]
    )[0]
    cell = feature.cells["research"]
    assert cell.matched == ["Exa", static]
    assert cell.other == []
    assert feature.variables["n_research_sources"] == 2.0
    assert feature.variables["uses_scraping"] == 1.0


def test_write_in_naming_an_option_and_an_extra_source_counts_both():
    feature = build_features([_respondent(research="OpenRouter web search. Tavily early on")])[0]
    assert feature.cells["research"].matched == ["Tavily"]
    assert feature.cells["research"].other == ["OpenRouter web search. Tavily early on"]
    assert feature.variables["n_research_sources"] == 2.0


def test_listed_write_in_research_source_adds_to_the_count():
    feature = build_features(
        [_respondent(research="Exa, Other AskNews, AskNews DeepNews, a few RSS feeds")]
    )[0]
    assert feature.cells["research"].other == ["a few RSS feeds"]
    assert feature.variables["n_research_sources"] == 4.0


def _research_answer(source_count: int) -> str:
    return ", ".join(config.MULTISELECT_VOCAB["research"][:source_count])


def test_moderate_research_sources_includes_both_bounds():
    fewest, most = config.MODERATE_RESEARCH_SOURCE_RANGE
    expected_by_count = {fewest - 1: 0.0, fewest: 1.0, most: 1.0, most + 1: 0.0}
    for source_count, expected in expected_by_count.items():
        feature = build_features([_respondent(research=_research_answer(source_count))])[0]
        assert feature.variables["n_research_sources"] == source_count
        assert feature.variables["moderate_research_sources"] == expected



def test_unrecognized_ordinal_value_is_none_not_zero():
    # A free-text hours answer that isn't a canonical bucket -> no midpoint.
    feature = build_features([_respondent(hours="about a weekend")])[0]
    assert feature.variables["hours_mid"] is None


def test_non_frontier_final_model_flags_false():
    feature = build_features([_respondent(final_model="Claude Sonnet 4.6")])[0]
    assert feature.frontier is False


def test_mini_variant_final_is_not_frontier():
    feature = build_features([_respondent(final_model="GPT-5.4 mini")])[0]
    assert feature.frontier is False


def test_final_model_text_naming_no_known_model_is_unknown_not_no():
    feature = build_features([_respondent(final_model="Our own forecasting system")])[0]
    assert feature.variables["frontier"] is None


def test_questions_missed_comes_from_the_leaderboard_join():
    respondent = _respondent()
    respondent.questions_missed = 12
    assert build_features([respondent])[0].variables["questions_missed"] == 12.0
    assert build_features([_respondent()])[0].variables["questions_missed"] is None
