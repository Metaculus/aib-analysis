"""End-to-end feature derivation from synthetic respondents.

build_features is where raw answers become the booleans, numeric variables, and
frontier flag that every downstream number depends on. These build Respondents
in memory (no CSVs, no leaderboard) and assert the derived features, including
the documented treatment of blank answers. Runs without env vars or local data.
"""

from __future__ import annotations

from aib_analysis.survey_analysis_v2 import config
from aib_analysis.survey_analysis_v2.features import build_features
from aib_analysis.survey_analysis_v2.loading import Respondent


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
        writeup_rating="Useful: I would have probably done worse without them",
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
    # writeup rating ordinal index: "Useful..." is second option (=1).
    assert feature.variables["writeup_rating_ord"] == 1.0


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
    assert feature.variables["hours_mid"] is None
    assert feature.variables["llm_calls_mid"] is None
    assert feature.variables["team_size"] is None
    assert feature.variables["writeup_rating_ord"] is None


def test_not_in_a_team_becomes_solo_one():
    feature = build_features([_respondent(team_size="Not in a team")])[0]
    assert feature.variables["team_size"] == 1.0


def test_write_in_research_captured_as_other_and_not_counted():
    feature = build_features(
        [_respondent(research="Exa, my custom homemade scraper")]
    )[0]
    cell = feature.cells["research"]
    assert cell.matched == ["Exa"]
    assert cell.other == ["my custom homemade scraper"]
    assert feature.variables["n_research_sources"] == 1.0  # write-in not counted


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
