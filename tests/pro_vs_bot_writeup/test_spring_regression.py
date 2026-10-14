"""Checks the template against the published Spring 2026 results post.

Runs only where the Spring 2026 simulation outputs exist under local/ (they are not in the repo).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aib_analysis.pro_vs_bot_writeup.results import (
    SeasonResults,
    compute_season_results,
)
from aib_analysis.pro_vs_bot_writeup.season_config import get_season_config

SPRING_SIMULATIONS = Path(__file__).resolve().parents[2] / "local" / "spring_2026_simulations"

pytestmark = pytest.mark.skipif(
    not SPRING_SIMULATIONS.exists(), reason="Spring 2026 simulation outputs are not in local/"
)

PUBLISHED_TOP_10 = [
    ("GreeneiBot2", 5442.02),
    ("Preseen-Atlas", 5292.83),
    ("SynapseSeer", 4993.10),
    ("manticAI", 4690.37),
    ("Preseen-Chestnut", 4162.26),
    ("cassi", 4093.36),
    ("pgodzinai", 4002.29),
    ("CumulativeBot", 3942.45),
    ("mmBot", 3816.22),
    ("jonahsingerbot", 3749.82),
]
PUBLISHED_BOT_TEAM = (
    "GreeneiBot2",
    "Preseen-Atlas",
    "manticAI",
    "SynapseSeer",
    "CumulativeBot",
    "pgodzinai",
    "laertes",
    "Preseen-Chestnut",
    "mmBot",
    "nostreambot",
)


@pytest.fixture(scope="module")
def spring_results() -> SeasonResults:
    return compute_season_results(SPRING_SIMULATIONS, get_season_config("spring_2026"))


def test_head_to_head_matches_published_post(spring_results: SeasonResults) -> None:
    head_to_head = spring_results.head_to_head
    assert head_to_head.question_count == 99
    assert round(head_to_head.bot_team_mean, 2) == -1.25
    assert tuple(round(bound, 2) for bound in head_to_head.ci) == (-4.87, 2.37)
    assert round(head_to_head.p_pros_better, 3) == 0.247
    assert (head_to_head.leader, head_to_head.evidence) == ("pros", "none")
    assert head_to_head.bot_team_members == PUBLISHED_BOT_TEAM


def test_leaderboards_match_published_post(spring_results: SeasonResults) -> None:
    top_10 = [(row.name, round(row.sum_of_scores, 2)) for row in spring_results.field.top_10]
    assert top_10 == PUBLISHED_TOP_10
    top_metac = spring_results.metac.top_baseline
    assert top_metac.name == "metac-gpt-5-1-high+asknews"
    assert round(top_metac.average_score, 2) == 11.32
    assert top_metac.ci is not None and tuple(round(bound, 2) for bound in top_metac.ci) == (7.18, 15.45)
    standing = spring_results.metac.current_standing
    assert (standing.rank, standing.bot_count) == (18, 173)


def test_individuals_and_team_sizes_match_published_post(spring_results: SeasonResults) -> None:
    individuals = spring_results.individuals
    assert individuals.pros_above_best_bot == 9
    assert (individuals.best_bot.name, individuals.best_bot.rank) == ("GreeneiBot2", 10)
    assert individuals.bots_above_lowest_pro == 2
    assert (individuals.rows[0].name, round(individuals.rows[0].sum_of_scores, 2)) == ("Pro A", 2505.09)
    assert round(spring_results.team_sizes[0].mean, 2) == -4.51


def test_question_counts_match_published_post(spring_results: SeasonResults) -> None:
    overview = spring_results.overview
    assert (overview.bot_question_count, overview.bot_scored_count, overview.bot_only_scored_count) == (327, 297, 198)
    assert round(100 * overview.binary_yes_count / overview.binary_scored_count) == 33
    assert spring_results.season_trend.change_p_value is not None and spring_results.season_trend.change_p_value < 0.05
