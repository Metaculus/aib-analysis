import logging

import pytest

from refactored_notebook.custom_types import ScoreType, UserType
from refactored_notebook.load_tournament import load_tournament
from refactored_notebook.process_tournament import get_leaderboard
from refactored_notebook.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_user,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("score_type", [ScoreType.SPOT_BASELINE, ScoreType.SPOT_PEER])
def test_simulated_tournament_leaderboard_ranking_good_forecast_higher(
    score_type: ScoreType,
):
    question = make_question_binary()
    user_good = make_user("good")
    user_medium = make_user("medium")
    user_bad = make_user("bad")
    # True resolution, so [0.9] is a good forecast, [0.1] is a bad forecast
    forecast_good = make_forecast(question, user_good, [0.9])
    forecast_medium = make_forecast(question, user_medium, [0.5])
    forecast_bad = make_forecast(question, user_bad, [0.1])
    tournament = SimulatedTournament(
        forecasts=[forecast_medium, forecast_good, forecast_bad]
    )
    leaderboard = get_leaderboard(tournament, score_type)

    entry_lists = [
        leaderboard.entries,
        leaderboard.entries_via_sum_of_scores(),
        leaderboard.entries_via_average_score(),
    ]
    for entry_list in entry_lists:
        assert entry_list[0].user.name == "good"
        assert entry_list[1].user.name == "medium"
        assert entry_list[2].user.name == "bad"


def test_leaderboard_pros(
    pro_tournament: SimulatedTournament,
):
    tournament = pro_tournament
    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    assert len(leaderboard.entries) == 10
    entries = leaderboard.entries_via_sum_of_scores()
    assert entries[0].user.name == "draaglom"
    assert entries[0].sum_of_scores == pytest.approx(511.338, abs=0.1)
    assert entries[1].user.name == "Jgalt"
    assert entries[1].sum_of_scores == pytest.approx(439.233, abs=0.1)
    assert entries[2].user.name == "MaciekK"
    assert entries[3].user.name == "SpottedBear"
    assert entries[9].user.name == "OpenSystem"
    assert entries[9].sum_of_scores == pytest.approx(-908.457, abs=0.1)

def test_leaderboard_bots(
    bot_tournament: SimulatedTournament,
):
    tournament = bot_tournament
    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    entries = leaderboard.entries_via_sum_of_scores()
    assert entries[0].user.name == "metac-o1"
    assert entries[0].sum_of_scores == pytest.approx(3631.123, abs=0.1)
    assert entries[1].user.name == "metac-o1-preview"
    assert entries[1].sum_of_scores == pytest.approx(3121.45, abs=0.1)
    assert entries[2].user.name == "manticAI"
    assert entries[3].user.name == "metac-Gemini-Exp-1206"
    assert entries[4].user.name == "acm_bot"
    assert entries[5].user.name == "metac-perplexity"
    assert entries[6].user.name == "GreeneiBot2"
    assert entries[7].user.name == "twsummerbot"
    assert entries[8].user.name == "cookics_bot_TEST"