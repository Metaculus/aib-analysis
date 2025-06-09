import logging

import pytest

from refactored_notebook.custom_types import ScoreType, UserType
from refactored_notebook.load_tournament import load_tournament
from refactored_notebook.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_user,
)
from refactored_notebook.process_tournament import get_leaderboard

logger = logging.getLogger(__name__)

@pytest.mark.parametrize("score_type", [ScoreType.SPOT_BASELINE, ScoreType.SPOT_PEER])
def test_simulated_tournament_leaderboard_ranking_good_forecast_higher(score_type: ScoreType):
    question = make_question_binary()
    user_good = make_user("good")
    user_medium = make_user("medium")
    user_bad = make_user("bad")
    # True resolution, so [0.9] is a good forecast, [0.1] is a bad forecast
    forecast_good = make_forecast(question, user_good, [0.9])
    forecast_medium = make_forecast(question, user_medium, [0.5])
    forecast_bad = make_forecast(question, user_bad, [0.1])
    tournament = SimulatedTournament(forecasts=[forecast_good, forecast_medium, forecast_bad])
    leaderboard = get_leaderboard(tournament, score_type)
    assert leaderboard.entries[0].user.name == "good"
    assert leaderboard.entries[1].user.name == "medium"
    assert leaderboard.entries[2].user.name == "bad"


def test_simulated_tournament_leaderboard_ranking_runs(pro_tournament: SimulatedTournament):
    tournament = pro_tournament
    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    assert len(leaderboard.entries) == 10