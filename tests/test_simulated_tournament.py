import pytest

from refactored_notebook.custom_types import ScoreType, UserType
from refactored_notebook.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_user,
)
from refactored_notebook.load_tournament_data import load_tournament


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
    leaderboard = tournament.get_leaderboard(score_type)
    assert leaderboard.entries[0].user.name == "good"
    assert leaderboard.entries[1].user.name == "medium"
    assert leaderboard.entries[2].user.name == "bad"


@pytest.mark.parametrize("file_path, user_type, num_users", [
    ("tests/test_data/pro_forecasts_q1.csv", UserType.PRO, 10),
    # ("tests/test_data/bot_forecasts_q1.csv", UserType.BOT, None),
])
def test_simulated_tournament_leaderboard_ranking_bot_forecasts(file_path: str, user_type: UserType, num_users: int):
    tournament = load_tournament(file_path, user_type)
    leaderboard = tournament.get_leaderboard(ScoreType.SPOT_PEER)
    if num_users:
        assert len(leaderboard.entries) == num_users
