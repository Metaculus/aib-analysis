import pytest

from refactored_notebook.custom_types import ScoreType, UserType
from refactored_notebook.load_tournament_data import load_tournament
from refactored_notebook.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_user,
)
import logging

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
    leaderboard = tournament.get_leaderboard(score_type)
    assert leaderboard.entries[0].user.name == "good"
    assert leaderboard.entries[1].user.name == "medium"
    assert leaderboard.entries[2].user.name == "bad"


def test_simulated_tournament_leaderboard_ranking_runs(pro_tournament: SimulatedTournament):
    tournament = pro_tournament
    leaderboard = tournament.get_leaderboard(ScoreType.SPOT_PEER)
    assert len(leaderboard.entries) == 10

def test_scores_are_calculated_from_pro_tournament(pro_tournament: SimulatedTournament):
    validate_scores_calculated_from_simulated_tournament(pro_tournament)

def test_scores_are_calculated_from_bot_tournament(bot_tournament: SimulatedTournament):
    validate_scores_calculated_from_simulated_tournament(bot_tournament)

def validate_scores_calculated_from_simulated_tournament(tournament: SimulatedTournament):
    # This test is mostly just running data validation.
    logger.info(f"Tournament has {len(tournament.questions)} questions")
    for question in tournament.questions:
        if question.question_id in [34247, 34730]:
            logger.info(f"Question {question.question_id} has resolution {question.resolution}.\n {question}")
        forecasters_on_question = tournament.get_forecasters_on_question(question.question_id)
        for user in forecasters_on_question:
            for score_type in ScoreType:
                if question.resolution is None:
                    continue

                score = tournament.get_spot_score_for_question(question.question_id, user.name, score_type)
                assert score.type == score_type

