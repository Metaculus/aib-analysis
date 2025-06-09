import logging

import pytest

from aib_analysis.custom_types import ScoreType, UserType
from aib_analysis.load_tournament import load_tournament
from aib_analysis.process_tournament import (
    combine_on_question_title_intersection,
    get_leaderboard,
)
from aib_analysis.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_question_mc,
    make_question_numeric,
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


class TestCombineTournaments:

    def test_combine_tournaments_drops_non_intersection_questions(self):
        # Create two tournaments with some overlapping and non-overlapping questions
        question1 = make_question_binary(question_text="Will it rain today?")
        question2 = make_question_binary(question_text="Will it rain tomorrow?")
        question3 = make_question_binary(question_text="Will it rain in 3 days?")

        user1 = make_user("user1")
        user2 = make_user("user2")
        user3 = make_user("user3")

        # Tournament 1 has questions 1 and 2
        forecast1_1 = make_forecast(question1, user1, [0.8])
        forecast1_2 = make_forecast(question2, user1, [0.7])
        tournament1 = SimulatedTournament(forecasts=[forecast1_1, forecast1_2])

        # Tournament 2 has questions 1 and 3
        forecast2_1 = make_forecast(question1, user2, [0.6])
        forecast2_3 = make_forecast(question3, user3, [0.5])
        tournament2 = SimulatedTournament(forecasts=[forecast2_1, forecast2_3])

        # Combine tournaments
        combined = combine_on_question_title_intersection(tournament1, tournament2)

        # Only question1 should be in the combined tournament
        assert len(combined.questions) == 1
        assert combined.questions[0].question_id == question1.question_id
        assert len(combined.users) == 2
        assert combined.users[0].name == user1.name
        assert combined.users[1].name == user2.name
        assert len(combined.forecasts) == 2
        question1_forecasts = [f for f in combined.forecasts if f.question.question_id == question1.question_id]
        assert len(question1_forecasts) == 2

    def test_combine_tournaments_on_question_intersection(
        self,
        pro_tournament: SimulatedTournament,
        bot_tournament: SimulatedTournament,
    ):
        pro_forecasts = pro_tournament.forecasts
        bot_forecasts = bot_tournament.forecasts
        pro_users = pro_tournament.users
        bot_users = bot_tournament.users
        combined_tournament = combine_on_question_title_intersection(pro_tournament, bot_tournament)
        assert len(combined_tournament.questions) == 99
        assert len(combined_tournament.forecasts) < len(pro_forecasts) + len(bot_forecasts)
        assert len(combined_tournament.users) < len(pro_users) + len(bot_users)
