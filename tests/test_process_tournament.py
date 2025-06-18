import logging

import pytest

from aib_analysis.data_structures.custom_types import (
    ForecastType,
    QuestionType,
    ScoreType,
)
from aib_analysis.data_structures.data_models import Question
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.process_tournament import (
    combine_tournaments,
    constrain_question_types,
    create_team_tournament,
    get_best_forecasters_from_tournament,
    get_leaderboard,
    smart_remove_questions_from_tournament,
)
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_tournament,
    make_user,
)

logger = logging.getLogger(__name__)


class TestLeaderboard:

    @pytest.mark.parametrize(
        "score_type", [ScoreType.SPOT_BASELINE, ScoreType.SPOT_PEER]
    )
    def test_simulated_tournament_leaderboard_ranking_good_forecast_higher(
        self,
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
        self,
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
        self,
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

    def test_leaderboard_binary_bots(
        self,
        bot_tournament: SimulatedTournament,
    ):
        binary_tournament = constrain_question_types(
            bot_tournament, [QuestionType.BINARY]
        )
        leaderboard = get_leaderboard(binary_tournament, ScoreType.SPOT_PEER)
        entries = leaderboard.entries_via_sum_of_scores()
        assert entries[0].user.name == "manticAI"
        assert entries[0].sum_of_scores == pytest.approx(1691.05, abs=0.1)
        assert entries[1].user.name == "pgodzinai"
        assert entries[12].user.name == "MWG"
        assert entries[12].sum_of_scores == pytest.approx(215.99, abs=0.1)
        assert entries[44].user.name == "InstitutPelFutur"
        assert entries[44].sum_of_scores == pytest.approx(-2114.76, abs=0.1)

    def test_leaderboard_multiple_choice_bots(
        self,
        bot_tournament: SimulatedTournament,
    ):
        multiple_choice_tournament = constrain_question_types(
            bot_tournament, [QuestionType.MULTIPLE_CHOICE]
        )
        leaderboard = get_leaderboard(multiple_choice_tournament, ScoreType.SPOT_PEER)
        entries = leaderboard.entries_via_sum_of_scores()
        assert entries[0].user.name == "metac-o1"
        assert entries[0].sum_of_scores == pytest.approx(964.41, abs=0.1)
        assert entries[1].user.name == "metac-Gemini-Exp-1206"
        assert entries[12].user.name == "Grizeu_Bot"
        assert entries[12].sum_of_scores == pytest.approx(189.96, abs=0.1)
        assert entries[43].user.name == "ajf-bot"
        assert entries[43].sum_of_scores == pytest.approx(-1342.38, abs=0.1)

    def test_leaderboard_numeric_bots(
        self,
        bot_tournament: SimulatedTournament,
    ):
        numeric_tournament = constrain_question_types(
            bot_tournament, [QuestionType.NUMERIC]
        )
        leaderboard = get_leaderboard(numeric_tournament, ScoreType.SPOT_PEER)
        entries = leaderboard.entries_via_sum_of_scores()
        assert entries[0].user.name == "metac-o1-preview"
        assert entries[0].sum_of_scores == pytest.approx(1683.54, abs=0.1)
        assert entries[1].user.name == "mmBot"
        assert entries[12].user.name == "metac-grok-2-1212"
        assert entries[12].sum_of_scores == pytest.approx(153.21, abs=0.1)
        assert entries[40].user.name == "minefrac1"
        assert entries[40].sum_of_scores == pytest.approx(-1348.27, abs=0.1)


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
        combined = combine_tournaments(tournament1, tournament2)

        # Only question1 should be in the combined tournament
        assert len(combined.questions) == 1
        assert combined.questions[0].question_id == question1.question_id
        assert len(combined.users) == 2
        assert combined.users[0].name == user1.name
        assert combined.users[1].name == user2.name
        assert len(combined.forecasts) == 2
        question1_forecasts = [
            f
            for f in combined.forecasts
            if f.question.question_id == question1.question_id
        ]
        assert len(question1_forecasts) == 2

    def test_combine_tournaments_pro_plus_bot(
        self,
        pro_tournament: SimulatedTournament,
        bot_tournament: SimulatedTournament,
    ):
        pro_forecasts = pro_tournament.forecasts
        bot_forecasts = bot_tournament.forecasts
        pro_users = pro_tournament.users
        bot_users = bot_tournament.users
        combined_tournament = combine_tournaments(pro_tournament, bot_tournament)
        assert len(combined_tournament.questions) == 98
        assert len(combined_tournament.forecasts) < len(pro_forecasts) + len(
            bot_forecasts
        )
        assert len(combined_tournament.users) < len(pro_users) + len(bot_users)


class TestCreateTeam:
    def test_create_team_from_leaderboard_none_size(self) -> None:
        tournament = make_tournament()
        team = get_best_forecasters_from_tournament(tournament, "all")
        assert len(team) == len(tournament.users)
        assert all(user in tournament.users for user in team)

    def test_create_team_from_leaderboard_sum_approach(self) -> None:
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

        team = get_best_forecasters_from_tournament(tournament, 2)
        assert len(team) == 2
        assert team[0].name == "good"
        assert team[1].name == "medium"

    def test_create_team_tournament(self) -> None:
        # Create two tournaments with different users and questions
        question1 = make_question_binary("Q1")
        question2 = make_question_binary("Q2")

        user1 = make_user("user1")
        user2 = make_user("user2")
        user3 = make_user("user3")
        user4 = make_user("user4")

        # Tournament 1 forecasts
        forecast1_1 = make_forecast(question1, user1, [0.8])
        forecast1_2 = make_forecast(question1, user2, [0.7])

        # Tournament 2 forecasts
        forecast2_1 = make_forecast(question1, user3, [0.6])
        forecast2_2 = make_forecast(question2, user4, [0.5])

        tournament1 = SimulatedTournament(forecasts=[forecast1_1, forecast1_2])
        tournament2 = SimulatedTournament(forecasts=[forecast2_1, forecast2_2])

        # Create team tournament with top 1 from each tournament
        team_tournament = create_team_tournament(
            tournament1, tournament2, [user1], [user3], "Team1", "Team2"
        )

        # Verify the combined tournament
        assert len(team_tournament.users) == 2
        assert any(user.name == "Team1" for user in team_tournament.users)
        assert any(user.name == "Team2" for user in team_tournament.users)

        # Verify forecasts were combined
        assert (
            len(team_tournament.forecasts) == 2
        )  # One forecast per team for the common question
        assert all(
            forecast.question.question_id == question1.question_id
            for forecast in team_tournament.forecasts
        )

        # Verify the aggregated users
        team1_user = next(
            user for user in team_tournament.users if user.name == "Team1"
        )
        team2_user = next(
            user for user in team_tournament.users if user.name == "Team2"
        )

        assert len(team1_user.aggregated_users) == 1
        assert len(team2_user.aggregated_users) == 1
        assert (
            team1_user.aggregated_users[0].name == "user1"
        )  # Best performer in tournament1
        assert (
            team2_user.aggregated_users[0].name == "user3"
        )  # Only user in tournament2

    @pytest.mark.skip(reason="Need more diverse mock tournaments in order to not have hash conflicts")
    def test_create_team_tournament_all_users(self) -> None:
        tournament1 = make_tournament()
        tournament2 = make_tournament()

        # Create team tournament with all users from each tournament
        team_tournament = create_team_tournament(
            tournament1, tournament2, "all", "all", "Team1", "Team2"
        )

        # Verify the combined tournament
        assert len(team_tournament.users) == 2
        assert any(user.name == "Team1" for user in team_tournament.users)
        assert any(user.name == "Team2" for user in team_tournament.users)

        # Verify the aggregated users
        team1_user = next(
            user for user in team_tournament.users if user.name == "Team1"
        )
        team2_user = next(
            user for user in team_tournament.users if user.name == "Team2"
        )

        assert len(team1_user.aggregated_users) == len(tournament1.users)
        assert len(team2_user.aggregated_users) == len(tournament2.users)
        assert all(user in team1_user.aggregated_users for user in tournament1.users)
        assert all(user in team2_user.aggregated_users for user in tournament2.users)

    def test_create_team_from_leaderboard_empty_tournament(self) -> None:
        empty_tournament = SimulatedTournament(forecasts=[])
        with pytest.raises(Exception):
            get_best_forecasters_from_tournament(empty_tournament, 5)

    def test_create_team_from_leaderboard_team_size_larger_than_users(self) -> None:
        tournament = make_tournament()
        team = get_best_forecasters_from_tournament(tournament, len(tournament.users) + 5)
        assert len(team) == len(tournament.users)
        assert all(user in tournament.users for user in team)

    def test_create_team_from_leaderboard_zero_team_size(self) -> None:
        tournament = make_tournament()
        with pytest.raises(ValueError):
            get_best_forecasters_from_tournament(tournament, 0)

    def test_create_team_tournament_no_common_questions(self) -> None:
        # Create two tournaments with completely different questions
        question1 = make_question_binary("Q1")
        question2 = make_question_binary("Q2")

        user1 = make_user("user1")
        user2 = make_user("user2")

        tournament1 = SimulatedTournament(
            forecasts=[make_forecast(question1, user1, [0.8])]
        )
        tournament2 = SimulatedTournament(
            forecasts=[make_forecast(question2, user2, [0.7])]
        )

        with pytest.raises(ValueError):
            create_team_tournament(tournament1, tournament2, "all", "all", "Team1", "Team2")

    def test_combine_pro_and_bot_tournament(
        self, bot_tournament: SimulatedTournament, pro_tournament: SimulatedTournament
    ):
        bot_team_size = 10
        pro_team_name = "Pro Team"
        bot_team_name = "Bot Team"
        team_2 = get_best_forecasters_from_tournament(bot_tournament, bot_team_size)
        aggregate_tournament = create_team_tournament(
            pro_tournament,
            bot_tournament,
            team_1="all",
            team_2=team_2,
            aggregate_name_1=pro_team_name,
            aggregate_name_2=bot_team_name,
        )
        assert len(aggregate_tournament.users) == 2
        assert len(aggregate_tournament.questions) == 98
        leaderboard = get_leaderboard(aggregate_tournament, ScoreType.SPOT_PEER)
        entries = leaderboard.entries_via_sum_of_scores()
        assert entries[0].user.name == pro_team_name
        assert entries[1].user.name == bot_team_name

        self._test_question_aggregate(
            aggregate_tournament, 34493, [0.377, 0.088, 0.175, 0.36]
        )  # https://www.metaculus.com/questions/31814/
        self._test_question_aggregate(
            aggregate_tournament, 31338, [0.886, 0.114]
        )  # https://www.metaculus.com/questions/31814/
        self._test_question_aggregate(
            aggregate_tournament, 33877, [0.25, 0.75]
        )  # https://www.metaculus.com/questions/34380/

    def _test_question_aggregate(
        self,
        aggregate_tournament: SimulatedTournament,
        question_id_to_test: int,
        expected_predictions: ForecastType,
    ):
        forecasts = aggregate_tournament.question_and_user_to_forecasts(
            question_id_to_test, "Pro Team"
        )
        assert (
            len(forecasts) == 1
        ), f"Expected 1 forecast for question {question_id_to_test}, but got {len(forecasts)}"
        forecast = forecasts[0]

        for actual_option_prediction, site_option_prediction in zip(
            forecast.prediction, expected_predictions  # type: ignore
        ):
            assert actual_option_prediction == pytest.approx(
                site_option_prediction, abs=0.03
            ), f"Question {question_id_to_test} forecast {forecast.prediction} does not match expected {expected_predictions}"


def test_constrain_question_types():
    tournament = make_tournament()
    binary_tournament = constrain_question_types(tournament, [QuestionType.BINARY])
    assert all(forecast.question.type == QuestionType.BINARY for forecast in binary_tournament.forecasts)
    assert all(question.type == QuestionType.BINARY for question in binary_tournament.questions)
    assert len(binary_tournament.forecasts) < len(tournament.forecasts)

    numeric_tournament = constrain_question_types(tournament, [QuestionType.NUMERIC])
    assert all(forecast.question.type == QuestionType.NUMERIC for forecast in numeric_tournament.forecasts)
    assert all(question.type == QuestionType.NUMERIC for question in numeric_tournament.questions)
    assert len(numeric_tournament.forecasts) < len(tournament.forecasts)

    multiple_choice_tournament = constrain_question_types(tournament, [QuestionType.MULTIPLE_CHOICE])
    assert all(forecast.question.type == QuestionType.MULTIPLE_CHOICE for forecast in multiple_choice_tournament.forecasts)
    assert all(question.type == QuestionType.MULTIPLE_CHOICE for question in multiple_choice_tournament.questions)
    assert len(multiple_choice_tournament.forecasts) < len(tournament.forecasts)

    binary_and_numeric_tournament = constrain_question_types(tournament, [QuestionType.BINARY, QuestionType.NUMERIC])
    assert all(forecast.question.type == QuestionType.BINARY or forecast.question.type == QuestionType.NUMERIC for forecast in binary_and_numeric_tournament.forecasts)
    assert all(question.type == QuestionType.BINARY or question.type == QuestionType.NUMERIC for question in binary_and_numeric_tournament.questions)
    assert len(binary_and_numeric_tournament.forecasts) < len(tournament.forecasts)

    binary_and_multiple_choice_tournament = constrain_question_types(tournament, [QuestionType.BINARY, QuestionType.MULTIPLE_CHOICE])
    assert all(forecast.question.type == QuestionType.BINARY or forecast.question.type == QuestionType.MULTIPLE_CHOICE for forecast in binary_and_multiple_choice_tournament.forecasts)
    assert all(question.type == QuestionType.BINARY or question.type == QuestionType.MULTIPLE_CHOICE for question in binary_and_multiple_choice_tournament.questions)
    assert len(binary_and_multiple_choice_tournament.forecasts) < len(tournament.forecasts)


class TestRemoveQuestions:
    def test_basic_question_removal(self) -> None:
        # Create a tournament with multiple questions
        question1 = make_question_binary("Q1")
        question2 = make_question_binary("Q2")
        question3 = make_question_binary("Q3")

        user1 = make_user("user1")
        user2 = make_user("user2")

        # Create forecasts for each question
        forecast1_1 = make_forecast(question1, user1, [0.8])
        forecast1_2 = make_forecast(question1, user2, [0.7])
        forecast2_1 = make_forecast(question2, user1, [0.6])
        forecast2_2 = make_forecast(question2, user2, [0.5])
        forecast3_1 = make_forecast(question3, user1, [0.4])
        forecast3_2 = make_forecast(question3, user2, [0.3])

        tournament = SimulatedTournament(
            forecasts=[forecast1_1, forecast1_2, forecast2_1, forecast2_2, forecast3_1, forecast3_2]
        )

        # Remove question2
        filtered_tournament = smart_remove_questions_from_tournament(tournament, [question2])

        # Verify results
        assert len(filtered_tournament.questions) == 2
        assert question1 in filtered_tournament.questions
        assert question3 in filtered_tournament.questions
        assert question2 not in filtered_tournament.questions
        assert len(filtered_tournament.forecasts) == 4  # 2 forecasts per remaining question

    def test_remove_multiple_questions(self) -> None:
        # Create a tournament with multiple questions
        question1 = make_question_binary("Q1")
        question2 = make_question_binary("Q2")
        question3 = make_question_binary("Q3")
        question4 = make_question_binary("Q4")

        user = make_user("user1")

        # Create one forecast per question
        forecast1 = make_forecast(question1, user, [0.8])
        forecast2 = make_forecast(question2, user, [0.7])
        forecast3 = make_forecast(question3, user, [0.6])
        forecast4 = make_forecast(question4, user, [0.5])

        tournament = SimulatedTournament(
            forecasts=[forecast1, forecast2, forecast3, forecast4]
        )
        assert len(tournament.questions) == 4
        assert len(tournament.forecasts) == 4

        # Remove question2 and question3
        filtered_tournament = smart_remove_questions_from_tournament(
            tournament, [question2, question3]
        )

        # Verify results
        assert len(filtered_tournament.questions) == 2
        assert question1 in filtered_tournament.questions
        assert question4 in filtered_tournament.questions
        assert question2 not in filtered_tournament.questions
        assert question3 not in filtered_tournament.questions
        assert len(filtered_tournament.forecasts) == 2  # One forecast per remaining question

    def test_remove_all_questions_raises_error(self) -> None:
        # Create a tournament with one question
        question = make_question_binary("Q1")
        user = make_user("user1")
        forecast = make_forecast(question, user, [0.8])

        tournament = SimulatedTournament(forecasts=[forecast])
        assert len(tournament.questions) == 1
        assert len(tournament.forecasts) == 1

        # Attempt to remove the only question
        with pytest.raises(ValueError, match="No forecasts left after removing"):
            smart_remove_questions_from_tournament(tournament, [question])

    def test_remove_nonexistent_question(self) -> None:
        # Create a tournament with one question
        question1 = make_question_binary("Q1")
        user = make_user("user1")
        forecast = make_forecast(question1, user, [0.8])

        tournament = SimulatedTournament(forecasts=[forecast])
        assert len(tournament.questions) == 1
        assert len(tournament.forecasts) == 1

        # Create a different question to try to remove
        question2 = make_question_binary("Q2")

        # Remove question2 (which isn't in the tournament)
        filtered_tournament = smart_remove_questions_from_tournament(tournament, [question2])

        # Verify tournament is unchanged
        assert len(filtered_tournament.questions) == 1
        assert question1 in filtered_tournament.questions
        assert len(filtered_tournament.forecasts) == 1

    def test_remove_questions_with_prequalified_matches(self) -> None:
        # Create two questions that are prequalified matches
        template_question = make_question_binary("How many Grammy awards will Taylor Swift win in 2025?")
        question1 = template_question.model_copy(update={"post_id": 31797, "options": ("0", "1", "2", "3 or more"), "question_id": 1111111})
        question2 = template_question.model_copy(update={"post_id": 31865, "options": ("0", "1", "2", "Greater than 2"), "question_id": 2222222})
        question3 = template_question

        user = make_user("user1")
        forecast1 = make_forecast(question1, user, [0.8])
        forecast3 = make_forecast(question3, user, [0.6])

        tournament = SimulatedTournament(forecasts=[forecast1, forecast3])
        assert len(tournament.questions) == 2
        assert len(tournament.forecasts) == 2

        # Remove question2
        filtered_tournament = smart_remove_questions_from_tournament(tournament, [question2])

        # Verify question1 was also removed due to prequalified match
        assert len(filtered_tournament.questions) == 1
        assert len(filtered_tournament.forecasts) == 1


    def test_remove_questions_with_hash_matches(self) -> None:
        # Create two questions with the same hash
        question1 = make_question_binary("Q1")
        question2 = make_question_binary("Q1")  # Same text to ensure same hash
        question3 = make_question_binary("Q3")  # Different text to ensure different hash

        user = make_user("user1")
        forecast1 = make_forecast(question1, user, [0.8])
        forecast2 = make_forecast(question2, user, [0.7])
        forecast3 = make_forecast(question3, user, [0.6])

        tournament = SimulatedTournament(forecasts=[forecast1, forecast2, forecast3])
        assert len(tournament.questions) == 3
        assert len(tournament.forecasts) == 3

        # Remove question2
        filtered_tournament = smart_remove_questions_from_tournament(tournament, [question2])

        # Verify question1 was also removed due to hash match
        assert len(filtered_tournament.questions) == 1
        assert len(filtered_tournament.forecasts) == 1