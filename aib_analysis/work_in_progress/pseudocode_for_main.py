from __future__ import annotations

from datetime import datetime
from typing import Callable, Literal

from pydantic import BaseModel

from aib_analysis.data_models import Forecast, Question, Score, User
from aib_analysis.simulated_tournament import SimulatedTournament

# TODO: Since I'm already creating spot score calculations,
#   I might as well just input forecasts rather than scores into the tournament
#   Though I will also need to check for spot scoring timing/
#   I should check that the scoring matches the original scoring though
# TODO: Rather than the seperate tournament creation for pros and bots, create a
#   "Create tournament from tournament" function that takes in a tournament and
#   a function that returns users. The function uses the tournament to make the new users
#   a new tournament with full scores is created.


def set_up_data(path_to_data: str) -> dict[str, SimulatedTournament]:

    def load_initial_tournament(path_to_data: str) -> dict[str, SimulatedTournament]:
        # Load the data
        # Match questions between the tournaments
        # Raise errors (or require manual matching) if there are differences in the questions
        bot_tournament = None
        pro_tournament = None
        return {
            "bot_tournament": bot_tournament,
            "pro_tournament": pro_tournament,
        }

    def caculate_spot_peer_score_for_user(all_forecasts_for_question: list[Forecast], user: User) -> Score:
        # Assert forecasts are all for the same question
        # Assert that there is only one forecast per user
        # Filter for last forecast of each user that is before the spot scoring time (possibly do in previous step)
        # Calculate the score for the user (weighted by question weight)
        raise NotImplementedError("Not implemented")

    def caculate_spot_baseline_score(forecasts_for_user: list[Forecast]) -> Score:
        # Find last forecast for user that is before the spot scoring time
        # Calculate the score for the user (weighted by question weight)
        raise NotImplementedError("Not implemented")

    def caculate_all_scores_for_forecasts(forecasts: list[Forecast]) -> list[Score]:
        # Find questions
        # For each question
            # For each user
                # Calculate spot peer score
                # Calculate spot baseline score
        raise NotImplementedError("Not implemented")

    def get_bot_team_user_with_size(original_tournament: SimulatedTournament, team_size: int) -> tuple[User, list[Forecast]]:
        # Create a new user for the team
        # Create forecasts for the team
        # Calculate the scores for the user
        raise NotImplementedError("Not implemented")

    def get_all_bot_teams_as_users(original_tournament: SimulatedTournament) -> list[tuple[User, list[Forecast]]]:
        users_and_forecasts = []
        for team_size in range(1, len(original_tournament.users)):
            users_and_forecasts.extend(get_bot_team_user_with_size(original_tournament, team_size))
        return users_and_forecasts

    def get_best_bot_team_user(bot_tournament: SimulatedTournament) -> list[tuple[User, list[Forecast]]]:
        # Simulate bot team tournament
        # Grab the user and forecasts for the best bot team
        raise NotImplementedError("Not implemented")

    def get_pro_median_user(pro_tournament: SimulatedTournament) -> list[tuple[User, list[Forecast]]]:
        # Create new user
        # Create forecasts for the median
        raise NotImplementedError("Not implemented")

    def get_pro_median_and_bot_median_users(bot_tournament: SimulatedTournament, pro_tournament: SimulatedTournament) -> list[tuple[User, list[Forecast]]]:
        # Get the pro median user
        # Get the bot median user
        # Return the two users and their forecasts
        raise NotImplementedError("Not implemented")

    def create_tournament(
        original_tournament: SimulatedTournament,
        new_users: list[tuple[User, list[Forecast]]],
        remove_all_old_users: bool = False
    ) -> SimulatedTournament:
        # TODO: Also add parameter for filtering questions (or choosing new ones like only binaries)
        # assert that the forecasts given each have a corresonding question and vise versa for each user
        # Create scores for the new users and recaculate for old users
        # Make a new tournament with all the new scores
        raise NotImplementedError("Not implemented")

    original_tournament = load_initial_tournament(path_to_data)
    original_bot_tournament = original_tournament["bot_tournament"]
    original_pro_tournament = original_tournament["pro_tournament"]
    bot_team_only_tournament = create_tournament(
        original_bot_tournament,
        get_all_bot_teams_as_users(original_bot_tournament),
        remove_all_old_users=True
    )
    pro_v_bot_head_to_head_tournament = create_tournament(
        original_bot_tournament,
        get_pro_median_and_bot_median_users(original_bot_tournament, original_pro_tournament),
        remove_all_old_users=True
    )

    return {
        "original_bot_tournament": original_bot_tournament,
        "original_pro_tournament": original_pro_tournament,
        "bot_team_only_tournament": bot_team_only_tournament,
        "pro_v_bot_head_to_head_tournament": pro_v_bot_head_to_head_tournament,
    }


def display_everything(score_sets: dict[str, SimulatedTournament]) -> None:

    forecasts_to_display = score_sets["original_bot_tournament"].forecasts

    def display_calibration_curve(forecasts: list[Forecast]) -> None:
        # Each user has its own line and a 90% confidence interval
        raise NotImplementedError("Not implemented")

    def display_discrimination_curve(forecasts: list[Forecast]) -> None:
        # Each user has its own bar
        raise NotImplementedError("Not implemented")

    def display_spot_peer_score_table(tournament: SimulatedTournament, users_to_display: list[User] | None = None) -> None:
        # Filter for peer scores
        # make sure all scores are peer scores
        # make sure that all scores use the same users for calculation

        # Add these stats as a property of the simulated tournament scores
            # Caculate average spot peer score
            # Caculate sum of spot peer scores
            # Find confidence interval w/ t test
            # find confidence interval with bootstrapping
            # Weighted question count (sum of weights)
        # Show in table with a row for each user
        # Filter by users_to_display if provided
        raise NotImplementedError("Not implemented")

    def display_best_and_worse_scoring_questions(tournament: SimulatedTournament) -> None:
        # Assert there are only 2 users
        # Find the score differences between each question
        # Show the top 5 and bottom 5 questions, forecasts for those questions, the resolution, and the score difference
        raise NotImplementedError("Not implemented")

    def display_general_tournament_stats(bot_tournament: SimulatedTournament, pro_tournament: SimulatedTournament) -> None:
        # Display num pro questions
        # Display num bot questions
        # Display num pro users
        # Display num bot users
        raise NotImplementedError("Not implemented")


    metac_bots = [user for user in score_sets["original_bot_tournament"].users if user.is_metac_bot]

    display_calibration_curve(forecasts_to_display)
    display_discrimination_curve(forecasts_to_display)
    display_spot_peer_score_table(score_sets["original_bot_tournament"])
    display_spot_peer_score_table(score_sets["original_bot_tournament"], users_to_display=metac_bots)
    display_spot_peer_score_table(score_sets["bot_team_only_tournament"])
    display_spot_peer_score_table(score_sets["pro_v_bot_head_to_head_tournament"])
    display_best_and_worse_scoring_questions(score_sets["pro_v_bot_head_to_head_tournament"])
    display_general_tournament_stats(score_sets["original_bot_tournament"], score_sets["original_pro_tournament"])