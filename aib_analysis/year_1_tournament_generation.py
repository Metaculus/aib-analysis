import logging
import os

from aib_analysis.data_structures.data_models import QuestionType, UserType
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.load_tournament import load_tournament
from aib_analysis.process_tournament import (
    combine_tournaments,
    constrain_question_types,
    create_team_tournament,
    get_best_forecasters_from_tournament,
    smart_remove_questions_from_tournament,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(pro_path: str, bot_path: str, quarterly_cup_path: str):
    initialize_logging()
    quarterly_cup_data_is_present = os.path.exists(quarterly_cup_path)
    bot_team_size = 10

    pro_tournament = grab_tournament_data(
        pro_path, UserType.PRO, "Pro Tournament"
    )
    save_tournament(pro_tournament, "pro_tournament.json")

    bot_tournament = grab_tournament_data(
        bot_path, UserType.BOT, "Bot Tournament"
    )
    save_tournament(bot_tournament, "bot_tournament.json", divide_into_types=True)

    bot_tournament_wo_pro_questions = smart_remove_questions_from_tournament(
        bot_tournament, pro_tournament.questions
    )
    save_tournament(bot_tournament_wo_pro_questions, "bot_tournament_wo_pro_questions.json", divide_into_types=True)

    metac_bot_users = [user for user in bot_tournament.users if user.is_metac_bot]
    metac_bot_forecasts = [
        forecast
        for user in metac_bot_users
        for forecast in bot_tournament.user_to_spot_forecasts(user.name)
    ]

    metac_bot_tournament = SimulatedTournament(
        name="Metac Bot Tournament",
        forecasts=metac_bot_forecasts,
    )
    save_tournament(metac_bot_tournament, "metac_bot_tournament.json")

    # Display regular participants
    regular_participant_users = [
        user for user in bot_tournament.users if not user.is_metac_bot
    ]
    regular_participant_forecasts = [
        forecast
        for user in regular_participant_users
        for forecast in bot_tournament.user_to_spot_forecasts(user.name)
    ]
    regular_participant_tournament = SimulatedTournament(
        name="Regular Participant Tournament",
        forecasts=regular_participant_forecasts,
    )
    save_tournament(regular_participant_tournament, "regular_participant_tournament.json")

    pro_with_bot_tourn = combine_tournaments(pro_tournament, bot_tournament)
    save_tournament(pro_with_bot_tourn, "pro_with_bot_tourn__no_teams.json", divide_into_types=True)

    bot_team_for_pro_comparison = get_best_forecasters_from_tournament(
        bot_tournament_wo_pro_questions, bot_team_size
    )
    pro_v_bot_tournament__teams = create_team_tournament(
        pro_tournament,
        bot_tournament,
        team_1="all",
        team_2=bot_team_for_pro_comparison,
        aggregate_name_1="Pro Team",
        aggregate_name_2="Bot Team",
    )
    save_tournament(pro_v_bot_tournament__teams, "pro_v_bot_tournament__teams.json", divide_into_types=True)

    cup_tournament = grab_tournament_data(
        quarterly_cup_path, UserType.BOT, "Quarterly Cup"
    )
    save_tournament(cup_tournament, "spot_scores_for_quarterly_cup.json")

    bot_tournament_wo_cup_questions = smart_remove_questions_from_tournament(
        bot_tournament, cup_tournament.questions
    )
    save_tournament(bot_tournament_wo_cup_questions, "bot_tournament_wo_cup_questions.json")

    bot_team_for_cup_comparison = get_best_forecasters_from_tournament(
        bot_tournament_wo_cup_questions, bot_team_size
    )
    cup_vs_bot_teams = create_team_tournament(
        cup_tournament,
        bot_tournament,
        team_1="all",
        team_2=bot_team_for_cup_comparison,
        aggregate_name_1="Cup Team (All forecasters)",
        aggregate_name_2="Bot Team",
    )
    save_tournament(cup_vs_bot_teams, "cup_vs_bot_teams.json")


def grab_tournament_data(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


def save_tournament(tournament: SimulatedTournament, path: str, divide_into_types: bool = False, folder = "local/cache/"):
    tournament_json = tournament.model_dump_json()

    with open(f"{folder}{path}", "w") as f:
        f.write(tournament_json)

    if divide_into_types:
        binary_combined_tournament = constrain_question_types(
            tournament, [QuestionType.BINARY]
        )
        multiple_choice_combined_tournament = constrain_question_types(
            tournament, [QuestionType.MULTIPLE_CHOICE]
        )
        numeric_combined_tournament = constrain_question_types(
            tournament, [QuestionType.NUMERIC]
        )
        with open(f"{folder}{path}_binary.json", "w") as f:
            f.write(binary_combined_tournament.model_dump_json())

        with open(f"{folder}{path}_multiple_choice.json", "w") as f:
            f.write(multiple_choice_combined_tournament.model_dump_json())

        with open(f"{folder}{path}_numeric.json", "w") as f:
            f.write(numeric_combined_tournament.model_dump_json())


if __name__ == "__main__":
    # Use `streamlit run main.py``
    pro_path = "input_data/pro_forecasts_q2.csv"
    bot_path = "input_data/bot_forecasts_q2.csv"
    quarterly_cup_path = "local/quarterly_cup_forecats_before_cp_reveal_time_q2.csv"
    main(pro_path, bot_path, quarterly_cup_path)