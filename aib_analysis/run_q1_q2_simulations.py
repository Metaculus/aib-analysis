import copy
import json
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.data_models import (
    QuestionType,
    User,
    UserType,
)
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.main_logic.load_tournament import load_tournament
from aib_analysis.main_logic.process_tournament import (
    combine_tournaments,
    constrain_question_types,
    create_team_tournament,
    get_best_forecasters_from_tournament,
    smart_remove_questions_from_tournament,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(
    pro_path: str,
    bot_path: str,
    quarterly_cup_path: str | None,
    output_folder: str,
):
    initialize_logging()
    bot_team_size = 10

    pro_tournament = grab_tournament_data(pro_path, UserType.PRO, "Pro Tournament")
    save_tournament(pro_tournament, "pro_tournament.json", folder=output_folder)

    bot_tournament_full = grab_tournament_data(
        bot_path, UserType.BOT, "Bot Tournament Full"
    )
    bot_tournament = SimulatedTournament(
        name="Bot Tournament (Only spot forecasts)",
        forecasts=bot_tournament_full.spot_forecasts,
    )
    save_tournament(
        bot_tournament,
        "bot_tournament.json",
        divide_into_types=True,
        folder=output_folder,
    )

    comparison_bot_users = get_comparison_bot_users(
        bot_tournament
    )  # Do this early so we can error out if we don't have right comparison bot users

    bot_tournament_wo_pro_questions = smart_remove_questions_from_tournament(
        bot_tournament, pro_tournament.questions
    )
    save_tournament(
        bot_tournament_wo_pro_questions,
        "bot_tournament_without_pro_questions.json",
        divide_into_types=True,
        folder=output_folder,
    )

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
    save_tournament(
        metac_bot_tournament, "metac_bot_tournament.json", folder=output_folder
    )

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
    save_tournament(
        regular_participant_tournament,
        "regular_participant_tournament.json",
        folder=output_folder,
    )

    pro_with_bot_tourn = combine_tournaments(pro_tournament, bot_tournament)
    save_tournament(
        pro_with_bot_tourn,
        "pro_with_bot_tourn__no_teams.json",
        divide_into_types=True,
        folder=output_folder,
    )

    pro_team = pro_tournament.users
    bot_team_for_pro_comparison = get_best_forecasters_from_tournament(
        bot_tournament_wo_pro_questions, bot_team_size
    )
    pro_v_bot_tournament__teams = create_team_tournament(
        pro_tournament,
        bot_tournament,
        team_1=pro_team,
        team_2=bot_team_for_pro_comparison,
        aggregate_name_1="Pro Team",
        aggregate_name_2="Bot Team",
    )
    save_tournament(
        pro_v_bot_tournament__teams,
        "pro_vs_bot_tournament__teams.json",
        divide_into_types=True,
        folder=output_folder,
    )

    # ------------------- Control/comparison Bots -------------------
    number_to_use = 99
    comparison_vs_bot__teams = create_team_tournament(
        pro_with_bot_tourn,
        pro_with_bot_tourn,
        team_1=comparison_bot_users,
        team_2=bot_team_for_pro_comparison,
        aggregate_name_1="Comparison Team",
        aggregate_name_2="Bot Team",
    )
    save_tournament(
        comparison_vs_bot__teams,
        "comparison_vs_bot__teams.json",
        folder=output_folder,
        counter_override=number_to_use,
    )
    comparison_vs_pros__teams = create_team_tournament(
        pro_with_bot_tourn,
        pro_with_bot_tourn,
        team_1=comparison_bot_users,
        team_2=pro_team,
        aggregate_name_1="Comparison Team",
        aggregate_name_2="Pro Team",
    )
    save_tournament(
        comparison_vs_pros__teams,
        "comparison_vs_pro__teams.json",
        folder=output_folder,
        counter_override=number_to_use,
    )

    # ------------------- Quarterly Cup -------------------
    if quarterly_cup_path is None:
        return

    cup_tournament = grab_tournament_data(
        quarterly_cup_path, UserType.BOT, "Quarterly Cup"
    )
    save_tournament(
        cup_tournament, "spot_scores_for_quarterly_cup.json", folder=output_folder
    )

    bot_tournament_wo_cup_questions = smart_remove_questions_from_tournament(
        bot_tournament, cup_tournament.questions
    )
    save_tournament(
        bot_tournament_wo_cup_questions,
        "bot_tournament_wo_cup_questions.json",
        folder=output_folder,
    )

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
    save_tournament(cup_vs_bot_teams, "cup_vs_bot_teams.json", folder=output_folder)


def get_comparison_bot_users(bot_tournament: SimulatedTournament) -> list[User]:
    comparison_bot_names = [
        "metac-gpt-4o+asknews",  # Q2 version of gpt-4o
        "metac-claude-3-5-sonnet-20240620+asknews",  # Q2 version of claude 3.5 sonnet
        "metac-gpt-4o",  # Q1 version of gpt-4o
        "metac-claude-3-5-sonnet-20240620",  # Q1 version of claude 3.5 sonnet
        "mf-bot-1",  # Q3/4 version of gpt-4o
        "mf-bot-3",  # Q3/4 version of claude 3.5 sonnet
        # "metac-claude-3-5-sonnet-latest+asknews", # Wasn't around in Q3
    ]
    comparison_bot_users = [
        user for user in bot_tournament.users if user.name in comparison_bot_names
    ]
    assert (
        len(comparison_bot_users) == 2
    ), f"Expected 2 control bot users, got {len(comparison_bot_users)}"
    return comparison_bot_users


def grab_tournament_data(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


counter = 0


def save_tournament(
    tournament_to_save: SimulatedTournament,
    file_name: str,
    divide_into_types: bool = False,
    folder: str = "local/cache/",
    counter_override: int | None = None,
):
    global counter
    if counter_override is None:
        counter += 1
        count_to_use = counter
    else:
        count_to_use = counter_override
    non_json_name = file_name.replace(".json", "")
    save_path = f"{folder}{count_to_use}_{non_json_name}"
    logger.info(f"Saving tournament {count_to_use} of {non_json_name}")
    os.makedirs(folder, exist_ok=True)

    _save_specific_tournament_to_file(tournament_to_save, f"{save_path}.json")

    if divide_into_types:
        binary_combined_tournament = constrain_question_types(
            tournament_to_save, [QuestionType.BINARY]
        )
        multiple_choice_combined_tournament = constrain_question_types(
            tournament_to_save, [QuestionType.MULTIPLE_CHOICE]
        )
        numeric_combined_tournament = constrain_question_types(
            tournament_to_save, [QuestionType.NUMERIC]
        )
        _save_specific_tournament_to_file(
            binary_combined_tournament, f"{save_path}__binary.json"
        )
        _save_specific_tournament_to_file(
            multiple_choice_combined_tournament, f"{save_path}__multiple_choice.json"
        )
        _save_specific_tournament_to_file(
            numeric_combined_tournament, f"{save_path}__numeric.json"
        )


def _save_specific_tournament_to_file(
    tournament_to_save: SimulatedTournament, save_path: str
):
    modified_tournament = copy.deepcopy(tournament_to_save)
    modified_tournament.forecasts = []
    SimulatedTournament.model_validate(modified_tournament)

    try:
        with open(save_path, "w") as f:
            f.write(modified_tournament.model_dump_json(indent=4))
    except Exception as original_error:
        # Provide more detailed error information
        logger.error(
            f"Failed to serialize tournament '{modified_tournament.name}' to JSON"
        )
        logger.error(f"Error type: {type(original_error).__name__}")
        logger.error(f"Error message: {str(original_error)}")
        logger.error(f"Number of scores: {len(modified_tournament.scores)}")
        logger.error(
            f"Number of spot forecasts: {len(modified_tournament.spot_forecasts)}"
        )
        logger.error(f"Number of forecasts: {len(modified_tournament.forecasts)}")
        logger.error(f"Number of questions: {len(modified_tournament.questions)}")
        logger.error(f"Number of users: {len(modified_tournament.users)}")

        for question in modified_tournament.questions:
            try:
                pydantic_json = question.model_dump_json()
            except Exception as e:
                logger.error(
                    f"Failed to serialize question '{question.question_id}' to JSON"
                )
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")

        for forecast in modified_tournament.forecasts:
            try:
                pydantic_json = forecast.model_dump_json()
            except Exception as e:
                logger.error(f"Failed to serialize forecast '{forecast.id}' to JSON")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                logger.error(f"Forecast: {forecast}")

        for score in modified_tournament.scores:
            try:
                pydantic_json = score.model_dump_json()
            except Exception as e:
                logger.error(f"Failed to serialize score '{score.id}' to JSON")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")

        raise original_error


if __name__ == "__main__":
    # main(
    #     pro_path="local/private_input_data/pro_forecasts_q3.csv",
    #     bot_path="local/private_input_data/bot_forecasts_q3.csv",
    #     quarterly_cup_path=None,
    #     output_folder="local/q3_2024_simulations/",
    # )

    # main(
    #     pro_path="local/private_input_data/pro_forecasts_q4.csv",
    #     bot_path="local/private_input_data/bot_forecasts_q4.csv",
    #     quarterly_cup_path=None,
    #     output_folder="local/q4_2024_simulations/",
    # )

    # main(
    #     pro_path="input_data/pro_forecasts_q1.csv",
    #     bot_path="input_data/bot_forecasts_q1.csv",
    #     quarterly_cup_path=None,  # "local/quarterly_cup_forecats_before_cp_reveal_time_q1.csv",
    #     output_folder="local/q1_2025_simulations/",
    # )

    main(
        pro_path="input_data/pro_forecasts_q2.csv",
        bot_path="input_data/bot_forecasts_q2.csv",
        quarterly_cup_path=None,
        output_folder="local/q2_2025_simulations/",
    )
