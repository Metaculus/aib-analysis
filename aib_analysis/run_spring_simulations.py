import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.data_models import User, UserType
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.main_logic.load_tournament import load_tournament
from aib_analysis.main_logic.process_tournament import (
    combine_tournaments,
    create_team_tournament,
    create_weighted_q3_spot_forecast_tourn,
    get_best_forecasters_from_tournament,
    save_tournament,
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

    local_counter: int = 0
    def next_count() -> int:
        nonlocal local_counter
        local_counter += 1
        return local_counter
    
    is_q3 = "q3" in output_folder.lower()
    is_q4 = "q4" in output_folder.lower()

    # ----------------------- Pros and Bot Tournaments -----------------------
    pro_tournament = grab_tournament_data(pro_path, UserType.PRO, "Pro Tournament")
    save_tournament(
        pro_tournament,
        "pro_tournament.json",
        folder=output_folder,
        counter_override=next_count(),
    )

    bot_tournament_full = grab_tournament_data(
        bot_path, UserType.BOT, "Bot Tournament Full"
    )
    if is_q3:
        bot_tournament = create_weighted_q3_spot_forecast_tourn(bot_tournament_full)
    else:
        bot_tournament = SimulatedTournament(
            name="Bot Tournament (Only spot forecasts)",
            forecasts=bot_tournament_full.spot_forecasts,
        )
    save_tournament(
        bot_tournament,
        "bot_tournament.json",
        divide_into_types=True,
        folder=output_folder,
        counter_override=next_count(),
    )

    comparison_bot_users = get_comparison_bot_users(bot_tournament)
    skip_comparison_team_outputs = False
    if len(comparison_bot_users) < 2:
        logger.warning(
            "Skipping comparison-team outputs: expected 2 or greater cross-tournament control "
            f"bots, found {len(comparison_bot_users)}: "
            f"{[user.name for user in comparison_bot_users]}"
        )
        skip_comparison_team_outputs = True

    bot_tournament_wo_pro_questions = smart_remove_questions_from_tournament(
        tournament=bot_tournament, questions_to_exclude=pro_tournament.questions
    )
    save_tournament(
        bot_tournament_wo_pro_questions,
        "bot_tournament_without_pro_questions.json",
        divide_into_types=True,
        folder=output_folder,
        counter_override=next_count(),
    )

    # ------------------------- Subdivisions of Bot Tournament -------------------------

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
        metac_bot_tournament,
        "metac_bot_tournament.json",
        folder=output_folder,
        counter_override=next_count(),
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
        counter_override=next_count(),
    )

    # ------------------------- Combine Pro and Bot Tournaments -------------------------

    use_pro_weights = False if is_q3 or is_q4 else True
    pro_with_bot_tourn = combine_tournaments(
        pro_tournament, bot_tournament, use_tourn_1_weights=use_pro_weights
    )
    save_tournament(
        pro_with_bot_tourn,
        "pro_with_bot_tourn__no_teams.json",
        divide_into_types=True,
        folder=output_folder,
        counter_override=next_count(),
    )

    team_comparison_counter = next_count()
    size_10_bot_team = None
    for bot_team_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100]:
        if bot_team_size > len(bot_tournament.users):
            continue
        pro_team = pro_tournament.users
        bot_team_for_pro_comparison = get_best_forecasters_from_tournament(
            bot_tournament_wo_pro_questions, bot_team_size
        )
        pro_v_bot_tournament__teams = create_team_tournament(
            tournament_1=pro_tournament,
            tournament_2=bot_tournament,
            team_1=pro_team,
            team_2=bot_team_for_pro_comparison,
            aggregate_name_1="Pro Team",
            aggregate_name_2="Bot Team",
            use_tourn_1_weights=use_pro_weights,
        )
        save_tournament(
            pro_v_bot_tournament__teams,
            f"pro_vs_bot_tournament__teams_size_{bot_team_size}.json",
            divide_into_types=False,
            folder=output_folder,
            counter_override=team_comparison_counter,
        )
        if bot_team_size == 10:
            save_tournament(
                pro_v_bot_tournament__teams,
                "pro_vs_bot_tournament__teams_size_10.json",
                divide_into_types=True,
                folder=output_folder,
                counter_override=next_count(),
            )
            size_10_bot_team = bot_team_for_pro_comparison

    # ------------------- Control/comparison Bots -------------------
    if not skip_comparison_team_outputs and size_10_bot_team is not None:
        number_to_use = 99
        comparison_vs_bot__teams = create_team_tournament(
            tournament_1=pro_with_bot_tourn,
            tournament_2=pro_with_bot_tourn,
            team_1=comparison_bot_users,
            team_2=size_10_bot_team,
            aggregate_name_1="Comparison Team",
            aggregate_name_2="Bot Team",
            use_tourn_1_weights=use_pro_weights,
        )
        save_tournament(
            comparison_vs_bot__teams,
            "comparison_vs_bot__teams.json",
            folder=output_folder,
            divide_into_types=True,
            counter_override=number_to_use,
        )
        comparison_vs_pros__teams = create_team_tournament(
            tournament_1=pro_with_bot_tourn,
            tournament_2=pro_with_bot_tourn,
            team_1=comparison_bot_users,
            team_2=pro_team,
            aggregate_name_1="Comparison Team",
            aggregate_name_2="Pro Team",
            use_tourn_1_weights=use_pro_weights,
        )
        save_tournament(
            comparison_vs_pros__teams,
            "comparison_vs_pro__teams.json",
            folder=output_folder,
            divide_into_types=True,
            counter_override=number_to_use,
        )

    # ------------------- Quarterly Cup -------------------
    if quarterly_cup_path is None:
        return

    cup_tournament = grab_tournament_data(
        quarterly_cup_path, UserType.BOT, "Quarterly Cup"
    )
    save_tournament(
        cup_tournament,
        "spot_scores_for_quarterly_cup.json",
        folder=output_folder,
        counter_override=next_count(),
    )

    bot_tournament_wo_cup_questions = smart_remove_questions_from_tournament(
        bot_tournament, cup_tournament.questions
    )
    save_tournament(
        bot_tournament_wo_cup_questions,
        "bot_tournament_wo_cup_questions.json",
        folder=output_folder,
        counter_override=next_count(),
    )

    cup_team_size = 10
    bot_team_for_cup_comparison = get_best_forecasters_from_tournament(
        bot_tournament_wo_cup_questions, cup_team_size
    )
    cup_vs_bot_teams = create_team_tournament(
        tournament_1=cup_tournament,
        tournament_2=bot_tournament,
        team_1="all",
        team_2=bot_team_for_cup_comparison,
        aggregate_name_1="Cup Team (All forecasters)",
        aggregate_name_2="Bot Team",
        use_tourn_1_weights=use_pro_weights,
    )
    save_tournament(
        cup_vs_bot_teams,
        "cup_vs_bot_teams.json",
        folder=output_folder,
        counter_override=next_count(),
    )


def get_comparison_bot_users(bot_tournament: SimulatedTournament) -> list[User]:
    # Cross-tournament control pair: same GPT + Claude Metaculus template lineage
    # across seasons (account names change; exactly two should match a given CSV).
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
    return comparison_bot_users


def grab_tournament_data(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


if __name__ == "__main__":
    main(
        pro_path="local/private_input_data/pro_forecasts_2026_spring.csv",
        bot_path="local/private_input_data/bot_forecasts_2026_spring.csv",
        quarterly_cup_path=None,
        output_folder="local/spring_2026_simulations_teams_comparison/",
    )