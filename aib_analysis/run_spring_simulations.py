import logging
import os
import sys
from typing import Literal

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.data_models import Forecast, User, UserType
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
from aib_analysis.math.aggregate import create_aggregated_user_at_spot_time
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def set_all_question_weights_to_one(
    tournament: SimulatedTournament,
) -> SimulatedTournament:
    """Rescore a tournament with every question weight forced to 1.0."""
    new_forecasts: list[Forecast] = [
        forecast.model_copy(
            update={
                "question": forecast.question.model_copy(update={"weight": 1.0}),
            }
        )
        for forecast in tournament.spot_forecasts
    ]
    return SimulatedTournament(name=tournament.name, forecasts=new_forecasts)


def remove_users_from_tournament(
    tournament: SimulatedTournament,
    usernames: set[str],
) -> SimulatedTournament:
    """Drop all forecasts from the given users, then rescore."""
    if not usernames:
        return tournament
    present = {user.name for user in tournament.users} & usernames
    missing = usernames - {user.name for user in tournament.users}
    if missing:
        logger.warning(
            f"Users to exclude not found in {tournament.name}: {sorted(missing)}"
        )
    if not present:
        return tournament
    filtered_forecasts = [
        forecast
        for forecast in tournament.forecasts
        if forecast.user.name not in present
    ]
    logger.info(
        f"Excluding {len(present)} user(s) from scoring in {tournament.name}: "
        f"{sorted(present)}"
    )
    return SimulatedTournament(
        name=f"{tournament.name} (excluded {', '.join(sorted(present))})",
        forecasts=filtered_forecasts,
    )


def main(
    pro_path: str,
    bot_path: str,
    quarterly_cup_path: str | None,
    output_folder: str,
    force_unit_weights: bool = False,
    usernames_to_exclude_from_scoring: list[str] | None = None,
):
    initialize_logging()

    local_counter: int = 0
    def next_count() -> int:
        nonlocal local_counter
        local_counter += 1
        return local_counter
    
    is_q3 = "q3" in output_folder.lower()
    is_q4 = "q4" in output_folder.lower()
    excluded_usernames = set(usernames_to_exclude_from_scoring or [])
    if force_unit_weights:
        logger.info(
            "force_unit_weights=True: all question weights will be set to 1.0 before scoring"
        )
    if excluded_usernames:
        logger.info(
            f"Excluding from scoring: {sorted(excluded_usernames)}"
        )

    # ----------------------- Pros and Bot Tournaments -----------------------
    pro_tournament = grab_tournament_data(pro_path, UserType.PRO, "Pro Tournament")
    if force_unit_weights:
        pro_tournament = set_all_question_weights_to_one(pro_tournament)
    if excluded_usernames:
        pro_tournament = remove_users_from_tournament(
            pro_tournament, excluded_usernames
        )
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
    if force_unit_weights:
        bot_tournament = set_all_question_weights_to_one(bot_tournament)
    if excluded_usernames:
        bot_tournament = remove_users_from_tournament(
            bot_tournament, excluded_usernames
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
    bot_team_sizes: list[int | Literal["all"]] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100, "all",
    ]
    for bot_team_size in bot_team_sizes:
        if bot_team_size != "all" and bot_team_size > len(bot_tournament.users):
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

    # ------------------- Pros + bots + team aggregates (group 9) -------------------
    if size_10_bot_team is not None:
        pro_team_names = {user.name for user in pro_tournament.users}
        bot_team_names = {user.name for user in size_10_bot_team}
        pro_team_in_combined = [
            user for user in pro_with_bot_tourn.users if user.name in pro_team_names
        ]
        bot_team_in_combined = [
            user for user in pro_with_bot_tourn.users if user.name in bot_team_names
        ]
        non_commercial_bots_in_combined = [
            user
            for user in pro_with_bot_tourn.users
            if user.type == UserType.BOT and user.name not in COMMERCIAL_BOT_NAMES
        ]
        if len(pro_team_in_combined) != len(pro_team_names):
            raise ValueError(
                f"Expected {len(pro_team_names)} pros in combined tournament, "
                f"found {len(pro_team_in_combined)}"
            )
        if len(bot_team_in_combined) != len(bot_team_names):
            raise ValueError(
                f"Expected {len(bot_team_names)} bot-team members in combined tournament, "
                f"found {len(bot_team_in_combined)}"
            )
        missing_commercial_bots = COMMERCIAL_BOT_NAMES - {
            user.name for user in pro_with_bot_tourn.users if user.type == UserType.BOT
        }
        if missing_commercial_bots:
            logger.warning(
                "Some listed commercial bots were not found in the combined "
                f"tournament: {sorted(missing_commercial_bots)}"
            )
        if len(non_commercial_bots_in_combined) == 0:
            raise ValueError("No non-commercial bots found for aggregate")

        logger.info(
            f"Non-Commercial Bot Team: aggregating "
            f"{len(non_commercial_bots_in_combined)} bots "
            f"(excluded commercial bots: {sorted(COMMERCIAL_BOT_NAMES)})"
        )

        # Keep teams out of the peer geometric mean so individual scores match group 6
        # while still ranking the aggregates against that same individual pool.
        team_forecast_batches: list[list[Forecast]] = []
        for team_users, team_name in [
            (pro_team_in_combined, "Pro Team"),
            (bot_team_in_combined, "Bot Team"),
            (non_commercial_bots_in_combined, "Non-Commercial Bot Team"),
        ]:
            team_aggregate = create_aggregated_user_at_spot_time(
                team_users, pro_with_bot_tourn, team_name
            )
            team_user = team_aggregate.user.model_copy(
                update={"exclude_from_aggregations": True}
            )
            team_forecast_batches.append(
                [
                    forecast.model_copy(update={"user": team_user})
                    for forecast in team_aggregate.aggregate_forecasts
                ]
            )

        pro_bots_with_teams = SimulatedTournament(
            name="Pro + Bot with Pro / Bot / Non-Commercial Bot Teams",
            forecasts=(
                list(pro_with_bot_tourn.forecasts)
                + [forecast for batch in team_forecast_batches for forecast in batch]
            ),
        )
        save_tournament(
            pro_bots_with_teams,
            "pro_with_bot_tourn__with_teams.json",
            divide_into_types=True,
            folder=output_folder,
            counter_override=next_count(),
        )

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


# Commercial bots excluded from the Non-Commercial Bot Team aggregate.
COMMERCIAL_BOT_NAMES: set[str] = {
    "Preseen-Atlas",
    "Preseen-Chestnut",
    "manticAI",
    "cassi",
    "futuresearch",
    "lightningrod",
    "Upskillbot",
}


if __name__ == "__main__":
    main(
        pro_path="local/private_input_data/pro_forecasts_2026_spring.csv",
        bot_path="local/private_input_data/bot_forecasts_2026_spring.csv",
        quarterly_cup_path=None,
        output_folder="local/spring_2026_simulations_teams_comparison_no_preseen_chestnut/",
        force_unit_weights=False,
        usernames_to_exclude_from_scoring=["Preseen-Chestnut"],
    )