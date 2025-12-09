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
    get_best_forecasters_from_tournament,
    save_tournament,
    smart_remove_questions_from_tournament,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(
    cp_path: str,
    bot_path: str,
    output_folder: str,
):
    initialize_logging()
    cp_size = 15

    local_counter: int = 0

    def next_count() -> int:
        nonlocal local_counter
        local_counter += 1
        return local_counter

    # ----------------------- CPs and Bot Tournaments -----------------------
    full_cp_tournament = grab_tournament_data(cp_path, UserType.CP, "CP Tournament")
    save_tournament(
        full_cp_tournament,
        "cp_tournament.json",
        folder=output_folder,
        counter_override=next_count(),
    )

    # Goal: >15 CP vs Bot team
    # Bot team chosen by: random selection of questions in tournament. Pick top 10 bots. choose 33% of total questions.
    # Comparison set chosen from left over, filtered for > 15 forecasters
    # Other alternative:
    # - Stratify by all possible counfounders. Question theme, question type, question easiness, CP size, etc.
    # Use fixed seed for reproducibility.

    quality_cp_forecasts = [
        forecast
        for forecast in full_cp_tournament.forecasts
        if forecast.forecasters_at_time is not None
        and forecast.forecasters_at_time >= cp_size
    ]
    quality_cp_tournament = SimulatedTournament(
        name="Comparison CP Tournament",
        forecasts=quality_cp_forecasts,
    )
    save_tournament(
        quality_cp_tournament,
        "quality_cp_tournament.json",
        folder=output_folder,
        divide_into_types=True,
        counter_override=next_count(),
    )

    comparison_cp_tournament = quality_cp_tournament
    save_tournament(
        comparison_cp_tournament,
        "comparison_cp_tournament.json",
        folder=output_folder,
        counter_override=next_count(),
    )

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
        counter_override=next_count(),
    )

    comparison_bot_users = get_comparison_bot_users(
        bot_tournament
    )  # Do this early so we can error out if we don't have right comparison bot users

    bot_team_qualification_tournament = smart_remove_questions_from_tournament(
        tournament=bot_tournament,
        questions_to_exclude=comparison_cp_tournament.questions,
    )
    save_tournament(
        bot_team_qualification_tournament,
        "bot_team_qualification_tournament.json",
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

    # ------------------------- Combine CP and Bot Tournaments -------------------------

    use_bot_tourn_weights = True
    use_weights_from_linked_cp_questions = not use_bot_tourn_weights
    cp_with_bot_tourn = combine_tournaments(
        comparison_cp_tournament,
        bot_tournament,
        use_tourn_1_weights=use_weights_from_linked_cp_questions,
    )
    save_tournament(
        cp_with_bot_tourn,
        "cp_with_bot_tourn__no_teams.json",
        divide_into_types=True,
        folder=output_folder,
        counter_override=next_count(),
    )

    team_comparison_counter = next_count()
    size_10_bot_team = None
    for bot_team_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100]:
        if bot_team_size > len(bot_tournament.users):
            continue
        cp_team = comparison_cp_tournament.users
        bot_team_for_cp_comparison = get_best_forecasters_from_tournament(
            bot_team_qualification_tournament, bot_team_size
        )
        cp_v_bot_tournament__teams = create_team_tournament(
            tournament_1=comparison_cp_tournament,
            tournament_2=bot_tournament,
            team_1=cp_team,
            team_2=bot_team_for_cp_comparison,
            aggregate_name_1="CP Team",
            aggregate_name_2="Bot Team",
            use_tourn_1_weights=use_weights_from_linked_cp_questions,
        )
        save_tournament(
            cp_v_bot_tournament__teams,
            f"cp_vs_bot_tournament__teams_size_{bot_team_size}.json",
            divide_into_types=False,
            folder=output_folder,
            counter_override=team_comparison_counter,
        )
        if bot_team_size == 10:
            save_tournament(
                cp_v_bot_tournament__teams,
                "cp_vs_bot_tournament__teams_size_10.json",
                divide_into_types=True,
                folder=output_folder,
                counter_override=next_count(),
            )
            size_10_bot_team = bot_team_for_cp_comparison

    # ------------------- Control/comparison Bots -------------------
    assert size_10_bot_team is not None, "Size 10 bot team is not None"
    number_to_use = 99
    comparison_vs_bot__teams = create_team_tournament(
        tournament_1=cp_with_bot_tourn,
        tournament_2=cp_with_bot_tourn,
        team_1=comparison_bot_users,
        team_2=size_10_bot_team,
        aggregate_name_1="Comparison Team",
        aggregate_name_2="Bot Team",
        use_tourn_1_weights=use_weights_from_linked_cp_questions,
    )
    save_tournament(
        comparison_vs_bot__teams,
        "comparison_vs_bot__teams.json",
        folder=output_folder,
        divide_into_types=True,
        counter_override=number_to_use,
    )
    comparison_vs_cps__teams = create_team_tournament(
        tournament_1=cp_with_bot_tourn,
        tournament_2=cp_with_bot_tourn,
        team_1=comparison_bot_users,
        team_2=cp_team,
        aggregate_name_1="Comparison Team",
        aggregate_name_2="CP Team",
        use_tourn_1_weights=use_weights_from_linked_cp_questions,
    )
    save_tournament(
        comparison_vs_cps__teams,
        "comparison_vs_cp__teams.json",
        folder=output_folder,
        divide_into_types=True,
        counter_override=number_to_use,
    )


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
    return comparison_bot_users


def grab_tournament_data(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


if __name__ == "__main__":

    main(
        cp_path="local/private_input_data/cp_forecasts_fall.csv",
        bot_path="local/private_input_data/bot_forecasts_fall.csv",
        output_folder="local/fall_2025_simulations_teams_comparison/",
    )
