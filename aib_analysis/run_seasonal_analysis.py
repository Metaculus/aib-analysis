import logging
import os
import random
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.data_models import Forecast, UserType
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


analysis_section_counter: int = 0


def next_count() -> int:
    global analysis_section_counter
    analysis_section_counter += 1
    return analysis_section_counter


def main(
    cp_path: str,
    bot_path: str,
    output_folder: str,
):
    initialize_logging()
    random_seed = 42
    main_cp_size = 15
    main_team_size = 10
    random.seed(random_seed)

    # ----------------------- Base CP and Bot Tournament -----------------------
    cp_tournament = grab_tournament_data(cp_path, UserType.CP, "CP Tournament")
    save_tournament(
        cp_tournament,
        "cp_tournament.json",
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

    # ----------------------- Train Test Split -----------------------
    test_set_fraction = 0.67  # Give extra room so that the cp forecaster filter works
    test_set_size = int(len(cp_tournament.questions) * test_set_fraction)
    test_set_save_number = next_count()

    full_cp_test_set_questions = random.sample(cp_tournament.questions, test_set_size)
    full_cp_test_set_forecasts = [
        forecast
        for forecast in cp_tournament.forecasts
        if forecast.question in full_cp_test_set_questions
    ]
    full_cp_test_set_tournament = SimulatedTournament(
        name="Full CP Test Set Tournament",
        forecasts=full_cp_test_set_forecasts,
    )
    save_tournament(
        full_cp_test_set_tournament,
        "full_cp_test_set_tournament.json",
        folder=output_folder,
        counter_override=test_set_save_number,
    )

    quality_cp_test_set_forecasts = filter_by_cp_size(
        full_cp_test_set_tournament, main_cp_size
    )
    quality_cp_test_set_tournament = SimulatedTournament(
        name="Quality CP Test Set Tournament",
        forecasts=quality_cp_test_set_forecasts,
    )
    save_tournament(
        quality_cp_test_set_tournament,
        "quality_cp_test_set_tournament.json",
        folder=output_folder,
        counter_override=test_set_save_number,
    )

    # ----------------------- Bot Qualification Tournament -----------------------

    bot_team_qualification_tournament = smart_remove_questions_from_tournament(
        tournament=bot_tournament,
        questions_to_exclude=full_cp_test_set_tournament.questions,
    )
    save_tournament(
        bot_team_qualification_tournament,
        "bot_team_qualification_tournament.json",
        divide_into_types=True,
        folder=output_folder,
        counter_override=next_count(),
    )

    # ------------------------- CP vs Bot team Tournaments -------------------------
    team_comparison_counter = next_count()
    use_bot_tourn_weights = True
    for bot_team_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100]:
        if bot_team_size > len(bot_tournament.users):
            continue
        cp_team = quality_cp_test_set_tournament.users
        bot_team_for_cp_comparison = get_best_forecasters_from_tournament(
            bot_team_qualification_tournament, bot_team_size
        )
        cp_v_bot_tournament__teams = create_team_tournament(
            tournament_1=quality_cp_test_set_tournament,
            tournament_2=bot_tournament,
            team_1=cp_team,
            team_2=bot_team_for_cp_comparison,
            aggregate_name_1="CP Team",
            aggregate_name_2="Bot Team",
            use_tourn_1_weights=not use_bot_tourn_weights,
        )
        save_tournament(
            cp_v_bot_tournament__teams,
            f"cp_vs_bot_tournament__teams_size_{bot_team_size}.json",
            divide_into_types=False,
            folder=output_folder,
            counter_override=team_comparison_counter,
        )
        if bot_team_size == main_team_size:
            save_tournament(
                cp_v_bot_tournament__teams,
                f"cp_vs_bot_tournament__teams_size_{main_team_size}.json",
                divide_into_types=True,
                folder=output_folder,
                counter_override=next_count(),
            )

    # ------------------------- Other Tournaments -------------------------
    process_metac_bot_tournament(bot_tournament, output_folder)
    process_bot_maker_only_tournament(bot_tournament, output_folder)
    process_cp_with_bot_tournament(cp_tournament, bot_tournament, output_folder, main_cp_size)


def process_metac_bot_tournament(
    bot_tournament: SimulatedTournament, output_folder: str
) -> None:
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


def process_bot_maker_only_tournament(
    bot_tournament: SimulatedTournament, output_folder: str
) -> None:
    # Display regular participants
    regular_bot_maker = [
        user for user in bot_tournament.users if not user.is_metac_bot
    ]
    regular_bot_maker_forecasts = [
        forecast
        for user in regular_bot_maker
        for forecast in bot_tournament.user_to_spot_forecasts(user.name)
    ]
    regular_bot_maker_tournament = SimulatedTournament(
        name="Bot Maker Only Tournament",
        forecasts=regular_bot_maker_forecasts,
    )
    save_tournament(
        regular_bot_maker_tournament,
        "bot_maker_only_tournament.json",
        folder=output_folder,
        counter_override=next_count(),
    )


def process_cp_with_bot_tournament(
    cp_tournament: SimulatedTournament,
    bot_tournament: SimulatedTournament,
    output_folder: str,
    cp_size: int,
) -> None:
    all_quality_cp_forecasts = filter_by_cp_size(cp_tournament, cp_size)
    all_quality_cp_tournament = SimulatedTournament(
        name="All Quality CP Tournament",
        forecasts=all_quality_cp_forecasts,
    )
    use_bot_tourn_weights = True
    cp_with_bot_tourn = combine_tournaments(
        all_quality_cp_tournament,
        bot_tournament,
        use_tourn_1_weights=not use_bot_tourn_weights,
    )
    save_tournament(
        cp_with_bot_tourn,
        "cp_with_bot_tourn__no_teams.json",
        divide_into_types=True,
        folder=output_folder,
        counter_override=next_count(),
    )


def filter_by_cp_size(tournament: SimulatedTournament, cp_size: int) -> list[Forecast]:
    return [
        forecast
        for forecast in tournament.forecasts
        if forecast.forecasters_at_time is not None
        and forecast.forecasters_at_time >= cp_size
    ]


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
