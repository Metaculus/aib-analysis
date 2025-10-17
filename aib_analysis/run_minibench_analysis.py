import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.data_models import (
    UserType,
)
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.main_logic.load_tournament import load_tournament
from aib_analysis.main_logic.process_tournament import (
    save_tournament,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(
    bot_path: str,
    output_folder: str,
):
    initialize_logging()

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


def grab_tournament_data(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


if __name__ == "__main__":
    main(
        bot_path="local/private_input_data/2025-09-1-minibench.csv",
        output_folder="local/minibench/2025-09-1-minibench/",
    )

    # main(
    #     bot_path="local/private_input_data/2025-09-15-minibench.csv",
    #     output_folder="local/minibench/2025-09-15-minibench/",
    # )


