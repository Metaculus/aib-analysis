# This file is run before any tests are run in order to configure tests

import logging
import dotenv
import pytest
import copy

from refactored_notebook.custom_types import UserType
from refactored_notebook.load_tournament_data import load_tournament
from refactored_notebook.simulated_tournament import SimulatedTournament

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    dotenv.load_dotenv()

    # TODO: Right now logging set up is not working. Not sure why.
    # initialize_logging()


def initialize_logging() -> None:
    file_name = "logs/latest.log"
    with open(file_name, "w") as f:
        f.write("")  # Clear the contents of the log file

    logging.basicConfig(
        level=logging.INFO,
        filename=file_name,
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - %(message)s"
        )
    )
    logging.getLogger().addHandler(console_handler)
    logger.info("Logging setup complete")


class CachedData:
    cached_loaded_pro_tournament: SimulatedTournament | None = None
    cached_loaded_bot_tournament: SimulatedTournament | None = None

@pytest.fixture(scope="function")
def pro_tournament() -> SimulatedTournament:
    if CachedData.cached_loaded_pro_tournament is None:
        file_path = "tests/test_data/pro_forecasts_q1.csv"
        user_type = UserType.PRO
        CachedData.cached_loaded_pro_tournament = load_tournament(file_path, user_type)
    return copy.deepcopy(CachedData.cached_loaded_pro_tournament)

@pytest.fixture(scope="function")
def bot_tournament() -> SimulatedTournament:
    if CachedData.cached_loaded_bot_tournament is None:
        file_path = "tests/test_data/bot_forecasts_q1.csv"
        user_type = UserType.BOT
        CachedData.cached_loaded_bot_tournament = load_tournament(file_path, user_type)
    return copy.deepcopy(CachedData.cached_loaded_bot_tournament)