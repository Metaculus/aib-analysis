# This file is run before any tests are run in order to configure tests

import copy
import logging
import os

import dotenv
import pytest

from aib_analysis.data_structures.custom_types import UserType
from aib_analysis.load_tournament import load_tournament
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    dotenv.load_dotenv()

    # TODO: Right now logging set up is not working. Not sure why.
    # initialize_logging()


def initialize_logging() -> None:
    enable_file_writing = os.getenv("ENABLE_FILE_WRITING", "true").lower() == "true"
    if enable_file_writing:
        os.environ["FILE_WRITING_ALLOWED"] = "TRUE"
        CustomLogger.setup_logging()
    else:
        os.environ["FILE_WRITING_ALLOWED"] = "FALSE"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - %(message)s",
        )
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