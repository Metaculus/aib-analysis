# This file is run before any tests are run in order to configure tests

import logging
import dotenv
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    dotenv.load_dotenv()
    set_up_logging()

def set_up_logging() -> None:
    file_name = "latest.log"
    with open(file_name, "w") as f:
        f.write("")  # Clear the contents of the log file

    logging.basicConfig(
        level=logging.INFO,
        filename=file_name,
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - %(message)s"))
    logging.getLogger().addHandler(console_handler)

