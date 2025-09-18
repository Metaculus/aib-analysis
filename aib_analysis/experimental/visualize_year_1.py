import logging
import os
import sys

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.visualize_tournament import display_individual_tournament
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(tournaments_folder: str):
    initialize_logging()

    st.title("AI Benchmarking Analysis")

    all_files_in_folder = os.listdir(tournaments_folder)
    all_files_in_folder.sort()
    files_grouped_by_first_number = {}
    for file in all_files_in_folder:
        first_character = file[0]
        first_number = int(first_character)
        files_grouped_by_first_number.setdefault(first_number, []).append(file)

    tabs = st.tabs(list(f"Group {first_number}" for first_number in files_grouped_by_first_number.keys()))
    for tab, entry in zip(tabs, files_grouped_by_first_number.items()):
        with tab:
            first_number, files = entry
            for file in files:
                logger.info(f"Displaying tournament {file}")
                if file.endswith(".json"):
                    tournament = grab_tournament_data(tournaments_folder, file)
                    tournament_name = file[2:].replace(".json", "").replace("__", " | ").replace("_", " ").title()
                    display_individual_tournament(tournament, tournament_name)

    st.write("---")
    st.info("Contact ben [at] metaculus [.com] with any questions about this data")

@st.cache_data(show_spinner="Loading tournament...") # Pausing caching since this may be the cause of reruns?
def grab_tournament_data(
    folder: str, file: str,
) -> SimulatedTournament:
    with open(os.path.join(folder, file)) as f:
        logger.info(f"Loading tournament {file}")
        tournament = SimulatedTournament.model_validate_json(f.read())
        logger.info(f"Finished loading tournament {file}")
    return tournament

if __name__ == "__main__":
    # Use `streamlit run main.py``
    tournaments_folder = "local/q1_2025_tournaments/"
    main(tournaments_folder)