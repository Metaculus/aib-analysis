import logging
import os
import sys

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.main_logic.visualize_tournament import (
    display_bot_v_pro_hypothesis_test,
    display_individual_tournament,
    display_unique_questions,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(tournaments_folder: str | None = None):
    initialize_logging()
    st.title("AI Benchmarking Analysis")

    tournaments_folder = st.text_input("Tournaments folder", value=tournaments_folder, on_change=lambda: st.rerun())

    if not tournaments_folder:
        st.write("No tournaments folder selected")
        folder_options = [f"local/{folder}" for folder in os.listdir("local/")]
        folder_options.extend([f"local/minibench/{folder}" for folder in os.listdir("local/minibench/")])
        folder_options.sort()
        st.write("- " + "\n- ".join(folder_options))
        return

    all_files_in_folder = os.listdir(tournaments_folder)
    all_files_in_folder.sort()
    files_grouped_by_first_number = {}
    for file in all_files_in_folder:
        first_number = file.split("_")[0]
        first_number = int(first_number)
        files_grouped_by_first_number.setdefault(first_number, []).append(file)

    with st.expander("Groups"):
        for first_number in files_grouped_by_first_number.keys():
            st.write(f"**Group {first_number}**")
            for file in files_grouped_by_first_number[first_number]:
                st.write(f"- {file}")

    hypothesis_test_tourns = [file for file in all_files_in_folder if "pro_vs_bot_tournament__teams.json" in file]

    # Create selectbox for group selection
    group_options = [f"Group {first_number}" for first_number in files_grouped_by_first_number.keys()]
    selected_group = st.selectbox("Select a group to display:", group_options)

    # Extract the group number from the selected option
    selected_group_number = int(selected_group.split()[-1])

    # Display tournaments for the selected group
    files: list[str] = files_grouped_by_first_number[selected_group_number]
    for file in files:
        logger.info(f"Displaying tournament {file}")
        if file.endswith(".json"):
            tournament = grab_tournament_data(tournaments_folder, file)
            tournament_name = file[2:].replace(".json", "").replace("__", " | ").replace("_", " ").title()
            if file in hypothesis_test_tourns:
                display_bot_v_pro_hypothesis_test(tournament, f"Hypothesis test for {tournament_name}")
            display_individual_tournament(tournament, tournament_name)

            if file == "7_pro_vs_bot_tournament__teams.json":
                comparison_tournament = grab_tournament_data(tournaments_folder, "1_pro_tournament.json")
                display_unique_questions(comparison_tournament, tournament)

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
    main(None)