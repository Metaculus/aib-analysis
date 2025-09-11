import logging
import os
import sys

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.data_models import UserType
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.load_tournament import load_tournament
from aib_analysis.process_tournament import (
    combine_tournaments,
    create_team_tournament,
    get_best_forecasters_from_tournament,
    smart_remove_questions_from_tournament,
)
from aib_analysis.visualize_tournament import (
    display_bot_v_pro_hypothesis_test,
    display_tournament_and_variations,
    display_unique_questions,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main(pro_path: str, bot_path: str, quarterly_cup_path: str):
    initialize_logging()
    quarterly_cup_data_is_present = os.path.exists(quarterly_cup_path)
    bot_team_size = 10

    st.title("AI Benchmarking Analysis")
    st.warning(
        "NOTE: If you interact with the page before the analysis is done running, it will occasionally rerun before finishing."
    )
    if not quarterly_cup_data_is_present:
        st.warning(
            "NOTE: Quarterly Cup data is not available in this environment. Some tabs will be disabled."
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Pro Tournament",
            "Bot Tournament",
            "Pros w/ Bots",
            "Pro vs Bot Teams",
            "Quarterly Cup",
            "Cup vs Bot Teams",
        ]
    )

    with tab1:
        pro_tournament = grab_tournament_data(
            pro_path, UserType.PRO, "Pro Tournament"
        )
        display_tournament_and_variations(pro_tournament, "Pro")

    with tab2:
        # Load and display bot tournament
        bot_tournament = grab_tournament_data(
            bot_path, UserType.BOT, "Bot Tournament"
        )
        display_tournament_and_variations(bot_tournament, "Bot", divide_into_types=True)

        bot_tournament_wo_pro_questions = smart_remove_questions_from_tournament(
            bot_tournament, pro_tournament.questions
        )
        display_tournament_and_variations(
            bot_tournament_wo_pro_questions, "Bot (No Pro Questions)"
        )

        # Display Metac bots
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
        display_tournament_and_variations(metac_bot_tournament, "Metac Bot")

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
        display_tournament_and_variations(regular_participant_tournament, "Regular Participant")

    with tab3:
        pro_with_bot_tourn = combine_tournaments(pro_tournament, bot_tournament)
        display_tournament_and_variations(
            pro_with_bot_tourn, "Pros w/ Bots (No Teams)", divide_into_types=True
        )

    with tab4:
        bot_team_for_pro_comparison = get_best_forecasters_from_tournament(
            bot_tournament_wo_pro_questions, bot_team_size
        )
        pro_bot_aggregate_tournament = create_team_tournament(
            pro_tournament,
            bot_tournament,
            team_1="all",
            team_2=bot_team_for_pro_comparison,
            aggregate_name_1="Pro Team",
            aggregate_name_2="Bot Team",
        )
        display_bot_v_pro_hypothesis_test(
            pro_bot_aggregate_tournament, "Pro vs Bot (Teams) Hypothesis Test"
        )
        display_tournament_and_variations(
            pro_bot_aggregate_tournament, "Pro vs Bot (Teams)", divide_into_types=True
        )
        st.write(
            f"---\nBot Team (best {bot_team_size} on non pro questions): {[user.name for user in bot_team_for_pro_comparison]}"
        )

    if not quarterly_cup_data_is_present:
        return

    with tab5:
        cup_tournament = grab_tournament_data(
            quarterly_cup_path, UserType.BOT, "Quarterly Cup"
        )
        display_tournament_and_variations(
            cup_tournament, "Spot Scores for Quarterly Cup"
        )

        bot_tournament_wo_cup_questions = smart_remove_questions_from_tournament(
            bot_tournament, cup_tournament.questions
        )
        display_tournament_and_variations(
            bot_tournament_wo_cup_questions, "Bot (No Cup Questions)"
        )

    with tab6:
        bot_team_for_cup_comparison = get_best_forecasters_from_tournament(
            bot_tournament_wo_cup_questions, bot_team_size
        )
        cup_vs_bot_teams = create_team_tournament(
            cup_tournament,
            bot_tournament,
            team_1="all",
            team_2=bot_team_for_cup_comparison,
            aggregate_name_1="Cup Team (All forecasters)",
            aggregate_name_2="Bot Team",
        )
        display_bot_v_pro_hypothesis_test(
            cup_vs_bot_teams, "Cup (All forecasters) vs Bot Teams Hypothesis Test"
        )
        display_tournament_and_variations(
            cup_vs_bot_teams, "Cup (All forecasters) vs Bot Teams"
        )
        display_unique_questions(cup_tournament, bot_tournament)
        st.write(
            f"---\nBot Team (best {bot_team_size} on non cup questions): {[user.name for user in bot_team_for_cup_comparison]}"
        )


# @st.cache_data(show_spinner="Loading tournaments...") # Pausing caching since this may be the cause of reruns?
def grab_tournament_data(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


if __name__ == "__main__":

    # Use `streamlit run main.py``
    pro_path = "input_data/pro_forecasts_q2.csv"
    bot_path = "input_data/bot_forecasts_q2.csv"
    quarterly_cup_path = "local/quarterly_cup_forecats_before_cp_reveal_time_q2.csv"
    main(pro_path, bot_path, quarterly_cup_path)