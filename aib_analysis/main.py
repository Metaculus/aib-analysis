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
    create_team,
    create_team_tournament,
)
from aib_analysis.visualize_tournament import (
    display_bot_v_pro_hypothesis_test,
    display_tournament_and_variations,
    display_unique_questions,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)


def main():
    initialize_logging()
    pro_path = "input_data/pro_forecasts_q1.csv"
    bot_path = "input_data/bot_forecasts_q1.csv"
    quarterly_cup_path = "local/quarterly_cup_forecats_before_cp_reveal_time_q1.csv"
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
        pro_tournament = load_and_cache_tournament(
            pro_path, UserType.PRO, "Pro Tournament"
        )
        display_tournament_and_variations(pro_tournament, "Pro")

    with tab2:
        bot_tournament = load_and_cache_tournament(
            bot_path, UserType.BOT, "Bot Tournament"
        )
        display_tournament_and_variations(bot_tournament, "Bot", divide_into_types=True)

        metac_bot_users = [
            user
            for user in bot_tournament.users
            if user.is_metac_bot
        ]
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

        non_metac_bot_users = [user for user in bot_tournament.users if user.is_metac_bot]
        best_metac_bot = create_team(metac_bot_tournament, 1)
        everyone_plus_best_metac_bot = non_metac_bot_users + best_metac_bot
        everyone_plus_best_metac_bot_forecasts = [
            forecast
            for user in everyone_plus_best_metac_bot
            for forecast in bot_tournament.user_to_spot_forecasts(user.name)
        ]
        everyone_plus_best_metac_bot_tournament = SimulatedTournament(
            name="Everyone Plus Best Metac Bot Tournament",
            forecasts=everyone_plus_best_metac_bot_forecasts,
        )
        display_tournament_and_variations(everyone_plus_best_metac_bot_tournament, "Everyone Plus Best Metac Bot")

    with tab3:
        pro_with_bot_tourn = combine_tournaments(pro_tournament, bot_tournament)
        display_tournament_and_variations(
            pro_with_bot_tourn, "Pros w/ Bots (No Teams)", divide_into_types=True
        )

    with tab4:
        pro_bot_aggregate_tournament = create_team_tournament(
            pro_tournament,
            bot_tournament,
            t1_size=None,
            t2_size=bot_team_size,
            aggregate_name_1="Pro Team",
            aggregate_name_2="Bot Team",
        )
        display_bot_v_pro_hypothesis_test(pro_bot_aggregate_tournament, "Pro vs Bot (Team) Hypothesis Test")
        display_tournament_and_variations(
            pro_bot_aggregate_tournament, "Pro vs Bot (Teams)", divide_into_types=True
        )

    if not quarterly_cup_data_is_present:
        return

    with tab5:
        cup_tournament = load_and_cache_tournament(
            quarterly_cup_path, UserType.BOT, "Quarterly Cup"
        )
        display_tournament_and_variations(cup_tournament, "Spot Score Quarterly Cup")

    with tab6:
        cup_vs_bot_teams = create_team_tournament(
            cup_tournament,
            bot_tournament,
            t1_size=None,
            t2_size=bot_team_size,
            aggregate_name_1="Cup Team (All forecasters)",
            aggregate_name_2="Bot Team",
        )
        display_bot_v_pro_hypothesis_test(cup_vs_bot_teams, "Cup (All forecasters) vs Bot Teams Hypothesis Test")
        display_tournament_and_variations(
            cup_vs_bot_teams, "Cup (All forecasters) vs Bot Teams"
        )

        cup_mvp_team_vs_bot_team = create_team_tournament(
            cup_tournament,
            bot_tournament,
            t1_size=bot_team_size,
            t2_size=bot_team_size,
            aggregate_name_1="Cup MVP Team",
            aggregate_name_2="Bot Team",
        )
        display_bot_v_pro_hypothesis_test(cup_mvp_team_vs_bot_team, "Cup MVP Team vs Bot Teams Hypothesis Test")
        display_tournament_and_variations(
            cup_mvp_team_vs_bot_team, "Cup MVP Team vs Bot Teams"
        )

        display_unique_questions(cup_tournament, bot_tournament)


@st.cache_data(show_spinner="Loading tournaments...")
def load_and_cache_tournament(
    path: str, user_type: UserType, tournament_name: str
) -> SimulatedTournament:
    return load_tournament(path, user_type, tournament_name)


if __name__ == "__main__":
    main()
