import os
import sys

import pandas as pd
import streamlit as st
import typeguard

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.aggregate import create_aggregated_user_at_spot_time
from aib_analysis.data_models import Forecast, Leaderboard, ScoreType, UserType
from aib_analysis.load_tournament import load_tournament
from aib_analysis.process_tournament import (
    combine_on_question_title_intersection,
    create_team_from_leaderboard,
    get_leaderboard,
)
from aib_analysis.simulated_tournament import SimulatedTournament
from conftest import initialize_logging


def main():
    initialize_logging()
    st.title("Tournament Forecast Explorer")
    pro_path = "input_data/pro_forecasts_q1.csv"
    bot_path = "input_data/bot_forecasts_q1.csv"
    pro_tournament = load_and_cache_tournament(pro_path, UserType.PRO)
    display_tournament(pro_tournament, "Pro")
    bot_tournament = load_and_cache_tournament(bot_path, UserType.BOT)
    display_tournament(bot_tournament, "Bot")
    combined_tournament = combine_on_question_title_intersection(
        pro_tournament, bot_tournament
    )
    display_tournament(combined_tournament, "Pro + Bot (No Teams)")
    pro_bot_aggregate_tournament = create_pro_bot_aggregate_tournament(
        pro_tournament, bot_tournament
    )
    display_tournament(pro_bot_aggregate_tournament, "Pro vs Bot (Teams)")


@st.cache_data(show_spinner="Loading tournaments...")
def load_and_cache_tournament(path: str, user_type: UserType) -> SimulatedTournament:
    return load_tournament(path, user_type)


def display_tournament(tournament: SimulatedTournament, name: str):
    st.subheader(f"{name} Tournament")
    with st.expander(f"{name} Tournament Forecasts"):
        display_forecasts(tournament)
    with st.expander(f"{name} Peer Leaderboard"):
        leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
        display_leaderboard(leaderboard)
    with st.expander(f"{name} Baseline Leaderboard"):
        leaderboard = get_leaderboard(tournament, ScoreType.SPOT_BASELINE)
        display_leaderboard(leaderboard)


def display_forecasts(tournament: SimulatedTournament):
    forecasts = tournament.forecasts
    if not forecasts:
        st.write("No forecasts available.")
        return
    # Convert forecasts to DataFrame for display
    data = [
        {
            "user": f.user.name,
            "user_type": f.user.type.value,
            "question": f.question.question_text,
            "question_type": f.question.type.value,
            "prediction": f.prediction,
            "prediction_time": f.prediction_time,
            "resolution": f.question.resolution,
            "weight": f.question.weight,
            "options": f.question.options,
            "range_max": f.question.range_max,
            "range_min": f.question.range_min,
        }
        for f in forecasts
    ]
    df = pd.DataFrame(data)
    # Truncate to first 100 rows for performance, allow filtering
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def display_leaderboard(leaderboard: Leaderboard):
    data = []
    for i, entry in enumerate(leaderboard.entries_via_sum_of_scores()):
        num_to_display = 5
        random_sample_of_scores = entry.randomly_sample_scores(num_to_display)
        top_n_scores = entry.top_n_scores(num_to_display)
        bottom_n_scores = entry.bottom_n_scores(num_to_display)
        confidence_level = 0.95
        try:
            confidence_interval = entry.get_confidence_interval(confidence_level)
            upper_bound = confidence_interval.upper_bound
            lower_bound = confidence_interval.lower_bound
        except ValueError:
            upper_bound = "Failed Normality"
            lower_bound = "Failed Normality"
        data.append(
            {
                "rank": i + 1,
                "user": entry.user.name,
                "user_type": entry.user.type.value,
                "sum_of_scores": entry.sum_of_scores,
                "average_score": entry.average_score,
                "average_upper_bound": upper_bound,
                "average_lower_bound": lower_bound,
                "num_questions": entry.question_count,
                "random_sample_of_scores": [
                    score.display_score_and_question()
                    for score in random_sample_of_scores
                ],
                "top_n_scores": [
                    score.display_score_and_question() for score in top_n_scores
                ],
                "bottom_n_scores": [
                    score.display_score_and_question() for score in bottom_n_scores
                ],
            }
        )
    st.write(f"**Confidence level**: {confidence_level}")
    df = pd.DataFrame(data)
    st.dataframe(
        df.sort_values(by="sum_of_scores", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def create_pro_bot_aggregate_tournament(
    pro_tournament: SimulatedTournament, bot_tournament: SimulatedTournament
) -> SimulatedTournament:
    pro_users = pro_tournament.users
    top_10_bot_users = create_team_from_leaderboard(
        bot_tournament, 10, ScoreType.SPOT_PEER, "sum"
    )
    pro_aggregate = create_aggregated_user_at_spot_time(
        pro_users, pro_tournament, "Pro Team"
    )
    bot_aggregate = create_aggregated_user_at_spot_time(
        top_10_bot_users, bot_tournament, "Bot Team"
    )

    pro_forecasts = typeguard.check_type(pro_aggregate.aggregate_forecasts, list[Forecast])
    bot_forecasts = typeguard.check_type(bot_aggregate.aggregate_forecasts, list[Forecast])

    pro_agg_tournament = SimulatedTournament(forecasts=pro_forecasts)
    bot_agg_tournament = SimulatedTournament(forecasts=bot_forecasts)
    return combine_on_question_title_intersection(
        pro_agg_tournament, bot_agg_tournament
    )


if __name__ == "__main__":
    main()
