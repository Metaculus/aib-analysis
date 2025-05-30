import os
import sys

import pandas as pd
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from refactored_notebook.data_models import Leaderboard, ScoreType, UserType
from refactored_notebook.load_tournament_data import load_tournament
from refactored_notebook.simulated_tournament import SimulatedTournament


def display_tournament(tournament: SimulatedTournament):
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
    for entry in leaderboard.entries:
        num_to_display = 5
        random_sample_of_scores = entry.randomly_sample_scores(num_to_display)
        top_n_scores = entry.top_n_scores(num_to_display)
        bottom_n_scores = entry.bottom_n_scores(num_to_display)
        data.append(
            {
                "user": entry.user.name,
                "user_type": entry.user.type.value,
                "sum_of_scores": entry.sum_of_scores,
                "average_score": entry.average_score,
                "num_questions": entry.question_count,
                "random_sample_of_scores": [score.display_score_and_question() for score in random_sample_of_scores],
                "top_n_scores": [score.display_score_and_question() for score in top_n_scores],
                "bottom_n_scores": [score.display_score_and_question() for score in bottom_n_scores]
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(
        df.sort_values(by="sum_of_scores", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data(show_spinner="Loading tournaments...")
def cached_load_tournament(path: str, user_type: UserType) -> SimulatedTournament:
    return load_tournament(path, user_type)


def main():
    st.title("Tournament Forecast Explorer")
    pro_path = "input_data/pro_forecasts_q1.csv"
    bot_path = "input_data/bot_forecasts_q1.csv"
    with st.expander("Pro Tournament Forecasts"):
        pro_tournament = cached_load_tournament(pro_path, UserType.PRO)
        display_tournament(pro_tournament)
    with st.expander("Pro Peer Leaderboard"):
        pro_leaderboard = pro_tournament.get_leaderboard(ScoreType.SPOT_PEER)
        display_leaderboard(pro_leaderboard)
    with st.expander("Pro Baseline Leaderboard"):
        pro_leaderboard = pro_tournament.get_leaderboard(ScoreType.SPOT_BASELINE)
        display_leaderboard(pro_leaderboard)

    with st.expander("Bot Tournament Forecasts"):
        bot_tournament = cached_load_tournament(bot_path, UserType.BOT)
        display_tournament(bot_tournament)
    with st.expander("Bot Peer Leaderboard"):
        bot_leaderboard = bot_tournament.get_leaderboard(ScoreType.SPOT_PEER)
        display_leaderboard(bot_leaderboard)


if __name__ == "__main__":
    main()
