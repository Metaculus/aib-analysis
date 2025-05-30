import os
import sys

import pandas as pd
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(top_level_dir)

from refactored_notebook.data_models import UserType
from refactored_notebook.main import load_tournament
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

@st.cache_data(show_spinner="Loading tournaments...")
def cached_load_tournament(path: str, user_type: UserType) -> SimulatedTournament:
    return load_tournament(path, user_type)

def main():
    st.title("Tournament Forecast Explorer")
    pro_path = "input_data/pro_forecasts_q1.csv"
    bot_path = "input_data/bot_forecasts_q1.csv"
    pro_tournament = cached_load_tournament(pro_path, UserType.PRO)
    bot_tournament = cached_load_tournament(bot_path, UserType.BOT)
    # with st.expander("Pro Tournament Forecasts", expanded=False):
    display_tournament(pro_tournament)
    display_tournament(bot_tournament)

if __name__ == "__main__":
    main()
