import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.main import main

if __name__ == "__main__":
    pro_path = "input_data/pro_forecasts_q1.csv"
    bot_path = "input_data/bot_forecasts_q1.csv"
    quarterly_cup_path = "local/quarterly_cup_forecats_before_cp_reveal_time_q1.csv"
    main(pro_path, bot_path, quarterly_cup_path)