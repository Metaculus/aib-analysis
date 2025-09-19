import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.front_end import main

if __name__ == "__main__":
    tournament_path = "local/q1_2025_simulations/"
    main(tournament_path)