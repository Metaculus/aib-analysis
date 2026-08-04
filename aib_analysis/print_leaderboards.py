import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from aib_analysis.main_logic.process_tournament import get_leaderboard
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.data_structures.custom_types import ScoreType

def print_leaderboard(file_path):
    print(f"Leaderboard for {file_path}:")
    with open(file_path, "r") as f:
        data = json.load(f)
        
    tournament = SimulatedTournament(**data)
    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    
    sorted_entries = leaderboard.entries_via_sum_of_scores()
    
    for i, entry in enumerate(sorted_entries[:20]):
        user_name = entry.user.name
        total_score = entry.sum_of_scores
        questions = entry.question_count
        print(f"{i+1}. {user_name}: {total_score:.3f} (Questions: {questions})")
    print()

if __name__ == "__main__":
    folder = "local/spring_2026_simulations_teams_comparison"
    try:
        print_leaderboard(os.path.join(folder, "5_pro_tournament__no_teams.json"))
    except Exception as e:
        print(f"Could not load pro: {e}")
        
    try:
        print_leaderboard(os.path.join(folder, "2_bot_tournament.json"))
    except Exception as e:
        print(f"Could not load bot: {e}")

