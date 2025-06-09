from refactored_notebook.data_models import Leaderboard, LeaderboardEntry, ScoreType, User
from refactored_notebook.simulated_tournament import SimulatedTournament


def get_leaderboard(tournament: SimulatedTournament, score_type: ScoreType) -> Leaderboard:
    entries = []
    for user in tournament.users:
        scores_of_type = tournament.user_to_scores(user.name, score_type)
        entries.append(LeaderboardEntry(scores=scores_of_type))
    return Leaderboard(entries=entries, type=score_type)


def get_ranking_by_spot_peer_score_lower_t_bound(
    tournament: SimulatedTournament, confidence_level: float
) -> list[tuple[User, float]]:
    # Get all spot peer scores
    # create a confidence interval for the spot peer score
    # Sort by lower bound
    raise NotImplementedError("Not implemented")

def get_ranking_by_spot_peer_score_sum(tournament: SimulatedTournament) -> list[tuple[User, float]]:
    # Get all spot peer scores
    # Sort by spot peer score
    raise NotImplementedError("Not implemented")

def get_ranking_by_spot_peer_score_bootstrap_lower_bound(
    tournament: SimulatedTournament, confidence_level: float
) -> list[tuple[User, float]]:
    # Get all spot peer scores
    # bootstrap the spot peer scores
    # create a confidence interval for the spot peer score
    # Sort by lower bound
    raise NotImplementedError("Not implemented")
