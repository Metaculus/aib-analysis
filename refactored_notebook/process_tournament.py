from refactored_notebook.data_models import Leaderboard, LeaderboardEntry, ScoreType, User
from refactored_notebook.simulated_tournament import SimulatedTournament


def get_leaderboard(tournament: SimulatedTournament, score_type: ScoreType) -> Leaderboard:
    entries = []
    for user in tournament.users:
        all_scores_of_user_for_tournament = tournament.user_to_scores(user.name)
        scores_of_type = [
            score
            for score in all_scores_of_user_for_tournament
            if score.type == score_type
        ]

        related_question_ids = set(
            [
                score.forecast.question.question_id
                for score in all_scores_of_user_for_tournament
            ]
        )
        if score_type.is_spot_score() and len(scores_of_type) != len(
            related_question_ids
        ):
            raise ValueError(
                f"Spot scores should have exactly one score per question. User {user.name} has {len(scores_of_type)} scores for {len(related_question_ids)} questions"
            )
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
