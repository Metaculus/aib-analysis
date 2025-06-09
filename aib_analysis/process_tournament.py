import copy

from aib_analysis.data_models import (
    Forecast,
    Leaderboard,
    LeaderboardEntry,
    Question,
    ScoreType,
    User,
)
from aib_analysis.simulated_tournament import SimulatedTournament

problem_questions = [
    "How many Grammy awards will Taylor Swift win in 2025?", # Pro/Bot question have different options (but the one that resolved was the same)
    "Which party will win the 2nd highest number of seats in the 2025 German federal election?", # Same as above
]

def get_leaderboard(
    tournament: SimulatedTournament, score_type: ScoreType
) -> Leaderboard:
    entries = []
    for user in tournament.users:
        scores_of_type = tournament.user_to_scores(user.name, score_type)
        entries.append(LeaderboardEntry(scores=scores_of_type))
    return Leaderboard(entries=entries, type=score_type)


def combine_on_question_title_intersection(
    tournament_1: SimulatedTournament, tournament_2: SimulatedTournament
) -> SimulatedTournament:
    combined_forecasts: list[Forecast] = []
    unique_question_titles = set()
    for question_1 in tournament_1.questions:
        for question_2 in tournament_2.questions:
            question_1_text = question_1.question_text.lower()
            question_2_text = question_2.question_text.lower()

            if question_1_text == question_2_text:
                if question_1_text in unique_question_titles:
                    continue
                else:
                    unique_question_titles.add(question_1_text)

                _assert_questions_match_in_important_ways(question_1, question_2)
                new_forecasts = []
                new_forecasts.extend(
                    tournament_1.question_to_forecasts(question_1.question_id)
                )
                new_forecasts.extend(
                    tournament_2.question_to_forecasts(question_2.question_id)
                )

                new_question = copy.deepcopy(question_1)
                for forecast in new_forecasts:
                    copied_forecast: Forecast = copy.deepcopy(forecast)
                    copied_forecast.question = new_question
                    combined_forecasts.append(copied_forecast)
    tournament = SimulatedTournament(forecasts=combined_forecasts)
    return tournament


def _assert_questions_match_in_important_ways(
    question_1: Question, question_2: Question
) -> None:
    question_1_text = question_1.question_text
    if question_1_text in problem_questions:
        return
    assert (
        question_1.type == question_2.type
    ), f"Question types do not match for {question_1_text}. {question_1.type} != {question_2.type}. URL: {question_1.url} vs {question_2.url}"
    assert (
        question_1.range_max == question_2.range_max
    ), f"Question range max does not match for {question_1_text}. {question_1.range_max} != {question_2.range_max}. URL: {question_1.url} vs {question_2.url}"
    assert (
        question_1.range_min == question_2.range_min
    ), f"Question range min does not match for {question_1_text}. {question_1.range_min} != {question_2.range_min}. URL: {question_1.url} vs {question_2.url}"
    assert (
        question_1.open_upper_bound == question_2.open_upper_bound
    ), f"Question open upper bound does not match for {question_1_text}. {question_1.open_upper_bound} != {question_2.open_upper_bound}. URL: {question_1.url} vs {question_2.url}"
    assert (
        question_1.open_lower_bound == question_2.open_lower_bound
    ), f"Question open lower bound does not match for {question_1_text}. {question_1.open_lower_bound} != {question_2.open_lower_bound}. URL: {question_1.url} vs {question_2.url}"
    assert (
        question_1.options == question_2.options
        ), f"Question options do not match for {question_1_text}. {question_1.options} != {question_2.options}. URL: {question_1.url} vs {question_2.url}"
    assert (
        question_1.spot_scoring_time == question_2.spot_scoring_time
    ), f"Question spot scoring times do not match for {question_1_text}. {question_1.spot_scoring_time} != {question_2.spot_scoring_time}. URL: {question_1.url} vs {question_2.url}"


def get_ranking_by_spot_peer_score_lower_t_bound(
    tournament: SimulatedTournament, confidence_level: float
) -> list[tuple[User, float]]:
    # Get all spot peer scores
    # create a confidence interval for the spot peer score
    # Sort by lower bound
    raise NotImplementedError("Not implemented")


def get_ranking_by_spot_peer_score_sum(
    tournament: SimulatedTournament,
) -> list[tuple[User, float]]:
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
