import logging
from datetime import timedelta, timezone
from typing import Literal

import numpy as np
import typeguard
from pydantic import BaseModel
from scipy.stats import binom

from aib_analysis.data_structures.custom_types import QuestionType
from aib_analysis.data_structures.data_models import (
    Forecast,
    Leaderboard,
    LeaderboardEntry,
    Question,
    ScoreType,
    User,
)
from aib_analysis.data_structures.problem_questions import ProblemManager
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.math.aggregate import create_aggregated_user_at_spot_time

logger = logging.getLogger(__name__)


def get_leaderboard(
    tournament: SimulatedTournament, score_type: ScoreType
) -> Leaderboard:
    entries = []
    for user in tournament.users:
        scores_of_type = tournament.user_to_scores(user.name, score_type)
        if scores_of_type:
            entries.append(LeaderboardEntry(scores=scores_of_type))
        else:
            logger.warning(
                f"No scores of type {score_type} for user {user.name} when creating leaderboard"
            )
    return Leaderboard(entries=entries, type=score_type)


def combine_tournaments(
    tournament_1: SimulatedTournament, tournament_2: SimulatedTournament
) -> SimulatedTournament:
    logger.info(f"Combining tournaments {tournament_1.name} and {tournament_2.name}")

    if (
        set([user.name for user in tournament_1.users])
        & set([user.name for user in tournament_2.users])
        != set()
    ):
        raise NotImplementedError(
            "Both tournaments have some of the same users. This is currently not supported."
        )

    combined_questions: list[Question] = tournament_1.questions + tournament_2.questions

    # Check if any question titles overlap between tournaments
    t1_titles = {q.question_text.lower().strip() for q in tournament_1.questions}
    t2_titles = {q.question_text.lower().strip() for q in tournament_2.questions}
    if not t1_titles & t2_titles:
        raise ValueError("No overlapping question titles found between tournaments")

    matching_hash_mapping: dict[str, list[Question]] = {}
    for question in combined_questions:
        tournamnet_matching_hash = question.get_hash_for_tournament_matching()
        matching_hash_mapping.setdefault(tournamnet_matching_hash, []).append(question)

    log_title_mapping_inconsistencies(tournament_1, tournament_2)

    combined_forecasts: list[Forecast] = []
    hash_matches = list(matching_hash_mapping.values())
    prequalified_matches = (
        ProblemManager.find_prequalified_matches_for_tournament_matching(
            combined_questions
        )
    )
    all_matches = hash_matches + prequalified_matches

    if len(all_matches) == 0:
        raise ValueError("No matches found between tournaments")

    for question_match in all_matches:
        if len(question_match) < 2:
            continue
        new_forecasts = _squash_questions_and_get_their_forecasts(
            question_match, tournament_1, tournament_2
        )
        combined_forecasts.extend(new_forecasts)

    return SimulatedTournament(
        forecasts=combined_forecasts, name=f"{tournament_1.name} + {tournament_2.name}"
    )


def log_title_mapping_inconsistencies(
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
) -> None:
    question_text_mapping: dict[str, list[Question]] = {}
    combined_questions: list[Question] = tournament_1.questions + tournament_2.questions
    for question in combined_questions:
        cleaned_question_text = question.question_text.lower().strip()
        question_text_mapping.setdefault(cleaned_question_text, []).append(question)

    for _, title_matched_questions in question_text_mapping.items():
        if len(title_matched_questions) < 2:
            continue

        if ProblemManager.dont_log_in_tournament_matching(title_matched_questions):
            logger.info(
                f"Prequalified Mismatch for tournament matching: {[q.url for q in title_matched_questions]}"
            )
            continue

        hashes = [q.get_hash_for_tournament_matching() for q in title_matched_questions]
        if all(hash == hashes[0] for hash in hashes):
            continue

        some_questions_in_t1 = any(
            q in tournament_1.questions for q in title_matched_questions
        )
        some_questions_in_t2 = any(
            q in tournament_2.questions for q in title_matched_questions
        )

        if some_questions_in_t1 and some_questions_in_t2:
            question_comparison_table = Question.question_comparison_table(
                title_matched_questions, tournament_1.questions, tournament_2.questions
            )
            logger.warning(
                f"\n# Text-matched questions have different tournament-matching hashes "
                "(NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):\n"
                f"{question_comparison_table}"
            )
        else:
            if ProblemManager.dont_log_in_duplicate_detection_within_tournament(
                title_matched_questions
            ):
                logger.info(
                    f"Prequalified duplicate within tournament: {[q.url for q in title_matched_questions]}"
                )
                continue
            urls = [q.url for q in title_matched_questions]
            logger.warning(
                f"During combining touranments, found duplicate question title in same tournament: {urls}"
            )


def _squash_questions_and_get_their_forecasts(
    questions: list[Question],
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
) -> list[Forecast]:
    question_t1, question_t2 = _validate_and_pair_tournament_questions(
        questions, tournament_1, tournament_2
    )

    logger.debug(f"Squashing questions '{question_t1.url}' and '{question_t2.url}'")

    t1_forecasts = tournament_1.question_to_forecasts(question_t1.question_id)
    t2_forecasts = tournament_2.question_to_forecasts(question_t2.question_id)
    forecasts_to_use: list[Forecast] = t1_forecasts + t2_forecasts

    max_weight = max(question_t1.weight, question_t2.weight)
    if question_t1.weight != question_t2.weight:
        logger.warning(
            f"Question weights are different: {question_t1.weight} != {question_t2.weight}. Using the max of the two weights ({max_weight})."
        )

    max_spot_scoring_time = max(
        question_t1.spot_scoring_time, question_t2.spot_scoring_time
    )
    if question_t1.spot_scoring_time != question_t2.spot_scoring_time:
        logger.warning(
            f"Question spot scoring times are different: {question_t1.spot_scoring_time} != {question_t2.spot_scoring_time}. Using the max of the two spot scoring times ({max_spot_scoring_time})."
        )
        allowed_days_apart = timedelta(days=2)
        if (
            abs(
                question_t1.spot_scoring_time.astimezone(timezone.utc)
                - question_t2.spot_scoring_time.astimezone(timezone.utc)
            )
            > allowed_days_apart
        ):
            raise ValueError(
                f"Question spot scoring times are more than {allowed_days_apart} days apart: {question_t1.spot_scoring_time} != {question_t2.spot_scoring_time}"
            )

    squashed_question = question_t1.model_copy(
        update={
            "notes": f"Combined {question_t1.url} (QID:{question_t1.question_id}) and {question_t2.url} (QID:{question_t2.question_id})\nQ1 Notes: {question_t1.notes}\nQ2 Notes: {question_t2.notes}",
            "project": f"{question_t1.project} and {question_t2.project}",
            "weight": max_weight,
            "spot_scoring_time": max(
                question_t1.spot_scoring_time, question_t2.spot_scoring_time
            ),
        }
    )
    combined_forecasts: list[Forecast] = []
    for forecast in forecasts_to_use:
        new_forecast: Forecast = forecast.model_copy(
            update={"question": squashed_question}
        )
        assert new_forecast.question == squashed_question
        combined_forecasts.append(new_forecast)
    return combined_forecasts


def _validate_and_pair_tournament_questions(
    questions: list[Question],
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
) -> tuple[Question, Question]:
    if len(questions) > 2:
        urls = [question.url for question in questions]
        raise ValueError(
            f"Found {len(questions)} questions with the same tournament matching hash. {urls}"
        )
    assert len(questions) == 2
    question_from_t1, question_from_t2 = questions

    if not question_from_t1 in tournament_1.questions:
        raise ValueError(f"Question {question_from_t1.url} not found in tournament_1")
    if not question_from_t2 in tournament_2.questions:
        raise ValueError(f"Question {question_from_t2.url} not found in tournament_2")
    return question_from_t1, question_from_t2


def constrain_question_types(
    tournament: SimulatedTournament, question_types: list[QuestionType]
) -> SimulatedTournament:
    filtered_forecasts = []
    for forecast in tournament.forecasts:
        if forecast.question.type in question_types:
            filtered_forecasts.append(forecast)
    return SimulatedTournament(forecasts=filtered_forecasts, name=f"{tournament.name} ({', '.join([qt.name for qt in question_types])})")


def smart_remove_questions_from_tournament(
    tournament: SimulatedTournament,
    questions_to_exclude: list[Question],
    use_tournament_matching_hash: bool = True,
) -> SimulatedTournament:
    if not use_tournament_matching_hash:
        raise NotImplementedError("Not implemented")

    final_questions_to_include = []
    all_matches_in_current_tournament: list[list[Question]] = []
    for current_question in tournament.questions:
        matches_with_current_question: list[Question] = []
        for question_to_exclude in questions_to_exclude:
            exclude_hash = question_to_exclude.get_hash_for_tournament_matching()
            current_hash = current_question.get_hash_for_tournament_matching()
            if current_hash == exclude_hash:
                logger.debug(
                    f"Question {current_question.url} is in the list of questions to exclude. Removing it from the tournament."
                )
                matches_with_current_question.append(question_to_exclude)
            elif ProblemManager.is_prequalified_for_tournament_matching(
                [current_question, question_to_exclude]
            ):
                logger.debug(
                    f"Question {current_question.url} is a prequalified match. Removing it from the tournament."
                )
                matches_with_current_question.append(question_to_exclude)
        if len(matches_with_current_question) == 0:
            final_questions_to_include.append(current_question)
        all_matches_in_current_tournament.append(matches_with_current_question)

    initial_questions_count = len(tournament.questions)
    num_questions_removed = initial_questions_count - len(final_questions_to_include)
    if num_questions_removed != len(questions_to_exclude):
        logger.warning(f"{len(questions_to_exclude)} questions were supposed to be removed from tournament. Instead, {num_questions_removed} removals were made.")

    for matches_with_current_question in all_matches_in_current_tournament:
        if len(matches_with_current_question) > 1:
            logger.warning(f"Question {current_question.url} has multiple matches with questions to exclude: {matches_with_current_question}")

    filtered_forecasts = [
        forecast
        for forecast in tournament.forecasts
        if forecast.question in final_questions_to_include
    ]
    if len(filtered_forecasts) == 0:
        raise ValueError(
            f"No forecasts left after removing {len(questions_to_exclude)} questions from {tournament.name}"
        )

    return SimulatedTournament(
        forecasts=filtered_forecasts,
        name=f"{tournament.name} ({len(questions_to_exclude)} Questions removed)",
    )


def get_best_forecasters_from_tournament(
    tournament: SimulatedTournament,
    num_users: int | Literal["all"],
) -> list[User]:
    if num_users == "all":
        return tournament.users
    if num_users > len(tournament.users):
        num_users = len(tournament.users)
        logger.warning(
            f"Team size is larger than the number of users in the tournament: {num_users} > {len(tournament.users)}. Using all users."
        )
    if num_users < 1:
        raise ValueError(f"Team size is less than 1: {num_users}")

    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    entries = leaderboard.entries_via_sum_of_scores()

    top_entries = entries[:num_users]
    users = [entry.user for entry in top_entries]
    return users


def create_team_tournament(
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
    team_1: list[User] | Literal["all"],
    team_2: list[User] | Literal["all"],
    aggregate_name_1: str,
    aggregate_name_2: str,
) -> SimulatedTournament:
    if team_1 == "all":
        team_1 = tournament_1.users
    if team_2 == "all":
        team_2 = tournament_2.users

    if len(team_1) == 0:
        raise ValueError(f"Team 1 is empty: {team_1}")
    if len(team_2) == 0:
        raise ValueError(f"Team 2 is empty: {team_2}")

    t1_aggregate = create_aggregated_user_at_spot_time(
        team_1, tournament_1, aggregate_name_1
    )
    t2_aggregate = create_aggregated_user_at_spot_time(
        team_2, tournament_2, aggregate_name_2
    )

    t1_forecasts = typeguard.check_type(
        t1_aggregate.aggregate_forecasts, list[Forecast]
    )
    t2_forecasts = typeguard.check_type(
        t2_aggregate.aggregate_forecasts, list[Forecast]
    )

    t1_agg_tournament = SimulatedTournament(
        forecasts=t1_forecasts, name=f"{tournament_1.name} ({aggregate_name_1})"
    )
    t2_agg_tournament = SimulatedTournament(
        forecasts=t2_forecasts, name=f"{tournament_2.name} ({aggregate_name_2})"
    )
    return combine_tournaments(t1_agg_tournament, t2_agg_tournament)


class Bin(BaseModel):
    lower_bound: float
    upper_bound: float
    lower_confidence_interval: float
    average_resolution: float | None
    upper_confidence_interval: float
    perfect_calibration: float
    forecast_count: int

    @property
    def bin_center(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2


class CalibrationCurve(BaseModel):
    curve: list[Bin]


def calculate_calibration_curve(input_forecasts: list[Forecast]) -> CalibrationCurve:
    predictions: list[float] = []
    resolutions: list[bool] = []
    weights: list[float] = []
    for f in input_forecasts:
        resolution = f.question.resolution
        if f.question.is_annulled_or_ambiguous:
            continue
        assert (
            f.question.type == QuestionType.BINARY
        ), "Calibration curve is only supported for binary questions"
        assert f.prediction is not None, "Forecast prediction is None"
        assert isinstance(resolution, bool), f"Resolution is not a bool: {resolution}"
        predictions.append(f.prediction[0])
        resolutions.append(resolution)
        weights.append(f.question.weight)
        # TODO: @Check should I check that each question only appears once (no duplicate questions)?

    calibration_curve_bins = []
    # Same number of forecasts in each bin
    quintiles = np.quantile(predictions, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_bounds = []
    for i in range(len(quintiles) - 1):
        bin_bounds.append((quintiles[i], quintiles[i + 1]))
    for p_min, p_max in bin_bounds:
        resolutions_for_bucket = []
        weights_for_bucket = []
        bin_center = (p_min + p_max) / 2
        for value, weight, resolution in zip(predictions, weights, resolutions):
            # For the last bin, include the upper bound
            if i == len(bin_bounds) - 1:
                if p_min <= value <= p_max:
                    resolutions_for_bucket.append(resolution)
                    weights_for_bucket.append(weight)
            else:
                if p_min <= value < p_max:
                    resolutions_for_bucket.append(resolution)
                    weights_for_bucket.append(weight)
        count = max(len(resolutions_for_bucket), 1)
        average_resolution = (
            np.average(resolutions_for_bucket, weights=weights_for_bucket)
            if sum(weights_for_bucket) > 0
            else None
        )
        lower_confidence_interval = binom.ppf(0.05, count, p_min) / count
        perfect_calibration = binom.ppf(0.50, count, bin_center) / count
        upper_confidence_interval = binom.ppf(0.95, count, p_max) / count

        calibration_curve_bins.append(
            Bin(
                lower_bound=p_min,
                upper_bound=p_max,
                lower_confidence_interval=float(lower_confidence_interval),
                average_resolution=(
                    float(average_resolution)
                    if average_resolution is not None
                    else None
                ),
                upper_confidence_interval=float(upper_confidence_interval),
                perfect_calibration=float(perfect_calibration),
                forecast_count=len(resolutions_for_bucket),
            )
        )

    return CalibrationCurve(curve=calibration_curve_bins)


def find_question_titles_unique_to_first_tournament(
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
) -> list[Question]:
    question_titles_2 = set([q.question_text for q in tournament_2.questions])
    return [
        q for q in tournament_1.questions if q.question_text not in question_titles_2
    ]
