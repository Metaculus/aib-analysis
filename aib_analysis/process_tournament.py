import copy
import logging
from typing import Literal

import numpy as np
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
from aib_analysis.data_structures.problem_questions import (
    poor_questions,
    problem_questions,
    ProblemGroup,
    ProblemType,
    SolutionType,
)
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.data_structures.problem_questions import ProblemManager

logger = logging.getLogger(__name__)


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
    if (
        set([user.name for user in tournament_1.users])
        & set([user.name for user in tournament_2.users])
        != set()
    ):
        raise NotImplementedError(
            "Both tournaments have some of the same users. This is currently not supported."
        )
    for question_1 in tournament_1.questions:
        for question_2 in tournament_2.questions:
            question_1_text = question_1.question_text.lower().strip()
            question_2_text = question_2.question_text.lower().strip()

            if question_1_text == question_2_text:
                if question_1_text in unique_question_titles:
                    continue
                else:
                    unique_question_titles.add(question_1_text)
                logger.info(
                    f"Found match for '{question_1_text}' vs '{question_2_text}'"
                )

                problem_group = ProblemManager.find_matching_problem_group([question_1, question_2])
                if problem_group:
                    solution = problem_group.solve_problem_group([question_1, question_2])
                    if solution == "skip_question":
                        logger.info(f"Skipping question {question_1_text} because it is in am unresolvable problem group")
                        continue
                else:
                    _assert_questions_match_in_important_ways(question_1, question_2)

                new_forecasts = []
                new_forecasts.extend(
                    tournament_1.question_to_forecasts(question_1.question_id)
                )
                new_forecasts.extend(
                    tournament_2.question_to_forecasts(question_2.question_id)
                )

                # TODO: @Check Are all relationships between objects kept correct? Should I also deep copy users? Or can I remove deep copying? Probably make this into a class function for sake of sfetry for new objects added to heirarchy.
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
    try:
        assert (
            question_1.type == question_2.type
        ), f"Question types do not match for {question_1_text}. {question_1.type} != {question_2.type}"
        assert (
            question_1.range_max == question_2.range_max
        ), f"Question range max does not match for {question_1_text}. {question_1.range_max} != {question_2.range_max}"
        assert (
            question_1.range_min == question_2.range_min
        ), f"Question range min does not match for {question_1_text}. {question_1.range_min} != {question_2.range_min}"
        assert (
            question_1.open_upper_bound == question_2.open_upper_bound
        ), f"Question open upper bound does not match for {question_1_text}. {question_1.open_upper_bound} != {question_2.open_upper_bound}"
        assert (
            question_1.open_lower_bound == question_2.open_lower_bound
        ), f"Question open lower bound does not match for {question_1_text}. {question_1.open_lower_bound} != {question_2.open_lower_bound}"
        assert (
            question_1.options == question_2.options
        ), f"Question options do not match for {question_1_text}. {question_1.options} != {question_2.options}"
        assert (
            question_1.spot_scoring_time == question_2.spot_scoring_time
        ), f"Question spot scoring times do not match for {question_1_text}. {question_1.spot_scoring_time} != {question_2.spot_scoring_time}"
    except AssertionError as e:
        if (
            ProblemManager.find_matching_problem_group([question_1, question_2])
            is not None
        ):
            ProblemManager.save_problem_group(
                ProblemGroup(
                    notes=f"AssertionError: {e}",
                    questions=[question_1, question_2],
                    problem_type=ProblemType.BETWEEN_TOURNAMENT,
                )
            )
        question_comparison_table = (
            f"\n{Question.question_comparison_table([question_1, question_2])}"
        )
        logger.error(f"AssertionError: {e}.\n{question_comparison_table}")
        raise e


def constrain_question_types(
    tournament: SimulatedTournament, question_types: list[QuestionType]
) -> SimulatedTournament:
    # TODO: @Check should I do a deep copy here?
    filtered_forecasts = []
    for forecast in tournament.forecasts:
        if forecast.question.type in question_types:
            filtered_forecasts.append(forecast)
    return SimulatedTournament(forecasts=filtered_forecasts)


def create_team_from_leaderboard(
    tournament: SimulatedTournament,
    team_size: int,
    score_type: ScoreType,
    approach: Literal["sum", "average", "average_lower_t"],
) -> list[User]:
    leaderboard = get_leaderboard(tournament, score_type)
    if approach == "sum":
        entries = leaderboard.entries_via_sum_of_scores()
    elif approach == "average":
        entries = leaderboard.entries_via_average_score()
    else:
        raise ValueError(f"Approach not supported: {approach}")
    top_entries = entries[:team_size]
    users = [entry.user for entry in top_entries]
    return users


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
