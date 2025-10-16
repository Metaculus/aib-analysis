import copy
import logging
import os
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
from aib_analysis.data_structures.problem_questions_2 import ProblemManager2
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.math.aggregate import create_aggregated_user_at_spot_time
from aib_analysis.data_structures.problem_questions_2 import (
    title_matched_questions_are_problematic,
)

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
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
    use_tourn_1_weights: bool,
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

    log_title_mapping_inconsistencies(tournament_1, tournament_2)

    matching_questions: dict[str, list[Question]] = {}
    for question_1 in tournament_1.questions:
        for question_2 in tournament_2.questions:
            hash_1 = question_1.get_hash_for_tournament_matching()
            hash_2 = question_2.get_hash_for_tournament_matching()
            hashes_match = hash_1 == hash_2

            should_be_forced_matched = ProblemManager2.should_be_forced_matched(
                question_1, question_2
            )

            if hashes_match or should_be_forced_matched:
                hash_already_used = matching_questions.get(hash_1, None) is not None
                if hash_already_used:
                    existing_questions = matching_questions[hash_1]
                    existing_urls = [question.url for question in existing_questions]
                    raise ValueError(
                        f"Question hash already had a question mapping. Existing questions: {existing_urls}. New match found: {question_1.url} and {question_2.url}"
                    )

                matching_questions[hash_1] = [question_1, question_2]

    if len(matching_questions) == 0:
        raise ValueError("No matches found between tournaments")

    combined_forecasts: list[Forecast] = []
    for question_match in matching_questions.values():
        if len(question_match) != 2:
            raise ValueError(
                f"Found {len(question_match)} questions in a match. Expected 2. {question_match}"
            )
        new_forecasts = _squash_questions_and_get_their_forecasts(
            question_match, tournament_1, tournament_2, use_tourn_1_weights
        )
        combined_forecasts.extend(new_forecasts)

    return SimulatedTournament(
        forecasts=combined_forecasts, name=f"{tournament_1.name} x {tournament_2.name}"
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
        title_matched_questions_are_problematic(
            title_matched_questions, log_results=True
        )


def _squash_questions_and_get_their_forecasts(
    questions: list[Question],
    tournament_1: SimulatedTournament,
    tournament_2: SimulatedTournament,
    use_tourn_1_weights: bool,
) -> list[Forecast]:
    question_t1, question_t2 = _validate_and_pair_tournament_questions(
        questions, tournament_1, tournament_2
    )

    logger.debug(f"Squashing questions '{question_t1.url}' and '{question_t2.url}'")

    t1_forecasts = tournament_1.question_to_forecasts(question_t1.question_id)
    t2_forecasts = tournament_2.question_to_forecasts(question_t2.question_id)
    forecasts_to_use: list[Forecast] = t1_forecasts + t2_forecasts

    if use_tourn_1_weights:
        squashed_weight = question_t1.weight
    else:
        squashed_weight = question_t2.weight

    if question_t1.weight != question_t2.weight:
        logger.warning(
            f"Question weights are different: {question_t1.weight} != {question_t2.weight}. Using tournament 1 weights are set to {use_tourn_1_weights} (this is tournament {tournament_1.name})."
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
            "weight": squashed_weight,
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
) -> SimulatedTournament | None:
    filtered_forecasts = []
    for forecast in tournament.forecasts:
        if forecast.question.type in question_types:
            filtered_forecasts.append(forecast)

    if len(filtered_forecasts) == 0:
        return None

    final_tournament = SimulatedTournament(
        forecasts=filtered_forecasts,
        name=f"{tournament.name} ({', '.join([qt.name for qt in question_types])})",
    )
    return final_tournament


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
            elif ProblemManager2.should_be_forced_matched(
                current_question, question_to_exclude
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
        logger.warning(
            f"{len(questions_to_exclude)} questions were supposed to be removed from tournament. Instead, {num_questions_removed} removals were made."
        )

    for matches_with_current_question in all_matches_in_current_tournament:
        if len(matches_with_current_question) > 1:
            logger.warning(
                f"Question {current_question.url} has multiple matches with questions to exclude: {matches_with_current_question}"
            )

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
    use_tourn_1_weights: bool,
    use_spot_scores: bool = True,
) -> SimulatedTournament:
    """
    Aggregates the forecasts of all users in each tournament.
    Then creates a new set of questions/forecasts based on the overlap between the two tournaments.
    Rescores the forecasts for this new set of questions/forecasts.
    """
    if not use_spot_scores:
        raise NotImplementedError("Not implemented")

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

    combined_tournament = combine_tournaments(
        t1_agg_tournament, t2_agg_tournament, use_tourn_1_weights=use_tourn_1_weights
    )
    return combined_tournament


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


counter = 0


def save_tournament(
    tournament_to_save: SimulatedTournament,
    file_name: str,
    divide_into_types: bool = False,
    folder: str = "local/cache/",
    counter_override: int | None = None,
):
    global counter
    if counter_override is None:
        counter += 1
        count_to_use = counter
    else:
        count_to_use = counter_override
    non_json_name = file_name.replace(".json", "")
    save_path = f"{folder}{count_to_use}_{non_json_name}"
    logger.info(f"Saving tournament {count_to_use} of {non_json_name}")
    os.makedirs(folder, exist_ok=True)

    _save_specific_tournament_to_file(tournament_to_save, f"{save_path}.json")

    if divide_into_types:
        binary_combined_tournament = constrain_question_types(
            tournament_to_save, [QuestionType.BINARY]
        )

        if binary_combined_tournament is not None:
            _save_specific_tournament_to_file(
                binary_combined_tournament, f"{save_path}__binary.json"
            )

        multiple_choice_combined_tournament = constrain_question_types(
            tournament_to_save, [QuestionType.MULTIPLE_CHOICE]
        )

        if multiple_choice_combined_tournament is not None:
            _save_specific_tournament_to_file(
                multiple_choice_combined_tournament,
                f"{save_path}__multiple_choice.json",
            )

        numeric_combined_tournament = constrain_question_types(
            tournament_to_save, [QuestionType.NUMERIC]
        )
        if numeric_combined_tournament is not None:
            _save_specific_tournament_to_file(
                numeric_combined_tournament, f"{save_path}__numeric.json"
            )


def _save_specific_tournament_to_file(
    tournament_to_save: SimulatedTournament, save_path: str
):
    modified_tournament = copy.deepcopy(tournament_to_save)
    modified_tournament.forecasts = []
    SimulatedTournament.model_validate(modified_tournament)

    try:
        with open(save_path, "w") as f:
            f.write(modified_tournament.model_dump_json(indent=4))
    except Exception as original_error:
        # Provide more detailed error information
        logger.error(
            f"Failed to serialize tournament '{modified_tournament.name}' to JSON"
        )
        logger.error(f"Error type: {type(original_error).__name__}")
        logger.error(f"Error message: {str(original_error)}")
        logger.error(f"Number of scores: {len(modified_tournament.scores)}")
        logger.error(
            f"Number of spot forecasts: {len(modified_tournament.spot_forecasts)}"
        )
        logger.error(f"Number of forecasts: {len(modified_tournament.forecasts)}")
        logger.error(f"Number of questions: {len(modified_tournament.questions)}")
        logger.error(f"Number of users: {len(modified_tournament.users)}")

        for question in modified_tournament.questions:
            try:
                pydantic_json = question.model_dump_json()
            except Exception as e:
                logger.error(
                    f"Failed to serialize question '{question.question_id}' to JSON"
                )
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")

        for forecast in modified_tournament.forecasts:
            try:
                pydantic_json = forecast.model_dump_json()
            except Exception as e:
                logger.error(f"Failed to serialize forecast '{forecast.id}' to JSON")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                logger.error(f"Forecast: {forecast}")

        for score in modified_tournament.scores:
            try:
                pydantic_json = score.model_dump_json()
            except Exception as e:
                logger.error(f"Failed to serialize score '{score.id}' to JSON")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")

        raise original_error


def create_weighted_q3_spot_forecast_tourn(
    tournament: SimulatedTournament,
) -> SimulatedTournament:
    """
    Relationships between questions are entered as tuples. These relationships
    will be used to perform logical consistency checks.

    Weights are assigned to questions based on relationships. This is a way to
    deal with correlations between questions.
    """

    # Scope sensitity list of tuples where the first entry should equal the sum of the others
    bot_scope_questions = [
        (26019, 26017, 26018),  # Starship launches
        (26098, 26096, 26097),  # SENSEX
        (26159, 26158, 26157),  # Geomagnetic storm July 28
        (26194, 26195, 26196),  # measles cases
        (26006, 26005, 26004),  # Trump lead over Biden
        (26642, 26643, 26644),  # spanish wikipedia
        (26700, 26701, 26702),  # market cap cryptocurrencies
        (27261, 27262, 27263),  # Geomagnetic storm Sept 11
    ]

    # Sum of each tuple should logically equal 1
    bot_sum_to_1_questions = [
        (25952, 25953, 25954),  # French PM party July 30
        (25957, 25958, 25959),  # Tour de France winner
        (26570, 26571, 26572, 26573),  # Warhammer
        (26574, 26575, 26576, 26577),  # H5 cases in US
        (26671, 26670, 26669),  # DOES NOT SUM TO EXACTLY 1 PM France Aug 31
        (27748, 27747, 27746, 27749),  # Speed Chess
        (27488, 27489, 27490, 27491, 27492, 27493),  # August CPI
        (27932, 27933, 27934, 27935),  # Chinese youth unemployment
        (27484, 27485, 27486, 27487),  # Fed rate cut Sept meeting
        (28045, 28044, 28043, 28042),  # Afd vote share
        (28038, 28039, 28040, 28041),  # Major Atlantic hurricanes
        (26776, 26777, 26778, 26779),  # Seattle-Tacoma-Bellevu Air Quality
    ]

    # parent, child, if_yes, if_no
    bot_conditional_pair = [(26917, 26918, 26919, 26920)]  # israel lebanon conflict

    # CDFs - Logically the probability of each successive question must not decrease
    bot_increasing_questions = [
        (26981, 26982, 26983, 26984, 26985, 26986),  # aircraft ADIZ
        (26977, 26978, 26979, 26980),  # hurricane energy
        (27548, 27547, 27546, 27545),  # mpox CDC risk level
        (28306, 28305, 28304, 28303, 28302),  # Gas prices in US Sept 30
    ]

    bot_repeated_questions = [
        (26646, 26021),  # mens 100m dash record
        (26555, 27021),  # USA gold silver
        (26210, 26917),  # israel invade lebanon
        (26781, 26304),  # ruto
        (26100, 27136),  # rfk drop out
        (25956, 27158),  # democrat brokered convention
        (26102, 27022),  # astronauts NOT EXACT REPEAT
        (26022, 27085),  # arrest warrants NOT EXACT REPEAT
        (26235, 27281),  # Buffett Indicator
        (26390, 27789),  # Bubble Magnificent 7
        (26024, 27161),  # QB Bo Nix starting for Broncos
        (26302, 27282),  # riots
        (25955, 27157),  # armed forces death US, China, Japan
        (26958, 27640),  # Youtube banned in Russia
        (25936, 27141),  # Crimean bridge attack
    ]

    bot_similar_questions = [
        (26915, 26916),  # harris favorability
        (26913, 26914),  # trump favorability
        (26193, 27733),  # debate on Sept 10
        (27886, 27968),  # Taylor Swift awards
        (27723, 27637),  # Best Rock VMAs
        (
            27583,
            27582,
            27584,
            27602,
            27603,
            27604,
        ),  # mpox Zambia, US, Angola, Russia, Japan, Mexico
        (26306, 26838),  # Richest people 250th > $10.2, 500th > 6.2
        (27887, 27969),  # Emmys Outstanding Limited or Anthology Series
        (28206, 28207, 28208, 28209, 28210),  # LMSYS leaderboard
        (28154, 28336),  # Nigeria Edo gubernatorial election
        (26407, 27897),  # Second Russian mobilization wave
        (27539, 26215),  # Nuclear weapons used
        (27606, 27607, 27608, 27609, 27610),  # Ukranian forces capture
        (26387, 27788),  # Will Tesla increase deliveries in Q3 2024
        (26821, 26959),  # VP debate
        (26212, 26213, 26214),  # number of dairy cow herds with H5N1
        (26639, 26640, 26641),  # Presidential debate 0, 1, or 2+
    ]

    ####### CREATE QUESTION WEIGHTS #########

    # Combine both lists of tuples
    all_question_tuples = (
        bot_scope_questions
        + bot_sum_to_1_questions
        + bot_increasing_questions
        + bot_similar_questions
        + bot_conditional_pair
    )

    # Do sanity checks
    all_weighted_post_ids = [
        post_id
        for tuple_questions in all_question_tuples
        for post_id in tuple_questions
    ]
    tournament_post_ids = [question.post_id for question in tournament.questions]
    assert len(set(all_weighted_post_ids)) == len(
        all_weighted_post_ids
    ), "All weighted post ids must be unique"
    assert len(set(tournament_post_ids)) == len(
        tournament_post_ids
    ), "All tournament post ids must be unique"
    weighted_ids_not_in_tournament = set(all_weighted_post_ids) - set(
        tournament_post_ids
    )
    assert (
        len(weighted_ids_not_in_tournament) == 0
    ), f"All weighted post ids must be in the tournament. weighted ids not in tournament: {weighted_ids_not_in_tournament}"
    union_of_post_ids = set(tournament_post_ids) | set(all_weighted_post_ids)
    assert len(union_of_post_ids) == len(
        tournament_post_ids
    ), "There should be no post ids not contained in the tournament id set"

    # Create an empty list to store the data
    data: dict[int, float] = {}

    # Process each tuple
    for tuple_questions in all_question_tuples:
        # Calculate the weight for each question in the tuple
        weight = np.log2(1 + len(tuple_questions)) / (1 + len(tuple_questions))

        # Add each question and its weight to the data list
        for post_id in tuple_questions:
            data[post_id] = weight

    # Process each tuple
    for tuple_questions in bot_repeated_questions:
        # 1st iteration has weight 1, 2nd has weight 1/2, 3rd weight 1/3....
        count = 1

        # Add each question and its weight to the data list
        for post_id in tuple_questions:
            weight = 1 / count
            data[post_id] = weight
            count += 1

    new_forecasts = []
    for forecast in tournament.spot_forecasts:
        question = forecast.question
        try:
            weight = data[question.post_id]
        except KeyError:
            logger.warning(
                f"Question {question.post_id} ({question.url}) not found in data"
            )
            weight = 1.0
        new_forecast = forecast.model_copy(
            update={"question": question.model_copy(update={"weight": weight})}
        )
        new_forecasts.append(new_forecast)
    return SimulatedTournament(name=tournament.name, forecasts=new_forecasts)
