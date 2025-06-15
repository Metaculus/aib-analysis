import logging

import pandas as pd
import pytest

from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)

logger = logging.getLogger(__name__)


def test_load_pro_data(pro_tournament: SimulatedTournament):
    validate_load_tournament_data(pro_tournament, 2840)


def test_load_bot_data(bot_tournament: SimulatedTournament):
    validate_load_tournament_data(bot_tournament, 33121)


def validate_load_tournament_data(
    tournament: SimulatedTournament, expected_forecast_count: int
):
    forecasts = tournament.forecasts
    questions = tournament.questions
    scores = tournament.scores
    users = tournament.users
    spot_forecasts = tournament.spot_forecasts

    assert (
        len(forecasts) == expected_forecast_count
    ), f"Expected {expected_forecast_count} forecasts, got {len(forecasts)}"
    assert len(questions) > 0, "Expected at least one question"
    assert len(users) > 0, "Expected at least one user"
    assert len(scores) > 0, "Expected at least one score"
    assert len(spot_forecasts) < len(
        forecasts
    ), f"Expected less than {len(forecasts)} spot forecasts, got {len(spot_forecasts)}"

    unique_questions = set(questions)
    unique_scores = set([score.id for score in scores])
    unique_forecasts = set([forecast.id for forecast in forecasts])
    unique_users = set([user.name for user in users])
    unique_spot_forecasts = set([forecast.id for forecast in spot_forecasts])

    assert len(unique_questions) == len(
        questions
    ), f"Expected all questions to be unique, got {len(unique_questions)} unique questions and {len(questions)} questions"
    assert len(unique_scores) == len(
        scores
    ), f"Expected all scores to be unique, got {len(unique_scores)} unique scores and {len(scores)} scores"
    assert len(unique_forecasts) == len(
        forecasts
    ), f"Expected all forecasts to be unique, got {len(unique_forecasts)} unique forecasts and {len(forecasts)} forecasts"
    assert len(unique_users) == len(
        users
    ), f"Expected all users to be unique, got {len(unique_users)} unique users and {len(users)} users"
    assert len(unique_spot_forecasts) == len(
        spot_forecasts
    ), f"Expected all spot forecasts to be unique, got {len(unique_spot_forecasts)} unique spot forecasts and {len(spot_forecasts)} spot forecasts"

    assert len(forecasts) > len(
        unique_questions
    ), f"Expected more forecasts than questions, got {len(forecasts)} forecasts and {len(unique_questions)} questions"
    assert (
        len(forecasts) > len(spot_forecasts) > len(questions) > len(users)
    ), f"This is a hueristic for number of objects in tournament. Got {len(scores)} scores, {len(forecasts)} forecasts, {len(spot_forecasts)} spot forecasts, {len(questions)} questions, and {len(users)} users"
    assert len(scores) > len(
        spot_forecasts
    ), f"Expected more scores than spot forecasts, got {len(scores)} scores and {len(spot_forecasts)} spot forecasts"

    # TODO: Make other validations here

@pytest.mark.skip(reason="This test is finicky and having problems with the hash function")
def test_bot_tournament_scores_against_csv(bot_tournament: SimulatedTournament) -> None:
    csv_path = "tests/test_data/imperfect_bot_scores_q1.csv"
    df = pd.read_csv(csv_path, nrows=10)
    df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S.%f%z"
    )

    # Create hash columns
    df["row_hash_without_score"] = df.apply(
        lambda row: make_hash(
            row['question_id'],
            row['forecaster_username'],
            row['forecast_timestamp'],
            row['score_type'],
            None
        ),
        axis=1
    )
    df["row_hash_with_score"] = df.apply(
        lambda row: make_hash(
            row['question_id'],
            row['forecaster_username'],
            row['forecast_timestamp'],
            row['score_type'],
            row['score']
        ),
        axis=1
    )

    scores = bot_tournament.scores
    for score in scores:
        score_forecast_timestamp = score.forecast.prediction_time.strftime(
            "%Y-%m-%d %H:%M:%S.%f%z"
        )

        # Create hash for current score
        score_hash_without_score = make_hash(
            score.forecast.question.question_id,
            score.forecast.user.name,
            score_forecast_timestamp,
            score.type.value,
            None
        )
        score_hash_with_score = make_hash(
            score.forecast.question.question_id,
            score.forecast.user.name,
            score_forecast_timestamp,
            score.type.value,
            score.score
        )

        # Match by hash with score
        matching_rows = df[df["row_hash_with_score"] == score_hash_with_score]

        if matching_rows.empty:
            # If no match with score, try matching without score
            closest_matches = df[df["row_hash_without_score"] == score_hash_without_score]

            message = (
                f"\nScore hash without score: {score_hash_without_score}\n"
                f"Score hash with score: {score_hash_with_score}\n"
                f"Matching row hashes without score: {closest_matches['row_hash_without_score'].tolist()}\n"
                f"Matching row hashes with score: {closest_matches['row_hash_with_score'].tolist()}\n"
                f"Score was: {score}\n"
                f"All closest matches: {closest_matches.to_dict(orient='records')}"
            )
            logger.error(message)
            assert False, f"No match found for score: {message}"
        elif len(matching_rows) > 1:
            logger.warning(
                f"Multiple matches found for score: {score}. The matching rows are {matching_rows}\n"
            )


def make_hash(question_id: int, user_name: str, forecast_timestamp: str, score_type: str, score: float | None) -> int:
    assert isinstance(question_id, int), f"Question ID must be an integer, got {type(question_id)}"
    assert isinstance(user_name, str), f"User name must be a string, got {type(user_name)}"
    assert isinstance(forecast_timestamp, str), f"Forecast timestamp must be a string, got {type(forecast_timestamp)}"
    assert isinstance(score_type, str), f"Score type must be a string, got {type(score_type)}"
    assert isinstance(score, float | None), f"Score must be a float or None, got {type(score)}"
    score_str = f"{score:4f}" if score is not None else "None"
    return hash(
        f"{question_id}_{user_name}_{forecast_timestamp}_{score_type}_{score_str}"
    )