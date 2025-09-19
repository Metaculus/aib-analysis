import logging
from datetime import datetime
import ast

import pandas as pd

from aib_analysis.data_structures.data_models import (
    Forecast,
    ForecastType,
    Question,
    QuestionType,
    ResolutionType,
    User,
    UserType,
)
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament

logger = logging.getLogger(__name__)


def load_tournament(
    forecast_file_path: str, user_type: UserType, tournament_name: str | None = None
) -> SimulatedTournament:
    logger.info(f"Start loading tournament from {forecast_file_path}")
    forecasts = []
    question_cache: dict[int, Question] = {}
    user_cache: dict[str, User] = {}

    dataframe = pd.read_csv(forecast_file_path, low_memory=False)

    logger.info(f"Loaded {len(dataframe)} forecast rows")
    log_every_n = 5000
    for i, (_, row) in enumerate(dataframe.iterrows()):
        should_log_parsing = i % log_every_n == 0
        if should_log_parsing:
            logger.info(f"Parsing forecast {i} of {len(dataframe)}")
        forecast, _, _ = _parse_forecast_row(
            row.to_dict(), user_type, question_cache, user_cache
        )
        forecasts.append(forecast)
    logger.info(f"Finished parsing {len(forecasts)} forecast rows")

    tournament = SimulatedTournament(forecasts=forecasts, name=tournament_name)
    _log_tournament_vs_dataframe_mismatches(tournament, dataframe)
    logger.info(f"Finished inializing tournament '{tournament.name}' from forecasts")

    return tournament


def _log_tournament_vs_dataframe_mismatches(
    tournament: SimulatedTournament, dataframe: pd.DataFrame
) -> None:
    dataframe_unique_question_ids: set[int] = set(dataframe["question_id"])
    dataframe_unique_users: set[str] = set(dataframe["forecaster"])

    tournament_unique_question_ids = set(
        [f.question.question_id for f in tournament.forecasts]
    )
    tournament_unique_users = set([f.user.name for f in tournament.forecasts])

    if dataframe_unique_question_ids != tournament_unique_question_ids:
        unique_to_dataframe = (
            dataframe_unique_question_ids - tournament_unique_question_ids
        )
        unique_to_tournament = (
            tournament_unique_question_ids - dataframe_unique_question_ids
        )
        logger.warning(
            f"Question ids in dataframe do not match question ids in tournament. IDs unique to dataframe: {unique_to_dataframe}, IDs unique to tournament: {unique_to_tournament}"
        )
    if dataframe_unique_users != tournament_unique_users:
        unique_to_dataframe = dataframe_unique_users - tournament_unique_users
        unique_to_tournament = tournament_unique_users - dataframe_unique_users
        logger.warning(
            f"Users in dataframe do not match users in tournament. Users unique to dataframe: {unique_to_dataframe}, Users unique to tournament: {unique_to_tournament}"
        )
    if len(tournament.forecasts) != len(dataframe):
        logger.warning(
            f"Number of forecasts ({len(tournament.forecasts)}) does not match number of rows in dataframe ({len(dataframe)})"
        )


def _parse_forecast_row(
    row: dict,
    user_type: UserType,
    question_cache: dict[int, Question],
    user_cache: dict[str, User],
) -> tuple[Forecast, Question, User]:
    prediction = _parse_forecast(row)
    resolution = _parse_resolution(row)
    question_id = int(row["question_id"])
    username = row["forecaster"]

    if question_id in question_cache:
        question = question_cache[question_id]
    else:
        question = Question(
            question_text=row["question_title"],
            resolution=resolution,
            weight=float(row["question_weight"]),
            spot_scoring_time=(
                pd.to_datetime(row["cp_reveal_time"])
                if pd.notnull(row.get("cp_reveal_time"))
                else pd.to_datetime(row["scheduled_close_time"])
            ),
            question_id=question_id,
            post_id=int(row["post_id"]),
            type=QuestionType(row["type"]),
            options=_parse_options(row),
            range_max=_parse_upper_bound(row),
            range_min=_parse_lower_bound(row),
            open_upper_bound=_parse_open_upper_bound(row),
            open_lower_bound=_parse_open_lower_bound(row),
            zero_point=_parse_zero_point(row),
            created_at=pd.to_datetime(row["created_at"]),
            project=row["project_title"],
        )
        question_cache[question_id] = question
    if username in user_cache:
        user = user_cache[username]
    else:
        user = User(
            name=username,
            type=user_type,
            aggregated_users=[],
        )
        user_cache[username] = user
    forecast = Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=pd.to_datetime(row["forecast_timestamp"]),
    )
    return forecast, question, user


def _parse_forecast(forecast_row: dict) -> ForecastType:
    row = forecast_row
    question_type = row["type"]
    if question_type == "binary":
        probability_yes = row["probability_yes"]
        if pd.notnull(probability_yes):
            prediction = [
                float(probability_yes),
                1 - float(probability_yes),
            ]
        else:
            prediction = None
    elif question_type == "multiple_choice":
        probability_yes_per_category = row["probability_yes_per_category"]
        if pd.notnull(probability_yes_per_category):
            prediction = eval(probability_yes_per_category)
        else:
            prediction = None
    elif question_type == "numeric":
        continuous_cdf = row["continuous_cdf"]
        if pd.notnull(continuous_cdf):
            prediction = eval(continuous_cdf)
        else:
            prediction = None
    else:
        prediction = None

    if prediction is None:
        raise ValueError(f"Invalid prediction: {prediction} for row {forecast_row}")
    return prediction


def _parse_resolution(forecast_row: dict) -> ResolutionType:
    q_type = forecast_row["type"]
    raw_resolution = forecast_row["resolution"]
    if pd.isnull(raw_resolution) or str(raw_resolution).lower() in [
        "annulled",
        "ambiguous",
    ]:
        return None
    if q_type == "binary":
        if str(raw_resolution).lower() in ["1", "true", "yes"]:
            return True
        if str(raw_resolution).lower() in ["0", "false", "no"]:
            return False
        raise ValueError(f"Invalid resolution: {raw_resolution}")
    elif q_type == "multiple_choice":
        return str(raw_resolution)
    elif q_type == "numeric":
        if raw_resolution == "above_upper_bound":
            return 1000000000000000000000000000000000.0  # Make it super obvious this is a fake number that is above upper bount
        if raw_resolution == "below_lower_bound":
            return -100000000000000000000000000000000.0
        return float(raw_resolution)

    return raw_resolution


def _parse_options(forecast_row: dict) -> tuple[str, ...] | None:
    if forecast_row["type"] == "multiple_choice":
        options = forecast_row.get("options")
        if options is not None and pd.notnull(options) and options != "":
            evaluated_options = tuple(ast.literal_eval(options))
            cleaned_options = [
                str(opt).strip().strip("'").strip('"') for opt in evaluated_options
            ]
            return tuple(cleaned_options)
        raise ValueError(f"Invalid options: {options}")
    return None


def _parse_upper_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        upper = forecast_row.get("range_max")
        if upper is not None and pd.notnull(upper) and upper != "":
            return float(upper)
        raise ValueError(f"Invalid upper bound: {upper}")
    return None


def _parse_lower_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        lower = forecast_row.get("range_min")
        if lower is not None and pd.notnull(lower) and lower != "":
            return float(lower)
        raise ValueError(f"Invalid lower bound: {lower}")
    return None


def _parse_zero_point(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        zero_point = forecast_row.get("zero_point")
        if pd.isna(zero_point):
            return None
        elif zero_point is not None and pd.notnull(zero_point) and zero_point != "":
            return float(zero_point)
        raise ValueError(f"Invalid zero point: {zero_point}")
    return None


def _parse_open_upper_bound(forecast_row: dict) -> bool | None:
    if forecast_row["type"] == "numeric":
        open_upper = forecast_row.get("open_upper_bound")
        if open_upper is not None and pd.notnull(open_upper) and open_upper != "":
            return _parse_truth_value(open_upper)
        raise ValueError(f"Invalid open upper bound: {open_upper}")
    return None


def _parse_open_lower_bound(forecast_row: dict) -> bool | None:
    if forecast_row["type"] == "numeric":
        open_lower = forecast_row.get("open_lower_bound")
        if open_lower is not None and pd.notnull(open_lower) and open_lower != "":
            return _parse_truth_value(open_lower)
        raise ValueError(f"Invalid open lower bound: {open_lower}")
    return None


def _parse_truth_value(string: str) -> bool:
    if str(string).lower() == "true":
        return True
    if str(string).lower() == "false":
        return False
    raise ValueError(f"Invalid value: {string}")
