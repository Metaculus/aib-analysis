import ast
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pendulum

from aib_analysis.data_structures.data_models import (
    Forecast,
    ForecastType,
    Question,
    QuestionType,
    ResolutionType,
    User,
    UserType,
)
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)

logger = logging.getLogger(__name__)


def load_tournament(
    forecast_file_path: str, user_type: UserType, tournament_name: str | None = None
) -> SimulatedTournament:
    logger.info(f"Start loading tournament from {forecast_file_path}")
    forecasts = []
    question_cache: dict[int, Question] = {}
    user_cache: dict[str, User] = {}

    dataframe = pd.read_csv(forecast_file_path, low_memory=False)
    question_to_remove = [
        "Will the same presidential candidate win Michigan and Wisconsin in the 2024 election?"
    ] # This question in Q4 has no scores in the Metaculus database due to mis-configuration (and we don't want to update a finalized tourn).
    dataframe = dataframe[~dataframe["question_title"].isin(question_to_remove)]
    assert isinstance(dataframe, pd.DataFrame)

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
    unscored_resolution_reason = _parse_unscored_resolution_reason(row)
    question_id = int(row["question_id"])
    username = row["forecaster"]

    if question_id in question_cache:
        question = question_cache[question_id]
    else:
        question = Question(
            question_text=row["question_title"],
            resolution=resolution,
            weight=float(row["question_weight"]),
            spot_scoring_time=_resolve_spot_scoring_time(row),
            question_id=question_id,
            post_id=int(row["post_id"]),
            type=QuestionType(row["type"]),
            options=_parse_options(row),
            range_max=_parse_upper_bound(row),
            range_min=_parse_lower_bound(row),
            open_upper_bound=_parse_open_upper_bound(row),
            open_lower_bound=_parse_open_lower_bound(row),
            zero_point=_parse_zero_point(row),
            inbound_outcome_count=_parse_inbound_outcome_count(row),
            created_at=pd.to_datetime(row["created_at"]),
            project=row["project_title"],
            unscored_resolution_reason=unscored_resolution_reason,
        )
        question_cache[question_id] = question
    if username in user_cache:
        user = user_cache[username]
    else:
        is_bot = _parse_optional_bool(row, "is_bot")
        if is_bot is None:
            actual_user_type = user_type
        else:
            actual_user_type = UserType.BOT if is_bot else UserType.PRO

        user = User(
            name=username,
            type=actual_user_type,
            aggregated_users=[],
            is_primary_bot=_parse_optional_bool(row, "is_primary_bot"),
            exclude_from_aggregations=_parse_optional_bool(
                row, "exclude_from_aggregations"
            ),
        )
        user_cache[username] = user

    try:
        end_time = pendulum.parse(row["forecast_endtime"])
        assert isinstance(end_time, datetime)
    except Exception:
        end_time = None

    prediction_time = pendulum.parse(row["forecast_timestamp"])
    assert isinstance(prediction_time, datetime)

    forecasters_at_time = row.get("forecasters_at_time")
    if forecasters_at_time is not None:
        forecasters_at_time = int(forecasters_at_time)
    else:
        forecasters_at_time = None

    forecast = Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=prediction_time,
        end_time=end_time,
        forecasters_at_time=forecasters_at_time,
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
    elif question_type == "numeric" or question_type == "discrete":
        continuous_cdf = row["continuous_cdf"]
        if pd.notnull(continuous_cdf):
            prediction = eval(continuous_cdf)
            for i, p in enumerate(prediction):
                prediction[i] = float(p)
                if abs(p - 1) < 1e-6:
                    prediction[i] = 1
                elif abs(p) < 1e-6:
                    prediction[i] = 0
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
    if pd.isnull(raw_resolution):
        return None
    if str(raw_resolution).lower() in [
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
    elif q_type == "numeric" or q_type == "discrete":
        if raw_resolution == "above_upper_bound":
            return 1000000000000000000000000000000000.0  # Make it super obvious this is a fake number that is above upper bount
        if raw_resolution == "below_lower_bound":
            return -100000000000000000000000000000000.0
        return float(raw_resolution)

    return raw_resolution


def _parse_unscored_resolution_reason(forecast_row: dict) -> str | None:
    """Why resolution is non-scoring: annulled/ambiguous from CSV, or blank (often deleted)."""
    raw_resolution = forecast_row["resolution"]
    if pd.isnull(raw_resolution):
        return "blank"
    lower_resolution = str(raw_resolution).lower()
    if lower_resolution == "annulled":
        return "annulled"
    if lower_resolution == "ambiguous":
        return "ambiguous"
    return None


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
    if forecast_row["type"] == "numeric" or forecast_row["type"] == "discrete":
        upper = forecast_row.get("range_max")
        if upper is not None and pd.notnull(upper) and upper != "":
            return float(upper)
        raise ValueError(f"Invalid upper bound: {upper}")
    return None


def _parse_lower_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric" or forecast_row["type"] == "discrete":
        lower = forecast_row.get("range_min")
        if lower is not None and pd.notnull(lower) and lower != "":
            return float(lower)
        raise ValueError(f"Invalid lower bound: {lower}")
    return None


def _parse_optional_datetime(row: dict, field_name: str) -> pd.Timestamp | None:
    value = row.get(field_name)
    if value is None or pd.isnull(value):
        return None
    return pd.to_datetime(value)


def _parse_optional_bool(row: dict, field_name: str) -> bool | None:
    if field_name not in row:
        return None
    value = row[field_name]
    if value is None or (isinstance(value, float) and pd.isnull(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
        return None
    return bool(value)


def _resolve_spot_scoring_time(row: dict) -> pd.Timestamp:
    """Match Metaculus Question.get_spot_scoring_time()."""
    spot_scoring_time = _parse_optional_datetime(row, "spot_scoring_time")
    if spot_scoring_time is not None:
        return spot_scoring_time

    cp_reveal_time = _parse_optional_datetime(row, "cp_reveal_time")
    open_time = _parse_optional_datetime(row, "open_time")
    if (
        cp_reveal_time is not None
        and open_time is not None
        and cp_reveal_time > open_time
    ):
        return cp_reveal_time

    actual_close_time = _parse_optional_datetime(row, "actual_close_time")
    if actual_close_time is not None:
        return actual_close_time

    scheduled_close_time = _parse_optional_datetime(row, "scheduled_close_time")
    if scheduled_close_time is not None:
        return scheduled_close_time

    raise ValueError(
        f"Could not resolve spot_scoring_time for question_id={row.get('question_id')}"
    )


def _parse_zero_point(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric" or forecast_row["type"] == "discrete":
        zero_point = forecast_row.get("zero_point")
        if pd.isna(zero_point):
            return None
        elif zero_point is not None and pd.notnull(zero_point) and zero_point != "":
            return float(zero_point)
        raise ValueError(f"Invalid zero point: {zero_point}")
    return None

def _parse_inbound_outcome_count(forecast_row: dict) -> int | None:
    if forecast_row["type"] == "discrete":
        inbound_outcome_count = forecast_row.get("inbound_outcome_count")
        if pd.isna(inbound_outcome_count):
            return None
        elif inbound_outcome_count is not None and pd.notnull(inbound_outcome_count) and inbound_outcome_count != "":
            return int(inbound_outcome_count)
        raise ValueError(f"Invalid inbound_outcome_count: {inbound_outcome_count}")
    return None


def _parse_open_upper_bound(forecast_row: dict) -> bool | None:
    if forecast_row["type"] == "numeric" or forecast_row["type"] == "discrete":
        open_upper = forecast_row.get("open_upper_bound")
        if open_upper is not None and pd.notnull(open_upper) and open_upper != "":
            return _parse_truth_value(open_upper)
        raise ValueError(f"Invalid open upper bound: {open_upper}")
    return None


def _parse_open_lower_bound(forecast_row: dict) -> bool | None:
    if forecast_row["type"] == "numeric" or forecast_row["type"] == "discrete":
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
