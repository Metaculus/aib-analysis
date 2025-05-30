from datetime import datetime

import pandas as pd

from refactored_notebook.data_models import (
    Forecast,
    ForecastType,
    Question,
    QuestionType,
    ResolutionType,
    User,
    UserType,
)
from refactored_notebook.simulated_tournament import SimulatedTournament


def load_tournament(
    forecast_file_path: str, user_type: UserType
) -> SimulatedTournament:
    forecasts = []
    question_cache: dict[int, Question] = {}
    user_cache: dict[str, User] = {}
    for _, row in pd.read_csv(forecast_file_path, low_memory=False).iterrows():
        forecast, _, _ = parse_forecast_row(
            row.to_dict(), user_type, question_cache, user_cache
        )
        forecasts.append(forecast)
    return SimulatedTournament(forecasts=forecasts)


def parse_forecast_row(
    row: dict,
    user_type: UserType,
    question_cache: dict[int, Question],
    user_cache: dict[str, User],
) -> tuple[Forecast, Question, User]:
    prediction = parse_forecast(row)
    resolution = parse_resolution(row)
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
            options=parse_options(row),
            upper_bound=parse_upper_bound(row),
            lower_bound=parse_lower_bound(row),
            open_upper_bound=parse_open_upper_bound(row),
            open_lower_bound=parse_open_lower_bound(row),
        )
        question_cache[question_id] = question

    if username in user_cache:
        user = user_cache[username]
    else:
        user = User(
            name=username,
            type=user_type,
            is_aggregate=False,
            aggregated_users=[],
        )
        user_cache[username] = user

    forecast = Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=pd.to_datetime(row["created_at"]),
        comment=None,
    )
    return forecast, question, user


def parse_forecast(forecast_row: dict) -> ForecastType:
    row = forecast_row
    if row["type"] == "binary":
        if pd.notnull(row["probability_yes"]):
            prediction = [
                float(row["probability_yes"]),
                1 - float(row["probability_yes"]),
            ]
        else:
            prediction = None
    elif row["type"] == "multiple_choice":
        if pd.notnull(row["probability_yes_per_category"]):
            prediction = eval(row["probability_yes_per_category"])
        else:
            prediction = None
    elif row["type"] == "numeric":
        if pd.notnull(row["continuous_cdf"]):
            prediction = eval(row["continuous_cdf"])
        else:
            prediction = None
    else:
        prediction = None
    return prediction


def parse_resolution(forecast_row: dict) -> ResolutionType:
    q_type = forecast_row["type"]
    raw_resolution = forecast_row["resolution"]
    if pd.isnull(raw_resolution):
        return None
    if q_type == "binary":
        if str(raw_resolution).lower() in ["1", "true", "yes"]:
            return True
        if str(raw_resolution).lower() in ["0", "false", "no"]:
            return False
        return None
    elif q_type == "multiple_choice":
        return str(raw_resolution)
    elif q_type == "numeric":
        try:
            return float(raw_resolution)
        except Exception:
            return None
    if str(raw_resolution).lower() in ["annulled", "ambiguous"]:
        return None
    return raw_resolution


def parse_options(forecast_row: dict) -> list[str] | None:
    if forecast_row["type"] == "multiple_choice":
        options = forecast_row.get("options")
        if options is not None and pd.notnull(options) and options != "":
            return eval(options)
        return None
    return None


def parse_upper_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        upper = forecast_row.get("range_max")
        if upper is not None and pd.notnull(upper) and upper != "":
            return float(upper)
        return None
    return None


def parse_lower_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        lower = forecast_row.get("range_min")
        if lower is not None and pd.notnull(lower) and lower != "":
            return float(lower)
        return None
    return None


def parse_open_upper_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        open_upper = forecast_row.get("open_upper_bound")
        if open_upper is not None and pd.notnull(open_upper) and open_upper != "":
            return parse_truth_value(open_upper)
        return None
    return None


def parse_open_lower_bound(forecast_row: dict) -> float | None:
    if forecast_row["type"] == "numeric":
        open_lower = forecast_row.get("open_lower_bound")
        if open_lower is not None and pd.notnull(open_lower) and open_lower != "":
            return parse_truth_value(open_lower)
        return None
    return None


def parse_truth_value(string: str) -> bool:
    if str(string).lower() == "true":
        return True
    if str(string).lower() == "false":
        return False
    raise ValueError(f"Invalid value: {string}")
