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


def parse_forecast(forecast_row: dict) -> ForecastType:
    row = forecast_row
    if row["type"] == "binary":
        if pd.notnull(row["probability_yes"]):
            prediction = [float(row["probability_yes"]), 1 - float(row["probability_yes"])]
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



def parse_forecast_row(row: dict, user_type: UserType) -> tuple[Forecast, Question, User]:
    prediction = parse_forecast(row)
    resolution = parse_resolution(row)
    question = Question(
        question_text=row["question_title"],
        resolution=resolution,
        weight=float(row["question_weight"]),
        spot_scoring_time=pd.to_datetime(row["cp_reveal_time"]) if pd.notnull(row.get("cp_reveal_time")) else pd.to_datetime(row["scheduled_close_time"]),
        question_id=int(row["question_id"]),
        post_id=int(row["post_id"]),
        type=QuestionType(row["type"]),
    )
    user = User(
        name=row["forecaster"],
        type=user_type,
        is_aggregate=False,
        aggregated_users=[],
    )
    forecast = Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=pd.to_datetime(row["created_at"]),
        comment=None,
    )
    return forecast, question, user


def load_tournament(forecast_file_path: str, user_type: UserType) -> SimulatedTournament:
    forecasts = []
    for _, row in pd.read_csv(forecast_file_path, low_memory=False).iterrows():
        forecast, _, _ = parse_forecast_row(row.to_dict(), user_type)
        forecasts.append(forecast)
    return SimulatedTournament(forecasts=forecasts)
