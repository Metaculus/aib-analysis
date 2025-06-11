import logging
from datetime import datetime

import numpy as np
import typeguard
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from aib_analysis.custom_types import (
    BinaryForecastType,
    MCForecastType,
    NumericForecastType,
    QuestionType,
    UserType,
)
from aib_analysis.data_models import Forecast, Question, User
from aib_analysis.simulated_tournament import SimulatedTournament

logger = logging.getLogger(__name__)


class AggregateForecast(BaseModel):
    aggregate: Forecast
    forecasts_inputted: list[Forecast]

    @property
    def relevant_question(self) -> Question:
        return self.aggregate.question

    @model_validator(mode="after")
    def all_forecasts_have_same_question(self) -> Self:
        if any(
            forecast.question != self.relevant_question
            for forecast in self.forecasts_inputted
        ):
            raise ValueError("All forecasts must have the same question")
        return self

    @model_validator(mode="after")
    def all_forecasts_have_different_users(self) -> Self:
        users_of_forecasts = list(
            set([forecast.user.name for forecast in self.forecasts_inputted])
        )
        if len(users_of_forecasts) != len(self.forecasts_inputted):
            raise ValueError(
                f"All forecasts must have different users, found: {users_of_forecasts} users and {len(self.forecasts_inputted)} forecasts"
            )
        return self


class AggregateUser(BaseModel):
    user: User
    forecasts: list[AggregateForecast]

    @property
    def relevant_questions(self) -> list[Question]:
        questions = list(
            set(forecast.relevant_question for forecast in self.forecasts)
        )
        return questions

    @property
    def forecasts_that_were_aggregated(self) -> list[Forecast]:
        forecasts = set(forecast.aggregate for forecast in self.forecasts)
        return list(forecasts)

    @model_validator(mode="after")
    def all_forecasts_have_same_user(self) -> Self:
        if any(
            forecast.aggregate.user != self.user
            for forecast in self.forecasts
        ):
            raise ValueError("All forecasts must have the same user")
        return self


def create_aggregated_user_at_spot_time(
    users: list[User],
    tournament: SimulatedTournament,
    aggregate_name: str,
) -> AggregateUser:
    """
    Takes in a list of users and tournament, and creates a aggregate forecast for each question in the tournament.
    These forecasts use a new 'aggregate' user with the name provided
    """
    each_users_forecasts = [tournament.user_to_spot_forecasts(user.name) for user in users]
    flattened_user_forecasts = [
        forecast
        for user_forecasts in each_users_forecasts
        for forecast in user_forecasts
    ]
    if len(flattened_user_forecasts) == 0:
        raise ValueError("No forecasts to aggregate")
    questions_of_forecasts = list(
        set(forecast.question for forecast in flattened_user_forecasts)
    )

    aggregate_user = User(
        name=aggregate_name, type=UserType.AGGREGATE, aggregated_users=users
    )
    aggregated_forecasts = []
    for question in questions_of_forecasts:
        forecasts_for_question = [
            forecast
            for forecast in flattened_user_forecasts
            if forecast.question == question
        ]
        aggregated_forecast = aggregate_forecasts(
            forecasts_for_question, aggregate_user, question.spot_scoring_time
        )
        aggregated_forecasts.append(aggregated_forecast)

    return AggregateUser(
        user=aggregate_user, forecasts=aggregated_forecasts
    )


def aggregate_forecasts(
    forecasts: list[Forecast], aggregate_user: User, time_of_aggregation: datetime
) -> AggregateForecast:
    if len(forecasts) == 0:
        raise ValueError("No forecasts to aggregate")

    for forecast in forecasts:
        if forecast.prediction_time > time_of_aggregation:
            raise ValueError(
                f"Forecast prediction time {forecast.prediction_time} is after the aggregation time {time_of_aggregation}"
            )

    if len(forecasts) == 1:
        users = set([forecast.user.name for forecast in forecasts])
        urls = set([forecast.question.url for forecast in forecasts])
        logger.warning(f"Only found one forecast to aggregate for question {urls} and users {users}")

    question_type = forecasts[0].question.type
    if question_type == QuestionType.BINARY:
        binary_predictions = [forecast.prediction for forecast in forecasts]
        binary_predictions = typeguard.check_type(binary_predictions, list[BinaryForecastType])
        aggregated_forecast = _aggregate_binary_forecasts(binary_predictions)
    elif question_type == QuestionType.MULTIPLE_CHOICE:
        mc_predictions = [forecast.prediction for forecast in forecasts]
        mc_predictions = typeguard.check_type(mc_predictions, list[MCForecastType])
        aggregated_forecast = _aggregate_mc_forecasts(mc_predictions)
    elif question_type == QuestionType.NUMERIC:
        numeric_predictions = [forecast.prediction for forecast in forecasts]
        numeric_predictions = typeguard.check_type(numeric_predictions, list[NumericForecastType])
        aggregated_forecast = _aggregate_numeric_forecasts(numeric_predictions)
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    return AggregateForecast(
        aggregate=Forecast(
            question=forecasts[0].question,
            user=aggregate_user,
            prediction=aggregated_forecast,
            prediction_time=forecasts[0].prediction_time,
        ),
        forecasts_inputted=forecasts,
    )


def _aggregate_binary_forecasts(
    forecasts: list[BinaryForecastType],
) -> BinaryForecastType:
    yes_probabilities = [forecast[0] for forecast in forecasts]
    median = float(np.nanmedian(yes_probabilities))
    return [median, 1 - median]

def _aggregate_mc_forecasts(forecasts: list[MCForecastType]) -> MCForecastType:
    forecasts_array = np.array(forecasts)
    mean_per_option = np.mean(forecasts_array, axis=0)
    return mean_per_option.tolist()
    # forecasts_array = np.array(forecasts)
    # median_per_option = np.nanmedian(forecasts_array, axis=0)
    # normalized = median_per_option / np.sum(median_per_option)
    # return normalized.tolist() # TODO: @Check: Is normalization of average better. The solution may be converting to and from log odds space and taking the median there

def _aggregate_numeric_forecasts(
    forecasts: list[NumericForecastType],
) -> NumericForecastType:
    cdfs = forecasts
    median_cdf: list[float] = np.median(
        np.array(cdfs), axis=0
    ).tolist()
    return median_cdf

