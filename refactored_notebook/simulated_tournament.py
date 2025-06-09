from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from typing_extensions import Self

from refactored_notebook.custom_types import AmbiguousResolutionType, UserType
from refactored_notebook.data_models import (
    Forecast,
    Leaderboard,
    LeaderboardEntry,
    Question,
    Score,
    ScoreType,
    User,
)

logger = logging.getLogger(__name__)


class SimulatedTournament(BaseModel):
    forecasts: list[Forecast]

    _user_cache: dict[str, User] = PrivateAttr(default_factory=dict)
    _question_cache: dict[int, Question] = PrivateAttr(default_factory=dict)
    _spot_forecasts_cache: list[Forecast] = PrivateAttr(
        default_factory=list
    )  # The latest forecast for each question that is before the spot scoring time
    _scores_cache: dict[str, Score] = PrivateAttr(default_factory=dict)

    @property
    def users(self) -> list[User]:
        assert self._user_cache is not None, "User cache is not initialized"
        return list(self._user_cache.values())

    @property
    def questions(self) -> list[Question]:
        assert self._question_cache is not None, "Question cache is not initialized"
        return list(self._question_cache.values())

    @property
    def scores(self) -> list[Score]:
        assert self._scores_cache is not None, "Scores cache is not initialized"
        return list(self._scores_cache.values())

    @property
    def spot_forecasts(self) -> list[Forecast]:
        assert self._spot_forecasts_cache is not None, "Spot forecasts cache is not initialized"
        return self._spot_forecasts_cache

    def question_to_forecasts(self, question_id: int) -> list[Forecast]:
        return [
            forecast
            for forecast in self.forecasts
            if forecast.question.question_id == question_id
        ]

    def question_to_scores(self, question_id: int) -> list[Score]:
        return [
            score
            for score in self.scores
            if score.forecast.question.question_id == question_id
        ]

    def user_to_scores(self, user_name: str) -> list[Score]:
        return [score for score in self.scores if score.forecast.user.name == user_name]

    def forecast_to_scores(self, forecast_id: str) -> list[Score]:
        return [score for score in self.scores if score.forecast.id == forecast_id]

    def user_question_type_to_scores(
        self, user_name: str, question_id: int, score_type: ScoreType
    ) -> list[Score]:
        return [
            score
            for score in self.scores
            if score.forecast.user.name == user_name
            and score.forecast.question.question_id == question_id
            and score.type == score_type
        ]


    @model_validator(mode="after")
    def initialize_tournament(self) -> Self:
        logger.info("Initializing caches")
        self._inialize_spot_forecast_cache()

        self._user_cache = {
            forecast.user.name: forecast.user for forecast in self.forecasts
        }
        self._question_cache = {
            forecast.question.question_id: forecast.question
            for forecast in self.forecasts
        }

        logger.info("Finished initializing non-scoring caches")

        log_every_n = 100
        all_scores: list[Score] = []
        for i, forecast in enumerate(self._spot_forecasts_cache):
            if i % log_every_n == 0:
                logger.info(
                    f"Caching scores for forecast {i} of {len(self._spot_forecasts_cache)}"
                )
            if isinstance(forecast.question.resolution, AmbiguousResolutionType):
                continue
            new_scores = self._calculate_spot_scores_for_forecast(forecast)
            for new_score in new_scores:
                all_scores.append(new_score)
        self._scores_cache = {score.id: score for score in all_scores}

        logger.info("Finished initializing scoring caches")
        return self

    def _inialize_spot_forecast_cache(self) -> None:
        spot_forecasts: dict[tuple[str, int], Forecast] = {}
        for forecast in self.forecasts:
            question = forecast.question
            user_name = forecast.user.name
            spot_time = question.spot_scoring_time
            if forecast.prediction_time >= spot_time:
                continue
            key = (user_name, question.question_id)
            current = spot_forecasts.get(key)
            if current is None or forecast.prediction_time > current.prediction_time:
                spot_forecasts[key] = forecast
        self._spot_forecasts_cache = list(spot_forecasts.values())


    def get_spot_score_for_question(
        self, question_id: int, user_name: str, score_type: ScoreType
    ) -> Score:
        assert (
            score_type.is_spot_score()
        ), "Only spot scores are supported for this method"
        scores = self.user_question_type_to_scores(user_name, question_id, score_type)
        assert len(scores) == 1, "Expected exactly for question for user if spot score"
        return scores[0]

    def get_forecasters_on_question(self, question_id: int) -> list[User]:
        unique_user_names = set(
            forecast.user.name
            for forecast in self.question_to_forecasts(question_id)
        )
        users = [self._user_cache[name] for name in unique_user_names]
        return users

    def _calculate_spot_scores_for_forecast(
        self, forecast_to_score: Forecast
    ) -> list[Score]:
        assert (
            forecast_to_score in self._spot_forecasts_cache
        ), "Forecast to score must be in spot forecasts cache"
        resolution = forecast_to_score.question.resolution
        if isinstance(resolution, AmbiguousResolutionType):
            return []

        spot_forecasts_from_others: list[Forecast] = (
            self.question_to_forecasts(forecast_to_score.question.question_id)
        )
        spot_forecasts_from_others.remove(forecast_to_score)
        spot_peer_score = forecast_to_score.get_spot_peer_score(
            resolution, spot_forecasts_from_others
        )
        spot_baseline_score = forecast_to_score.get_spot_baseline_score(resolution)

        scores = [spot_peer_score, spot_baseline_score]
        return scores

