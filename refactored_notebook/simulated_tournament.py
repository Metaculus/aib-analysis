from __future__ import annotations

import logging

from pydantic import BaseModel, PrivateAttr, model_validator
from typing_extensions import Self

from refactored_notebook.custom_types import AmbiguousResolutionType
from refactored_notebook.data_models import (
    Forecast,
    Question,
    Score,
    ScoreType,
    User,
)

logger = logging.getLogger(__name__)


class SimulatedTournament(BaseModel):
    forecasts: list[Forecast]

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
        assert (
            self._spot_forecasts_cache is not None
        ), "Spot forecasts cache is not initialized"
        return list(self._spot_forecasts_cache.values())

    _user_cache: dict[str, User] = PrivateAttr(default_factory=dict)
    _question_cache: dict[int, Question] = PrivateAttr(default_factory=dict)
    _spot_forecasts_cache: dict[str, Forecast] = PrivateAttr(
        default_factory=dict
    )  # The latest forecast for each question that is before the spot scoring time
    _scores_cache: dict[str, Score] = PrivateAttr(default_factory=dict)

    _question_to_spot_forecasts_cache: dict[int, list[Forecast]] = PrivateAttr(
        default_factory=dict
    )


    def question_to_spot_forecasts(self, question_id: int) -> list[Forecast]:
        if len(self._question_to_spot_forecasts_cache) == 0:
            for forecast in self.spot_forecasts:
                question_id_to_cache = forecast.question.question_id
                if question_id_to_cache not in self._question_to_spot_forecasts_cache:
                    self._question_to_spot_forecasts_cache[question_id_to_cache] = []
                self._question_to_spot_forecasts_cache[question_id_to_cache].append(forecast)
        spot_forecasts = self._question_to_spot_forecasts_cache[question_id]
        return spot_forecasts.copy()

    def user_to_scores(
        self, user_name: str, score_type: ScoreType | None = None
    ) -> list[Score]:
        scores = [
            score for score in self.scores if score.forecast.user.name == user_name
        ]
        if score_type is not None:
            scores = [score for score in scores if score.type == score_type]
        return scores.copy()

    def get_spot_score_for_question_and_user(
        self, question_id: int, user_name: str, score_type: ScoreType
    ) -> Score:
        assert (
            score_type.is_spot_score()
        ), "Only spot scores are supported for this method"
        scores = [
            score
            for score in self.scores
            if score.forecast.user.name == user_name
            and score.forecast.question.question_id == question_id
            and score.type == score_type
        ]
        assert len(scores) == 1, "Expected exactly for question for user if spot score"
        return scores[0]

    @model_validator(mode="after")
    def initialize_tournament(self) -> Self:
        logger.info("Initializing caches")
        self._initialize_spot_forecast_cache()

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
        for i, forecast in enumerate(self.spot_forecasts):
            should_log_scoring = i % log_every_n == 0
            if should_log_scoring:
                logger.info(
                    f"Caching scores for forecast {i} of {len(self.spot_forecasts)}"
                )
            if isinstance(forecast.question.resolution, AmbiguousResolutionType):
                continue
            new_scores = self._calculate_spot_scores_for_forecast(forecast)
            all_scores.extend(new_scores)
        self._scores_cache = {score.id: score for score in all_scores}

        logger.info("Finished initializing scoring caches")
        return self

    def _initialize_spot_forecast_cache(self) -> None:
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
        self._spot_forecasts_cache = {
            forecast.id: forecast for forecast in spot_forecasts.values()
        }

    def _calculate_spot_scores_for_forecast(
        self, forecast_to_score: Forecast
    ) -> list[Score]:
        if forecast_to_score.id not in self._spot_forecasts_cache:
            raise ValueError("Forecast to score must be in spot forecasts cache")

        resolution = forecast_to_score.question.resolution
        if isinstance(resolution, AmbiguousResolutionType):
            return []

        spot_forecasts_from_others: list[Forecast] = self.question_to_spot_forecasts(
            forecast_to_score.question.question_id
        )
        spot_forecasts_from_others.remove(forecast_to_score)

        spot_peer_score = forecast_to_score.get_spot_peer_score(
            resolution, spot_forecasts_from_others
        )
        spot_baseline_score = forecast_to_score.get_spot_baseline_score(resolution)
        scores = [spot_peer_score, spot_baseline_score]
        return scores
