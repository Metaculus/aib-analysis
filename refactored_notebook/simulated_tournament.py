from __future__ import annotations

import logging

from pydantic import BaseModel, PrivateAttr, model_validator
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

    _question_to_forecast_cache: dict[int, list[Forecast]] = PrivateAttr(
        default_factory=dict
    )

    _question_to_scores_cache: dict[int, list[Score]] = PrivateAttr(default_factory=dict)
    _forecast_to_scores_cache: dict[str, list[Score]] = PrivateAttr(default_factory=dict)
    _user_to_scores_cache: dict[str, list[Score]] = PrivateAttr(default_factory=dict)
    _user__question__type__to_scores_cache: dict[str, list[Score]] = PrivateAttr(
        default_factory=dict
    )

    def _get_spot_forecasts(self) -> list[Forecast]:
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
        return list(spot_forecasts.values())

    @model_validator(mode="after")
    def initialize_caches(self) -> Self:
        logger.info("Initializing caches")
        self._spot_forecasts_cache = self._get_spot_forecasts()

        self._user_cache = {
            forecast.user.name: forecast.user for forecast in self.forecasts
        }
        self._question_cache = {
            forecast.question.question_id: forecast.question
            for forecast in self.forecasts
        }
        self._question_to_forecast_cache = {
            forecast.question.question_id: [forecast] for forecast in self.forecasts
        }

        logger.info("Finished initializing caches part 1")

        log_every_n = 100
        for i, forecast in enumerate(self._spot_forecasts_cache):
            if i % log_every_n == 0:
                logger.info(f"Caching scores for forecast {i} of {len(self._spot_forecasts_cache)}")
            if isinstance(forecast.question.resolution, AmbiguousResolutionType):
                continue
            new_scores = self._calculate_spot_scores_for_forecast(forecast)
            for new_score in new_scores:
                self._add_new_score_to_caches(new_score)

        logger.info("Finished initializing caches part 2")

        return self

    def _add_new_score_to_caches(self, score: Score) -> None:
        forecast = score.forecast
        question = forecast.question
        question_id = question.question_id
        hash = self._get_hash_for_user_question_score_type(
            forecast.user.name, forecast.question.question_id, score.type
        )

        if question_id not in self._question_to_scores_cache:
            self._question_to_scores_cache[question_id] = []
        if forecast.id not in self._forecast_to_scores_cache:
            self._forecast_to_scores_cache[forecast.id] = []
        if forecast.user.name not in self._user_to_scores_cache:
            self._user_to_scores_cache[forecast.user.name] = []
        if hash not in self._user__question__type__to_scores_cache:
            self._user__question__type__to_scores_cache[hash] = []

        self._question_to_scores_cache[question_id].append(score)
        self._forecast_to_scores_cache[forecast.id].append(score)
        self._user_to_scores_cache[forecast.user.name].append(score)

        current_scores = self._user__question__type__to_scores_cache[hash]
        if score.type.is_spot_score() and current_scores:
            assert (
                len(current_scores) == 1
            ), "Spot scores should have exactly one score per question"
        else:
            self._user__question__type__to_scores_cache[hash].append(score)

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
        assert (
            self._question_to_scores_cache is not None
        ), "Question score cache is not initialized"
        scores = list(self._question_to_scores_cache.values())
        flat_scores = [score for sublist in scores for score in sublist]
        return flat_scores

    def get_leaderboard(self, score_type: ScoreType) -> Leaderboard:
        entries = []
        for user in self.users:
            cache = self._user_to_scores_cache
            all_scores_of_user_for_tournament = cache[user.name]
            scores_of_type = [
                score
                for score in all_scores_of_user_for_tournament
                if score.type == score_type
            ]

            related_question_ids = set(
                [
                    score.forecast.question.question_id
                    for score in all_scores_of_user_for_tournament
                ]
            )
            if score_type.is_spot_score() and len(scores_of_type) != len(
                related_question_ids
            ):
                raise ValueError(
                    f"Spot scores should have exactly one score per question. User {user.name} has {len(scores_of_type)} scores for {len(related_question_ids)} questions"
                )
            entries.append(LeaderboardEntry(scores=scores_of_type))
        return Leaderboard(entries=entries, type=score_type)

    def get_spot_score_for_question(
        self, question_id: int, user_name: str, score_type: ScoreType
    ) -> Score:
        assert (
            score_type.is_spot_score()
        ), "Only spot scores are supported for this method"
        user_question_score_type_hash = self._get_hash_for_user_question_score_type(
            user_name, question_id, score_type
        )
        assert (
            self._user__question__type__to_scores_cache is not None
        ), "User question score type cache is not initialized"
        scores = self._user__question__type__to_scores_cache[user_question_score_type_hash]
        assert len(scores) == 1, "Expected exactly for question for user if spot score"
        return scores[0]

    def get_forecasters_on_question(self, question_id: int) -> list[User]:
        assert (
            self._question_to_scores_cache is not None
        ), "Question score cache is not initialized"
        unique_user_names = set(
            forecast.user.name
            for forecast in self._question_to_forecast_cache[question_id]
        )
        users = [self._user_cache[name] for name in unique_user_names]
        return users

    def get_ranking_by_spot_peer_score_lower_t_bound(
        self, confidence_level: float
    ) -> list[tuple[User, float]]:
        # Get all spot peer scores
        # create a confidence interval for the spot peer score
        # Sort by lower bound
        raise NotImplementedError("Not implemented")

    def get_ranking_by_spot_peer_score_sum(self) -> list[tuple[User, float]]:
        # Get all spot peer scores
        # Sort by spot peer score
        raise NotImplementedError("Not implemented")

    def get_ranking_by_spot_peer_score_bootstrap_lower_bound(
        self, confidence_level: float
    ) -> list[tuple[User, float]]:
        # Get all spot peer scores
        # bootstrap the spot peer scores
        # create a confidence interval for the spot peer score
        # Sort by lower bound
        raise NotImplementedError("Not implemented")

    def _get_hash_for_user_question_score_type(
        self, user_name: str, question_id: int, score_type: ScoreType
    ) -> str:
        return f"{user_name}_{question_id}_{score_type}"

    def _calculate_spot_scores_for_forecast(
        self, forecast_to_score: Forecast
    ) -> list[Score]:
        assert (
            forecast_to_score in self._spot_forecasts_cache
        ), "Forecast to score must be in spot forecasts cache"
        resolution = forecast_to_score.question.resolution
        if isinstance(resolution, AmbiguousResolutionType):
            return []

        spot_forecasts_from_others: list[Forecast] = []
        forecasts_for_question = self._question_to_forecast_cache[forecast_to_score.question.question_id]
        for forecast in forecasts_for_question:
            if (
                forecast in self._spot_forecasts_cache
                and forecast.user != forecast_to_score.user
            ):
                spot_forecasts_from_others.append(forecast)

        spot_peer_score = forecast_to_score.get_spot_peer_score(
            resolution, spot_forecasts_from_others
        )
        spot_baseline_score = forecast_to_score.get_spot_baseline_score(resolution)

        scores = [spot_peer_score, spot_baseline_score]
        return scores
