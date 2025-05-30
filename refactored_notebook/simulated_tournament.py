from __future__ import annotations

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from refactored_notebook.custom_types import AmbiguousResolutionType
from refactored_notebook.data_models import (
    Forecast,
    Leaderboard,
    LeaderboardEntry,
    Question,
    Score,
    ScoreType,
    User,
)


class SimulatedTournament(BaseModel):
    forecasts: list[Forecast]
    _user_cache: dict[str, User] | None = None # Cache of users by name
    _question_cache: dict[int, Question] | None = None # Cache of questions by question_id
    _score_cache: dict[str, list[Score]] | None = None # Cache of scores by forecast id

    @property
    def users(self) -> list[User]:
        if self._user_cache is None:
            self._user_cache = {
                forecast.user.name: forecast.user for forecast in self.forecasts
            }
        return list(self._user_cache.values())

    @property
    def questions(self) -> list[Question]:
        if self._question_cache is None:
            self._question_cache = {
                forecast.question.question_id: forecast.question
                for forecast in self.forecasts
            }
        return list(self._question_cache.values())

    @property
    def scores(self) -> list[Score]:
        if self._score_cache is not None:
            scores = list(self._score_cache.values())
            flat_scores = [score for sublist in scores for score in sublist]
            return flat_scores
        self._score_cache = {}
        spot_peer_scores = []
        spot_baseline_scores = []
        for forecast in self.forecasts:
            if isinstance(forecast.question.resolution, AmbiguousResolutionType):
                continue
            forecasts_from_other_users = [
                f
                for f in self.forecasts
                if f.question == forecast.question and f.user != forecast.user
            ]
            spot_peer_scores.append(
                forecast.get_spot_peer_score(
                    forecast.question.resolution, forecasts_from_other_users
                )
            )
            spot_baseline_scores.append(
                forecast.get_spot_baseline_score(forecast.question.resolution)
            )
            self._score_cache[forecast.id] = spot_peer_scores + spot_baseline_scores
        return spot_peer_scores + spot_baseline_scores

    def get_user_by_name(self, name: str) -> User:
        if self._user_cache is None:
            self._user_cache = {
                forecast.user.name: forecast.user for forecast in self.forecasts
            }
        return self._user_cache[name]

    def get_leaderboard(self, score_type: ScoreType) -> Leaderboard:
        peer_scores = [score for score in self.scores if score.type == score_type]
        user_score_map: dict[str, list[Score]] = {}

        for score in peer_scores:
            forecast_author = score.forecast.user
            if forecast_author.name not in user_score_map:
                user_score_map[forecast_author.name] = [score]
            user_score_map[forecast_author.name].append(score)

        entries = [
            LeaderboardEntry(user=self.get_user_by_name(username), scores=scores)
            for username, scores in user_score_map.items()
        ]
        return Leaderboard(entries=entries, type=score_type)

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
