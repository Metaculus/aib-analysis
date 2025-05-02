from __future__ import annotations

from pydantic import BaseModel
from refactored_notebook.data_models import User, Question, Forecast, Score


class SimulatedTournament(BaseModel):
    forecasts: list[Forecast]

    @property
    def users(self) -> set[User]:
        users = set()
        for forecast in self.forecasts:
            users.add(forecast.user)
        return users

    @property
    def questions(self) -> set[Question]:
        questions = set()
        for forecast in self.forecasts:
            questions.add(forecast.question)
        return questions

    @property
    def scores(self) -> list[Score]:
        spot_peer_scores = []
        spot_baseline_scores = []
        for forecast in self.forecasts:
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
        return spot_peer_scores + spot_baseline_scores

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
