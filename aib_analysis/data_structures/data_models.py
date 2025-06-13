from __future__ import annotations

import random
from datetime import datetime

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from aib_analysis.data_structures.custom_types import (
    AnnulledAmbiguousResolutionType,
    ForecastType,
    QuestionType,
    ResolutionType,
    ScoreType,
    UserType,
)
from aib_analysis.math.scoring import calculate_baseline_score, calculate_peer_score
from aib_analysis.math.stats import ConfidenceInterval, ConfidenceIntervalCalculator


class Forecast(BaseModel):
    question: Question
    user: User
    prediction: ForecastType
    prediction_time: datetime
    _id: str | None = None

    @property
    def id(self) -> str:
        if self._id is None:
            id = f"U{self.user.name}_"
            id += f"Q{self.question.question_id}_"
            id += f"F{self.prediction_time.strftime('%Y-%m-%d_%H-%M-%S_%f')}_"
            if self.prediction:
                id += f"P{sum(self.prediction)}"
            else:
                id += f"P{None}"
            self._id = id
        return self._id

    def get_spot_baseline_score(self, resolution: ResolutionType) -> Score:
        q = self.question
        score_value = calculate_baseline_score(
            forecast=self.prediction,
            resolution=resolution,
            question_weight=q.weight,
            q_type=q.type.value,
            options=q.options,
            range_min=q.range_min,
            range_max=q.range_max,
            open_upper_bound=q.open_upper_bound,
            open_lower_bound=q.open_lower_bound,
        )

        return Score(
            score=score_value,
            type=ScoreType.SPOT_BASELINE,
            forecast=self,
            users_used_in_scoring=None,
        )

    def get_spot_peer_score(
        self, resolution: ResolutionType, other_users_forecasts: list[Forecast]
    ) -> Score:
        other_preds = [f.prediction for f in other_users_forecasts]
        users_used_in_scoring = [f.user for f in other_users_forecasts]
        if self.user in users_used_in_scoring:
            raise ValueError(
                "Forecast Author cannot be in other users forecasts list for peer score"
            )
        q = self.question
        score_value = calculate_peer_score(
            forecast=self.prediction,
            forecast_for_other_users=other_preds,
            resolution=resolution,
            question_weight=q.weight,
            q_type=q.type.value,
            options=q.options,
            range_min=q.range_min,
            range_max=q.range_max,
        )
        return Score(
            score=score_value,
            type=ScoreType.SPOT_PEER,
            forecast=self,
            users_used_in_scoring=[f.user for f in other_users_forecasts],
        )

    @model_validator(mode="after")
    def check_prediction_type_matches(self) -> Self:
        if self.prediction is not None:
            q_type = self.question.type
            if q_type == QuestionType.BINARY:
                if not (
                    isinstance(self.prediction, list)
                    and len(self.prediction) in (1, 2)
                    and all(0 <= p <= 1 for p in self.prediction)
                ):
                    raise ValueError(
                        "Prediction must be a list of 1 or 2 floats for binary questions."
                    )
            elif q_type == QuestionType.MULTIPLE_CHOICE:
                if not (
                    isinstance(self.prediction, list)
                    and len(self.prediction) >= 2
                    and all(isinstance(p, float) for p in self.prediction)
                ):
                    raise ValueError(
                        "Prediction must be a list of floats (length >= 2) for multiple choice questions."
                    )
                if abs(sum(self.prediction) - 1) > 1e-6:
                    raise ValueError("Prediction must sum to 1 for multiple choice questions.")
            elif q_type == QuestionType.NUMERIC:
                if not (
                    isinstance(self.prediction, list)
                    and len(self.prediction) == 201
                    and all(isinstance(p, float) for p in self.prediction)
                ):
                    raise ValueError(
                        "Prediction must be a list of 201 floats for numeric questions."
                    )
        if self.prediction is not None:
            for p in self.prediction:
                if not (0 <= p <= 1):
                    raise ValueError(f"Prediction must be between 0 and 1, got {p}. Full prediction: {self.prediction}")
        return self


    def __hash__(self) -> int:
        # Convert options list to tuple for hashing if it exists
        hash_tuple = (
            self.id,
            self.question.question_id,
            self.user.name,
            self.prediction,
            self.prediction_time,
        )
        return hash(hash_tuple)



class Score(BaseModel):
    score: float
    type: ScoreType
    forecast: Forecast
    users_used_in_scoring: list[User] | None  # Empty if baseline

    @property
    def id(self) -> str:
        return f"{self.forecast.id}_{self.type.value}"

    @model_validator(mode="after")
    def check_forecast_resolution_not_none(self) -> Self:
        if self.forecast.question.resolution is None or self.forecast.question.is_annulled_or_ambiguous:
            raise ValueError(
                "Forecast's question resolution must not be None. You cannot assign a score to ambiguous/annulled resolution"
            )
        return self

    def display_score_and_question(self) -> str:
        return f"({self.score:.3f}) {self.forecast.question.url}"


class Question(BaseModel):
    question_id: int
    type: QuestionType
    question_text: str
    resolution: ResolutionType
    options: list[str] | None
    range_max: float | None
    range_min: float | None
    open_upper_bound: bool | None
    open_lower_bound: bool | None
    weight: float
    post_id: int
    spot_scoring_time: datetime

    @property
    def is_annulled_or_ambiguous(self) -> bool:
        return isinstance(self.resolution, AnnulledAmbiguousResolutionType)

    @model_validator(mode="after")
    def check_resolution_type_matches(self) -> Self:
        if self.resolution is not None:
            if self.type == QuestionType.BINARY and not isinstance(
                self.resolution, bool
            ):
                raise ValueError("Resolution must be a boolean for binary questions.")
            if self.type == QuestionType.MULTIPLE_CHOICE and not isinstance(
                self.resolution, str
            ):
                raise ValueError(
                    "Resolution must be a string for multiple choice questions."
                )
            if self.type == QuestionType.NUMERIC and not isinstance(
                self.resolution, float
            ):
                raise ValueError("Resolution must be a float for numeric questions.")
        return self

    @model_validator(mode="after")
    def check_misc_constraints(self) -> Self:
        if not (0 <= self.weight <= 1):
            raise ValueError("Weight must be between 0 and 1.")
        if not self.question_text.strip():
            raise ValueError("Question text must not be empty.")
        if self.options is not None and len(self.options) < 2:
            raise ValueError(
                "Multiple choice questions must have at least two options."
            )
        if "PRACTICE" in self.question_text:
            raise ValueError("Practice questions are not allowed.")
        return self

    @model_validator(mode="after")
    def check_question_type_has_right_fields(self) -> Self:
        if self.type == QuestionType.NUMERIC:
            if (
                self.range_max is None
                or self.range_min is None
                or self.open_upper_bound is None
                or self.open_lower_bound is None
            ):
                raise ValueError(
                    "Numeric questions must have all bound information (upper_bound, lower_bound, open_upper_bound, open_lower_bound)."
                )
        if self.type == QuestionType.MULTIPLE_CHOICE:
            if not self.options or len(self.options) < 2:
                raise ValueError(
                    "Multiple choice questions must have at least two options."
                )
        return self

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"

    def __hash__(self) -> int:
        # Convert options list to tuple for hashing if it exists
        options_tuple = tuple(self.options) if self.options is not None else None
        # Create a tuple of all fields that should be used for hashing
        hash_tuple = (
            self.question_id,
            self.type,
            self.question_text,
            self.resolution,
            options_tuple,
            self.range_max,
            self.range_min,
            self.open_upper_bound,
            self.open_lower_bound,
            self.weight,
            self.post_id,
            self.spot_scoring_time,
        )
        return hash(hash_tuple)


class User(BaseModel):
    name: str
    type: UserType
    aggregated_users: list[User]

    @property
    def is_metac_bot(self) -> bool:
        return "metac-" in self.name


class Leaderboard(BaseModel):
    entries: list[LeaderboardEntry]
    type: ScoreType

    @model_validator(mode="after")
    def check_all_entries_same_score_type(self: Self) -> Self:
        if self.entries:
            flat_scores: list[Score] = [
                score for entry in self.entries for score in entry.scores
            ]
            score_types = {score.type for score in flat_scores}
            if len(score_types) > 1:
                raise ValueError(
                    f"All entries must have the same score type, found: {score_types}"
                )
            if self.type != list(score_types)[0]:
                raise ValueError(
                    f"Leaderboard type {self.type} does not match score type {list(score_types)[0]}"
                )
        return self

    @model_validator(mode="after")
    def sort_entries(self: Self) -> Self:
        self.entries.sort(key=lambda x: x.sum_of_scores, reverse=True)
        return self

    def entries_via_sum_of_scores(self) -> list[LeaderboardEntry]:
        return sorted(self.entries, key=lambda x: x.sum_of_scores, reverse=True)

    def entries_via_average_score(self) -> list[LeaderboardEntry]:
        return sorted(self.entries, key=lambda x: x.average_score, reverse=True)


class LeaderboardEntry(BaseModel):
    scores: list[Score]

    @model_validator(mode="after")
    def check_single_user(self: Self) -> Self:
        user_names = {score.forecast.user.name for score in self.scores}
        if len(user_names) != 1:
            raise ValueError(
                f"Leaderboard entry should have exactly one user, found: {user_names}"
            )
        return self

    @model_validator(mode="after")
    def check_all_scores_same_type(self: Self) -> Self:
        score_types = {score.type for score in self.scores}
        if len(score_types) > 1:
            raise ValueError(
                f"All scores must have the same type, found: {score_types}"
            )
        return self

    @property
    def user(self) -> User:
        users = [score.forecast.user for score in self.scores]
        first_user = users[0]
        return first_user

    @property
    def sum_of_scores(self) -> float:
        return sum(score.score for score in self.scores)

    @property
    def average_score(self) -> float:
        return self.sum_of_scores / len(self.scores)

    @property
    def question_count(self) -> int:
        return len(set([score.forecast.question.question_id for score in self.scores]))

    def top_n_scores(self, n: int) -> list[Score]:
        return sorted(self.scores, key=lambda x: x.score, reverse=True)[:n]

    def bottom_n_scores(self, n: int) -> list[Score]:
        return sorted(self.scores, key=lambda x: x.score, reverse=False)[:n]

    def randomly_sample_scores(self, n: int) -> list[Score]:
        return random.sample(self.scores, n)

    def get_confidence_interval(self, confidence_level: float = 0.95) -> ConfidenceInterval:
        score_observations = [score.score for score in self.scores]
        return ConfidenceIntervalCalculator.confidence_interval_from_observations(score_observations, confidence_level)