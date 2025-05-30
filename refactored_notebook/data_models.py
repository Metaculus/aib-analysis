from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator
from typing_extensions import Self

ResolutionType = (
    bool | str | float | None
)  # binary, MC, numeric, or 'annulled/ambiguous'
ForecastType = (
    list[float] | None
)  # binary: [p_yes, p_no], multiple choice: [p_a, p_b, p_c], numeric: [p_0, p_1, p_2, ...]


class UserType(Enum):
    PRO = "pro"
    BOT = "bot"
    CP = "cp"


class QuestionType(Enum):
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"


class Forecast(BaseModel):
    question: Question
    user: User
    prediction: ForecastType
    prediction_time: datetime
    comment: str | None = None

    def get_spot_baseline_score(self, resolution: ResolutionType) -> Score:
        raise NotImplementedError("Not implemented")

    def get_spot_peer_score(
        self, resolution: ResolutionType, other_users_forecasts: list[Forecast]
    ) -> Score:
        # assert only one forecast per user
        # assert that forecasts are in time range of question
        raise NotImplementedError("Not implemented")

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
            elif q_type == QuestionType.NUMERIC:
                if not (
                    isinstance(self.prediction, list)
                    and len(self.prediction) == 201
                    and all(isinstance(p, float) for p in self.prediction)
                ):
                    raise ValueError(
                        "Prediction must be a list of 201 floats for numeric questions."
                    )
        return self


class Score(BaseModel):
    score: float
    type: Literal["spot_peer", "spot_baseline"]
    forecast: Forecast
    users_used_in_scoring: list[User] | None  # Empty if baseline


class Question(BaseModel):
    question_text: str
    type: QuestionType
    resolution: ResolutionType
    weight: float
    spot_scoring_time: datetime
    question_id: int
    post_id: int

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

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"


class User(BaseModel):
    name: str
    type: UserType
    is_aggregate: bool
    aggregated_users: list[User]

    @property
    def is_metac_bot(self) -> bool:
        return "metac-" in self.name
