from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel

ResolutionType = bool | str | float | None # binary, MC, numeric, or 'annulled/ambiguous'
ForecastType = list[float] | None # binary: [p_yes, p_no], multiple choice: [p_a, p_b, p_c], numeric: [p_0, p_1, p_2, ...]

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

    def get_spot_peer_score(self, resolution: ResolutionType, other_users_forecasts: list[Forecast]) -> Score:
        # assert only one forecast per user
        # assert that forecasts are in time range of question
        raise NotImplementedError("Not implemented")

class Score(BaseModel):
    score: float
    type: Literal["spot_peer", "spot_baseline"]
    forecast: Forecast
    users_used_in_scoring: list[User] | None # Empty if baseline

class Question(BaseModel):
    question_text: str
    type: QuestionType
    resolution: ResolutionType
    weight: float
    spot_scoring_time: datetime
    question_id: int
    post_id: int

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

