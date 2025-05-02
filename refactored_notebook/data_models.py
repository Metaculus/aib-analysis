from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from enum import Enum

class ResolutionType(Enum):
    YES = "yes"
    NO = "no"
    ANNULLED = "annulled"
    AMBIGUOUS = "ambiguous"

class Forecast(BaseModel):
    question: Question
    user: User
    prediction: list[float] # binary, MC, or numeric
    predcition_for_correct_answer: float
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
    users_used_in_scoring: list[User] | None# Empty if baseline

class Question(BaseModel):
    question_text: str
    resolution: ResolutionType
    weight: float
    spot_scoring_time: datetime

class User(BaseModel):
    name: str
    type: Literal["pro", "bot", "cp"]
    is_aggregate: bool
    aggregated_users: list[User]

    @property
    def is_metac_bot(self) -> bool:
        return "metac-" in self.name

