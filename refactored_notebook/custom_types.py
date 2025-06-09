from enum import Enum

AmbiguousResolutionType = type(None)
BinaryResolutionType = bool
MCResolutionType = str
NumericResolutionType = float
ResolutionType = BinaryResolutionType | MCResolutionType | NumericResolutionType | AmbiguousResolutionType

BinaryForecastType = list[float]
MCForecastType = list[float]
NumericForecastType = list[float]
ForecastType = BinaryForecastType | MCForecastType | NumericForecastType | None # binary: [p_yes, p_no], multiple choice: [p_a, p_b, p_c], numeric: [p_0, p_1, p_2, ...]


class UserType(Enum):
    PRO = "pro"
    BOT = "bot"
    CP = "cp"


class QuestionType(Enum):
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"


class ScoreType(Enum):
    SPOT_PEER = "spot_peer"
    SPOT_BASELINE = "spot_baseline"

    def is_spot_score(self) -> bool:
        return self in {ScoreType.SPOT_PEER, ScoreType.SPOT_BASELINE}


