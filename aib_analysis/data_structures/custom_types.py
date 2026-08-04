from enum import Enum

AnnulledAmbiguousResolutionType = type(None)
BinaryResolutionType = bool
MCResolutionType = str
NumericResolutionType = float
DiscreteResolutionType = float
ResolutionType = BinaryResolutionType | MCResolutionType | NumericResolutionType | DiscreteResolutionType | AnnulledAmbiguousResolutionType

BinaryForecastType = list[float] # binary: [p_yes, p_no]
MCForecastType = list[float] # multiple choice: [p_option_a, p_option_b, p_option_c],
NumericForecastType = list[float] # numeric: [p_0, p_1, p_2, ..., p_200] (201 value cdf)
DiscreteForecastType = list[float] # discrete: cdf of variable length
ForecastType = BinaryForecastType | MCForecastType | NumericForecastType | DiscreteForecastType | None

class UserType(Enum):
    PRO = "pro"
    BOT = "bot"
    CP = "community_prediction"
    AGGREGATE = "aggregate"

class QuestionType(Enum):
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    DISCRETE = "discrete"


class ScoreType(Enum):
    SPOT_PEER = "spot_peer"
    SPOT_BASELINE = "spot_baseline"

    def is_spot_score(self) -> bool:
        return self in {ScoreType.SPOT_PEER, ScoreType.SPOT_BASELINE}


