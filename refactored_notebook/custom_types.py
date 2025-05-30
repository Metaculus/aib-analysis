from enum import Enum

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


class ScoreType(Enum):
    SPOT_PEER = "spot_peer"
    SPOT_BASELINE = "spot_baseline"
