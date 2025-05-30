from datetime import datetime

from refactored_notebook.custom_types import QuestionType, UserType
from refactored_notebook.data_models import Question, User, Forecast


def make_user(name: str, user_type: UserType = UserType.PRO) -> User:
    return User(name=name, type=user_type, is_aggregate=False, aggregated_users=[])

def make_question_binary() -> Question:
    return Question(
        post_id=1,
        question_id=1,
        type=QuestionType.BINARY,
        question_text="Will it rain tomorrow?",
        resolution=True,
        options=None,
        range_max=None,
        range_min=None,
        open_upper_bound=None,
        open_lower_bound=None,
        weight=1.0,
        spot_scoring_time=datetime(2025, 5, 30, 0, 46, 31),
    )

def make_question_mc() -> Question:
    return Question(
        post_id=2,
        question_id=2,
        type=QuestionType.MULTIPLE_CHOICE,
        question_text="Which color?",
        resolution="Red",
        options=["Red", "Blue", "Green"],
        range_max=None,
        range_min=None,
        open_upper_bound=None,
        open_lower_bound=None,
        weight=1.0,
        spot_scoring_time=datetime(2025, 5, 30, 0, 46, 31),
    )

def make_quesQuestionType() -> Question:
    return Question(
        post_id=3,
        question_id=3,
        type=QuestionType.NUMERIC,
        question_text="How many apples?",
        resolution=42.0,
        options=None,
        range_max=100.0,
        range_min=0.0,
        open_upper_bound=False,
        open_lower_bound=False,
        weight=1.0,
        spot_scoring_time=datetime(2025, 5, 0, 46, 31),
    )

def make_forecast(question: Question, user: User, prediction: list[float]) -> Forecast:
    return Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=datetime(2025, 5, 30, 0, 46, 31),
    )