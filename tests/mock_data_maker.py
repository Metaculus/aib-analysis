import random
from datetime import datetime

from refactored_notebook.custom_types import QuestionType, UserType
from refactored_notebook.data_models import Forecast, Question, User


def make_user(name: str, user_type: UserType = UserType.PRO) -> User:
    return User(name=name, type=user_type, is_aggregate=False, aggregated_users=[])

def make_question_binary(question_text: str = "Will it rain tomorrow?") -> Question:
    return Question(
        post_id=random.randint(30000, 40000),
        question_id=random.randint(30000, 40000),
        type=QuestionType.BINARY,
        question_text=question_text,
        resolution=True,
        options=None,
        range_max=None,
        range_min=None,
        open_upper_bound=None,
        open_lower_bound=None,
        weight=1.0,
        spot_scoring_time=datetime(2030, 5, 30, 0, 46, 31),
    )

def make_question_mc() -> Question:
    return Question(
        post_id=random.randint(30000, 40000),
        question_id=random.randint(30000, 40000),
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

def make_question_numeric() -> Question:
    return Question(
        post_id=random.randint(30000, 40000),
        question_id=random.randint(30000, 40000),
        type=QuestionType.NUMERIC,
        question_text="How many apples?",
        resolution=42.0,
        options=None,
        range_max=100.0,
        range_min=0.0,
        open_upper_bound=False,
        open_lower_bound=False,
        weight=1.0,
        spot_scoring_time=datetime(2025, 5, 30, 0, 46, 31),
    )

def make_forecast(question: Question, user: User, prediction: list[float]) -> Forecast:
    if len(prediction) == 1:
        prediction = [prediction[0], 1 - prediction[0]]
    return Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=datetime(2025, 5, 30, 0, 46, 31),
    )