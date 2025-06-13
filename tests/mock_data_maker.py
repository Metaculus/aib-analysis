import random
from datetime import datetime

from aib_analysis.data_structures.custom_types import QuestionType, UserType
from aib_analysis.data_structures.data_models import Forecast, Question, User
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from tests.test_scoring import generate_cdf, Percentile

def make_user(name: str, user_type: UserType = UserType.PRO) -> User:
    return User(name=name, type=user_type, aggregated_users=[])

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
        created_at=datetime(2030, 5, 20, 0, 46, 31),
    )

def make_question_mc() -> Question:
    return Question(
        post_id=random.randint(30000, 40000),
        question_id=random.randint(30000, 40000),
        type=QuestionType.MULTIPLE_CHOICE,
        question_text="Which color?",
        resolution="Red",
        options=("Red", "Blue", "Green"),
        range_max=None,
        range_min=None,
        open_upper_bound=None,
        open_lower_bound=None,
        weight=1.0,
        spot_scoring_time=datetime(2025, 5, 30, 0, 46, 31),
        created_at=datetime(2030, 5, 20, 0, 46, 31),
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
        created_at=datetime(2030, 5, 20, 0, 46, 31),
    )

def make_forecast(question: Question, user: User, prediction: list[float], forecast_time: datetime = datetime(2025, 5, 30, 0, 46, 31)) -> Forecast:
    if len(prediction) == 1:
        prediction = [prediction[0], 1 - prediction[0]]
    return Forecast(
        question=question,
        user=user,
        prediction=prediction,
        prediction_time=forecast_time,
    )


def make_tournament() -> SimulatedTournament:
    questions = [make_question_binary(), make_question_mc(), make_question_numeric()]
    users = [make_user(f"User {i}") for i in range(1, 10)]

    forecasts = []
    for question in questions:
        for user in users:
            # Create a simple forecast for each user-question pair
            if question.type == QuestionType.BINARY:
                prediction = [0.5, 0.5]  # [p_yes, p_no]
            elif question.type == QuestionType.MULTIPLE_CHOICE:
                prediction = [0.33, 0.33, 0.34]  # [p_red, p_blue, p_green]
            else:  # NUMERIC
                prediction = generate_cdf(
                    [
                        Percentile(value=20, probability_below=0.1),
                        Percentile(value=50, probability_below=0.9),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=False,
                    open_upper_bound=False,
                )

            forecast = Forecast(
                question=question,
                user=user,
                prediction=prediction,
                prediction_time=datetime(2024, 12, 1)
            )
            forecasts.append(forecast)
    return SimulatedTournament(forecasts=forecasts)