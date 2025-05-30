import pytest

from refactored_notebook.data_models import Question, UserType
from refactored_notebook.load_tournament_data import load_tournament

@pytest.mark.parametrize("user_type, file_name, expected_forecast_count", [
    (UserType.PRO, "pro_forecasts_q1.csv", 2840),
    (UserType.BOT, "bot_forecasts_q1.csv", 33121),
])
def test_load_tournament(user_type: UserType, file_name: str, expected_forecast_count: int):
    tournament = load_tournament(f"tests/test_data/{file_name}", user_type)
    forecasts = tournament.forecasts

    unique_question_objects: list[Question] = []
    for forecast in forecasts:
        if forecast.question not in unique_question_objects:
            unique_question_objects.append(forecast.question)
    assert len(unique_question_objects) < len(forecasts) / 2

    assert len(forecasts) == expected_forecast_count
