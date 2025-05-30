from refactored_notebook.data_models import Question
from refactored_notebook.simulated_tournament import SimulatedTournament


def test_load_pro_data(pro_tournament: SimulatedTournament):
    validate_load_tournament_data(pro_tournament, 2840)


def test_load_bot_data(bot_tournament: SimulatedTournament):
    validate_load_tournament_data(bot_tournament, 33121)


def validate_load_tournament_data(
    tournament: SimulatedTournament, expected_forecast_count: int
):
    forecasts = tournament.forecasts

    unique_question_objects: list[Question] = []
    for forecast in forecasts:
        if forecast.question not in unique_question_objects:
            unique_question_objects.append(forecast.question)
    assert len(unique_question_objects) < len(forecasts) / 2

    assert len(forecasts) == expected_forecast_count
