import pytest
from refactored_notebook.main import load_tournament
from refactored_notebook.data_models import Question, UserType

def test_load_tournament():
    pro_tournament = load_tournament("tests/test_data/pro_forecasts_q1.csv", UserType.PRO)
    pro_forecasts = pro_tournament.forecasts
    assert len(pro_forecasts) == 2840

    all_question_objects: list[Question] = []
    for forecast in pro_forecasts:
        if forecast.question not in all_question_objects:
            all_question_objects.append(forecast.question)
    assert len(all_question_objects) == len(pro_forecasts)
