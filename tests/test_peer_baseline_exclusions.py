import math

import pytest

from aib_analysis.data_structures.custom_types import ScoreType
from aib_analysis.data_structures.data_models import Forecast, Question
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import make_forecast, make_question_binary, make_user

PREDICTION_BY_USER = {"high": 0.9, "middle": 0.5, "low": 0.2}


def make_questions() -> tuple[Question, Question]:
    first = make_question_binary("First question?").model_copy(
        update={"question_id": 1, "post_id": 1}
    )
    second = make_question_binary("Second question?").model_copy(
        update={"question_id": 2, "post_id": 2}
    )
    return first, second


def make_forecasts(questions: tuple[Question, Question]) -> list[Forecast]:
    return [
        make_forecast(question, make_user(user_name), [prediction])
        for question in questions
        for user_name, prediction in PREDICTION_BY_USER.items()
    ]


def spot_peer_scores(tournament: SimulatedTournament) -> dict[tuple[int, str], float]:
    return {
        (score.forecast.question.question_id, score.forecast.user.name): score.score
        for score in tournament.scores
        if score.type == ScoreType.SPOT_PEER
    }


def test_excluded_user_is_left_out_of_the_baseline_only_on_listed_questions() -> None:
    questions = make_questions()
    forecasts = make_forecasts(questions)
    without_exclusions = spot_peer_scores(SimulatedTournament(forecasts=forecasts))
    with_exclusions = spot_peer_scores(
        SimulatedTournament(forecasts=forecasts, peer_baseline_exclusions={1: {"low"}})
    )

    assert with_exclusions.keys() == without_exclusions.keys()
    for user_name in PREDICTION_BY_USER:
        assert with_exclusions[(2, user_name)] == without_exclusions[(2, user_name)]

    baseline_without_low = math.sqrt(0.9 * 0.5)
    expected_high_score = 100 * (2 / 1) * math.log(0.9 / baseline_without_low)
    assert with_exclusions[(1, "high")] == pytest.approx(expected_high_score)
    assert with_exclusions[(1, "high")] != pytest.approx(without_exclusions[(1, "high")])


def test_exclusion_for_unknown_question_raises() -> None:
    forecasts = make_forecasts(make_questions())
    with pytest.raises(ValueError):
        SimulatedTournament(forecasts=forecasts, peer_baseline_exclusions={999: {"low"}})


def test_exclusion_for_unknown_user_raises() -> None:
    forecasts = make_forecasts(make_questions())
    with pytest.raises(ValueError):
        SimulatedTournament(forecasts=forecasts, peer_baseline_exclusions={1: {"nobody"}})
