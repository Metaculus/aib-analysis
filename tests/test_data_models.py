from datetime import datetime

import pytest

from aib_analysis.custom_types import QuestionType, ScoreType, UserType
from aib_analysis.data_models import Leaderboard, LeaderboardEntry, Question
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_user,
)


def test_leaderboard_entry_and_leaderboard_binary():
    user = make_user("alice")
    question = make_question_binary()
    forecast = make_forecast(question, user, [0.8])
    score = forecast.get_spot_baseline_score(question.resolution)
    entry = LeaderboardEntry(scores=[score])
    leaderboard = Leaderboard(entries=[entry], type=ScoreType.SPOT_BASELINE)
    assert leaderboard.entries[0].user.name == "alice"
    assert leaderboard.entries[0].sum_of_scores == pytest.approx(score.score)
    assert leaderboard.entries[0].average_score == pytest.approx(score.score)


def test_leaderboard_type_consistency():
    user = make_user("dave")
    question = make_question_binary()
    forecast = make_forecast(question, user, [0.5])
    score = forecast.get_spot_baseline_score(question.resolution)
    entry = LeaderboardEntry(scores=[score])
    # Should succeed
    Leaderboard(entries=[entry], type=ScoreType.SPOT_BASELINE)
    # Should fail if type mismatches
    with pytest.raises(ValueError):
        Leaderboard(entries=[entry], type=ScoreType.SPOT_PEER)

def test_leaderboard_ranking_good_forecast_higher():
    question = make_question_binary()
    user_good = make_user("good")
    user_medium = make_user("medium")
    user_bad = make_user("bad")
    # True resolution, so [0.9] is a good forecast, [0.1] is a bad forecast
    forecast_good = make_forecast(question, user_good, [0.9])
    forecast_medium = make_forecast(question, user_medium, [0.5])
    forecast_bad = make_forecast(question, user_bad, [0.1])
    score_good = forecast_good.get_spot_baseline_score(question.resolution)
    score_medium = forecast_medium.get_spot_baseline_score(question.resolution)
    score_bad = forecast_bad.get_spot_baseline_score(question.resolution)
    entry_good = LeaderboardEntry(scores=[score_good])
    entry_medium = LeaderboardEntry(scores=[score_medium])
    entry_bad = LeaderboardEntry(scores=[score_bad])
    leaderboard = Leaderboard(entries=[entry_good, entry_medium, entry_bad], type=ScoreType.SPOT_BASELINE)
    # Sort entries by sum_of_scores descending
    sorted_entries = sorted(leaderboard.entries, key=lambda e: e.sum_of_scores, reverse=True)
    assert sorted_entries[0].user.name == "good"
    assert sorted_entries[1].user.name == "medium"
    assert sorted_entries[2].user.name == "bad"

def test_question_validation_errors():
    # Invalid resolution for binary
    with pytest.raises(ValueError):
        Question(
            post_id=1,
            question_id=1,
            type=QuestionType.BINARY,
            question_text="Test binary",
            resolution="not_bool",
            options=None,
            range_max=None,
            range_min=None,
            open_upper_bound=None,
            open_lower_bound=None,
            weight=1.0,
            spot_scoring_time=datetime(2025, 1, 1),
        )
    # Invalid resolution for multiple choice
    with pytest.raises(ValueError):
        Question(
            post_id=2,
            question_id=2,
            type=QuestionType.MULTIPLE_CHOICE,
            question_text="Test MC",
            resolution=123,
            options=["A", "B"],
            range_max=None,
            range_min=None,
            open_upper_bound=None,
            open_lower_bound=None,
            weight=1.0,
            spot_scoring_time=datetime(2025, 1, 1),
        )
    # Invalid resolution for numeric
    with pytest.raises(ValueError):
        Question(
            post_id=3,
            question_id=3,
            type=QuestionType.NUMERIC,
            question_text="Test numeric",
            resolution="not_float",
            options=None,
            range_max=10.0,
            range_min=0.0,
            open_upper_bound=False,
            open_lower_bound=False,
            weight=1.0,
            spot_scoring_time=datetime(2025, 1, 1),
        )
    # Invalid weight
    with pytest.raises(ValueError):
        Question(
            post_id=4,
            question_id=4,
            type=QuestionType.BINARY,
            question_text="Test weight",
            resolution=True,
            options=None,
            range_max=None,
            range_min=None,
            open_upper_bound=None,
            open_lower_bound=None,
            weight=1.5,
            spot_scoring_time=datetime(2025, 1, 1),
        )
    # Empty question text
    with pytest.raises(ValueError):
        Question(
            post_id=5,
            question_id=5,
            type=QuestionType.BINARY,
            question_text="   ",
            resolution=True,
            options=None,
            range_max=None,
            range_min=None,
            open_upper_bound=None,
            open_lower_bound=None,
            weight=1.0,
            spot_scoring_time=datetime(2025, 1, 1),
        )
    # MC with less than two options
    with pytest.raises(ValueError):
        Question(
            post_id=6,
            question_id=6,
            type=QuestionType.MULTIPLE_CHOICE,
            question_text="Test MC options",
            resolution="A",
            options=["A"],
            range_max=None,
            range_min=None,
            open_upper_bound=None,
            open_lower_bound=None,
            weight=1.0,
            spot_scoring_time=datetime(2025, 1, 1),
        )
    # Numeric missing bounds
    with pytest.raises(ValueError):
        Question(
            post_id=7,
            question_id=7,
            type=QuestionType.NUMERIC,
            question_text="Test numeric bounds",
            resolution=1.0,
            options=None,
            range_max=None,
            range_min=0.0,
            open_upper_bound=None,
            open_lower_bound=None,
            weight=1.0,
            spot_scoring_time=datetime(2025, 1, 1),
        )
