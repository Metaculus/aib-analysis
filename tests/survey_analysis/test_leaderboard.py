"""Questions missed after a bot's first forecast, the leaderboard cache, and the season-start check.

Runs without env vars or local data.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aib_analysis.survey_analysis import leaderboard, season_notes
from aib_analysis.survey_analysis.leaderboard import (
    LeaderboardRow,
    count_questions_open_after_first_forecast,
    load_cached_leaderboard,
    save_leaderboard,
)


def _at_hour(hour: int) -> datetime:
    return datetime(2026, 6, 1, hour, tzinfo=timezone.utc)


def _forecast(question_id: int, hour: int) -> SimpleNamespace:
    return SimpleNamespace(
        prediction_time=_at_hour(hour),
        question=SimpleNamespace(question_id=question_id),
        user=SimpleNamespace(name="late-bot"),
    )


SCORING_TIME_BY_QUESTION = {1: _at_hour(2), 2: _at_hour(5), 3: _at_hour(7), 4: _at_hour(9)}


def test_questions_scored_before_the_first_forecast_are_not_open_to_the_bot():
    forecasts = [_forecast(2, hour=4), _forecast(4, hour=8)]
    assert count_questions_open_after_first_forecast(forecasts, SCORING_TIME_BY_QUESTION) == 3


def test_questions_missed_is_open_questions_minus_forecast_questions():
    row = LeaderboardRow(rank=3, bot_name="b", sum_spot_peer=1.0, question_count=2, questions_open_after_start=3)
    assert row.questions_missed == 1


def test_forecast_on_a_question_scored_before_the_first_forecast_raises():
    scoring_times = {**SCORING_TIME_BY_QUESTION, 5: _at_hour(3)}
    forecasts = [_forecast(5, hour=4), _forecast(2, hour=4)]
    with pytest.raises(ValueError, match="scored before its first forecast"):
        count_questions_open_after_first_forecast(forecasts, scoring_times)


def test_leaderboard_cache_round_trips_questions_missed(tmp_path):
    path = str(tmp_path / "lb.csv")
    rows = [LeaderboardRow(rank=1, bot_name="b", sum_spot_peer=1.5, question_count=320, questions_open_after_start=326)]
    save_leaderboard(rows, path)
    assert load_cached_leaderboard(path)[0].questions_missed == 6


def test_old_leaderboard_cache_without_open_question_counts_is_rejected(tmp_path):
    path = tmp_path / "lb.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "bot_name", "sum_spot_peer", "question_count", "season_question_count"])
        writer.writerow([1, "b", 1.0, 300, 326])
    with pytest.raises(ValueError, match="--refresh"):
        load_cached_leaderboard(str(path))


def test_season_start_must_match_the_first_scored_forecast(monkeypatch):
    first_forecast = SimpleNamespace(
        prediction_time=datetime.combine(season_notes.SEASON_START_DATE, datetime.min.time(), timezone.utc)
    )
    leaderboard._check_season_start([[first_forecast]])
    monkeypatch.setattr(season_notes, "SEASON_START_DATE", datetime(2020, 1, 1).date())
    with pytest.raises(ValueError, match="SEASON_START_DATE"):
        leaderboard._check_season_start([[first_forecast]])
