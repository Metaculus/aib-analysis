"""Build the season's sum-of-spot-peer-score leaderboard.

Reuses the existing tournament machinery: load the saved bot tournament, build a
SPOT_PEER leaderboard, and rank bots by sum of spot peer score. The result is
cached to a small CSV so the report can rerun without touching the large
tournament JSON.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from aib_analysis.data_structures.custom_types import ScoreType
from aib_analysis.data_structures.data_models import Forecast
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.main_logic.process_tournament import get_leaderboard
from aib_analysis.survey_analysis import config, season_notes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    bot_name: str
    sum_spot_peer: float
    question_count: int
    questions_open_after_start: int

    @property
    def questions_missed(self) -> int:
        return self.questions_open_after_start - self.question_count


def count_questions_open_after_first_forecast(
    forecasts: list[Forecast], scoring_time_by_question: dict[int, datetime]
) -> int:
    """Count the scored questions whose spot-scoring time came after the bot's first forecast.

    The tournament data has no question open time, so a question counts as open
    to the bot if it had not yet been scored when the bot first forecast.
    """
    first_forecast_time = min(forecast.prediction_time for forecast in forecasts)
    open_after_start = {
        question_id
        for question_id, scoring_time in scoring_time_by_question.items()
        if scoring_time > first_forecast_time
    }
    forecast_after_scoring = {
        forecast.question.question_id for forecast in forecasts
    } - open_after_start
    if forecast_after_scoring:
        raise ValueError(
            f"{forecasts[0].user.name} has scored forecasts on questions scored before its first "
            f"forecast: {sorted(forecast_after_scoring)}"
        )
    return len(open_after_start)


def compute_leaderboard(tournament_json: str) -> list[LeaderboardRow]:
    logger.info("Loading bot tournament from %s (large file, please wait)", tournament_json)
    with open(tournament_json) as handle:
        tournament = SimulatedTournament.model_validate_json(handle.read())
    logger.info("Loaded tournament with %d bots; building spot-peer leaderboard", len(tournament.users))

    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    entries = leaderboard.entries_via_sum_of_scores()
    forecasts_by_entry = [[score.forecast for score in entry.scores] for entry in entries]
    _check_season_start(forecasts_by_entry)
    scoring_time_by_question = {
        forecast.question.question_id: forecast.question.spot_scoring_time
        for forecasts in forecasts_by_entry
        for forecast in forecasts
    }
    rows = [
        LeaderboardRow(
            rank=rank,
            bot_name=entry.user.name,
            sum_spot_peer=entry.sum_of_scores,
            question_count=entry.question_count,
            questions_open_after_start=count_questions_open_after_first_forecast(
                forecasts, scoring_time_by_question
            ),
        )
        for rank, (entry, forecasts) in enumerate(zip(entries, forecasts_by_entry), start=1)
    ]
    logger.info(
        "Leaderboard built: %d ranked bots over %d scored questions",
        len(rows),
        len(scoring_time_by_question),
    )
    return rows


def _check_season_start(forecasts_by_entry: list[list[Forecast]]) -> None:
    first_forecast_date = min(
        forecast.prediction_time for forecasts in forecasts_by_entry for forecast in forecasts
    ).date()
    if first_forecast_date != season_notes.SEASON_START_DATE:
        raise ValueError(
            f"season_notes.SEASON_START_DATE is {season_notes.SEASON_START_DATE}, but the tournament's "
            f"first scored forecast was on {first_forecast_date}. Update SEASON_START_DATE (it sets the "
            "frontier release cutoff)."
        )


def save_leaderboard(rows: list[LeaderboardRow], cache_csv: str) -> None:
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    with open(cache_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "bot_name", "sum_spot_peer", "question_count", "questions_open_after_start"])
        for row in rows:
            writer.writerow(
                [
                    row.rank,
                    row.bot_name,
                    f"{row.sum_spot_peer:.6f}",
                    row.question_count,
                    row.questions_open_after_start,
                ]
            )
    logger.info("Wrote leaderboard cache to %s", cache_csv)


def load_cached_leaderboard(cache_csv: str) -> list[LeaderboardRow]:
    with open(cache_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        if "questions_open_after_start" not in (reader.fieldnames or []):
            raise ValueError(
                f"{cache_csv} predates the questions_open_after_start column; rerun with --refresh"
            )
        return [
            LeaderboardRow(
                rank=int(row["rank"]),
                bot_name=row["bot_name"],
                sum_spot_peer=float(row["sum_spot_peer"]),
                question_count=int(row["question_count"]),
                questions_open_after_start=int(row["questions_open_after_start"]),
            )
            for row in reader
        ]


def get_leaderboard_rows(refresh: bool = False) -> list[LeaderboardRow]:
    """Return leaderboard rows, using the cache unless refresh is requested."""
    if not refresh and os.path.exists(config.LEADERBOARD_CACHE_CSV):
        logger.info("Using cached leaderboard at %s", config.LEADERBOARD_CACHE_CSV)
        return load_cached_leaderboard(config.LEADERBOARD_CACHE_CSV)
    rows = compute_leaderboard(config.BOT_TOURNAMENT_JSON)
    save_leaderboard(rows, config.LEADERBOARD_CACHE_CSV)
    return rows
