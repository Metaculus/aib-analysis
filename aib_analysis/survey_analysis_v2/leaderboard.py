"""Build the Spring 2026 sum-of-spot-peer-score leaderboard.

Reuses the existing tournament machinery: load the saved bot tournament, build a
SPOT_PEER leaderboard, and rank bots by sum of spot peer score. The result is
cached to a small CSV so the report can rerun without touching the 327MB
tournament JSON.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass

from aib_analysis.data_structures.custom_types import ScoreType
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.main_logic.process_tournament import get_leaderboard
from aib_analysis.survey_analysis_v2 import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    bot_name: str
    sum_spot_peer: float
    question_count: int


def compute_leaderboard(tournament_json: str) -> list[LeaderboardRow]:
    logger.info("Loading bot tournament from %s (large file, please wait)", tournament_json)
    with open(tournament_json) as handle:
        tournament = SimulatedTournament.model_validate_json(handle.read())
    logger.info("Loaded tournament with %d bots; building spot-peer leaderboard", len(tournament.users))

    leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
    entries = leaderboard.entries_via_sum_of_scores()
    rows = [
        LeaderboardRow(
            rank=rank,
            bot_name=entry.user.name,
            sum_spot_peer=entry.sum_of_scores,
            question_count=entry.question_count,
        )
        for rank, entry in enumerate(entries, start=1)
    ]
    logger.info("Leaderboard built: %d ranked bots", len(rows))
    return rows


def save_leaderboard(rows: list[LeaderboardRow], cache_csv: str) -> None:
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    with open(cache_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "bot_name", "sum_spot_peer", "question_count"])
        for row in rows:
            writer.writerow([row.rank, row.bot_name, f"{row.sum_spot_peer:.6f}", row.question_count])
    logger.info("Wrote leaderboard cache to %s", cache_csv)


def load_cached_leaderboard(cache_csv: str) -> list[LeaderboardRow]:
    with open(cache_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            LeaderboardRow(
                rank=int(row["rank"]),
                bot_name=row["bot_name"],
                sum_spot_peer=float(row["sum_spot_peer"]),
                question_count=int(row["question_count"]),
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
