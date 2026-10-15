"""Predict a tournament's official leaderboard after Metaculus rescores it with current user flags.

Scores the forecast CSV twice: once leaving each listed user out of the peer baseline on questions
Metaculus scored inside that user's exclusion window (the flags it had at the time), and once with
current flags (what a rescore computes). The first run must reproduce the official per-question
scores exactly; if it does, the second run is the post-rescore leaderboard.
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.custom_types import ScoreType, UserType
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.main_logic.load_tournament import load_tournament
from aib_analysis.main_logic.official_leaderboard import (
    OfficialLeaderboardEntry,
    build_official_leaderboard,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)

SCORE_TOLERANCE = 1e-6
QuestionUserKey = tuple[int, str]


@dataclass(frozen=True)
class BaselineExclusionWindow:
    user_name: str
    excluded_from: datetime | None
    excluded_until: datetime

    def contains(self, moment: datetime) -> bool:
        after_start = self.excluded_from is None or moment >= self.excluded_from
        return after_start and moment < self.excluded_until

    @classmethod
    def from_arguments(cls, user_name: str, excluded_from: str, excluded_until: str) -> "BaselineExclusionWindow":
        return cls(
            user_name=user_name,
            excluded_from=None if excluded_from == "start" else datetime.fromisoformat(excluded_from),
            excluded_until=datetime.fromisoformat(excluded_until),
        )


def spot_peer_scores(tournament: SimulatedTournament) -> dict[QuestionUserKey, float]:
    return {
        (score.forecast.question.question_id, score.forecast.user.name): score.score
        for score in tournament.scores
        if score.type == ScoreType.SPOT_PEER
    }


def saved_spot_peer_scores(tournament_json_path: Path) -> dict[QuestionUserKey, float]:
    logger.info(f"Loading saved simulation {tournament_json_path}")
    with open(tournament_json_path) as file_handle:
        raw_tournament = json.load(file_handle)
    return {
        (score["forecast"]["question"]["question_id"], score["forecast"]["user"]["name"]): score["score"]
        for score in raw_tournament["scores_cache"].values()
        if score["type"] == ScoreType.SPOT_PEER.value
    }


def load_official_scores(
    official_scores_path: Path, question_weights: dict[int, float]
) -> tuple[dict[QuestionUserKey, float], dict[int, datetime]]:
    """Read official per-question spot peer scores (unweighted, as Metaculus stores them) and weight them."""
    with open(official_scores_path) as file_handle:
        rows = json.load(file_handle)
    unknown_question_ids = {row["question_id"] for row in rows} - set(question_weights)
    if unknown_question_ids:
        raise ValueError(
            f"Official scores reference questions missing from the CSV: {sorted(unknown_question_ids)}"
        )
    weighted_scores = {
        (row["question_id"], row["user"]): row["score"] * question_weights[row["question_id"]]
        for row in rows
    }
    scored_at = {row["question_id"]: datetime.fromisoformat(row["edited_at"]) for row in rows}
    return weighted_scores, scored_at


def compare_scores(
    label: str,
    expected: dict[QuestionUserKey, float],
    actual: dict[QuestionUserKey, float],
) -> list[str]:
    problems = []
    only_expected = expected.keys() - actual.keys()
    only_actual = actual.keys() - expected.keys()
    if only_expected or only_actual:
        problems.append(
            f"{label}: {len(only_expected)} scores missing, {len(only_actual)} unexpected "
            f"(examples: {sorted(only_expected)[:3]}, {sorted(only_actual)[:3]})"
        )
    shared_keys = expected.keys() & actual.keys()
    differences = {key: abs(expected[key] - actual[key]) for key in shared_keys}
    over_tolerance = {key: value for key, value in differences.items() if value > SCORE_TOLERANCE}
    logger.info(
        f"{label}: {len(shared_keys)} (question, user) scores compared, "
        f"max |diff| = {max(differences.values()):.2e}, over {SCORE_TOLERANCE:g}: {len(over_tolerance)}"
    )
    if over_tolerance:
        worst = sorted(over_tolerance.items(), key=lambda item: item[1], reverse=True)[:5]
        problems.append(f"{label}: {len(over_tolerance)} scores differ, worst {worst}")
    return problems


def total_by_user(scores: dict[QuestionUserKey, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for (_, user_name), score in scores.items():
        totals[user_name] = totals.get(user_name, 0.0) + score
    return totals


def baseline_exclusions_at(
    scoring_times: dict[int, datetime], windows: list[BaselineExclusionWindow]
) -> dict[int, set[str]]:
    exclusions: dict[int, set[str]] = {}
    for question_id, scoring_time in scoring_times.items():
        excluded_users = {window.user_name for window in windows if window.contains(scoring_time)}
        if excluded_users:
            exclusions[question_id] = excluded_users
    return exclusions


def write_comparison(
    official_board: list[OfficialLeaderboardEntry],
    predicted_board: list[OfficialLeaderboardEntry],
    path: Path,
) -> None:
    predicted_by_user = {entry.user_name: entry for entry in predicted_board}
    with open(path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            ["user", "excluded_from_ranking", "official_rank", "official_score",
             "predicted_rank", "predicted_score", "rank_change", "score_change"]
        )
        for entry in official_board:
            predicted = predicted_by_user.get(entry.user_name)
            if predicted is None:
                writer.writerow([entry.user_name, entry.is_excluded_from_ranking, entry.rank,
                                 f"{entry.score:.4f}", "", "", "", ""])
                continue
            writer.writerow(
                [entry.user_name, entry.is_excluded_from_ranking, entry.rank, f"{entry.score:.4f}",
                 predicted.rank, f"{predicted.score:.4f}", entry.rank - predicted.rank,
                 f"{predicted.score - entry.score:.4f}"]
            )


def main(
    bot_csv_path: Path,
    official_scores_path: Path,
    exclusion_windows: list[BaselineExclusionWindow],
    saved_simulation_path: Path | None,
    output_folder: Path,
) -> None:
    initialize_logging()
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Scoring with current flags (what a rescore computes)")
    current_flags = load_tournament(str(bot_csv_path), UserType.BOT, "Current flags")
    question_weights = {question.question_id: question.weight for question in current_flags.questions}
    scored_question_ids = {
        question.question_id
        for question in current_flags.questions
        if not question.is_annulled_or_ambiguous
    }

    official_scores, official_scored_at = load_official_scores(official_scores_path, question_weights)
    problems = []
    if set(official_scored_at) != scored_question_ids:
        problems.append(
            f"Scored questions differ between the CSV and the official scores: "
            f"{sorted(set(official_scored_at) ^ scored_question_ids)}"
        )
    exclusions = baseline_exclusions_at(official_scored_at, exclusion_windows)
    for window in exclusion_windows:
        affected = sum(1 for users in exclusions.values() if window.user_name in users)
        logger.info(
            f"Scoring with flags at scoring time: {window.user_name} left out of the peer baseline on "
            f"{affected} questions scored from {window.excluded_from or 'the start'} until {window.excluded_until}"
        )
    flags_at_scoring_time = SimulatedTournament(
        name="Flags at scoring time",
        forecasts=current_flags.forecasts,
        peer_baseline_exclusions=exclusions,
    )

    problems += compare_scores(
        "Official vs flags at scoring time", official_scores, spot_peer_scores(flags_at_scoring_time)
    )
    predicted_scores = spot_peer_scores(current_flags)
    if saved_simulation_path is not None:
        problems += compare_scores(
            "Saved simulation vs current flags", saved_spot_peer_scores(saved_simulation_path), predicted_scores
        )

    excluded_from_ranking = {
        user.name
        for user in current_flags.users
        if user.type != UserType.BOT or user.is_primary_bot is False
    }
    logger.info(
        f"Excluded from ranking (humans are assumed excluded; non-primary bots always are): "
        f"{sorted(excluded_from_ranking)}"
    )
    official_board = build_official_leaderboard(total_by_user(official_scores), excluded_from_ranking)
    reproduced_board = build_official_leaderboard(
        total_by_user(spot_peer_scores(flags_at_scoring_time)), excluded_from_ranking
    )
    predicted_board = build_official_leaderboard(total_by_user(predicted_scores), excluded_from_ranking)
    reproduced_ranks = {entry.user_name: entry.rank for entry in reproduced_board}
    rank_mismatches = [entry.user_name for entry in official_board if reproduced_ranks.get(entry.user_name) != entry.rank]
    if rank_mismatches:
        problems.append(f"Ranks from flags at scoring time differ from official totals for {rank_mismatches}")

    comparison_path = output_folder / "official_vs_predicted_leaderboard.csv"
    write_comparison(official_board, predicted_board, comparison_path)
    predicted_ranks = {entry.user_name: entry.rank for entry in predicted_board}
    ranked_entries = [entry for entry in official_board if not entry.is_excluded_from_ranking]
    changed = [entry for entry in ranked_entries if predicted_ranks.get(entry.user_name, entry.rank) != entry.rank]
    first_changed_rank = min((entry.rank for entry in changed), default=None)
    largest_move = max((abs(predicted_ranks[entry.user_name] - entry.rank) for entry in changed), default=0)
    logger.info(
        f"After rescoring: {len(changed)} of {len(ranked_entries)} ranked bots change rank "
        f"(first change at official rank {first_changed_rank}, largest move {largest_move}). "
        f"Wrote {comparison_path}"
    )

    if problems:
        raise ValueError("Rescore check failed:\n- " + "\n- ".join(problems))
    logger.info("Rescore check passed: the official scores are reproduced exactly with flags at scoring time")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-csv", required=True, type=Path)
    parser.add_argument(
        "--official-scores", required=True, type=Path,
        help="JSON list of official spot peer scores: question_id, user, score (unweighted), edited_at",
    )
    parser.add_argument(
        "--baseline-exclusion", required=True, action="append", nargs=3,
        metavar=("USER", "FROM", "UNTIL"),
        help="Leave USER out of the peer baseline on questions resolved in [FROM, UNTIL). "
        "FROM is an ISO time or 'start'. Repeat for several users or windows.",
    )
    parser.add_argument("--saved-simulation", type=Path, default=None)
    parser.add_argument("--output-folder", required=True, type=Path)
    arguments = parser.parse_args()
    main(
        bot_csv_path=arguments.bot_csv,
        official_scores_path=arguments.official_scores,
        exclusion_windows=[
            BaselineExclusionWindow.from_arguments(*window_arguments)
            for window_arguments in arguments.baseline_exclusion
        ],
        saved_simulation_path=arguments.saved_simulation,
        output_folder=arguments.output_folder,
    )
