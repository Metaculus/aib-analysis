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
EXCLUDED_FROM_RANKING_STATUS = 2
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


@dataclass(frozen=True)
class OfficialLeaderboardRow:
    score: float
    rank: int
    exclusion_status: int


def load_official_leaderboard(official_leaderboard_path: Path) -> dict[str, OfficialLeaderboardRow]:
    with open(official_leaderboard_path) as file_handle:
        rows = json.load(file_handle)
    return {
        row["user"]: OfficialLeaderboardRow(
            score=row["score"], rank=row["rank"], exclusion_status=row["exclusion_status"]
        )
        for row in rows
    }


def compare_leaderboard(
    official_leaderboard: dict[str, OfficialLeaderboardRow],
    reproduced_board: list[OfficialLeaderboardEntry],
) -> list[str]:
    reproduced_by_user = {entry.user_name: entry for entry in reproduced_board}
    if reproduced_by_user.keys() != official_leaderboard.keys():
        only_official = sorted(official_leaderboard.keys() - reproduced_by_user.keys())
        only_reproduced = sorted(reproduced_by_user.keys() - official_leaderboard.keys())
        return [f"Leaderboard users differ: only official {only_official}, only reproduced {only_reproduced}"]
    score_differences = {
        user_name: abs(row.score - reproduced_by_user[user_name].score)
        for user_name, row in official_leaderboard.items()
    }
    rank_mismatches = sorted(
        user_name for user_name, row in official_leaderboard.items()
        if row.rank != reproduced_by_user[user_name].rank
    )
    logger.info(
        f"Official leaderboard vs flags at scoring time: {len(official_leaderboard)} entries, "
        f"max |score diff| = {max(score_differences.values()):.2e}, rank mismatches: {len(rank_mismatches)}"
    )
    problems = []
    over_tolerance = sorted(user for user, difference in score_differences.items() if difference > SCORE_TOLERANCE)
    if over_tolerance:
        problems.append(f"Leaderboard scores differ for {over_tolerance}")
    if rank_mismatches:
        problems.append(f"Leaderboard ranks differ for {rank_mismatches}")
    return problems


def write_leaderboard_comparison(
    official_leaderboard: dict[str, OfficialLeaderboardRow],
    reproduced_board: list[OfficialLeaderboardEntry],
    predicted_board: list[OfficialLeaderboardEntry],
    path: Path,
) -> None:
    reproduced_by_user = {entry.user_name: entry for entry in reproduced_board}
    predicted_by_user = {entry.user_name: entry for entry in predicted_board}
    with open(path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            ["user", "exclusion_status", "official_rank", "official_score",
             "reproduced_rank", "reproduced_score", "predicted_rank", "predicted_score",
             "rank_change", "score_change"]
        )
        for user_name, row in sorted(official_leaderboard.items(), key=lambda item: -item[1].score):
            reproduced = reproduced_by_user[user_name]
            predicted = predicted_by_user[user_name]
            writer.writerow(
                [user_name, row.exclusion_status, row.rank, f"{row.score:.6f}",
                 reproduced.rank, f"{reproduced.score:.6f}", predicted.rank, f"{predicted.score:.6f}",
                 row.rank - predicted.rank, f"{predicted.score - row.score:.6f}"]
            )


def write_question_comparison(
    official_scores: dict[QuestionUserKey, float],
    reproduced_scores: dict[QuestionUserKey, float],
    predicted_scores: dict[QuestionUserKey, float],
    exclusions: dict[int, set[str]],
    path: Path,
) -> None:
    with open(path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            ["question_id", "user", "official_score", "reproduced_score", "predicted_score",
             "left_out_of_baseline_at_scoring"]
        )
        for question_id, user_name in sorted(official_scores):
            key = (question_id, user_name)
            writer.writerow(
                [question_id, user_name, f"{official_scores[key]:.9f}",
                 f"{reproduced_scores[key]:.9f}", f"{predicted_scores[key]:.9f}",
                 ";".join(sorted(exclusions.get(question_id, set())))]
            )


def main(
    bot_csv_path: Path,
    official_scores_path: Path,
    official_leaderboard_path: Path,
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

    reproduced_scores = spot_peer_scores(flags_at_scoring_time)
    problems += compare_scores("Official vs flags at scoring time", official_scores, reproduced_scores)
    predicted_scores = spot_peer_scores(current_flags)
    if saved_simulation_path is not None:
        problems += compare_scores(
            "Saved simulation vs current flags", saved_spot_peer_scores(saved_simulation_path), predicted_scores
        )

    official_leaderboard = load_official_leaderboard(official_leaderboard_path)
    excluded_from_ranking = {
        user_name
        for user_name, row in official_leaderboard.items()
        if row.exclusion_status >= EXCLUDED_FROM_RANKING_STATUS
    }
    logger.info(f"Using the official exclusion statuses: {len(excluded_from_ranking)} entries do not use up a rank")
    reproduced_board = build_official_leaderboard(total_by_user(reproduced_scores), excluded_from_ranking)
    predicted_board = build_official_leaderboard(total_by_user(predicted_scores), excluded_from_ranking)
    problems += compare_leaderboard(official_leaderboard, reproduced_board)

    leaderboard_path = output_folder / "official_vs_predicted_leaderboard.csv"
    question_path = output_folder / "official_vs_reproduced_scores_by_question.csv"
    write_leaderboard_comparison(official_leaderboard, reproduced_board, predicted_board, leaderboard_path)
    write_question_comparison(official_scores, reproduced_scores, predicted_scores, exclusions, question_path)
    predicted_ranks = {entry.user_name: entry.rank for entry in predicted_board}
    ranked_users = [user_name for user_name in official_leaderboard if user_name not in excluded_from_ranking]
    changed = [
        user_name for user_name in ranked_users
        if predicted_ranks[user_name] != official_leaderboard[user_name].rank
    ]
    first_changed_rank = min((official_leaderboard[user_name].rank for user_name in changed), default=None)
    largest_move = max(
        (abs(predicted_ranks[user_name] - official_leaderboard[user_name].rank) for user_name in changed), default=0
    )
    logger.info(
        f"After rescoring: {len(changed)} of {len(ranked_users)} ranked entries change rank "
        f"(first change at official rank {first_changed_rank}, largest move {largest_move}). "
        f"Wrote {leaderboard_path} and {question_path}"
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
        "--official-leaderboard", required=True, type=Path,
        help="JSON list of official leaderboard entries: user, score, rank, exclusion_status",
    )
    parser.add_argument(
        "--baseline-exclusion", required=True, action="append", nargs=3,
        metavar=("USER", "FROM", "UNTIL"),
        help=(
            "Leave USER out of the peer baseline on questions Metaculus scored in [FROM, UNTIL). "
            "FROM is an ISO time or 'start'. Repeat for several users or windows."
        ),
    )
    parser.add_argument("--saved-simulation", type=Path, default=None)
    parser.add_argument("--output-folder", required=True, type=Path)
    arguments = parser.parse_args()
    main(
        bot_csv_path=arguments.bot_csv,
        official_scores_path=arguments.official_scores,
        official_leaderboard_path=arguments.official_leaderboard,
        exclusion_windows=[
            BaselineExclusionWindow.from_arguments(*window_arguments)
            for window_arguments in arguments.baseline_exclusion
        ],
        saved_simulation_path=arguments.saved_simulation,
        output_folder=arguments.output_folder,
    )
