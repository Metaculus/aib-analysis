"""Summarize one season's simulation JSON outputs into a markdown report with tables and charts."""

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)

from aib_analysis.math.stats import (
    ConfidenceIntervalCalculator,
    MeanHypothesisCalculator,
)
from conftest import initialize_logging

logger = logging.getLogger(__name__)

CONFIDENCE_LEVEL = 0.95
TIE_TOLERANCE = 1e-9
BOT_COLOR = "#2a78d6"
PRO_COLOR = "#eb6834"
AGGREGATE_COLOR = "#1baf7a"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"

# Bot team vs Pro team head-to-head (mean spot peer score of the bot team with 95% CI)
# as published in the results write-ups of previous seasons.
PREVIOUS_SEASON_HEAD_TO_HEAD: list[dict[str, Any]] = [
    {"season": "Q3 2024", "mean": -11.3, "lower": -21.8, "upper": -0.7, "questions": None},
    {"season": "Q4 2024", "mean": -8.9, "lower": -18.8, "upper": 1.0, "questions": None},
    {"season": "Q1 2025", "mean": -17.7, "lower": -28.3, "upper": -7.0, "questions": None},
    {"season": "Q2 2025", "mean": -20.03, "lower": -28.63, "upper": -11.41, "questions": 93},
    {"season": "Spring 2026", "mean": -1.25, "lower": -4.87, "upper": 2.37, "questions": 99},
]


@dataclass(frozen=True)
class ScoredQuestion:
    question_id: int
    question_type: str
    score: float


@dataclass
class UserResult:
    name: str
    user_type: str
    aggregated_user_count: int
    scores: list[ScoredQuestion]

    @property
    def sum_of_scores(self) -> float:
        return sum(scored.score for scored in self.scores)

    @property
    def average_score(self) -> float:
        return self.sum_of_scores / len(self.scores)

    @property
    def question_count(self) -> int:
        return len({scored.question_id for scored in self.scores})

    @property
    def is_metac_bot(self) -> bool:
        return "metac-" in self.name.lower() or "mf-bot-" in self.name.lower()

    def confidence_interval(self) -> tuple[float, float] | None:
        return confidence_interval_or_none([scored.score for scored in self.scores])


class RawTournament:
    """Thin reader over a saved SimulatedTournament JSON (avoids slow pydantic validation)."""

    def __init__(self, path: Path) -> None:
        logger.info(f"Loading {path.name}")
        with open(path) as file_handle:
            self.raw: dict[str, Any] = json.load(file_handle)
        self.path = path

    def spot_peer_results(self) -> dict[str, UserResult]:
        results: dict[str, UserResult] = {}
        for score in self.raw["scores_cache"].values():
            if score["type"] != "spot_peer":
                continue
            forecast = score["forecast"]
            user = forecast["user"]
            question = forecast["question"]
            result = results.setdefault(
                user["name"],
                UserResult(
                    name=user["name"],
                    user_type=user["type"],
                    aggregated_user_count=len(user["aggregated_users"]),
                    scores=[],
                ),
            )
            result.scores.append(
                ScoredQuestion(
                    question_id=question["question_id"],
                    question_type=question["type"],
                    score=score["score"],
                )
            )
        return results

    def questions(self) -> dict[int, dict[str, Any]]:
        questions: dict[int, dict[str, Any]] = {}
        for forecast in self.raw["forecasts"]:
            question = forecast["question"]
            questions.setdefault(question["question_id"], question)
        return questions

    def forecasting_user_names(self) -> set[str]:
        return {forecast["user"]["name"] for forecast in self.raw["forecasts"]}


def is_scored(question: dict[str, Any]) -> bool:
    return (
        question["resolution"] is not None
        and question["unscored_resolution_reason"] is None
    )


def confidence_interval_or_none(observations: list[float]) -> tuple[float, float] | None:
    try:
        interval = ConfidenceIntervalCalculator.confidence_interval_from_observations(
            observations, CONFIDENCE_LEVEL
        )
    except (ValueError, AssertionError) as error:
        logger.debug(f"No confidence interval for {len(observations)} observations: {error}")
        return None
    return interval.lower_bound, interval.upper_bound


def ranked(results: dict[str, UserResult]) -> list[UserResult]:
    return sorted(results.values(), key=lambda result: result.sum_of_scores, reverse=True)


def format_number(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def format_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "n/a (normality assumption failed with n < 30)"
    return f"[{interval[0]:.2f}, {interval[1]:.2f}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def leaderboard_rows(results: list[UserResult]) -> list[list[str]]:
    rows = []
    for rank, result in enumerate(results, start=1):
        interval = result.confidence_interval()
        rows.append(
            [
                str(rank),
                result.name,
                format_number(result.sum_of_scores),
                format_number(result.average_score),
                format_interval(interval),
                str(result.question_count),
                result.user_type,
            ]
        )
    return rows


LEADERBOARD_HEADERS = [
    "Rank",
    "User",
    "Sum of spot peer scores",
    "Average score",
    "95% CI of average",
    "Questions scored",
    "Type",
]


def write_leaderboard_csv(results: list[UserResult], path: Path) -> None:
    with open(path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            ["rank", "user", "sum_of_scores", "average_score", "ci_lower", "ci_upper", "questions_scored", "user_type"]
        )
        for rank, result in enumerate(results, start=1):
            interval = result.confidence_interval()
            writer.writerow(
                [
                    rank,
                    result.name,
                    f"{result.sum_of_scores:.4f}",
                    f"{result.average_score:.4f}",
                    "" if interval is None else f"{interval[0]:.4f}",
                    "" if interval is None else f"{interval[1]:.4f}",
                    result.question_count,
                    result.user_type,
                ]
            )


def _style_axes(axes: plt.Axes) -> None:
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID_COLOR)
    axes.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED_COLOR, labelsize=8)
    axes.title.set_color(TEXT_COLOR)
    axes.xaxis.label.set_color(MUTED_COLOR)
    axes.yaxis.label.set_color(MUTED_COLOR)


def plot_leaderboard(
    results: list[UserResult], title: str, path: Path, show_labels: bool
) -> None:
    names = [result.name for result in results]
    averages = [result.average_score for result in results]
    intervals = [result.confidence_interval() for result in results]
    lower_errors = [
        0 if interval is None else average - interval[0]
        for average, interval in zip(averages, intervals)
    ]
    upper_errors = [
        0 if interval is None else interval[1] - average
        for average, interval in zip(averages, intervals)
    ]
    colors = [PRO_COLOR if result.user_type == "pro" else BOT_COLOR for result in results]
    width = max(6.0, min(18.0, 0.32 * len(results)))
    figure, axes = plt.subplots(figsize=(width, 5))
    positions = range(len(results))
    axes.bar(positions, averages, color=colors, width=0.7)
    axes.errorbar(
        positions, averages, yerr=[lower_errors, upper_errors],
        fmt="none", ecolor=MUTED_COLOR, elinewidth=1, capsize=2,
    )
    axes.axhline(0, color=MUTED_COLOR, linewidth=0.8)
    axes.set_title(title, fontsize=11, loc="left")
    axes.set_ylabel("Average spot peer score (95% CI)")
    if show_labels:
        axes.set_xticks(list(positions))
        axes.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    else:
        axes.set_xticks([])
        axes.set_xlabel("Participants ordered by sum of spot peer scores (left = 1st place)")
    _style_axes(axes)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_histogram(scores: list[float], title: str, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.hist(scores, bins=30, color=BOT_COLOR, edgecolor="white", linewidth=0.5)
    axes.axvline(0, color=MUTED_COLOR, linewidth=1, linestyle="--")
    axes.set_title(title, fontsize=11, loc="left")
    axes.set_xlabel("Bot team head-to-head score (negative = Pro team scored better)")
    axes.set_ylabel("Questions")
    _style_axes(axes)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_points_with_intervals(
    labels: list[str],
    means: list[float],
    intervals: list[tuple[float, float] | None],
    title: str,
    y_label: str,
    x_label: str,
    path: Path,
    connect_points: bool,
) -> None:
    lower_errors = [0 if ci is None else mean - ci[0] for mean, ci in zip(means, intervals)]
    upper_errors = [0 if ci is None else ci[1] - mean for mean, ci in zip(means, intervals)]
    figure, axes = plt.subplots(figsize=(max(6.0, 0.6 * len(labels) + 2), 4.5))
    positions = range(len(labels))
    if connect_points:
        axes.plot(positions, means, color=BOT_COLOR, linewidth=2, zorder=2)
    axes.errorbar(
        positions, means, yerr=[lower_errors, upper_errors], fmt="o", color=BOT_COLOR,
        ecolor=MUTED_COLOR, elinewidth=1, capsize=3, markersize=6, zorder=3,
    )
    axes.axhline(0, color=MUTED_COLOR, linewidth=0.8)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_title(title, fontsize=11, loc="left")
    axes.set_ylabel(y_label)
    axes.set_xlabel(x_label)
    _style_axes(axes)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def question_type_counts(questions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        counts[question["type"]] = counts.get(question["type"], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def format_type_counts(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    return ", ".join(
        f"{count} {question_type} ({count / total:.0%})" for question_type, count in counts.items()
    )


def overview_section(
    bot_tournament: RawTournament, pro_tournament: RawTournament, combined: RawTournament,
    bot_only: RawTournament,
) -> str:
    bot_results = bot_tournament.spot_peer_results()
    bots_with_scores = [result for result in bot_results.values() if result.user_type == "bot"]
    humans_with_scores = [result for result in bot_results.values() if result.user_type != "bot"]
    metac_bots = [result for result in bots_with_scores if result.is_metac_bot]
    all_bot_questions = list(bot_tournament.questions().values())
    scored_bot_questions = [question for question in all_bot_questions if is_scored(question)]
    annulled_bot_questions = [question for question in all_bot_questions if not is_scored(question)]
    scored_binary = [q for q in scored_bot_questions if q["type"] == "binary"]
    binary_yes = sum(1 for q in scored_binary if q["resolution"] is True)
    weighted_questions = [q for q in scored_bot_questions if q["weight"] != 1.0]
    average_questions_per_bot = sum(r.question_count for r in bots_with_scores) / len(bots_with_scores)
    average_questions_per_metac_bot = sum(r.question_count for r in metac_bots) / len(metac_bots)

    pro_results = pro_tournament.spot_peer_results()
    pros_with_scores = [result for result in pro_results.values() if result.user_type == "pro"]
    all_pro_questions = list(pro_tournament.questions().values())
    scored_pro_questions = [question for question in all_pro_questions if is_scored(question)]
    combined_questions = [q for q in combined.questions().values() if is_scored(q)]
    bot_only_questions = [q for q in bot_only.questions().values() if is_scored(q)]

    lines = [
        "## Tournament overview",
        "",
        "**Bot tournament**",
        "",
        (
            f"- Bots with at least one scored forecast: {len(bots_with_scores)} "
            f"({len(metac_bots)} Metac Bots, {len(bots_with_scores) - len(metac_bots)} non-Metaculus bots). "
            f"{len(bot_tournament.forecasting_user_names())} accounts forecast at all (including on annulled questions only)."
        ),
        (
            f"- Human accounts kept in the bot tournament for peer-score parity (hidden from bot leaderboards): "
            f"{len(humans_with_scores)}"
        ),
        (
            f"- Questions: {len(all_bot_questions)} total, {len(annulled_bot_questions)} annulled/ambiguous, "
            f"{len(scored_bot_questions)} scored"
        ),
        f"- Scored question types: {format_type_counts(question_type_counts(scored_bot_questions))}",
        (
            f"- Binary resolutions: {binary_yes} Yes ({binary_yes / len(scored_binary):.0%}), "
            f"{len(scored_binary) - binary_yes} No ({1 - binary_yes / len(scored_binary):.0%})"
        ),
        f"- Scored questions with weight below 1: {len(weighted_questions)}",
        (
            f"- Average scored questions per bot: {average_questions_per_bot:.0f} "
            f"(Metac Bots: {average_questions_per_metac_bot:.0f})"
        ),
        (
            f"- Bot-only questions (bot tournament minus the Pro questions, used to pick the bot team): "
            f"{len(bot_only_questions)} scored"
        ),
        "",
        "**Pro tournament**",
        "",
        f"- Pros with at least one scored forecast: {len(pros_with_scores)}",
        (
            f"- Questions: {len(all_pro_questions)} total, {len(all_pro_questions) - len(scored_pro_questions)} "
            f"annulled/ambiguous, {len(scored_pro_questions)} scored"
        ),
        f"- Scored question types: {format_type_counts(question_type_counts(scored_pro_questions))}",
        "",
        "**Pro vs Bot overlap**",
        "",
        (
            f"- Questions matched between the two tournaments: {len(combined_questions)} "
            f"({format_type_counts(question_type_counts(combined_questions))})"
        ),
    ]
    return "\n".join(lines)


def metac_section(metac_tournament: RawTournament, output_folder: Path) -> str:
    results = ranked(metac_tournament.spot_peer_results())
    questions = [q for q in metac_tournament.questions().values() if is_scored(q)]
    average_questions = sum(result.question_count for result in results) / len(results)
    plot_leaderboard(
        results, "Metac Bots only: average spot peer score",
        output_folder / "metac_bot_leaderboard.png", show_labels=True,
    )
    write_leaderboard_csv(results, output_folder / "metac_bot_leaderboard.csv")
    lines = [
        "## How do LLMs compare? (Metac Bots only)",
        "",
        (
            f"{len(results)} Metac Bots scored against one another on {len(questions)} scored questions "
            f"(average Metac Bot forecast on {average_questions:.0f} of them)."
        ),
        "",
        markdown_table(LEADERBOARD_HEADERS, leaderboard_rows(results)),
        "",
        "![Metac bot leaderboard](metac_bot_leaderboard.png)",
    ]
    return "\n".join(lines)


def bot_leaderboard_section(bot_tournament: RawTournament, output_folder: Path, top_n: int) -> str:
    results = [r for r in ranked(bot_tournament.spot_peer_results()) if r.user_type == "bot"]
    plot_leaderboard(
        results[:10], "Bot tournament: top 10 by sum of spot peer scores",
        output_folder / "bot_leaderboard_top10.png", show_labels=True,
    )
    plot_leaderboard(
        results, "Bot tournament: all bots ordered by sum of spot peer scores",
        output_folder / "bot_leaderboard_full.png", show_labels=False,
    )
    write_leaderboard_csv(results, output_folder / "bot_leaderboard.csv")
    metac_ranks = [
        (rank, result.name) for rank, result in enumerate(results, start=1) if result.is_metac_bot
    ]
    lines = [
        f"## Which bot strategy is best? (Bot tournament, {len(results)} bots)",
        "",
        f"Top {top_n} of {len(results)} bots (full leaderboard in `bot_leaderboard.csv`):",
        "",
        markdown_table(LEADERBOARD_HEADERS, leaderboard_rows(results[:top_n])),
        "",
        f"- Best Metac Bot: {metac_ranks[0][1]} in {metac_ranks[0][0]}th place",
        f"- Metac Bots in the top 20: {sum(1 for rank, _ in metac_ranks if rank <= 20)}",
        "",
        "![Top 10 bots](bot_leaderboard_top10.png)",
        "",
        "![Full bot leaderboard](bot_leaderboard_full.png)",
    ]
    return "\n".join(lines)


def hypothesis_summary(bot_scores: list[float], pro_scores: list[float]) -> list[str]:
    lines = []
    try:
        two_sided = MeanHypothesisCalculator.test_if_mean_is_equal_to_than_hypothesis_mean(
            bot_scores, 0, CONFIDENCE_LEVEL
        )
        pros_better = MeanHypothesisCalculator.test_if_mean_is_greater_than_hypothesis_mean(
            pro_scores, 0, CONFIDENCE_LEVEL
        )
        lines.append(
            f"- Two-sided t-test that the bot team mean differs from 0: p = {two_sided.p_value:.3f} "
            f"(Shapiro normality test passes: {two_sided.shapiro_test_passes})"
        )
        lines.append(
            f"- One-sided t-test that the Pro team beats the bot team: p = {pros_better.p_value:.3f}"
        )
    except (ValueError, AssertionError) as error:
        lines.append(f"- Hypothesis tests unavailable: {error}")
    return lines


def head_to_head_section(team_tournament: RawTournament, output_folder: Path) -> str:
    results = team_tournament.spot_peer_results()
    bot_team = results["Bot Team"]
    pro_team = results["Pro Team"]
    bot_scores = [scored.score for scored in bot_team.scores]
    pro_scores = [scored.score for scored in pro_team.scores]
    bot_wins = sum(1 for score in bot_scores if score > TIE_TOLERANCE)
    pro_wins = sum(1 for score in bot_scores if score < -TIE_TOLERANCE)
    ties = len(bot_scores) - bot_wins - pro_wins
    plot_histogram(
        bot_scores, "Bot team head-to-head scores vs the Pro team, by question",
        output_folder / "head_to_head_histogram.png",
    )

    type_rows: list[list[str]] = []
    type_labels: list[str] = []
    type_means: list[float] = []
    type_intervals: list[tuple[float, float] | None] = []
    type_groups = {
        "binary": ["binary"],
        "multiple_choice": ["multiple_choice"],
        "numeric": ["numeric"],
        "discrete": ["discrete"],
        "numeric + discrete": ["numeric", "discrete"],
    }
    for label, types in type_groups.items():
        scores = [scored.score for scored in bot_team.scores if scored.question_type in types]
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        interval = confidence_interval_or_none(scores)
        type_rows.append([label, str(len(scores)), format_number(mean), format_interval(interval)])
        if label != "numeric + discrete":
            type_labels.append(label)
            type_means.append(mean)
            type_intervals.append(interval)
    plot_points_with_intervals(
        type_labels, type_means, type_intervals,
        "Bot team head-to-head score by question type", "Average head-to-head score (95% CI)",
        "Question type", output_folder / "head_to_head_by_type.png", connect_points=False,
    )

    lines = [
        "## Are bots better than human Pros? (Bot team vs Pro team)",
        "",
        (
            f"Bot team members ({bot_team.aggregated_user_count} bots, top of the bot-only leaderboard): "
            + ", ".join(sorted(aggregated_user_names(team_tournament, "Bot Team")))
        ),
        "",
        f"- Questions compared: {len(bot_scores)}",
        (
            f"- Bot team average head-to-head score: {bot_team.average_score:.2f}, "
            f"95% CI {format_interval(bot_team.confidence_interval())}"
        ),
        (
            f"- Pro team average head-to-head score: {pro_team.average_score:.2f}, "
            f"95% CI {format_interval(pro_team.confidence_interval())}"
        ),
        f"- Question wins: Pros {pro_wins}, bots {bot_wins}, ties {ties}",
        *hypothesis_summary(bot_scores, pro_scores),
        "",
        "![Head-to-head histogram](head_to_head_histogram.png)",
        "",
        "**By question type (bot team head-to-head score)**",
        "",
        markdown_table(["Question type", "Questions", "Average score", "95% CI"], type_rows),
        "",
        "![Head-to-head by type](head_to_head_by_type.png)",
    ]
    return "\n".join(lines)


def aggregated_user_names(tournament: RawTournament, aggregate_name: str) -> list[str]:
    for forecast in tournament.raw["forecasts"]:
        user = forecast["user"]
        if user["name"] == aggregate_name:
            return [member["name"] for member in user["aggregated_users"]]
    raise ValueError(f"No forecasts by aggregate user {aggregate_name!r} in {tournament.path.name}")


def individual_section(
    combined: RawTournament, pro_user_names: set[str], output_folder: Path, top_n: int
) -> str:
    all_results = ranked(combined.spot_peer_results())
    stray_humans = [
        result.name
        for result in all_results
        if result.user_type != "bot" and result.name not in pro_user_names
    ]
    results = [result for result in all_results if result.name not in stray_humans]
    pro_names = [result.name for result in results if result.user_type == "pro"]
    pro_aliases = {name: f"Pro {chr(ord('A') + index)}" for index, name in enumerate(pro_names)}
    with open(output_folder / "pro_name_key.csv", "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["alias", "username"])
        writer.writerows(sorted(pro_aliases.items()))
    write_leaderboard_csv(results, output_folder / "individual_leaderboard.csv")

    rows = leaderboard_rows(results[:top_n])
    for row in rows:
        row[1] = pro_aliases.get(row[1], row[1])
    pros_in_top_10 = sum(1 for result in results[:10] if result.user_type == "pro")
    lowest_pro_rank = max(rank for rank, r in enumerate(results, start=1) if r.user_type == "pro")
    bots_above_lowest_pro = sum(
        1 for r in results[:lowest_pro_rank] if r.user_type == "bot"
    )
    anonymized_results = [
        UserResult(pro_aliases.get(r.name, r.name), r.user_type, r.aggregated_user_count, r.scores)
        for r in results[:top_n]
    ]
    plot_leaderboard(
        anonymized_results, f"Individual Pros (orange) and bots (blue) on the shared questions: top {top_n}",
        output_folder / "individual_leaderboard_top.png", show_labels=True,
    )
    lines = [
        "## Individual Pros vs individual bots (shared questions only)",
        "",
        (
            f"{len(results)} participants ({len(pro_names)} Pros, {len(results) - len(pro_names)} bots) "
            f"scored together on the shared questions. Pros are anonymized in rank order (key in `pro_name_key.csv`). "
            f"Human accounts from the bot tournament are left out of this table: {len(stray_humans)}."
        ),
        "",
        f"- Pros in the top 10: {pros_in_top_10}",
        f"- Bots ranked above the lowest-ranked Pro (rank {lowest_pro_rank}): {bots_above_lowest_pro}",
        "",
        markdown_table(LEADERBOARD_HEADERS, rows),
        "",
        "![Individual leaderboard](individual_leaderboard_top.png)",
    ]
    return "\n".join(lines)


def team_aggregates_section(with_teams: RawTournament) -> str:
    results = ranked(with_teams.spot_peer_results())
    rows = []
    for rank, result in enumerate(results, start=1):
        if result.user_type != "aggregate":
            continue
        rows.append(
            [
                result.name,
                str(result.aggregated_user_count),
                format_number(result.sum_of_scores),
                format_number(result.average_score),
                format_interval(result.confidence_interval()),
                str(result.question_count),
                str(rank),
            ]
        )
    lines = [
        "## Team aggregates scored against the individual pool",
        "",
        (
            "Each team is the aggregate forecast of its members, scored as one extra participant "
            "among all individual Pros and bots on the shared questions (teams are excluded from the peer baseline)."
        ),
        "",
        markdown_table(
            ["Team", "Members", "Sum of scores", "Average score", "95% CI of average", "Questions", "Rank among individuals + teams"],
            rows,
        ),
    ]
    return "\n".join(lines)


def team_size_section(simulations_folder: Path, output_folder: Path) -> str:
    size_pattern = re.compile(r"pro_vs_bot_tournament__teams_size_(\d+|all)\.json$")
    tournaments_by_size: dict[str, RawTournament] = {}
    for path in sorted(simulations_folder.iterdir()):
        match = size_pattern.search(path.name)
        if match and match.group(1) not in tournaments_by_size:
            tournaments_by_size[match.group(1)] = RawTournament(path)

    rows = []
    labels: list[str] = []
    means: list[float] = []
    intervals: list[tuple[float, float] | None] = []
    for size_label, tournament in sorted(
        tournaments_by_size.items(), key=lambda item: item[1].spot_peer_results()["Bot Team"].aggregated_user_count
    ):
        bot_team = tournament.spot_peer_results()["Bot Team"]
        interval = bot_team.confidence_interval()
        rows.append(
            [
                str(bot_team.aggregated_user_count),
                format_number(bot_team.average_score),
                format_interval(interval),
                str(bot_team.question_count),
            ]
        )
        labels.append(str(bot_team.aggregated_user_count))
        means.append(bot_team.average_score)
        intervals.append(interval)
    plot_points_with_intervals(
        labels, means, intervals, "Bot team head-to-head score vs the Pro team, by bot team size",
        "Average head-to-head score (95% CI)", "Bot team size (top N bots from the bot-only leaderboard)",
        output_folder / "bot_team_size_sensitivity.png", connect_points=True,
    )
    lines = [
        "## How much does bot team size affect aggregation strength?",
        "",
        "Bot teams of different sizes (top N bots from the bot-only leaderboard) against the same Pro team:",
        "",
        markdown_table(["Bot team size", "Average head-to-head score", "95% CI", "Questions"], rows),
        "",
        "![Bot team size sensitivity](bot_team_size_sensitivity.png)",
    ]
    return "\n".join(lines)


def seasons_section(
    season_name: str, bot_team: UserResult, output_folder: Path
) -> str:
    current_interval = bot_team.confidence_interval()
    seasons = PREVIOUS_SEASON_HEAD_TO_HEAD + [
        {
            "season": season_name,
            "mean": bot_team.average_score,
            "lower": None if current_interval is None else current_interval[0],
            "upper": None if current_interval is None else current_interval[1],
            "questions": bot_team.question_count,
        }
    ]
    rows = [
        [
            season["season"],
            format_number(season["mean"]),
            format_number(season["lower"]),
            format_number(season["upper"]),
            "n/a" if season["questions"] is None else str(season["questions"]),
        ]
        for season in seasons
    ]
    plot_points_with_intervals(
        [season["season"] for season in seasons],
        [season["mean"] for season in seasons],
        [
            None if season["lower"] is None else (season["lower"], season["upper"])
            for season in seasons
        ],
        "Bot team vs Pro team head-to-head score by season", "Bot team average head-to-head score (95% CI)",
        "Season", output_folder / "head_to_head_by_season.png", connect_points=False,
    )
    lines = [
        "## Team performance over seasons",
        "",
        (
            "Bot team head-to-head score against the Pro team (0 = equal, negative = Pros better). "
            "Earlier seasons are the published figures; the current season comes from this run."
        ),
        "",
        markdown_table(["Season", "Bot team mean", "95% CI lower", "95% CI upper", "Questions"], rows),
        "",
        "![Head-to-head by season](head_to_head_by_season.png)",
    ]
    return "\n".join(lines)


def find_single_file(simulations_folder: Path, suffix: str, first_if_many: bool = False) -> Path:
    name_pattern = re.compile(r"\d+_" + re.escape(suffix))
    matches = sorted(path for path in simulations_folder.iterdir() if name_pattern.fullmatch(path.name))
    if first_if_many and len(matches) > 1:
        return matches[0]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '<number>_{suffix}' in {simulations_folder}, found {matches}"
        )
    return matches[0]


def main(simulations_folder: Path, output_folder: Path, season_name: str, top_n: int) -> None:
    initialize_logging()
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Summarizing {simulations_folder} into {output_folder}")

    pro_tournament = RawTournament(find_single_file(simulations_folder, "pro_tournament.json"))
    bot_tournament = RawTournament(find_single_file(simulations_folder, "bot_tournament.json"))
    bot_only = RawTournament(find_single_file(simulations_folder, "bot_tournament_without_pro_questions.json"))
    combined = RawTournament(find_single_file(simulations_folder, "pro_with_bot_tourn__no_teams.json"))
    sections = [overview_section(bot_tournament, pro_tournament, combined, bot_only)]
    logger.info("Overview done")

    sections.append(metac_section(RawTournament(find_single_file(simulations_folder, "metac_bot_tournament.json")), output_folder))
    sections.append(bot_leaderboard_section(bot_tournament, output_folder, top_n))
    del bot_tournament
    logger.info("Bot leaderboards done")

    bot_only_results = [r for r in ranked(bot_only.spot_peer_results()) if r.user_type == "bot"]
    sections.append(
        "\n".join(
            [
                "## Bot team qualification (bot-only questions)",
                "",
                "Top 10 bots on the bot-only questions form the bot team:",
                "",
                markdown_table(LEADERBOARD_HEADERS, leaderboard_rows(bot_only_results[:10])),
            ]
        )
    )
    del bot_only

    team_tournament = RawTournament(
        find_single_file(simulations_folder, "pro_vs_bot_tournament__teams_size_10.json", first_if_many=True)
    )
    sections.append(head_to_head_section(team_tournament, output_folder))
    sections.append(
        individual_section(combined, pro_tournament.forecasting_user_names(), output_folder, top_n)
    )
    sections.append(
        team_aggregates_section(
            RawTournament(find_single_file(simulations_folder, "individual_pros_vs_bots_vs_various_teams.json"))
        )
    )
    logger.info("Head-to-head and individual sections done")
    sections.append(team_size_section(simulations_folder, output_folder))
    sections.append(
        seasons_section(season_name, team_tournament.spot_peer_results()["Bot Team"], output_folder)
    )

    report_path = output_folder / "results_summary.md"
    report_path.write_text(
        f"# {season_name} FutureEval simulation results\n\n"
        f"Generated from `{simulations_folder}`. All scores are spot peer scores.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    logger.info(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations-folder", required=True, type=Path)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--season-name", default="Summer 2026")
    parser.add_argument("--top-n", type=int, default=20)
    arguments = parser.parse_args()
    main(
        simulations_folder=arguments.simulations_folder,
        output_folder=arguments.output_folder or arguments.simulations_folder / "results_summary",
        season_name=arguments.season_name,
        top_n=arguments.top_n,
    )
