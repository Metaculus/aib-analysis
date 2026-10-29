"""Static PNG charts for the write-up, sized for a forum post at 2x resolution.

Colors follow one rule across every chart: bots are blue and Pros are orange. Other marks
are gray. The palette passed the dataviz skill's colorblind validator on a white surface.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl-pro-vs-bot-writeup"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from aib_analysis.pro_vs_bot_writeup.phrasing import format_compact
from aib_analysis.pro_vs_bot_writeup.results import (
    TOP_N,
    BotFieldAccuracy,
    FieldSummary,
    IndividualResults,
    LeaderboardRow,
    MetacResults,
    SeasonTrend,
    TeamAccuracy,
    TeamSizeResult,
    bot_label,
    is_metac_bot,
    require_head_to_head_vs_main,
)

logger = logging.getLogger(__name__)

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BOT_COLOR = "#2a78d6"
BOT_COLOR_LIGHT = "#86b6ef"
PRO_COLOR = "#eb6834"
PROMISING_COLOR = "#1baf7a"
DEEMPHASIS = "#b9b7af"
BAND_FILL = "#efeee9"
FIGURE_WIDTH_INCHES = 7.2
DPI = 200
TITLE_BLOCK_INCHES = 0.72
LEGEND_ROW_INCHES = 0.28
COUNT_COLUMN_INCHES = 0.82
LABEL_CHARACTER_WIDTH_INCHES = 0.068
MAX_FIELD_LABELS = 4

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    }
)


def _new_figure(
    height_inches: float,
    left_inches: float = 0.75,
    right_inches: float = 0.25,
    bottom_inches: float = 0.6,
    has_legend: bool = False,
) -> tuple[Figure, Axes]:
    top_inches = TITLE_BLOCK_INCHES + (LEGEND_ROW_INCHES if has_legend else 0)
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, height_inches))
    figure.subplots_adjust(
        left=left_inches / FIGURE_WIDTH_INCHES,
        right=1 - right_inches / FIGURE_WIDTH_INCHES,
        bottom=bottom_inches / height_inches,
        top=1 - top_inches / height_inches,
    )
    return figure, axes


def _add_titles(figure: Figure, title: str, subtitle: str, legend_handles: list[Artist] | None = None) -> None:
    height_inches = figure.get_figheight()
    left = 0.12 / FIGURE_WIDTH_INCHES
    figure.text(left, 1 - 0.12 / height_inches, title, ha="left", va="top", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    figure.text(left, 1 - 0.38 / height_inches, subtitle, ha="left", va="top", fontsize=9, color=TEXT_SECONDARY)
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(left - 0.006, 1 - 0.62 / height_inches),
            ncols=len(legend_handles),
            handletextpad=0.4,
            columnspacing=1.6,
            borderaxespad=0,
        )


def _style_grid(axes: Axes, axis: str) -> None:
    axes.grid(axis=axis, color=GRIDLINE, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(which="both", length=0)
    if axis == "x":
        axes.spines["left"].set_visible(False)


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    logger.info(f"Saved chart {path.name}")


def _left_margin_for_labels(labels: list[str]) -> float:
    return 0.25 + LABEL_CHARACTER_WIDTH_INCHES * max(len(label) for label in labels)


def _zero_line(axes: Axes, horizontal: bool) -> None:
    if horizontal:
        axes.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    else:
        axes.axvline(0, color=BASELINE, linewidth=1, zorder=1)


def _dot(color: str, label: str) -> Line2D:
    return Line2D([], [], marker="o", linestyle="none", color=color, markersize=6.5, label=label)


def season_trend_chart(trend: SeasonTrend, path: Path) -> None:
    figure, axes = _new_figure(3.9)
    positions = list(range(len(trend.points)))
    axes.plot(positions, [point.mean for point in trend.points], color=GRIDLINE, linewidth=1.5, zorder=2)
    for position, point in zip(positions, trend.points):
        color = BOT_COLOR if point.is_current else BOT_COLOR_LIGHT
        axes.plot([position, position], point.ci, color=color, linewidth=2, solid_capstyle="round", zorder=3)
        axes.plot(position, point.mean, "o", color=color, markersize=8 if point.is_current else 6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    current = trend.current
    axes.annotate(
        format_compact(current.mean), xy=(positions[-1], current.ci[1]), xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT_PRIMARY,
    )
    _zero_line(axes, horizontal=True)
    axes.annotate("Bots and Pros equal", xy=(0, 0), xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=TEXT_MUTED)
    axes.text(0.005, 0.98, "Bots better", transform=axes.transAxes, ha="left", va="top", fontsize=8, color=TEXT_MUTED)
    axes.text(0.005, 0.02, "Pros better", transform=axes.transAxes, ha="left", va="bottom", fontsize=8, color=TEXT_MUTED)
    lowest = min(point.ci[0] for point in trend.points)
    highest = max(point.ci[1] for point in trend.points)
    padding = 0.14 * (highest - lowest)
    axes.set_ylim(min(lowest, 0) - padding, max(highest, 0) + padding)
    axes.set_xticks(positions, [point.season_label for point in trend.points])
    axes.set_xlim(-0.5, len(positions) - 0.5)
    axes.set_ylabel("Bot team head-to-head score")
    _style_grid(axes, "y")
    _add_titles(figure, "Bot team vs Pro team, by season", "Average head-to-head score per question, with 95% confidence intervals")
    _save(figure, path)


def team_size_chart(team_sizes: list[TeamSizeResult], main_team_size: int, path: Path) -> None:
    figure, axes = _new_figure(3.9, bottom_inches=0.75)
    sizes = np.array([result.team_size for result in team_sizes], dtype=float)
    means = np.array([result.mean for result in team_sizes])
    lowers = np.array([result.ci[0] if result.ci else result.mean for result in team_sizes])
    uppers = np.array([result.ci[1] if result.ci else result.mean for result in team_sizes])
    axes.fill_between(sizes, lowers, uppers, color=BOT_COLOR, alpha=0.14, linewidth=0, zorder=2)
    axes.plot(sizes, means, color=BOT_COLOR, linewidth=2, zorder=3)
    axes.plot(sizes, means, "o", color=BOT_COLOR, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
    main_team = next((result for result in team_sizes if result.team_size == main_team_size), None)
    if main_team is not None:
        axes.plot(main_team.team_size, main_team.mean, "o", markersize=11, markerfacecolor="none", markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.2, zorder=5)
        axes.annotate(
            f"Main bot team ({main_team_size})", xy=(main_team.team_size, main_team.mean), xytext=(0, 14),
            textcoords="offset points", ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY,
        )
    _zero_line(axes, horizontal=True)
    axes.annotate("Bots and Pros equal", xy=(1, 0), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=TEXT_MUTED)
    _team_size_x_axis(axes, team_sizes, {})
    axes.set_ylabel("Head-to-head score vs the Pro team")
    _style_grid(axes, "y")
    _add_titles(figure, "Bot team vs Pro team, by team size", "Average head-to-head score against the Pro team, with 95% confidence band")
    _save(figure, path)


def team_size_vs_main_chart(team_sizes: list[TeamSizeResult], main_team_size: int, path: Path) -> None:
    require_head_to_head_vs_main(team_sizes, main_team_size)
    figure, axes = _new_figure(3.9, bottom_inches=0.75, has_legend=True)
    for result in team_sizes:
        if result.head_to_head_vs_main_mean is None:
            continue
        ci = result.head_to_head_vs_main_ci
        excludes_zero = ci is not None and (ci[0] > 0 or ci[1] < 0)
        if ci is not None:
            axes.plot([result.team_size, result.team_size], ci, color=BOT_COLOR, linewidth=2, alpha=0.55, solid_capstyle="round", zorder=2)
        axes.plot(
            result.team_size, result.head_to_head_vs_main_mean, "o", markersize=7, zorder=3,
            color=BOT_COLOR if excludes_zero else SURFACE, markeredgecolor=BOT_COLOR, markeredgewidth=1.5,
        )
    _zero_line(axes, horizontal=True)
    axes.annotate(f"Equal to the top {main_team_size}", xy=(1, 0), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=TEXT_MUTED)
    _team_size_x_axis(axes, team_sizes, {main_team_size: f"{main_team_size}\n(reference)"})
    axes.set_ylabel(f"Head-to-head score vs the top {main_team_size}")
    _style_grid(axes, "y")
    legend_handles = [
        _dot(BOT_COLOR, "95% CI excludes 0"),
        Line2D([], [], marker="o", linestyle="none", color=SURFACE, markeredgecolor=BOT_COLOR, markeredgewidth=1.5, markersize=6.5, label="95% CI includes 0"),
    ]
    _add_titles(
        figure,
        f"Bot team vs top-{main_team_size} bot team, by team size",
        f"Average head-to-head score against the top-{main_team_size} bot team, with 95% confidence intervals",
        legend_handles,
    )
    _save(figure, path)


def _team_size_x_axis(axes: Axes, team_sizes: list[TeamSizeResult], extra_tick_labels: dict[int, str]) -> None:
    sizes = [result.team_size for result in team_sizes]
    largest = team_sizes[-1]
    tick_labels = {size: str(size) for size in (1, 2, 3, 5, 10, 30, 50, 100) if size in sizes}
    tick_labels[largest.team_size] = f"{largest.team_size}\n(all bots)" if largest.is_all_bots else str(largest.team_size)
    tick_labels.update(extra_tick_labels)
    ticks = sorted(tick_labels)
    axes.set_xscale("log")
    axes.xaxis.set_major_locator(FixedLocator(ticks))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_xticklabels([tick_labels[tick] for tick in ticks])
    axes.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.15)
    axes.set_xlabel("Bot team size (top N bots on the bot-only questions, log scale)")


def _dot_interval_rows(axes: Axes, rows: list[LeaderboardRow], colors: list[str], labels: list[str]) -> None:
    positions = list(range(len(rows)))
    for position, row, color in zip(positions, rows, colors):
        if row.ci is not None:
            axes.plot(row.ci, [position, position], color=color, linewidth=2, alpha=0.55, solid_capstyle="round", zorder=2)
        axes.plot(row.average_score, position, "o", color=color, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
        axes.annotate(
            str(row.question_count), xy=(1, position), xycoords=("axes fraction", "data"), xytext=(10, 0),
            textcoords="offset points", ha="left", va="center", fontsize=8, color=TEXT_MUTED,
        )
    axes.annotate("Questions", xy=(1, -0.6), xycoords=("axes fraction", "data"), xytext=(10, 2), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=TEXT_MUTED)
    axes.set_yticks(positions, labels)
    axes.tick_params(axis="y", labelcolor=TEXT_PRIMARY)
    axes.set_ylim(len(rows) - 0.4, -0.6)


def individual_leaderboard_chart(individuals: IndividualResults, shown_count: int, path: Path) -> None:
    rows = individuals.rows[:shown_count]
    labels = [f"{row.rank}. {row.name}" for row in rows]
    figure, axes = _new_figure(
        0.25 * len(rows) + 1.75, left_inches=_left_margin_for_labels(labels), right_inches=COUNT_COLUMN_INCHES, has_legend=True
    )
    _dot_interval_rows(axes, rows, [PRO_COLOR if row.user_type == "pro" else BOT_COLOR for row in rows], labels)
    axes.set_xlabel("Average spot peer score (95% CI)")
    _style_grid(axes, "x")
    _add_titles(
        figure,
        f"Pros and bots as individuals: top {shown_count}",
        f"On the {individuals.question_count} shared questions, ranked by sum of spot peer scores",
        [_dot(PRO_COLOR, "Pro forecaster"), _dot(BOT_COLOR, "Bot")],
    )
    _save(figure, path)


def top_bots_chart(field: FieldSummary, shown_count: int, non_primary_bots: tuple[str, ...], path: Path) -> None:
    rows = field.rows[:shown_count]
    labels = [f"{row.rank}. {bot_label(row.name, non_primary_bots)}" for row in rows]
    figure, axes = _new_figure(
        0.25 * len(rows) + 1.75, left_inches=_left_margin_for_labels(labels), right_inches=COUNT_COLUMN_INCHES, has_legend=True
    )
    _dot_interval_rows(axes, rows, [DEEMPHASIS if is_metac_bot(row.name) else BOT_COLOR for row in rows], labels)
    axes.set_xlabel("Average spot peer score (95% CI)")
    _style_grid(axes, "x")
    _add_titles(
        figure,
        f"Bot tournament: top {len(rows)} of {len(field.rows)} bots",
        f"On {field.scored_question_count} questions, ranked by sum of spot peer scores",
        [_dot(BOT_COLOR, "Bot maker's bot"), _dot(DEEMPHASIS, "Metac Bot (run by Metaculus)")],
    )
    _save(figure, path)


def metac_chart(metac: MetacResults, shown_count: int, path: Path) -> None:
    rows = metac.baseline_rows[:shown_count]
    labels = [f"{row.rank}. {row.name}" for row in rows]
    figure, axes = _new_figure(0.25 * len(rows) + 1.45, left_inches=_left_margin_for_labels(labels), right_inches=COUNT_COLUMN_INCHES)
    _dot_interval_rows(axes, rows, [BOT_COLOR] * len(rows), labels)
    _zero_line(axes, horizontal=False)
    axes.set_xlabel("Average spot peer score against all Metac Bots (95% CI)")
    _style_grid(axes, "x")
    _add_titles(
        figure,
        f"Same-prompt Metac Bots: top {len(rows)} of {len(metac.baseline_rows)}",
        "Same prompt, different model or search provider. Ranked by sum of spot peer scores",
    )
    _save(figure, path)


@dataclass(frozen=True)
class _AccuracyRow:
    label: str
    percents: list[float]
    ci: tuple[float, float] | None = None
    color: str = TEXT_MUTED
    marker: str = "o"
    is_header: bool = False


def _accuracy_rows(field_accuracy: BotFieldAccuracy, team_accuracy: TeamAccuracy) -> list[_AccuracyRow]:
    def percent_ci(ci: tuple[float, float] | None) -> tuple[float, float] | None:
        return (100 * ci[0], 100 * ci[1]) if ci is not None else None

    return [
        _AccuracyRow(f"Bot tournament: {field_accuracy.binary_question_count} binary questions", [], is_header=True),
        _AccuracyRow(
            f"All {len(field_accuracy.eligible_bots)} regular bots",
            [100 * bot.typical_probability for bot in field_accuracy.eligible_bots],
            color=DEEMPHASIS,
        ),
        _AccuracyRow(f"Top bot ({field_accuracy.top_bot.name})", [100 * field_accuracy.top_bot.typical_probability], color=BOT_COLOR),
        _AccuracyRow("Median bot", [100 * field_accuracy.median_probability], color=BOT_COLOR),
        _AccuracyRow("Lowest-scoring bot", [100 * field_accuracy.worst_bot.typical_probability], color=BOT_COLOR),
        _AccuracyRow(
            f"Always the Yes rate ({100 * field_accuracy.yes_rate:.0f}%)", [100 * field_accuracy.base_rate_probability], marker="D"
        ),
        _AccuracyRow(f"Pro comparison: {team_accuracy.binary_question_count} shared binary questions", [], is_header=True),
        _AccuracyRow("Pro team", [100 * team_accuracy.pro_team_probability], percent_ci(team_accuracy.pro_team_ci), PRO_COLOR),
        _AccuracyRow("Bot team", [100 * team_accuracy.bot_team_probability], percent_ci(team_accuracy.bot_team_ci), BOT_COLOR),
        _AccuracyRow(
            f"Always the Yes rate ({100 * team_accuracy.yes_rate:.0f}%)", [100 * team_accuracy.base_rate_probability], marker="D"
        ),
    ]


def accuracy_chart(field_accuracy: BotFieldAccuracy, team_accuracy: TeamAccuracy, path: Path) -> None:
    rows = _accuracy_rows(field_accuracy, team_accuracy)
    figure, axes = _new_figure(0.34 * len(rows) + 1.5, left_inches=_left_margin_for_labels([row.label for row in rows]))
    jitter = np.random.default_rng(0)
    all_values = [value for row in rows for value in row.percents] + [
        bound for row in rows if row.ci is not None for bound in row.ci
    ]
    for position, row in enumerate(rows):
        if row.is_header or not row.percents:
            continue
        if len(row.percents) > 1:
            offsets = jitter.uniform(-0.22, 0.22, len(row.percents))
            axes.scatter(row.percents, position + offsets, s=12, color=row.color, edgecolors=SURFACE, linewidths=0.5, zorder=3)
            continue
        value = row.percents[0]
        if row.ci is not None:
            axes.plot(row.ci, [position, position], color=row.color, linewidth=2, alpha=0.55, solid_capstyle="round", zorder=2)
        axes.plot(value, position, row.marker, color=row.color, markersize=7 if row.marker == "o" else 5.5, markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
        label_x = row.ci[1] if row.ci is not None else value
        axes.annotate(f"{value:.1f}%", xy=(label_x, position), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=TEXT_PRIMARY)
    axes.axvline(50, color=BASELINE, linewidth=1, zorder=1)
    axes.annotate("Coin flip (50%)", xy=(50, -0.6), xytext=(3, 2), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=TEXT_MUTED)
    axes.set_yticks(range(len(rows)), [row.label for row in rows])
    for tick_label, row in zip(axes.get_yticklabels(), rows):
        tick_label.set_color(TEXT_PRIMARY)
        if row.is_header:
            tick_label.set_fontweight("bold")
    axes.set_ylim(len(rows) - 0.4, -0.6)
    axes.set_xlim(min(min(all_values), 50) - 4, max(all_values) + 8)
    axes.xaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    axes.set_xlabel("Typical probability given to the outcome that happened (geometric mean)")
    _style_grid(axes, "x")
    _add_titles(
        figure,
        "How much probability forecasters put on what happened",
        "Binary questions only. Higher is better; values are only comparable within each group of questions",
    )
    _save(figure, path)


def _field_color(row: LeaderboardRow, field: FieldSummary) -> str:
    if row in field.top_10:
        return BOT_COLOR
    if row in field.promising:
        return PROMISING_COLOR
    return DEEMPHASIS


def _field_legend() -> list[Artist]:
    return [
        _dot(BOT_COLOR, f"Top {TOP_N} by sum of scores"),
        _dot(PROMISING_COLOR, "Fewer questions, high average"),
        _dot(DEEMPHASIS, "Other bots"),
    ]


def field_rank_chart(field: FieldSummary, min_questions: int, path: Path) -> None:
    shown = [row for row in field.rows if row.question_count >= min_questions]
    figure, axes = _new_figure(4.6, has_legend=True)
    for position, row in enumerate(shown, start=1):
        color = _field_color(row, field)
        emphasized = color != DEEMPHASIS
        if row.ci is not None:
            axes.plot([position, position], row.ci, color=color, linewidth=1.6 if emphasized else 0.8, alpha=0.6 if emphasized else 0.45, zorder=3 if emphasized else 2)
        axes.plot(position, row.average_score, "o", color=color, markersize=4.5 if emphasized else 2.5, zorder=4 if emphasized else 2)
    for label_index, row in enumerate(row for row in shown if row in field.promising):
        position = shown.index(row) + 1
        above = label_index % 2 == 0
        anchor = row.ci[1 if above else 0] if row.ci is not None else row.average_score
        axes.annotate(
            f"{row.name} ({row.question_count} questions)", xy=(position, anchor), xytext=(0, 4 if above else -4),
            textcoords="offset points", ha="left", va="bottom" if above else "top", fontsize=7.5, color=TEXT_PRIMARY,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1, "alpha": 0.85}, zorder=5,
        )
    _zero_line(axes, horizontal=True)
    axes.set_xlim(0, len(shown) + 1)
    axes.set_xlabel("Leaderboard rank (by sum of spot peer scores)")
    axes.set_ylabel("Average spot peer score (95% CI)")
    _style_grid(axes, "y")
    _add_titles(
        figure,
        f"Every bot in leaderboard order ({len(shown)} bots)",
        f"Bots with at least {min_questions} scored questions; lines are 95% confidence intervals",
        _field_legend(),
    )
    _save(figure, path)


def field_chart(field: FieldSummary, min_questions: int, path: Path) -> list[LeaderboardRow]:
    """Plot every bot with at least min_questions scored questions; returns the bots left out."""
    shown = [row for row in field.rows if row.question_count >= min_questions]
    hidden = [row for row in field.rows if row.question_count < min_questions]
    figure, axes = _new_figure(4.8, has_legend=True)
    question_grid = np.linspace(min_questions, max(row.question_count for row in shown) * 1.03, 200)
    band = np.array([field.luck_band_half_width(question_count) for question_count in question_grid])
    axes.fill_between(question_grid, -band, band, color=BAND_FILL, linewidth=0, zorder=1)
    _zero_line(axes, horizontal=True)
    for row in shown:
        color = _field_color(row, field)
        emphasized = color != DEEMPHASIS
        if emphasized and row.ci is not None:
            axes.plot([row.question_count, row.question_count], row.ci, color=color, linewidth=1.6, alpha=0.6, zorder=3)
        axes.plot(
            row.question_count, row.average_score, "o", color=color, markersize=5 if emphasized else 3.2,
            markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4 if emphasized else 2,
        )
    labeled = [row for row in field.promising if row.question_count >= min_questions][:MAX_FIELD_LABELS]
    if field.uniform_reference is not None:
        labeled.append(field.uniform_reference)
    for row in labeled:
        axes.annotate(
            row.name, xy=(row.question_count, row.average_score), xytext=(-6, 0), textcoords="offset points",
            ha="right", va="center", fontsize=7.5, color=TEXT_PRIMARY,
        )
    axes.set_xlabel("Scored questions forecast")
    axes.set_ylabel("Average spot peer score")
    axes.set_xlim(0, question_grid[-1] * 1.02)
    _style_grid(axes, "y")
    _add_titles(
        figure,
        "Average score vs questions forecast",
        "Shaded band: where about 95% of bots exactly as good as the typical bot would land, by luck alone",
        _field_legend(),
    )
    _save(figure, path)
    return hidden
