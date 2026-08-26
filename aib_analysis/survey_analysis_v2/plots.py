"""Chart generation. Distribution-by-group bars and score-relationship charts.

All charts are saved as PNGs under the output charts/ folder and referenced by
relative path in the report.
"""

from __future__ import annotations

import logging
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl-survey-v2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aib_analysis.survey_analysis_v2 import config
from aib_analysis.survey_analysis_v2.features import RespondentFeatures

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
    }
)


# Aggregated bar for respondents whose answer had leftover write-in ("Other") text.
OTHER_LABEL = "Other (write-in)"


def _answered(features: list[RespondentFeatures], slug: str, group: str) -> list[RespondentFeatures]:
    return [
        feature
        for feature in features
        if (group == "everyone" or group in feature.respondent.groups)
        and feature.cells.get(slug)
        and feature.cells[slug].raw.strip()
    ]


def _has_any_other(features: list[RespondentFeatures], slug: str) -> bool:
    return any(
        feature.cells[slug].other
        for feature in features
        if feature.cells.get(slug) and feature.cells[slug].raw.strip()
    )


def _wrap(label: str, width: int = 42) -> str:
    if len(label) <= width:
        return label
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


def _group_legend_label(features: list[RespondentFeatures], slug: str, group: str) -> str:
    n = len(_answered(features, slug, group))
    return f"{config.GROUP_LABELS[group]} (n={n})"


def plot_multiselect_distribution(
    features: list[RespondentFeatures],
    slug: str,
    title: str,
    options: list[str],
    out_path: str,
    top_n: int = 14,
) -> None:
    overall_counts = {option: 0 for option in options}
    for feature in features:
        for option in feature.cells[slug].matched:
            if option in overall_counts:
                overall_counts[option] += 1
    ranked = [
        option
        for option, _count in sorted(overall_counts.items(), key=lambda kv: kv[1], reverse=True)
        if overall_counts[option] > 0
    ][:top_n]
    if not ranked:
        logger.warning("No options to plot for %s", slug)
        return
    ranked = list(reversed(ranked))  # highest at top
    if _has_any_other(features, slug):
        ranked = [OTHER_LABEL] + ranked  # "Other" pinned to the bottom

    _plot_grouped_horizontal(features, slug, title, ranked, out_path)


def plot_categorical_distribution(
    features: list[RespondentFeatures],
    slug: str,
    title: str,
    categories: list[str],
    out_path: str,
) -> None:
    present = [
        category
        for category in categories
        if any(category in feature.cells[slug].matched for feature in features)
    ]
    if not present:
        logger.warning("No categories to plot for %s", slug)
        return
    ordered = list(reversed(present))
    if _has_any_other(features, slug):
        ordered = [OTHER_LABEL] + ordered  # "Other" pinned to the bottom
    _plot_grouped_horizontal(features, slug, title, ordered, out_path)


def _plot_grouped_horizontal(
    features: list[RespondentFeatures],
    slug: str,
    title: str,
    options_bottom_to_top: list[str],
    out_path: str,
) -> None:
    groups = config.CHART_GROUP_ORDER
    group_answered = {group: _answered(features, slug, group) for group in groups}

    n_options = len(options_bottom_to_top)
    bar_height = 0.82 / len(groups)
    positions = list(range(n_options))

    fig_height = max(2.6, 0.62 * n_options + 1.6)
    fig, axis = plt.subplots(figsize=(9.5, fig_height))

    for group_index, group in enumerate(groups):
        answered = group_answered[group]
        denom = len(answered) or 1
        percentages = []
        for option in options_bottom_to_top:
            if option == OTHER_LABEL:
                selected = sum(1 for feature in answered if feature.cells[slug].other)
            else:
                selected = sum(1 for feature in answered if option in feature.cells[slug].matched)
            percentages.append(100.0 * selected / denom)
        offsets = [pos + (group_index - (len(groups) - 1) / 2) * bar_height for pos in positions]
        axis.barh(
            offsets,
            percentages,
            height=bar_height,
            color=config.GROUP_COLORS[group],
            label=_group_legend_label(features, slug, group),
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([_wrap(option) for option in options_bottom_to_top], fontsize=8.5)
    axis.set_xlabel("Share of group that selected this (%)")
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axis.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved chart %s", os.path.basename(out_path))


# Minimum respondents per bar. Bars for smaller groups are suppressed so no bar
# can reveal an individual bot's public peer score.
MIN_CELL_SIZE = 5


def plot_group_means(
    labeled_scores: list[tuple[str, list[float]]],
    title: str,
    out_path: str,
    min_cell: int = MIN_CELL_SIZE,
) -> bool:
    """Horizontal bars of mean peer score per answer group.

    Groups with fewer than `min_cell` scored respondents are dropped (privacy).
    Returns True only if at least two groups survive and a chart was written.
    """
    shown = [(label, scores) for label, scores in labeled_scores if len(scores) >= min_cell]
    if len(shown) < 2:
        logger.info("Skipping score chart for %s (too few groups >= %d)", os.path.basename(out_path), min_cell)
        return False

    shown = list(reversed(shown))  # first group at the top
    labels = [_wrap(f"{label} (n={len(scores)})", 34) for label, scores in shown]
    means = [sum(scores) / len(scores) for _label, scores in shown]
    positions = list(range(len(shown)))

    fig_height = max(2.4, 0.62 * len(shown) + 1.2)
    fig, axis = plt.subplots(figsize=(6.6, fig_height))
    axis.barh(positions, means, color="#2f80ed", height=0.62)
    axis.axvline(0, color="#667085", linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlabel("Mean of bots' average spot peer score")
    axis.set_title(_wrap(title, 44), fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved chart %s", os.path.basename(out_path))
    return True
