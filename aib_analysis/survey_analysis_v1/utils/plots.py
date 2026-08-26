"""Charts for the survey report.

All figures are written as PNG into `config.charts_dir` and returned as a list
of `Chart` records so the report can embed them with captions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aib_analysis.survey_analysis_v1.analysis import (
    AnalysisResults,
    category_counts,
    split_top_bottom,
)
from aib_analysis.survey_analysis_v1.loading import Respondent
from aib_analysis.survey_analysis_v1.statistics import TestFamily, pearson_ci

logger = logging.getLogger(__name__)

WINNER_COLOR = "#2f6f4e"
LOSER_COLOR = "#b5651d"
NEUTRAL = "#4a5568"
ACCENT = "#7b4fa8"
GRID = "#d9dee5"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


@dataclass
class Chart:
    slug: str
    title: str
    caption: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


def _save(fig: plt.Figure, charts_dir: Path, slug: str, title: str, caption: str) -> Chart:
    charts_dir.mkdir(parents=True, exist_ok=True)
    path = charts_dir / f"{slug}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return Chart(slug=slug, title=title, caption=caption, path=path)


def build_all_charts(results: AnalysisResults) -> list[Chart]:
    _style()
    dataset = results.dataset
    config = dataset.config
    charts_dir = config.charts_dir
    charts: list[Chart] = []

    charts.append(_cohort_chart(results, charts_dir))
    charts.append(_model_tier_chart(results, charts_dir))
    charts.append(_rate_gap_chart(results, charts_dir))
    charts.append(_effect_vs_significance_chart(results.winner_family, charts_dir))
    charts.append(_correction_ladder_chart(results, charts_dir))
    charts.append(_ordinal_distribution_chart(results, charts_dir))
    charts.append(_research_breadth_chart(results, charts_dir))
    charts.append(_score_scatter_chart(results, charts_dir))
    charts.append(_within_winner_forest(results, charts_dir))
    charts.append(_top_vs_bottom_chart(results, charts_dir))
    charts.append(_ensemble_chart(results, charts_dir))
    charts.append(_categorical_chart(results, charts_dir))

    return [c for c in charts if c is not None]


# ---------------------------------------------------------------------------


def _cohort_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    dataset = results.dataset
    groups = {
        "Prize winners": len(dataset.winners),
        "Non-winners\n(competed)": len(dataset.non_winners),
        "MiniBench only": len(dataset.minibench_only),
        "No record": len(dataset.cohort("unknown")),
    }
    groups = {k: v for k, v in groups.items() if v > 0}

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = [WINNER_COLOR, LOSER_COLOR, NEUTRAL, "#9aa5b1"][: len(groups)]
    bars = ax.bar(list(groups), list(groups.values()), color=colors, width=0.6)
    ax.bar_label(bars, padding=2, fontsize=9, fontweight="bold")
    ax.set_ylabel("Respondents")
    ax.set_title(f"{dataset.config.season} survey composition (n={len(dataset.respondents)})")
    ax.set_ylim(0, max(groups.values()) * 1.2)
    ax.grid(axis="x", visible=False)

    return _save(
        fig,
        charts_dir,
        "01_cohort_composition",
        "Survey composition",
        "MiniBench-only respondents never forecast in the main tournament, so they "
        "sit outside the winner comparison.",
    )


def _model_tier_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    dataset = results.dataset
    tiers = ["frontier", "mid", "legacy", "unknown"]
    labels = ["Frontier", "Mid-tier", "Legacy", "Unreported"]

    winner_counts = [
        sum(1 for r in dataset.winners if r.features.get("model_tier") == t) for t in tiers
    ]
    loser_counts = [
        sum(1 for r in dataset.non_winners if r.features.get("model_tier") == t)
        for t in tiers
    ]

    x = np.arange(len(tiers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    b1 = ax.bar(x - width / 2, winner_counts, width, label="Winners", color=WINNER_COLOR)
    b2 = ax.bar(x + width / 2, loser_counts, width, label="Non-winners", color=LOSER_COLOR)
    ax.bar_label(b1, padding=2, fontsize=8)
    ax.bar_label(b2, padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Respondents")
    ax.set_title("Final-prediction model tier")
    ax.legend()
    ax.grid(axis="x", visible=False)

    return _save(
        fig,
        charts_dir,
        "02_model_tier",
        "Model tier by cohort",
        "Tier is assigned from the model named for the final prediction, taking the "
        "best model listed when several are given.",
    )


def _rate_gap_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    """Winner vs non-winner adoption rate for every binary feature."""
    family = results.winner_family
    binary_names = {f.name for f in results.dataset.config.binary_features}
    rows = [
        r
        for r in family.results
        if r.feature in binary_names and r.group_a is not None and r.p_raw is not None
    ]
    rows.sort(key=lambda r: (r.group_a or 0) - (r.group_b or 0))

    fig, ax = plt.subplots(figsize=(8.4, max(4.5, 0.30 * len(rows))))
    y = np.arange(len(rows))

    for i, row in enumerate(rows):
        ax.plot(
            [row.group_b, row.group_a],
            [i, i],
            color=GRID,
            linewidth=2.0,
            zorder=1,
            solid_capstyle="round",
        )
    ax.scatter(
        [r.group_b for r in rows], y, color=LOSER_COLOR, s=34, zorder=3, label="Non-winners"
    )
    ax.scatter(
        [r.group_a for r in rows], y, color=WINNER_COLOR, s=34, zorder=3, label="Winners"
    )

    labels = []
    for row in rows:
        mark = row.stars
        labels.append(f"{row.label} {mark}".strip())
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Share of cohort reporting the practice (%)")
    ax.set_xlim(-4, 104)
    ax.set_title("Practice adoption: winners vs non-winners")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.text(
        0.5,
        -0.055,
        "** survives Bonferroni   * survives FDR   . raw p<0.05",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        color=NEUTRAL,
    )

    return _save(
        fig,
        charts_dir,
        "03_practice_adoption",
        "Practice adoption gaps",
        "Each line links the non-winner rate to the winner rate for one reported "
        "practice. Markers show which findings survive multiplicity correction.",
    )


def _effect_vs_significance_chart(family: TestFamily, charts_dir: Path) -> Chart:
    """Effect size against evidence strength, with both correction thresholds."""
    rows = [r for r in family.results if r.p_raw is not None and r.effect is not None]
    binary = [r for r in rows if r.effect_label == "pp gap"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    xs = [r.effect for r in binary]
    ys = [-math.log10(max(r.p_raw, 1e-12)) for r in binary]  # type: ignore[arg-type]
    colors = [
        WINNER_COLOR if r.significant_bonferroni else (ACCENT if r.significant_fdr else NEUTRAL)
        for r in binary
    ]
    ax.scatter(xs, ys, c=colors, s=46, alpha=0.85, edgecolor="white", linewidth=0.6)

    for row, x, y in zip(binary, xs, ys):
        if row.significant_raw or abs(x) > 30:
            ax.annotate(
                row.label,
                (x, y),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7,
                color="#1f2933",
            )

    ax.axhline(-math.log10(0.05), color=NEUTRAL, linestyle=":", linewidth=1.0)
    ax.text(
        ax.get_xlim()[0], -math.log10(0.05), " raw p = 0.05", va="bottom", fontsize=7, color=NEUTRAL
    )

    if family.size:
        bonf = family.alpha / family.size
        ax.axhline(-math.log10(bonf), color=WINNER_COLOR, linestyle="--", linewidth=1.2)
        ax.text(
            ax.get_xlim()[0],
            -math.log10(bonf),
            f" Bonferroni p = {bonf:.4f}  (alpha {family.alpha} / {family.size} tests)",
            va="bottom",
            fontsize=7,
            color=WINNER_COLOR,
        )

    ax.axvline(0, color=GRID, linewidth=1.0)
    ax.set_xlabel("Winner rate minus non-winner rate (percentage points)")
    ax.set_ylabel("Evidence strength  (-log10 raw p)")
    ax.set_title(f"Effect size vs evidence, family of {family.size} tests")

    return _save(
        fig,
        charts_dir,
        "04_effect_vs_evidence",
        "Effect size against evidence",
        "Points above the dashed line survive Bonferroni correction for the whole "
        "family. Points above the dotted line only clear an uncorrected p of 0.05.",
    )


def _correction_ladder_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    """How many findings survive each level of correction, per family."""
    families = results.families
    levels = ["raw", "fdr", "bonferroni"]
    level_labels = ["Raw p < 0.05", "FDR q < 0.05", "Bonferroni"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(families))
    width = 0.26
    colors = [NEUTRAL, ACCENT, WINNER_COLOR]

    for i, (level, label, color) in enumerate(zip(levels, level_labels, colors)):
        counts = [len(f.survivors(level)) for f in families]
        bars = ax.bar(x + (i - 1) * width, counts, width, label=label, color=color)
        ax.bar_label(bars, padding=2, fontsize=8)

    ax.set_xticks(
        x, [f"{f.name}\n({f.size} tests)" for f in families], fontsize=8
    )
    ax.set_ylabel("Findings surviving")
    ax.set_title("Survivors at each correction level")
    ax.legend(ncol=3, loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    top = max(
        [len(f.survivors("raw")) for f in families] + [1]
    )
    ax.set_ylim(0, top * 1.45)

    return _save(
        fig,
        charts_dir,
        "05_correction_ladder",
        "Survivors by correction level",
        "The gap between the grey and green bars is the cost of testing many "
        "features at once.",
    )


def _ordinal_distribution_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    dataset = results.dataset
    specs = list(dataset.config.ordinal_features)
    fig, axes = plt.subplots(1, len(specs), figsize=(3.3 * len(specs), 3.6))
    if len(specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, specs):
        winner_vals = [
            float(r.features[spec.name])
            for r in dataset.winners
            if r.features.get(spec.name) is not None
        ]
        loser_vals = [
            float(r.features[spec.name])
            for r in dataset.non_winners
            if r.features.get(spec.name) is not None
        ]
        data = [v for v in (winner_vals, loser_vals) if v]
        if not data:
            ax.axis("off")
            continue

        positions = list(range(1, len(data) + 1))
        parts = ax.violinplot(data, positions=positions, showextrema=False, widths=0.75)
        for body, color in zip(parts["bodies"], [WINNER_COLOR, LOSER_COLOR]):
            body.set_facecolor(color)
            body.set_alpha(0.25)

        rng = np.random.default_rng(7)
        for pos, vals, color in zip(positions, data, [WINNER_COLOR, LOSER_COLOR]):
            jitter = rng.normal(0, 0.10, len(vals))
            ax.scatter(
                np.array([pos] * len(vals)) + jitter,
                vals,
                s=20,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            ax.hlines(np.median(vals), pos - 0.28, pos + 0.28, color=color, linewidth=2.2, zorder=4)

        if spec.log_scale and min(min(d) for d in data) >= 0:
            ax.set_yscale("symlog", linthresh=0.05)
        ax.set_xticks(positions, ["Winners", "Non-win"][: len(data)], fontsize=8)
        ax.set_title(spec.label, fontsize=9.5)
        ax.set_ylabel(spec.unit, fontsize=8)
        ax.grid(axis="x", visible=False)

    fig.suptitle("Effort and spend distributions", fontsize=11, fontweight="bold", y=1.01)
    return _save(
        fig,
        charts_dir,
        "06_effort_distributions",
        "Effort and spend",
        "Bucketed survey answers mapped to midpoints. Horizontal bars are medians; "
        "log-scaled axes where the range spans orders of magnitude.",
    )


def _research_breadth_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    dataset = results.dataset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    max_sources = 0
    for group in (dataset.winners, dataset.non_winners):
        for r in group:
            v = r.features.get("n_research_sources")
            if v is not None:
                max_sources = max(max_sources, int(v))  # type: ignore[arg-type]
    bins = np.arange(0, max_sources + 2) - 0.5

    for group, color, label in (
        (dataset.winners, WINNER_COLOR, "Winners"),
        (dataset.non_winners, LOSER_COLOR, "Non-winners"),
    ):
        vals = [
            int(r.features["n_research_sources"])  # type: ignore[arg-type]
            for r in group
            if r.features.get("n_research_sources") is not None
        ]
        if vals:
            ax1.hist(vals, bins=bins, alpha=0.6, color=color, label=f"{label} (n={len(vals)})",
                     density=True)
    ax1.set_xlabel("Distinct research sources")
    ax1.set_ylabel("Share of cohort")
    ax1.set_title("Research breadth")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", visible=False)

    scored = [
        r
        for r in dataset.compared
        if r.total_score is not None and r.features.get("n_research_sources") is not None
    ]
    xs = [int(r.features["n_research_sources"]) for r in scored]  # type: ignore[arg-type]
    ys = [r.total_score for r in scored]
    colors = [WINNER_COLOR if r.is_winner else LOSER_COLOR for r in scored]
    ax2.scatter(xs, ys, c=colors, s=40, alpha=0.85, edgecolor="white", linewidth=0.5)
    if len(set(xs)) > 1:
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(min(xs), max(xs), 50)
        ax2.plot(grid, slope * grid + intercept, color=NEUTRAL, linestyle="--", linewidth=1.2)
    ax2.axhline(0, color=GRID, linewidth=1.0)
    ax2.set_xlabel("Distinct research sources")
    ax2.set_ylabel("Total spot peer score")
    ax2.set_title("Sources vs score")
    ax2.grid(axis="x", visible=False)

    return _save(
        fig,
        charts_dir,
        "07_research_breadth",
        "Research breadth",
        "Counts how many distinct tools from the survey catalog each respondent "
        "named. Free-text tools outside the catalog are not counted.",
    )


def _score_scatter_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    dataset = results.dataset
    scored = sorted(
        [r for r in dataset.compared if r.total_score is not None and r.participant_rank],
        key=lambda r: r.participant_rank,  # type: ignore[arg-type,return-value]
    )

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    xs = [r.participant_rank for r in scored]
    ys = [r.total_score for r in scored]
    colors = [WINNER_COLOR if r.is_winner else LOSER_COLOR for r in scored]
    ax.bar(xs, ys, color=colors, width=0.8)
    ax.axhline(0, color="#1f2933", linewidth=0.9)

    winners = [r for r in scored if r.is_winner]
    if winners:
        cutoff = max(r.participant_rank for r in winners)  # type: ignore[type-var]
        ax.axvline(cutoff + 0.5, color=ACCENT, linestyle="--", linewidth=1.3)
        ax.text(
            cutoff + 1.2,
            max(ys) * 0.78,
            f"prize cutoff\n(participant rank {cutoff})",
            fontsize=8,
            color=ACCENT,
        )

    ax.set_xlabel("Participant rank (Metaculus reference and pre-seen bots removed)")
    ax.set_ylabel("Total spot peer score")
    ax.set_title("Where survey respondents landed on the leaderboard")
    ax.grid(axis="x", visible=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=WINNER_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=LOSER_COLOR),
    ]
    ax.legend(handles, ["Prize winner", "Non-winner"], loc="upper right")

    return _save(
        fig,
        charts_dir,
        "08_leaderboard_position",
        "Leaderboard position of respondents",
        "Only survey respondents are shown, so gaps are bots whose makers did not "
        "respond.",
    )


def _within_winner_forest(results: AnalysisResults, charts_dir: Path) -> Chart:
    family = results.within_winner_family
    rows = [r for r in family.results if r.p_raw is not None and r.effect is not None]
    rows.sort(key=lambda r: r.effect)  # type: ignore[arg-type,return-value]
    rows = [r for r in rows if abs(r.effect) > 0.01]  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.29 * len(rows))))
    y = np.arange(len(rows))

    for i, row in enumerate(rows):
        lo, hi = pearson_ci(float(row.effect), row.n)  # type: ignore[arg-type]
        if not math.isnan(lo):
            ax.plot([lo, hi], [i, i], color=GRID, linewidth=1.8, zorder=1)
        color = (
            WINNER_COLOR
            if row.significant_bonferroni
            else (ACCENT if row.significant_fdr else (NEUTRAL if row.significant_raw else "#a0aec0"))
        )
        ax.scatter([row.effect], [i], color=color, s=34, zorder=3)

    ax.axvline(0, color="#1f2933", linewidth=0.9)
    ax.set_yticks(y, [f"{r.label} {r.stars}".strip() for r in rows], fontsize=8)
    ax.set_xlabel("Pearson r with total score, prize winners only")
    ax.set_title(
        f"Within-winner gradients (n={len(results.dataset.winners)}, {family.size} tests)"
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-1, 1)

    return _save(
        fig,
        charts_dir,
        "09_within_winner_forest",
        "Within-winner correlations",
        "Bars are 95% Fisher-z intervals. Nearly all cross zero, which is what a "
        "sample this size can support.",
    )


def _top_vs_bottom_chart(results: AnalysisResults, charts_dir: Path) -> Chart:
    dataset = results.dataset
    top, bottom = split_top_bottom(dataset.winners, dataset.config.top_group_size)
    binary = list(dataset.config.binary_features)

    gaps = []
    for spec in binary:
        top_flags = [r.features.get(spec.name) for r in top]
        bot_flags = [r.features.get(spec.name) for r in bottom]
        tv = [bool(f) for f in top_flags if f is not None]
        bv = [bool(f) for f in bot_flags if f is not None]
        if len(tv) < 3 or len(bv) < 3:
            continue
        top_rate = 100 * sum(tv) / len(tv)
        bot_rate = 100 * sum(bv) / len(bv)
        gaps.append((spec.label, top_rate, bot_rate, top_rate - bot_rate))

    gaps.sort(key=lambda g: g[3])
    gaps = gaps[:6] + gaps[-6:] if len(gaps) > 12 else gaps

    fig, ax = plt.subplots(figsize=(8.0, max(3.6, 0.34 * len(gaps))))
    y = np.arange(len(gaps))
    colors = [WINNER_COLOR if g[3] >= 0 else LOSER_COLOR for g in gaps]
    bars = ax.barh(y, [g[3] for g in gaps], color=colors, height=0.62)
    ax.bar_label(
        bars,
        labels=[f"{g[1]:.0f}% vs {g[2]:.0f}%" for g in gaps],
        padding=4,
        fontsize=7.5,
    )
    ax.set_yticks(y, [g[0] for g in gaps], fontsize=8)
    ax.axvline(0, color="#1f2933", linewidth=0.9)
    ax.set_xlabel(
        f"Adoption gap, top {len(top)} winners minus remaining {len(bottom)} (pp)"
    )
    ax.set_title("What separates the very top from the rest of the winners")
    ax.grid(axis="y", visible=False)
    pad = max(abs(g[3]) for g in gaps) * 0.45 if gaps else 1
    ax.set_xlim(min(g[3] for g in gaps) - pad, max(g[3] for g in gaps) + pad)

    return _save(
        fig,
        charts_dir,
        "10_top_vs_bottom_winners",
        "Top winners vs other winners",
        "Descriptive only. These splits are not corrected and the subgroups are "
        "small, so treat them as directions worth checking next season.",
    )


def _ensemble_chart(results: AnalysisResults, charts_dir: Path) -> Chart | None:
    dataset = results.dataset
    counts = category_counts(dataset.compared, "ensemble_combination")
    if not counts:
        return None

    def shorten(label: str) -> str:
        label = label.split(",")[0]
        return label[:44] + ("..." if len(label) > 44 else "")

    merged: dict[str, list[Respondent]] = {}
    for r in dataset.compared:
        raw = r.features.get("ensemble_combination")
        if not raw:
            continue
        merged.setdefault(shorten(str(raw)), []).append(r)

    rows = sorted(merged.items(), key=lambda kv: -len(kv[1]))[:8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    y = np.arange(len(rows))
    win = [sum(1 for r in v if r.is_winner) for _, v in rows]
    lose = [sum(1 for r in v if not r.is_winner) for _, v in rows]
    ax1.barh(y, win, color=WINNER_COLOR, height=0.6, label="Winners")
    ax1.barh(y, lose, left=win, color=LOSER_COLOR, height=0.6, label="Non-winners")
    ax1.set_yticks(y, [k for k, _ in rows], fontsize=7.5)
    ax1.set_xlabel("Respondents")
    ax1.set_title("How ensembles are combined")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", visible=False)
    ax1.invert_yaxis()

    scored_rows = [(k, [r for r in v if r.total_score is not None]) for k, v in rows]
    scored_rows = [(k, v) for k, v in scored_rows if len(v) >= 2]
    if scored_rows:
        positions = np.arange(len(scored_rows))
        for pos, (_, group) in zip(positions, scored_rows):
            vals = [r.total_score for r in group]
            jitter = np.random.default_rng(3).normal(0, 0.06, len(vals))
            ax2.scatter(
                np.array([pos] * len(vals)) + jitter,
                vals,
                s=26,
                color=[WINNER_COLOR if r.is_winner else LOSER_COLOR for r in group],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
            )
            ax2.hlines(np.median(vals), pos - 0.26, pos + 0.26, color=NEUTRAL, linewidth=2.0)
        ax2.set_xticks(positions, [k for k, _ in scored_rows], rotation=35, ha="right", fontsize=7)
        ax2.axhline(0, color=GRID, linewidth=1.0)
        ax2.set_ylabel("Total spot peer score")
        ax2.set_title("Score by combination method")
        ax2.grid(axis="x", visible=False)
    else:
        ax2.axis("off")

    return _save(
        fig,
        charts_dir,
        "11_ensemble_methods",
        "Ensemble combination",
        "Respondents selecting several methods are grouped by the first one they "
        "listed.",
    )


def _categorical_chart(results: AnalysisResults, charts_dir: Path) -> Chart | None:
    dataset = results.dataset
    specs = [s for s in dataset.config.categorical_features if s.order]
    if not specs:
        return None

    fig, axes = plt.subplots(1, len(specs), figsize=(4.4 * len(specs), 3.8))
    if len(specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, specs):
        categories = list(spec.order)
        win = [
            sum(1 for r in dataset.winners if r.features.get(spec.name) == c)
            for c in categories
        ]
        lose = [
            sum(1 for r in dataset.non_winners if r.features.get(spec.name) == c)
            for c in categories
        ]
        y = np.arange(len(categories))
        ax.barh(y, win, color=WINNER_COLOR, height=0.6, label="Winners")
        ax.barh(y, lose, left=win, color=LOSER_COLOR, height=0.6, label="Non-winners")
        short = [c if len(c) <= 34 else c[:32] + "..." for c in categories]
        ax.set_yticks(y, short, fontsize=7.5)
        ax.set_xlabel("Respondents")
        ax.set_title(spec.label, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8)

    return _save(
        fig,
        charts_dir,
        "12_categorical_breakdown",
        "Single-select answers",
        "Distribution of the single-choice survey questions across cohorts.",
    )
