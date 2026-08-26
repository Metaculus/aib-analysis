"""Generate the giant Markdown review report.

This report exists so a reviewer can confirm the analysis "just works" by
looking at the output rather than reading the code. Every derived number is
shown next to the raw survey text it came from.

Layout:
  0. Cohort reconciliation  - every respondent, their prize, cohort, and score
  1. Distribution analysis  - one chart per raw survey question, with a parse
                              ledger showing how non-standard answers were read
  2. Feature analysis       - one section per derived feature, ordered by best
                              p-value, with distributions across groups, the
                              tests already run, and a random sample of the
                              underlying rows
  3. Spot-check appendix    - file index and how to regenerate

Nothing here computes a statistic. It only lays out results computed upstream.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aib_analysis.survey_analysis_v1.analysis import AnalysisResults, split_top_bottom
from aib_analysis.survey_analysis_v1.config import SeasonConfig
from aib_analysis.survey_analysis_v1.loading import Dataset, Respondent
from aib_analysis.survey_analysis_v1.statistics import TestFamily, TestResult
from aib_analysis.survey_analysis_v1.utils.plots import (
    ACCENT,
    GRID,
    LOSER_COLOR,
    NEUTRAL,
    WINNER_COLOR,
    _style,
)

logger = logging.getLogger(__name__)

TOP_N = 10  # "top group" = top 10 respondents by total score
SAMPLE_SEED = 20260824
GROUP_COLORS = [ACCENT, WINNER_COLOR, LOSER_COLOR, NEUTRAL]

# How each raw survey question is drawn in section 1. Presentation only.
QUESTION_RENDER: dict[str, str] = {
    "research": "multiselect",
    "forecasting_strategies": "multiselect",
    "development": "multiselect",
    "aggregation_method": "multiselect",
    "ensemble_combination": "multiselect",
    "verification_env": "multiselect",
    "hours": "numeric",
    "llm_calls": "numeric",
    "cost_per_q": "numeric",
    "iterations": "numeric",
    "team_size": "numeric",
    "respondent_type": "categorical",
    "research_vs_reasoning": "categorical",
    "writeup_rating": "categorical",
    "changed_since_last": "categorical",
    "minibench_opinion": "multiselect",
    "share_code_publicly": "categorical",
    "share_individual": "categorical",
    "final_model": "model_tier",
    "support_model": "freetext",
    "lessons": "freetext",
    "abandoned": "freetext",
    "other_comments": "freetext",
    "code_link": "skip",
    "timestamp": "skip",
    "bot_name": "skip",
    "self_reported_rank": "categorical",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _top_by_score(respondents: list[Respondent], n: int) -> list[Respondent]:
    scored = [r for r in respondents if r.total_score is not None]
    return sorted(scored, key=lambda r: r.total_score, reverse=True)[:n]  # type: ignore[arg-type,return-value]


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _escape_cell(text: str | None, limit: int = 90) -> str:
    """Make a raw survey answer safe for a Markdown table cell."""
    if not text:
        return "_(blank)_"
    flat = " ".join(str(text).split()).replace("|", "\\|")
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat


def _derived_str(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓ True" if value else "✗ False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ---------------------------------------------------------------------------
# Section 1 charts
# ---------------------------------------------------------------------------


def _bar_by_cohort(
    ax: plt.Axes,
    categories: list[str],
    winner_counts: list[int],
    loser_counts: list[int],
    xlabel: str,
) -> None:
    y = np.arange(len(categories))
    ax.barh(y, winner_counts, color=WINNER_COLOR, height=0.62, label="Winners")
    ax.barh(y, loser_counts, left=winner_counts, color=LOSER_COLOR, height=0.62, label="Non-winners")
    short = [c if len(c) <= 40 else c[:38] + "…" for c in categories]
    ax.set_yticks(y, short, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")


def _catalog_for_source(config: SeasonConfig, source: str) -> list[tuple[str, tuple[str, ...]]]:
    """Parsed options for a multi-select question: (label, needles) per feature."""
    return [
        (spec.label, spec.needles)
        for spec in config.binary_features
        if spec.source == source
    ]


def _draw_multiselect(
    dataset: Dataset, field: str, out: Path, slug: str
) -> tuple[Path, str, str]:
    config = dataset.config
    catalog = _catalog_for_source(config, field)
    winners, losers = dataset.winners, dataset.non_winners

    def count(group: list[Respondent], needles: tuple[str, ...]) -> int:
        total = 0
        for r in group:
            cell = (r.answer(field) or "").lower()
            if cell and any(n.lower() in cell for n in needles):
                total += 1
        return total

    rows = [
        (label, count(winners, needles), count(losers, needles))
        for label, needles in catalog
    ]
    rows.sort(key=lambda t: -(t[1] + t[2]))

    fig, ax = plt.subplots(figsize=(9.0, max(2.6, 0.34 * len(rows) + 0.6)))
    _bar_by_cohort(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        "Respondents selecting the option",
    )
    ax.set_title(f"{field}: option counts")
    path = out / f"{slug}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Parse ledger: comma-split fragments not covered by any catalog needle.
    unmatched: dict[str, int] = {}
    for r in dataset.respondents:
        cell = r.answer(field)
        if not cell:
            continue
        for frag in re.split(r",\s*", cell):
            frag = frag.strip()
            if not frag:
                continue
            low = frag.lower()
            covered = any(
                n.lower() in low or low in n.lower()
                for _, needles in catalog
                for n in needles
            )
            if not covered:
                unmatched[frag] = unmatched.get(frag, 0) + 1

    ledger = ""
    if unmatched:
        top = sorted(unmatched.items(), key=lambda kv: -kv[1])[:15]
        ledger = (
            "\n**Raw fragments not matched to any parsed option** "
            "(comma-split, may include split artifacts from options that "
            "themselves contain commas):\n\n"
            "| fragment | count |\n|---|---|\n"
            + "".join(f"| {_escape_cell(k, 70)} | {v} |\n" for k, v in top)
        )

    if rows:
        top_opt = rows[0]
        gaps = sorted(
            rows,
            key=lambda t: abs(
                (t[1] / max(len(winners), 1)) - (t[2] / max(len(losers), 1))
            ),
            reverse=True,
        )
        g = gaps[0]
        insight = (
            f"Most-selected option is **{top_opt[0]}** "
            f"({top_opt[1] + top_opt[2]} of {len(winners) + len(losers)}). "
            f"Largest winner vs non-winner divergence is **{g[0]}** "
            f"({g[1]}/{len(winners)} winners vs {g[2]}/{len(losers)} non-winners)."
        )
    else:
        insight = "No parsed options for this question."
    return path, insight, ledger


def _draw_numeric(
    dataset: Dataset, field: str, feature_name: str | None, out: Path, slug: str
) -> tuple[Path, str, str]:
    """Bar of raw-answer buckets ordered by mapped midpoint, plus a parse ledger."""
    config = dataset.config
    spec = next((s for s in config.ordinal_features if s.source == field), None)

    winners, losers = dataset.winners, dataset.non_winners

    # Distinct raw answers with their mapped midpoint.
    mapping: dict[str, float | None] = {}
    for r in dataset.respondents:
        raw = (r.answer(field) or "").strip()
        if not raw:
            continue
        midpoint = spec.evaluate(raw) if spec else None
        mapping.setdefault(raw, midpoint)

    def sort_key(item: tuple[str, float | None]) -> tuple[int, float]:
        _, mid = item
        return (1, 0.0) if mid is None else (0, mid)

    ordered = sorted(mapping.items(), key=sort_key)

    def count(group: list[Respondent], raw: str) -> int:
        return sum(1 for r in group if (r.answer(field) or "").strip() == raw)

    fig, ax = plt.subplots(figsize=(9.0, max(2.6, 0.34 * len(ordered) + 0.6)))
    _bar_by_cohort(
        ax,
        [k for k, _ in ordered],
        [count(winners, k) for k, _ in ordered],
        [count(losers, k) for k, _ in ordered],
        "Respondents",
    )
    ax.set_title(f"{field}: answer distribution (ordered by parsed value)")
    path = out / f"{slug}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    ledger = (
        "\n**Parse ledger** (raw answer → numeric value used). "
        "Rows with — were left out of the numeric tests:\n\n"
        "| raw answer | parsed value |\n|---|---|\n"
        + "".join(
            f"| {_escape_cell(k, 70)} | {_fmt(v, 3) if v is not None else '—'} |\n"
            for k, v in ordered
        )
    )

    unparsed = [k for k, v in ordered if v is None]
    insight_bits = []
    if spec and feature_name:
        wv = [
            r.features[feature_name]
            for r in winners
            if r.features.get(feature_name) is not None
        ]
        lv = [
            r.features[feature_name]
            for r in losers
            if r.features.get(feature_name) is not None
        ]
        if wv and lv:
            insight_bits.append(
                f"Median parsed value is **{np.median([float(x) for x in wv]):g}** for winners "
                f"vs **{np.median([float(x) for x in lv]):g}** for non-winners."
            )
    if unparsed:
        insight_bits.append(
            f"{len(unparsed)} distinct free-text answer(s) could not be mapped and were dropped: "
            + ", ".join(f'"{_escape_cell(u, 40)}"' for u in unparsed[:3])
        )
    return path, " ".join(insight_bits) or "See distribution above.", ledger


def _draw_categorical(
    dataset: Dataset, field: str, out: Path, slug: str
) -> tuple[Path, str, str]:
    winners, losers = dataset.winners, dataset.non_winners
    values: dict[str, list[int]] = {}
    for group_idx, group in enumerate((winners, losers)):
        for r in group:
            raw = (r.answer(field) or "").strip()
            if not raw:
                continue
            values.setdefault(raw, [0, 0])[group_idx] += 1

    ordered = sorted(values.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:14]

    fig, ax = plt.subplots(figsize=(9.0, max(2.6, 0.34 * len(ordered) + 0.6)))
    _bar_by_cohort(
        ax,
        [k for k, _ in ordered],
        [v[0] for _, v in ordered],
        [v[1] for _, v in ordered],
        "Respondents",
    )
    ax.set_title(f"{field}: answer distribution")
    path = out / f"{slug}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if ordered:
        top = ordered[0]
        insight = (
            f"Most common answer is **{_escape_cell(top[0], 60)}** "
            f"({top[1][0] + top[1][1]} of {len(winners) + len(losers)})."
        )
    else:
        insight = "No answers to this question."
    return path, insight, ""


def _draw_model_tier(
    dataset: Dataset, out: Path, slug: str
) -> tuple[Path, str, str]:
    config = dataset.config
    winners, losers = dataset.winners, dataset.non_winners
    tiers = ["frontier", "mid", "legacy", "unknown"]

    def count(group: list[Respondent], tier: str) -> int:
        return sum(1 for r in group if r.features.get("model_tier") == tier)

    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    _bar_by_cohort(
        ax,
        tiers,
        [count(winners, t) for t in tiers],
        [count(losers, t) for t in tiers],
        "Respondents",
    )
    ax.set_title("final_model: parsed tier")
    path = out / f"{slug}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    ledger_rows: dict[str, str] = {}
    for r in dataset.respondents:
        raw = (r.answer("final_model") or "").strip()
        if raw:
            ledger_rows.setdefault(raw, str(r.features.get("model_tier")))
    ledger = (
        "\n**Parse ledger** (raw model string → assigned tier):\n\n"
        "| raw answer | tier |\n|---|---|\n"
        + "".join(
            f"| {_escape_cell(k, 70)} | {v} |\n"
            for k, v in sorted(ledger_rows.items(), key=lambda kv: kv[1])
        )
    )
    frontier = count(winners, "frontier") + count(losers, "frontier")
    total = len(winners) + len(losers)
    insight = (
        f"**{frontier} of {total}** competing respondents named a frontier model "
        f"for their final prediction."
    )
    return path, insight, ledger


# ---------------------------------------------------------------------------
# Section 2 charts
# ---------------------------------------------------------------------------


@dataclass
class FeatureDescriptor:
    name: str
    label: str
    kind: str  # "binary" | "numeric"
    source: str
    definition: str
    value_key: str


def _feature_descriptors(config: SeasonConfig) -> list[FeatureDescriptor]:
    out: list[FeatureDescriptor] = []
    for spec in config.binary_features:
        rule = "True if the answer contains: " + " OR ".join(f'"{n}"' for n in spec.needles)
        if spec.negate:
            rule = "True if the answer does NOT contain: " + " OR ".join(
                f'"{n}"' for n in spec.needles
            )
        out.append(
            FeatureDescriptor(spec.name, spec.label, "binary", spec.source, rule, spec.name)
        )
    for spec in config.ordinal_features:
        pairs = ", ".join(f'"{k}"→{v:g}' for k, v in list(spec.mapping.items())[:6])
        out.append(
            FeatureDescriptor(
                spec.name,
                spec.label,
                "numeric",
                spec.source,
                f"Bucket midpoints ({pairs}, …)",
                spec.name,
            )
        )
    for spec in config.count_features:
        out.append(
            FeatureDescriptor(
                spec.name,
                spec.label,
                "numeric",
                spec.source,
                f"Count of catalog options present ({len(spec.catalog)} in catalog)",
                spec.name,
            )
        )
    for spec in config.categorical_features:
        if not spec.ordinal_scores:
            continue
        pairs = ", ".join(f'"{k[:20]}"→{v:g}' for k, v in spec.ordinal_scores.items())
        out.append(
            FeatureDescriptor(
                spec.name,
                spec.label,
                "numeric",
                spec.source,
                f"Ordinal score ({pairs})",
                f"{spec.name}_score",
            )
        )
    return out


def _feature_chart(
    dataset: Dataset, desc: FeatureDescriptor, out: Path
) -> Path:
    groups = [
        ("Top 10", _top_by_score(dataset.respondents, TOP_N)),
        ("Winners", dataset.winners),
        ("Non-winners", dataset.non_winners),
        ("All", dataset.respondents),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8))

    if desc.kind == "binary":
        rates, labels = [], []
        for gi, (gname, group) in enumerate(groups):
            flags = [r.features.get(desc.value_key) for r in group]
            valid = [bool(f) for f in flags if f is not None]
            pct = 100 * sum(valid) / len(valid) if valid else 0
            rates.append(pct)
            labels.append(f"{gname}\n({sum(valid)}/{len(valid)})")
        bars = ax1.bar(range(len(groups)), rates, color=GROUP_COLORS, width=0.62)
        ax1.bar_label(bars, labels=[f"{r:.0f}%" for r in rates], padding=2, fontsize=8)
        ax1.set_xticks(range(len(groups)), labels, fontsize=8)
        ax1.set_ylabel("Adoption (%)")
        ax1.set_ylim(0, 105)
        ax1.set_title("Adoption by group")
        ax1.grid(axis="x", visible=False)

        scored = [r for r in dataset.compared if r.total_score is not None]
        true_scores = [r.total_score for r in scored if r.features.get(desc.value_key)]
        false_scores = [
            r.total_score for r in scored if r.features.get(desc.value_key) is False
        ]
        data = [d for d in (true_scores, false_scores) if d]
        positions = list(range(len(data)))
        rng = np.random.default_rng(SAMPLE_SEED)
        for pos, vals, color in zip(positions, data, [WINNER_COLOR, NEUTRAL]):
            jitter = rng.normal(0, 0.06, len(vals))
            ax2.scatter(
                np.array([pos] * len(vals)) + jitter, vals, s=26, color=color,
                alpha=0.8, edgecolor="white", linewidth=0.4,
            )
            ax2.hlines(np.median(vals), pos - 0.25, pos + 0.25, color=color, linewidth=2.2)
        ax2.set_xticks(positions, ["True", "False"][: len(data)], fontsize=9)
        ax2.axhline(0, color=GRID, linewidth=1.0)
        ax2.set_ylabel("Total spot-peer score")
        ax2.set_title("Score by feature value")
        ax2.grid(axis="x", visible=False)
    else:
        data, labels = [], []
        for gname, group in groups:
            vals = [
                float(r.features[desc.value_key])
                for r in group
                if r.features.get(desc.value_key) is not None
            ]
            data.append(vals)
            labels.append(f"{gname}\n(n={len(vals)})")
        positions = list(range(len(data)))
        rng = np.random.default_rng(SAMPLE_SEED)
        for pos, vals, color in zip(positions, data, GROUP_COLORS):
            if not vals:
                continue
            jitter = rng.normal(0, 0.07, len(vals))
            ax1.scatter(
                np.array([pos] * len(vals)) + jitter, vals, s=24, color=color,
                alpha=0.8, edgecolor="white", linewidth=0.4,
            )
            ax1.hlines(np.median(vals), pos - 0.25, pos + 0.25, color=color, linewidth=2.2)
        ax1.set_xticks(positions, labels, fontsize=8)
        ax1.set_ylabel(desc.label)
        ax1.set_title("Value by group")
        ax1.grid(axis="x", visible=False)

        scored = [
            (float(r.features[desc.value_key]), r.total_score, r.is_winner)
            for r in dataset.compared
            if r.features.get(desc.value_key) is not None and r.total_score is not None
        ]
        if scored:
            xs = np.array([s[0] for s in scored])
            ys = np.array([s[1] for s in scored])
            cols = [WINNER_COLOR if s[2] else LOSER_COLOR for s in scored]
            ax2.scatter(xs, ys, c=cols, s=34, alpha=0.8, edgecolor="white", linewidth=0.5)
            if len(set(xs)) > 1:
                slope, intercept = np.polyfit(xs, ys, 1)
                grid = np.linspace(xs.min(), xs.max(), 40)
                ax2.plot(grid, slope * grid + intercept, color=NEUTRAL, linestyle="--", linewidth=1.2)
        ax2.axhline(0, color=GRID, linewidth=1.0)
        ax2.set_xlabel(desc.label)
        ax2.set_ylabel("Total spot-peer score")
        ax2.set_title("Value vs score")
        ax2.grid(axis="x", visible=False)

    fig.suptitle(desc.label, fontsize=12, fontweight="bold", y=1.02)
    path = out / f"feat_{desc.name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_review_report(results: AnalysisResults) -> Path:
    _style()
    dataset = results.dataset
    config = dataset.config
    charts_dir = config.review_charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# {config.season} survey — review report\n")
    lines.append(
        "Every derived number is shown next to the raw survey text it came from, "
        "so this report can be checked by reading it rather than the code. "
        "Charts live in `review_charts/`.\n"
    )
    lines.append(_toc())

    lines.append(_section_cohorts(dataset))
    lines.append(_section_distributions(dataset, charts_dir))
    lines.append(_section_features(results, charts_dir))
    lines.append(_section_appendix(results))

    path = config.review_report_path
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote review report %s", path)

    render_review_html(path)
    return path


def render_review_html(md_path: Path) -> Path | None:
    """Render the review Markdown to a self-contained HTML with Quarto.

    Best-effort. The Markdown keeps relative image references so it stays small
    and diff-friendly and renders in editors like VS Code. Quarto embeds the
    chart PNGs into a single HTML file that opens in any browser with no
    external files. Skipped with a note if Quarto is not installed.
    """
    quarto = shutil.which("quarto")
    if quarto is None:
        logger.info(
            "Quarto not installed; skipping HTML render. To build it manually: "
            "cd %s && quarto render %s --to html --embed-resources --standalone",
            md_path.parent,
            md_path.name,
        )
        return None

    try:
        subprocess.run(
            [
                quarto,
                "render",
                md_path.name,
                "--to",
                "html",
                "--embed-resources",
                "--standalone",
            ],
            cwd=md_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        tail = (error.stderr or error.stdout or "")[-600:]
        logger.warning("Quarto render failed; the Markdown is still valid.\n%s", tail)
        return None

    html_path = md_path.with_suffix(".html")
    logger.info("Rendered self-contained review HTML %s", html_path)
    return html_path


def _toc() -> str:
    return (
        "## Contents\n\n"
        "0. [Cohort reconciliation](#0-cohort-reconciliation)\n"
        "1. [Distribution analysis](#1-distribution-analysis)\n"
        "2. [Feature analysis](#2-feature-analysis)\n"
        "3. [Spot-check appendix](#3-spot-check-appendix)\n"
    )


def _section_cohorts(dataset: Dataset) -> str:
    out = ["## 0. Cohort reconciliation\n"]
    out.append(
        "Every survey respondent, their prize, whether they competed in the main "
        "tournament (`in_aib`), the cohort assigned, and their leaderboard score. "
        "This is the join everything else depends on. Sorted by score.\n"
    )
    counts = {
        "winner": len(dataset.winners),
        "non_winner": len(dataset.non_winners),
        "minibench_only": len(dataset.minibench_only),
        "unknown": len(dataset.cohort("unknown")),
    }
    out.append(
        "**Cohorts:** "
        + ", ".join(f"{k} = {v}" for k, v in counts.items())
        + f" (total {len(dataset.respondents)})\n"
    )
    out.append(
        "| bot name | prize (USD) | in_aib | cohort | participant rank | total score |\n"
        "|---|---:|:---:|:---:|---:|---:|"
    )
    ranked = sorted(
        dataset.respondents,
        key=lambda r: (r.total_score is None, -(r.total_score or 0)),
    )
    for r in ranked:
        out.append(
            f"| {_escape_cell(r.bot_name, 40)} | {r.prize:,.0f} | "
            f"{'yes' if r.in_tournament else 'no'} | {r.cohort} | "
            f"{r.participant_rank if r.participant_rank else '—'} | "
            f"{_fmt(r.total_score, 0)} |"
        )
    return "\n".join(out) + "\n"


def _section_distributions(dataset: Dataset, charts_dir: Path) -> str:
    out = ["## 1. Distribution analysis\n"]
    out.append(
        "One chart per raw survey question. Multi-select questions show each "
        "parsed option; numeric questions show a parse ledger mapping every raw "
        "answer to the value used. Bars split winners (green) from non-winners "
        "(orange); MiniBench-only respondents are omitted from these charts.\n"
    )
    resolved = dataset.resolved_columns
    ordinal_by_source = {s.source: s.name for s in dataset.config.ordinal_features}

    index = 0
    for field, question in resolved.items():
        kind = QUESTION_RENDER.get(field, "categorical")
        if kind == "skip":
            continue
        index += 1
        slug = f"q{index:02d}_{field}"
        out.append(f"### 1.{index} {question}\n")
        out.append(f"_Survey field: `{field}`_\n")

        if kind == "freetext":
            answered = [
                r for r in dataset.respondents if (r.answer(field) or "").strip()
            ]
            out.append(
                f"Free-text question, {len(answered)} of {len(dataset.respondents)} "
                f"answered. Not charted. Sample responses:\n"
            )
            rng = np.random.default_rng(SAMPLE_SEED + index)
            sample = rng.choice(answered, size=min(4, len(answered)), replace=False) if answered else []
            for r in sample:
                out.append(f"> **{_escape_cell(r.bot_name, 30)}:** {_escape_cell(r.answer(field), 160)}")
            out.append("")
            continue

        if kind == "multiselect":
            path, insight, ledger = _draw_multiselect(dataset, field, charts_dir, slug)
        elif kind == "numeric":
            path, insight, ledger = _draw_numeric(
                dataset, field, ordinal_by_source.get(field), charts_dir, slug
            )
        elif kind == "model_tier":
            path, insight, ledger = _draw_model_tier(dataset, charts_dir, slug)
        else:
            path, insight, ledger = _draw_categorical(dataset, field, charts_dir, slug)

        out.append(f"![{field}](review_charts/{path.name})\n")
        out.append(f"**Read:** {insight}\n")
        if ledger:
            out.append(ledger)
    return "\n".join(out) + "\n"


def _section_features(results: AnalysisResults, charts_dir: Path) -> str:
    dataset = results.dataset
    config = dataset.config
    descriptors = _feature_descriptors(config)

    by_family: dict[str, dict[str, TestResult]] = {
        fam.name: {r.feature: r for r in fam.results} for fam in results.families
    }

    def best_p(name: str) -> float:
        ps = [
            fam[name].p_raw
            for fam in by_family.values()
            if name in fam and fam[name].p_raw is not None
        ]
        return min(ps) if ps else 2.0

    descriptors.sort(key=lambda d: best_p(d.name))

    out = ["## 2. Feature analysis\n"]
    out.append(
        f"Each derived feature, ordered by its most significant p-value across the "
        f"three test families. For every feature: distribution across the top "
        f"{TOP_N} respondents, winners, non-winners, and all; the relationship to "
        f"total score; the tests already run; and a random sample of the underlying "
        f"rows so you can trace the derivation.\n"
    )
    out.append(
        "Nothing survives Bonferroni or BH correction, so read the p-values as a "
        "ranking of where the signal is, not as findings.\n"
    )

    for rank, desc in enumerate(descriptors, 1):
        stars = ""
        for fam in by_family.values():
            tr = fam.get(desc.name)
            if tr and tr.stars:
                stars = tr.stars
                break
        out.append(f"### 2.{rank} {desc.label} {stars}\n")
        out.append(f"- **Survey field:** `{desc.source}`")
        out.append(f"- **Rule:** {desc.definition}")
        out.append(f"- **Best raw p across families:** {_fmt_p(best_p(desc.name) if best_p(desc.name) <= 1 else None)}\n")

        chart = _feature_chart(dataset, desc, charts_dir)
        out.append(f"![{desc.name}](review_charts/{chart.name})\n")

        out.append("**Tests run:**\n")
        out.append("| family | group A | group B | effect | n | raw p | Bonferroni p | BH q |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for fam in results.families:
            tr = by_family[fam.name].get(desc.name)
            if tr is None:
                continue
            out.append(
                f"| {fam.name} | {_fmt(tr.group_a, 1)} | {_fmt(tr.group_b, 1)} | "
                f"{_fmt(tr.effect, 3)} ({tr.effect_label}) | {tr.n} | "
                f"{_fmt_p(tr.p_raw)} | {_fmt_p(tr.p_bonferroni)} | {_fmt_p(tr.q_value)} |"
            )
        out.append("")
        out.append(_feature_spot_check(dataset, desc))
    return "\n".join(out) + "\n"


def _feature_spot_check(dataset: Dataset, desc: FeatureDescriptor) -> str:
    """Random sample of respondents: raw answer next to the derived value."""
    rng = np.random.default_rng(hash(desc.name) % (2**32))
    pool = [r for r in dataset.respondents if (r.answer(desc.source) or "").strip()]
    if not pool:
        return "_No answers to sample._\n"
    sample = rng.choice(pool, size=min(6, len(pool)), replace=False)
    lines = [
        "**Spot-check** (random respondents, raw answer → derived value):\n",
        "| bot | raw answer | derived | cohort | score |",
        "|---|---|:---:|:---:|---:|",
    ]
    for r in sample:
        derived = r.features.get(desc.value_key)
        lines.append(
            f"| {_escape_cell(r.bot_name, 24)} | {_escape_cell(r.answer(desc.source), 70)} | "
            f"{_derived_str(derived)} | {r.cohort} | {_fmt(r.total_score, 0)} |"
        )
    return "\n".join(lines) + "\n"


def _section_appendix(results: AnalysisResults) -> str:
    config = results.dataset.config
    out = ["## 3. Spot-check appendix\n"]
    out.append(
        "**Full per-respondent feature matrix:** `data/respondent_features.csv` "
        "(one row per respondent, every derived feature).\n"
    )
    out.append("**Full test results with corrected p-values:**")
    for fam in results.families:
        out.append(f"- `data/tests_{fam.name}.csv` ({fam.size} tests)")
    out.append("")
    out.append("**Leaderboard used:** `data/spring_2026_leaderboard.csv`\n")
    out.append("**Regenerate this report:**\n")
    out.append("```bash")
    out.append(
        "poetry run python aib_analysis/survey_analysis_v1/run_survey_analysis.py "
        "--season spring-2026"
    )
    out.append("```\n")
    out.append(
        "The leaderboard CSV is built separately from the tournament JSON with "
        "`aib_analysis/survey_analysis_v1/build_leaderboard_csv.py`.\n"
    )
    out.append(
        f"**How significance is judged:** three families of {results.winner_family.size} "
        f"tests each, corrected separately. Bonferroni threshold is "
        f"{config.alpha}/{results.winner_family.size} = "
        f"{config.alpha / results.winner_family.size:.5f} on the raw p. "
        f"Benjamini-Hochberg q-values control the false discovery rate. "
        f"Binary features use Fisher exact, ordinal features use Mann-Whitney U, "
        f"correlations use Pearson.\n"
    )
    return "\n".join(out) + "\n"
