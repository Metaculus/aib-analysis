"""Generate parsing_decisions.md: the rules, straight from config.py.

This doc explains every choice the parser makes. It is generated from the same
constants the analysis runs on, so it cannot drift from the code.
"""

from __future__ import annotations

import logging
import os

from aib_analysis.survey_analysis_v2 import config

logger = logging.getLogger(__name__)


def _model_registry_table() -> list[str]:
    lines = [
        "| Model | Release date | High power | After cutoff | Frontier | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in config.MODEL_REGISTRY:
        lines.append(
            f"| {model.display} | {model.release_date_str} | {_yn(model.high_power)} | "
            f"{_yn(model.released_after_cutoff)} | {_yn(model.is_frontier)} | {model.note} |"
        )
    return lines


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def _midpoint_table(title: str, mapping: dict[str, float]) -> list[str]:
    lines = [f"**{title}**", "", "| Bucket | Midpoint used |", "| --- | --- |"]
    for bucket, midpoint in mapping.items():
        lines.append(f"| {bucket} | {midpoint} |")
    lines.append("")
    return lines


def generate_parsing_decisions() -> str:
    lines: list[str] = []
    lines.append("# Parsing decisions")
    lines.append("")
    lines.append(
        "This doc lists every rule used to turn the raw Google Forms export into the numbers behind "
        "the report. It is generated from `aib_analysis/survey_analysis_v2/config.py`, so it always "
        "matches the code that ran."
    )
    lines.append("")

    lines.append("## Groups")
    lines.append("")
    lines.append("- Winner: won any AIB prize (prize-stats `winner_count > 0` or `aib_prize > 0`).")
    lines.append(
        f"- Top 10: top {config.TOP_N_FOR_TOP_GROUP} bots by sum of spot peer score on the full "
        "leaderboard, computed from the saved bot tournament."
    )
    lines.append("- Non-winner: every responding bot that is not a winner.")
    lines.append("- Groups overlap: a top-10 bot is almost always also a winner.")
    lines.append("")

    lines.append("## Frontier definition")
    lines.append("")
    lines.append(
        f"A bot is frontier if its final-prediction model is high powered and "
        f"{config.FRONTIER_RELEASE_CUTOFF_LABEL}. High power excludes mini, nano, flash, fast, haiku, "
        "free-tier, and other small variants. Release dates were looked up online in August 2026; "
        "`after cutoff` is computed from the date against the 2025-11-01 cutoff. A couple of dates are "
        "marked approximate in the note column. Fix any date below and rerun."
    )
    lines.append("")
    lines.extend(_model_registry_table())
    lines.append("")
    lines.append(
        "Tokens treated as vague or as parsing artifacts, never counted as frontier: "
        + ", ".join(sorted(config.MODEL_TOKENS_IGNORED))
        + "."
    )
    lines.append("")

    lines.append("## Multi-select parsing")
    lines.append("")
    lines.append(
        "Multi-select cells are not split on commas, because option text contains commas. Instead "
        "each known option is matched as a substring (longest first) and removed; whatever is left is "
        "recorded as an 'Other' write-in. Write-ins are aggregated into an 'Other (write-in)' bar in "
        "the distribution charts, but they do not feed the habit features or the correlations. Option "
        "vocabularies:"
    )
    lines.append("")
    for slug, vocab in config.MULTISELECT_VOCAB.items():
        lines.append(f"**{config.COLUMNS[slug]}** (`{slug}`)")
        lines.append("")
        for option in vocab:
            lines.append(f"- {option}")
        lines.append("")

    lines.append("## Single-select parsing")
    lines.append("")
    lines.append(
        "Single-select cells are kept only if they match a canonical option exactly (case-insensitive). "
        "Anything else counts toward the 'Other (write-in)' bar in the distribution chart, is logged in "
        "the review doc, and is excluded from the correlations. Canonical options:"
    )
    lines.append("")
    for slug, vocab in config.SINGLE_SELECT_VOCAB.items():
        lines.append(f"**{config.COLUMNS[slug]}** (`{slug}`): " + "; ".join(vocab))
        lines.append("")

    lines.append("## Numeric midpoints")
    lines.append("")
    lines.append(
        "Bucketed answers are converted to a single number for correlation. Mixed-unit hours are "
        "mapped to approximate total hours."
    )
    lines.append("")
    lines.extend(_midpoint_table("Iterations", config.ITERATIONS_MIDPOINT))
    lines.extend(_midpoint_table("Total active hours", config.HOURS_MIDPOINT))
    lines.extend(_midpoint_table("LLM calls per question", config.LLM_CALLS_MIDPOINT))
    lines.extend(_midpoint_table("Cost per question (USD)", config.COST_MIDPOINT))

    lines.append("## Habit features")
    lines.append("")
    lines.append(
        "Each habit is a yes/no flag set when the substring appears in one of the selected "
        "(canonical) options of a multi-select column. Matching the parsed options rather than the "
        "raw text means a free-text write-in cannot trip a flag."
    )
    lines.append("")
    lines.append("| Feature | Yes means | Column | Matched substring |")
    lines.append("| --- | --- | --- | --- |")
    for feature in config.BOOLEAN_FEATURES:
        lines.append(
            f"| {feature.label} | {feature.definition} | {config.COLUMNS[feature.column_slug]} "
            f"| `{feature.match_substring}` |"
        )
    lines.append("")

    lines.append("## Performance metric and correlation methods")
    lines.append("")
    lines.append(
        "- Performance is each bot's average spot peer score: its summed spot peer score divided by the "
        "number of scored questions it forecast. This isolates per-question skill from how many "
        "questions a bot answered."
    )
    lines.append(
        f"- Correlations only include bots with at least `MIN_QUESTIONS_FOR_CORRELATION` = "
        f"{config.MIN_QUESTIONS_FOR_CORRELATION} scored questions, so a bot with a few questions cannot "
        "swing a result with one lucky forecast. Distributions and the winner/top-10 groups still use "
        "every in-scope participant."
    )
    lines.append(
        "- Winner and top-10 groups are defined on total (summed) score, matching the official "
        "leaderboard and prizes; only the performance metric for correlations uses the average."
    )
    lines.append(
        "- Yes/no and plain-number traits vs peer score: Pearson's r. (For a yes/no trait coded 0/1 "
        "this is the same number as a point-biserial correlation.)"
    )
    lines.append("- Ranked or counted traits vs peer score: Spearman rank correlation.")
    lines.append("- Winners vs non-winners on a yes/no trait: Fisher exact test on the 2x2 table.")
    lines.append(
        "- Correlations need at least 8 paired, varying observations; otherwise they are reported as "
        "insufficient data."
    )
    return "\n".join(lines)


def write_parsing_decisions() -> None:
    markdown = generate_parsing_decisions()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.PARSING_DECISIONS_MD, "w") as handle:
        handle.write(markdown)
    logger.info("Wrote parsing decisions to %s", config.PARSING_DECISIONS_MD)
