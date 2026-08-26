"""Generate the main analysis report (spring_survey_analysis.md).

One section per structured question: a distribution chart split by non-winners,
winners, and top-10 bots, plus one or two correlations against sum of spot peer
score described in plain language. Prose follows the project writing style: direct,
specific, sentence-case headers, no hype vocabulary.
"""

from __future__ import annotations

import logging
import os

from aib_analysis.survey_analysis_v2 import config, plots, stats
from aib_analysis.survey_analysis_v2.features import (
    RespondentFeatures,
    variable_spec,
)

logger = logging.getLogger(__name__)


def _rel(path: str) -> str:
    return os.path.relpath(path, config.OUTPUT_DIR)


def _fmt_p(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _correlation_sentence(result: stats.CorrelationResult) -> str:
    if result.coefficient is None:
        return f"- {result.label}: not enough data to estimate ({result.note}, n = {result.n})."
    flag = " Statistically significant." if result.is_significant else ""
    return (
        f"- {result.label}: {result.method} r = {result.coefficient:+.2f} "
        f"({_fmt_p(result.p_value)}, n = {result.n}). This is {result.direction_phrase()}.{flag}"
    )


def _model_options(features: list[RespondentFeatures], slug: str, top_n: int = 14) -> list[str]:
    counts: dict[str, int] = {}
    for feature in features:
        for option in feature.cells[slug].matched:
            counts[option] = counts.get(option, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _count in ranked[:top_n]]


def _numeric_categories(features: list[RespondentFeatures], slug: str) -> list[str]:
    values = {
        feature.cells[slug].matched[0]
        for feature in features
        if feature.cells[slug].matched
    }
    return sorted(values, key=lambda value: int(value) if value.isdigit() else 9999)


def _distribution_chart(
    features: list[RespondentFeatures], spec, index: int
) -> str | None:
    filename = f"q{index:02d}_{spec.slug}_dist.png"
    out_path = os.path.join(config.CHARTS_DIR, filename)
    title = spec.title

    if spec.kind == "multiselect":
        plots.plot_multiselect_distribution(
            features, spec.slug, title, config.MULTISELECT_VOCAB[spec.slug], out_path
        )
    elif spec.kind == "model":
        plots.plot_multiselect_distribution(
            features, spec.slug, title, _model_options(features, spec.slug), out_path
        )
    elif spec.kind == "ordinal":
        plots.plot_categorical_distribution(
            features, spec.slug, title, config.ORDINAL_ORDER[spec.slug], out_path
        )
    elif spec.kind == "single_select":
        plots.plot_categorical_distribution(
            features, spec.slug, title, config.SINGLE_SELECT_VOCAB[spec.slug], out_path
        )
    elif spec.kind == "numeric":
        plots.plot_categorical_distribution(
            features, spec.slug, title, _numeric_categories(features, spec.slug), out_path
        )
    else:
        return None

    return filename if os.path.exists(out_path) else None


# Correlation keys whose score chart groups respondents by a canonical answer bucket.
_ORDINAL_SLUG_FOR_KEY: dict[str, str] = {
    "iterations_mid": "iterations",
    "hours_mid": "hours",
    "llm_calls_mid": "llm_calls",
    "cost_mid": "cost_per_q",
    "research_vs_reasoning_ord": "research_vs_reasoning",
    "writeup_rating_ord": "writeup_rating",
}
# Correlation keys binned into numeric ranges (no clean answer bucket).
_NUMERIC_BIN_KEYS: set[str] = {"n_research_sources"}


def _labeled_scores_binary(
    features: list[RespondentFeatures], key: str
) -> list[tuple[str, list[float]]]:
    no_scores = [f.score for f in features if f.variables.get(key) == 0.0 and f.score is not None]
    yes_scores = [f.score for f in features if f.variables.get(key) == 1.0 and f.score is not None]
    return [("No", no_scores), ("Yes", yes_scores)]


def _labeled_scores_ordinal(
    features: list[RespondentFeatures], slug: str
) -> list[tuple[str, list[float]]]:
    groups: list[tuple[str, list[float]]] = []
    for category in config.ORDINAL_ORDER[slug]:
        scores = [
            f.score
            for f in features
            if f.cells[slug].matched and f.cells[slug].matched[0] == category and f.score is not None
        ]
        groups.append((category, scores))
    return groups


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _labeled_scores_numeric(
    features: list[RespondentFeatures], key: str, n_bins: int = 3
) -> list[tuple[str, list[float]]]:
    pairs = sorted(
        (f.variables[key], f.score)
        for f in features
        if f.variables.get(key) is not None and f.score is not None
    )
    total = len(pairs)
    if total < 2 * plots.MIN_CELL_SIZE:
        n_bins = 2
    groups: list[tuple[str, list[float]]] = []
    for bin_index in range(n_bins):
        low = bin_index * total // n_bins
        high = (bin_index + 1) * total // n_bins
        chunk = pairs[low:high]
        if not chunk:
            continue
        value_min, value_max = chunk[0][0], chunk[-1][0]
        label = _fmt_num(value_min) if value_min == value_max else f"{_fmt_num(value_min)}–{_fmt_num(value_max)}"
        scores = [score for _value, score in chunk]
        if groups and groups[-1][0] == label:  # merge equal-labeled adjacent bins
            groups[-1] = (label, groups[-1][1] + scores)
        else:
            groups.append((label, scores))
    return groups


def _correlation_chart(
    features: list[RespondentFeatures], key: str, index: int
) -> str | None:
    spec = variable_spec(key)
    filename = f"q{index:02d}_{key}_score.png"
    out_path = os.path.join(config.CHARTS_DIR, filename)

    if spec.kind == "binary":
        labeled = _labeled_scores_binary(features, key)
    elif key in _ORDINAL_SLUG_FOR_KEY:
        labeled = _labeled_scores_ordinal(features, _ORDINAL_SLUG_FOR_KEY[key])
    elif key in _NUMERIC_BIN_KEYS:
        labeled = _labeled_scores_numeric(features, key)
    else:
        return None  # e.g. team_size: no meaningful bucketing, correlation number only

    created = plots.plot_group_means(labeled, f"{spec.label}: average peer score", out_path)
    return filename if created else None


def _group_counts(features: list[RespondentFeatures]) -> dict[str, int]:
    counts = {group: 0 for group in config.GROUP_ORDER}
    for feature in features:
        for group in feature.respondent.groups:
            counts[group] += 1
    return counts


def generate_report(features: list[RespondentFeatures]) -> str:
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    for stale in os.listdir(config.CHARTS_DIR):
        if stale.endswith(".png"):
            os.remove(os.path.join(config.CHARTS_DIR, stale))
    # Analysis scope: only bots that actually competed in the scored FutureEval
    # tournament. MiniBench-only participants (0 FutureEval forecasts) and any
    # unjoinable bot are excluded from distributions and correlations alike; they
    # stay listed in the parsing review doc.
    in_scope = [f for f in features if f.respondent.matched_leaderboard_name]
    excluded = [f for f in features if not f.respondent.matched_leaderboard_name]
    # Correlations and score charts use only bots with enough scored questions
    # for a stable per-question average. Distributions and groups use all of in_scope.
    corr_pool = [f for f in in_scope if f.respondent.meets_correlation_minimum]
    counts = _group_counts(in_scope)
    scored = corr_pool

    lines: list[str] = []
    lines.append("# Spring 2026 FutureEval bot-maker survey")
    lines.append("")
    lines.append(_intro(in_scope, excluded, corr_pool, counts))
    lines.append("")

    for index, spec in enumerate(config.QUESTION_SPECS, start=1):
        lines.append(f"## {spec.title}")
        lines.append("")
        lines.append(f"> Survey question: *{config.COLUMNS[spec.slug]}*")
        lines.append("")
        dist = _distribution_chart(in_scope, spec, index)
        if dist:
            # Empty alt text so Quarto does not add a caption that duplicates the
            # chart's own title.
            lines.append(f"![]({_rel(os.path.join(config.CHARTS_DIR, dist))})")
            lines.append("")

        if spec.correlations:
            lines.append("Relationship with performance (average spot peer score):")
            lines.append("")
            for corr_index, key in enumerate(spec.correlations):
                result = stats.correlate_with_score(scored, key)
                lines.append(_correlation_sentence(result))
                if corr_index == 0:
                    chart = _correlation_chart(scored, key, index)
                    if chart:
                        lines.append("")
                        lines.append(f"![]({_rel(os.path.join(config.CHARTS_DIR, chart))})")
            lines.append("")
        lines.append("")

    lines.append(_headline_tests_section(in_scope, scored))
    lines.append("")
    lines.append(_frontier_section(in_scope, scored))
    lines.append("")
    lines.append(_caveats_section(in_scope, counts))
    return "\n".join(lines)


def _intro(in_scope, excluded, corr_pool, counts) -> str:
    total = len(in_scope) + len(excluded)
    parts = [
        f"{total} bot makers answered the Spring 2026 survey. This report covers the "
        f"{len(in_scope)} whose bot competed in the scored FutureEval tournament. It shows, for each "
        "structured question, how answers were distributed and how they relate to bot performance.",
        "",
        f"A bot is called \"frontier\" if the model it used for its final prediction is high powered (a "
        f"flagship model, not a mini, flash, or fast variant) and was {config.FRONTIER_RELEASE_CUTOFF_LABEL}.",
        "",
        "Performance is a bot's average spot peer score in the Spring 2026 FutureEval tournament. Its "
        "spot peer score on a question compares its forecast at scoring time against the geometric mean "
        "of its peers; averaging over the bot's questions gives a per-question skill measure that does "
        "not reward simply answering more questions.",
        "",
        "One label note: in the per-question performance charts below, each bar is the mean of the "
        "individual bots' averages in that group (one bot, one vote), so a bar sits below any single "
        "strong bot because it blends strong and weak bots. The axis is labeled \"Mean of bots' average "
        "spot peer score\" to reflect this.",
        "",
        "Three groups appear in every distribution chart:",
        "",
        f"- Non-winners: {counts['non_winner']} FutureEval participants who did not win an AIB prize.",
        f"- Winners: {counts['winner']} participants who won an AIB prize.",
        f"- Top 10 (peer score): {counts['top_10']} participants whose bot placed in the top 10 of "
        "the full 180-bot leaderboard by total score. This group overlaps with winners.",
        "",
        "Charts show the share within each group, since the groups differ in size.",
        "",
        f"Correlations use a stricter set: the {len(corr_pool)} bots that forecast at least "
        f"{config.MIN_QUESTIONS_FOR_CORRELATION} scored questions, so a bot with only a few questions "
        "cannot swing a result with one lucky forecast. The distributions above still use all "
        f"{len(in_scope)} participants; only the performance correlations apply the question floor.",
        "",
        "How to read the correlations: r runs from -1 to +1. Values near 0 mean no relationship, "
        "positive means the trait goes with a higher average peer score, negative with a lower one. A "
        "p-value below 0.05 means the pattern is unlikely to be chance. With this few bots, treat "
        "single results as suggestive, not proof.",
    ]
    if excluded:
        parts.append("")
        parts.append(
            f"{len(excluded)} respondents are excluded from this report: they made no forecasts in the "
            "scored FutureEval tournament (MiniBench-only participants, plus one bot with no matching "
            "tournament record). They are listed in the parsing review doc."
        )
    return "\n".join(parts)


def _headline_tests_section(features, scored) -> str:
    lines = ["## Repeat of the Fall headline tests", ""]
    lines.append(
        "The Fall 2025 analysis singled out research breadth and a handful of habits. Here are the "
        "same checks on Spring data. The correlation column uses average spot peer score on the "
        f"{len(scored)} bots with at least {config.MIN_QUESTIONS_FOR_CORRELATION} questions; the winner "
        "and non-winner rates use all in-scope participants."
    )
    lines.append("")
    lines.append("| Trait | Correlation with avg peer score | Winner rate | Non-winner rate | Fisher p |")
    lines.append("| --- | --- | --- | --- | --- |")

    keys = [
        "n_research_sources",
        "uses_asknews",
        "uses_exa",
        "uses_aggregation",
        "uses_base_rates",
        "uses_self_critique",
        "uses_similar_qs",
        "manual_review",
        "uses_pastcasting",
        "vs_community",
        "uses_minibench",
        "hours_mid",
        "cost_mid",
        "llm_calls_mid",
    ]
    for key in keys:
        spec = variable_spec(key)
        corr = stats.correlate_with_score(scored, key)
        corr_text = (
            f"{corr.method} r = {corr.coefficient:+.2f} ({_fmt_p(corr.p_value)})"
            if corr.coefficient is not None
            else "insufficient data"
        )
        if spec.kind == "binary":
            comparison = stats.compare_winner_rate(features, key, spec.label)
            winner_rate = f"{comparison.winner_rate * 100:.0f}%"
            non_rate = f"{comparison.non_winner_rate * 100:.0f}%"
            fisher = _fmt_p(comparison.fisher_p).replace("p = ", "").replace("p ", "")
        else:
            winner_rate = non_rate = fisher = "—"
        lines.append(f"| {spec.label} | {corr_text} | {winner_rate} | {non_rate} | {fisher} |")
    return "\n".join(lines)


def _frontier_section(features, scored) -> str:
    frontier_scores = [f.score for f in scored if f.frontier]
    other_scores = [f.score for f in scored if not f.frontier]
    corr = stats.correlate_with_score(scored, "frontier")

    lines = ["## Frontier vs non-frontier final models", ""]
    lines.append(
        "A bot counts as frontier if its final-prediction model is high powered and "
        f"{config.FRONTIER_RELEASE_CUTOFF_LABEL} (for example GPT-5.4, not GPT-5.4 mini). The exact "
        "model-to-date mapping is the model registry in `parsing_decisions.md`, and the per-bot "
        "verdicts are in `parsing_review.md`. That mapping is the main thing worth checking."
    )
    lines.append("")
    n_frontier = sum(1 for f in features if f.frontier)
    lines.append(f"- Frontier bots among respondents: {n_frontier} of {len(features)}.")
    if frontier_scores and other_scores:
        lines.append(
            f"- Mean of bots' average peer score, frontier: "
            f"{sum(frontier_scores) / len(frontier_scores):.2f} (n = {len(frontier_scores)}); "
            f"non-frontier: {sum(other_scores) / len(other_scores):.2f} (n = {len(other_scores)}). "
            f"Among bots with at least {config.MIN_QUESTIONS_FOR_CORRELATION} questions."
        )
    lines.append(_correlation_sentence(corr))
    return "\n".join(lines)


def _caveats_section(features, counts) -> str:
    lines = ["## Caveats", ""]
    lines.append(
        f"- Small samples. Only {counts['top_10']} top-10 bots answered the survey, so the orange "
        "bars move a lot with one response."
    )
    lines.append(
        "- Overlapping groups. Nearly every top-10 bot is also a winner, so those two series are not "
        "independent."
    )
    lines.append(
        "- Self-reported answers. Model names, hours, and costs are what makers typed, cleaned to "
        "fixed options. Write-in answers show up as an \"Other (write-in)\" bar in the distributions "
        "but are left out of the correlations."
    )
    lines.append(
        "- Correlation only. None of these links prove that a habit caused a better score."
    )
    lines.append(
        "- Metric vs groups. Performance here is average spot peer score, while winners and the top 10 "
        "were decided on total score. The two rank bots similarly but not identically."
    )
    lines.append(
        "- Top 10 is the raw ranking. It uses total spot peer score over all ~180 bots, so it can "
        "include bots the official prize board marks ineligible: Preseen-Chestnut ranks 5th here but is "
        "excluded there, which puts nostreambot at 11th (just outside) where the official board shows "
        "it 10th. This shifts one bot in or out of the top-10 group."
    )
    return "\n".join(lines)


def write_report(features: list[RespondentFeatures]) -> None:
    markdown = generate_report(features)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.REPORT_MD, "w") as handle:
        handle.write(markdown)
    logger.info("Wrote report to %s", config.REPORT_MD)
