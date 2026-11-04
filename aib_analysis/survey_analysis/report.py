"""Generate the main analysis report (config.REPORT_MD).

One section per structured question: a distribution chart split by non-winners,
winners, and top-10 bots, plus one or two correlations against sum of spot peer
score described in plain language. Prose follows the project writing style: direct,
specific, sentence-case headers, no hype vocabulary.

With blind=True, every result for a precommitted or alternative hypothesis is
withheld (its correlations, score charts, and the group split of its question's
distribution chart), so the picks can be reviewed before anyone sees how they did.
"""

from __future__ import annotations

import logging
import os
import string
from dataclasses import replace
from datetime import date

from aib_analysis.survey_analysis import (
    config,
    loading,
    parsing,
    plots,
    precommitted,
    season_notes,
    stats,
)
from aib_analysis.survey_analysis.features import (
    NUMERIC_VARIABLE_SPECS,
    RespondentFeatures,
    variable_spec,
)
from aib_analysis.survey_analysis.leaderboard import get_leaderboard_rows

logger = logging.getLogger(__name__)


# Stable Metaculus links (not season-specific). Season-specific links live in
# season_notes.py.
_FUTUREEVAL_URL = "https://www.metaculus.com/futureeval/"
_PEER_SCORE_URL = "https://www.metaculus.com/help/scores-faq/#peer-score"
_MINIBENCH_URL = "https://www.metaculus.com/aib/minibench/"

_WITHHELD = "withheld until the precommitted hypotheses are reviewed"

# Identifies the research option for Metaculus/Manifold search tools, which
# overlaps with the similar-questions strategy without counting toward it.
_MARKET_SEARCH_TOOL_SUBSTRING = "metaculus/manifold"


def _rel(path: str) -> str:
    return os.path.relpath(path, config.OUTPUT_DIR)


def _fmt_p(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


_BOOLEAN_DEFINITIONS: dict[str, str] = {
    feature.key: feature.definition for feature in config.BOOLEAN_FEATURES
}


def _correlation_sentence(result: stats.CorrelationResult, q_value: float | None) -> str:
    if result.coefficient is None:
        return f'"{result.label}" has too few observations to estimate a correlation (n = {result.n}).'
    significant = q_value is not None and q_value < config.EVIDENCE_SIGNIFICANT_Q
    sig_clause = " and is significant after false-discovery correction" if significant else ""
    stats_str = (
        f"{result.method} r = {result.coefficient:+.2f}, {_fmt_p(result.p_value)}, "
        f"q = {_fmt_pq(q_value)}, n = {result.n}"
    )
    sentence = f'"{result.label}" shows {result.direction_phrase()}{sig_clause} ({stats_str}).'
    definition = _BOOLEAN_DEFINITIONS.get(result.key)
    if definition:
        sentence += f" Yes means {definition}."
    return sentence


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
    features: list[RespondentFeatures], spec, index: int, groups: list[str]
) -> str | None:
    filename = f"q{index:02d}_{spec.slug}_dist.png"
    out_path = os.path.join(config.CHARTS_DIR, filename)
    title = spec.title

    if spec.kind == "multiselect":
        plots.plot_multiselect_distribution(
            features, spec.slug, title, config.MULTISELECT_VOCAB[spec.slug], out_path, groups=groups
        )
    elif spec.kind == "model":
        plots.plot_multiselect_distribution(
            features, spec.slug, title, _model_options(features, spec.slug), out_path, groups=groups
        )
    elif spec.kind == "ordinal":
        categories = config.ORDINAL_ORDER[spec.slug] + config.UNORDERED_OPTIONS.get(spec.slug, [])
        plots.plot_categorical_distribution(
            features, spec.slug, title, categories, out_path, groups=groups
        )
    elif spec.kind == "single_select":
        plots.plot_categorical_distribution(
            features, spec.slug, title, config.SINGLE_SELECT_VOCAB[spec.slug], out_path, groups=groups
        )
    elif spec.kind == "numeric":
        plots.plot_categorical_distribution(
            features, spec.slug, title, _numeric_categories(features, spec.slug), out_path, groups=groups
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
}
# Correlation keys binned into numeric ranges (no clean answer bucket).
_NUMERIC_BIN_KEYS: set[str] = {"n_research_sources", "final_model_release", "questions_missed"}


def _fmt_release_ordinal(value: float) -> str:
    """Render a stored day-ordinal release date as a YYYY-MM bucket label."""
    return date.fromordinal(int(value)).strftime("%Y-%m")


# Per-key formatter for numeric bin labels; defaults to _fmt_num when absent.
_NUMERIC_LABEL_FMT = {"final_model_release": _fmt_release_ordinal}


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
    features: list[RespondentFeatures], key: str, n_bins: int = 3, value_fmt=_fmt_num
) -> list[tuple[str, list[float]]]:
    pairs = sorted(
        (f.variables[key], f.score)
        for f in features
        if f.variables.get(key) is not None and f.score is not None
    )
    total = len(pairs)
    if total < 2 * plots.MIN_CELL_SIZE:
        n_bins = 2
    # Merge any bins that end up with the same range label (e.g. heavily tied
    # values), preserving first-seen order. Robust to non-adjacent duplicates.
    merged: dict[str, list[float]] = {}
    order: list[str] = []
    for bin_index in range(n_bins):
        low = bin_index * total // n_bins
        high = (bin_index + 1) * total // n_bins
        chunk = pairs[low:high]
        if not chunk:
            continue
        value_min, value_max = chunk[0][0], chunk[-1][0]
        label = value_fmt(value_min) if value_min == value_max else f"{value_fmt(value_min)} to {value_fmt(value_max)}"
        if label not in merged:
            merged[label] = []
            order.append(label)
        merged[label].extend(score for _value, score in chunk)
    return [(label, merged[label]) for label in order]


def _correlation_chart(
    features: list[RespondentFeatures], key: str, prefix: str
) -> tuple[str | None, str | None]:
    """Return (chart_filename, None) if drawn, else (None, exclusion_reason)."""
    spec = variable_spec(key)
    filename = f"{prefix}_{key}_score.png"
    out_path = os.path.join(config.CHARTS_DIR, filename)

    if spec.kind == "binary":
        labeled = _labeled_scores_binary(features, key)
    elif key in _ORDINAL_SLUG_FOR_KEY:
        labeled = _labeled_scores_ordinal(features, _ORDINAL_SLUG_FOR_KEY[key])
    elif key in _NUMERIC_BIN_KEYS:
        labeled = _labeled_scores_numeric(
            features, key, value_fmt=_NUMERIC_LABEL_FMT.get(key, _fmt_num)
        )
    else:
        return None, "Graph excluded: this variable is not bucketed for a group chart"

    created = plots.plot_group_means(labeled, f"{spec.label}: average peer score", out_path)
    if created:
        return filename, None
    return None, "Graph excluded due to low bucket count"


def _spec_effect_size(
    spec, feature_results: dict[str, "stats.CorrelationResult"], hidden_keys: frozenset[str]
) -> float:
    """A question's strongest shown correlation effect, for ordering. Questions with
    no shown performance correlation sort last via -1, so a withheld result cannot
    move its question up the page."""
    effects = [
        abs(feature_results[key].coefficient)
        for key in spec.correlations
        if key in feature_results
        and key not in hidden_keys
        and feature_results[key].coefficient is not None
    ]
    return max(effects) if effects else -1.0


def _group_counts(features: list[RespondentFeatures]) -> dict[str, int]:
    counts = {group: 0 for group in config.GROUP_ORDER}
    for feature in features:
        for group in feature.respondent.groups:
            counts[group] += 1
    return counts


def _all_feature_keys() -> list[tuple[str, str]]:
    """Every measured feature as (key, label), booleans then numerics, deduped."""
    pairs = [(f.key, f.label) for f in config.BOOLEAN_FEATURES]
    pairs += [(s.key, s.label) for s in NUMERIC_VARIABLE_SPECS]
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for key, label in pairs:
        if key not in seen:
            seen.add(key)
            unique.append((key, label))
    return unique


def _benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """False-discovery-rate adjusted q-values, in the input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvalues = [1.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        original = order[rank]
        adjusted = pvalues[original] * m / (rank + 1)
        running_min = min(running_min, adjusted)
        qvalues[original] = min(running_min, 1.0)
    return qvalues


def _fmt_pq(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _feature_stats(
    corr_pool: list[RespondentFeatures],
) -> tuple[dict[str, "stats.CorrelationResult"], dict[str, float]]:
    """Correlate every measured feature and compute one shared q-value map.

    Returns (results_by_key, q_by_key). Precommitted hypotheses carry their one-sided
    p and a Holm-adjusted q; every other feature gets a Benjamini-Hochberg q across
    the exploratory features. The same q is reused wherever a feature appears.
    """
    results = {key: stats.correlate_with_score(corr_pool, key) for key, _label in _all_feature_keys()}
    q_by_key: dict[str, float] = {}
    for hypothesis_result in precommitted.evaluate_precommitted(corr_pool):
        key = hypothesis_result.hypothesis.key
        results[key] = replace(results[key], p_value=hypothesis_result.one_sided_p)
        if hypothesis_result.holm_p is not None:
            q_by_key[key] = hypothesis_result.holm_p
    exploratory = [
        (key, res)
        for key, res in results.items()
        if key not in precommitted.PRECOMMITTED_KEYS and res.p_value is not None
    ]
    qvalues = _benjamini_hochberg([res.p_value for _key, res in exploratory])
    q_by_key.update({key: qvalues[i] for i, (key, _res) in enumerate(exploratory)})
    return results, q_by_key


def _evidence_summary_section(
    results: dict[str, "stats.CorrelationResult"],
    q_by_key: dict[str, float],
    hidden_keys: frozenset[str],
) -> str:
    tested = sum(1 for res in results.values() if res.p_value is not None)
    exploratory_tested = sum(
        1 for key, res in results.items() if key not in precommitted.PRECOMMITTED_KEYS and res.p_value is not None
    )
    n_obs = max((res.n for res in results.values()), default=0)
    labels = dict(_all_feature_keys())

    def effect(key: str) -> float:
        coef = results[key].coefficient
        return abs(coef) if coef is not None else -1.0

    # Flat ordering by |r| descending; features with no estimate sort last. A
    # stable sort keeps the _all_feature_keys() order among ties.
    ordered_keys = sorted(
        (key for key, _label in _all_feature_keys() if key not in hidden_keys),
        key=lambda key: -effect(key),
    )

    lines = ["## Evidence summary", ""]
    lines.append(
        "Below is a list of every measured survey feature versus performance (average spot peer "
        "score), ordered by correlation magnitude."
    )
    lines.append("")
    lines.append(
        f"In the list, r is correlation, p is the uncorrected p-value, and q is the "
        f"Benjamini-Hochberg p-value adjusted for testing all {exploratory_tested} exploratory features. "
        f"For the {len(precommitted.PRECOMMITTED_KEYS)} precommitted hypotheses (marked \"precommitted: "
        "yes\"), p is one-sided and q is Holm-adjusted across those hypotheses (see Appendix A). \"Significant\" means "
        f"q < {config.EVIDENCE_SIGNIFICANT_Q}. This analysis aims to be primarily hypothesis-generating. "
        f"We can't conclude much with n ≈ {n_obs} bots and {tested} features tested. We also expect "
        "many of these features to be confounded. However, FutureEval hosts the largest group of "
        "publicly competing AI forecasters, and so the results are worth indexing on."
    )
    lines.append("")
    for key in ordered_keys:
        res = results[key]
        label = labels[key]
        q_value = q_by_key.get(key)
        if res.coefficient is None:
            lines.append(f"- **{label}**: insufficient data (n = {res.n})")
            continue
        significant = q_value is not None and q_value < config.EVIDENCE_SIGNIFICANT_Q
        precommitted_marker = " · precommitted: yes" if key in precommitted.PRECOMMITTED_KEYS else ""
        lines.append(
            f"- **{label}**: r = {res.coefficient:+.2f} · p = {_fmt_pq(res.p_value)} · "
            f"q = {_fmt_pq(q_value)} · significant: {'yes' if significant else 'no'}{precommitted_marker}"
        )
    hidden_labels = [label for key, label in _all_feature_keys() if key in hidden_keys]
    if hidden_labels:
        lines.append(
            f"- **{len(hidden_labels)} features {_WITHHELD}**: " + "; ".join(hidden_labels)
        )
    return "\n".join(lines)


def generate_report(features: list[RespondentFeatures], blind: bool = False) -> str:
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    for stale in os.listdir(config.CHARTS_DIR):
        if stale.endswith(".png"):
            os.remove(os.path.join(config.CHARTS_DIR, stale))
    hidden_keys = precommitted.BLINDED_KEYS if blind else frozenset()
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
    feature_results, q_by_key = _feature_stats(corr_pool)
    leaderboard_rows = get_leaderboard_rows()
    leaderboard_size = len(leaderboard_rows)

    lines: list[str] = []
    lines.append(f"# {season_notes.SEASON_LABEL} FutureEval bot-maker survey")
    lines.append("")
    lines.append(_intro(in_scope, excluded, corr_pool))
    lines.append("")
    ordered_specs = sorted(
        config.QUESTION_SPECS,
        key=lambda spec: _spec_effect_size(spec, feature_results, hidden_keys),
        reverse=True,
    )
    lines.append(_evidence_summary_section(feature_results, q_by_key, hidden_keys))
    lines.append("")
    no_correlation = "with no shown performance correlation" if blind else "with no performance correlation"
    lines.append(
        "The question sections below are ordered by their strongest correlation with performance "
        f"(largest |r| first). Questions {no_correlation} come last. The evidence summary above is "
        "ordered by |r| alone."
    )
    lines.append("")

    for index, spec in enumerate(ordered_specs, start=1):
        has_hidden_result = any(key in hidden_keys for key in spec.correlations)
        lines.append(f"## {spec.title}")
        lines.append("")
        if spec.slug in config.DERIVED_SPEC_SOURCES:
            lines.append(f"> {config.DERIVED_SPEC_SOURCES[spec.slug]}")
        else:
            lines.append(f"> Survey question: *{config.COLUMNS[spec.slug]}*")
        lines.append("")
        groups = ["everyone"] if has_hidden_result else config.CHART_GROUP_ORDER
        dist = _distribution_chart(in_scope, spec, index, groups)
        if dist:
            # Empty alt text so Quarto does not add a caption that duplicates the
            # chart's own title.
            lines.append(f"![]({_rel(os.path.join(config.CHARTS_DIR, dist))})")
            lines.append("")
            if has_hidden_result:
                lines.append(f"The split by winners and top 10 is {_WITHHELD}.")
                lines.append("")

        if spec.correlations:
            for key in spec.correlations:
                if key in hidden_keys:
                    lines.append(f'"{variable_spec(key).label}": {_WITHHELD}.')
                    lines.append("")
                    continue
                result = feature_results[key]
                chart, reason = _correlation_chart(scored, key, f"q{index:02d}")
                sentence = _correlation_sentence(result, q_by_key.get(key))
                if key in precommitted.PRECOMMITTED_KEYS:
                    sentence += " Precommitted: p is one-sided and q is Holm-adjusted."
                if chart is None:
                    sentence += f" [{reason}]"
                lines.append(sentence)
                lines.append("")
                if chart is not None:
                    lines.append(f"![]({_rel(os.path.join(config.CHARTS_DIR, chart))})")
                    lines.append("")
        lines.append("")

    lines.append(_methodology_appendix(in_scope, excluded, corr_pool, counts, leaderboard_size))
    lines.append("")
    lines.append(_caveats_section(in_scope, corr_pool, counts, leaderboard_size))
    lines.append("")
    return "\n".join(lines)


def _excluded_detail(excluded) -> str:
    """Characterize the excluded respondents, computing the no-record tail.

    Every excluded bot lacks a scored-leaderboard row; those that also match no
    prize-sheet owner have no tournament record at all, and that count is derived
    here so the sentence can never drift from the data.
    """
    no_record = sum(1 for f in excluded if f.respondent.prize_match_kind == "none")
    detail = season_notes.EXCLUDED_MINIBENCH_LABEL.replace(
        "MiniBench-only", f"[MiniBench-only]({_MINIBENCH_URL})", 1
    )
    if no_record == 1:
        detail += ", plus one bot with no matching tournament record"
    elif no_record > 1:
        detail += f", plus {no_record} bots with no matching tournament record"
    return detail


def _intro(in_scope, excluded, corr_pool) -> str:
    takeaway_bullets = "\n".join(f"* {item}" for item in season_notes.MAIN_TAKEAWAYS)
    return "\n".join(
        [
            f"Every season Metaculus sends out a survey to bot makers who participated in the "
            f"[FutureEval]({_FUTUREEVAL_URL}) AI forecasting benchmark. This analysis compares answers to "
            f"this survey with performance in the "
            f"[{season_notes.TOURNAMENT_NAME}]({season_notes.TOURNAMENT_URL}) to help surface what works "
            "and what doesn't when forecasting the future.",
            "",
            "## Main takeaways",
            "",
            takeaway_bullets,
            "",
            "## Introduction",
            "",
            f"{len(in_scope) + len(excluded)} bot makers answered the {season_notes.SEASON_LABEL} survey, and "
            f"this analysis covers the {len(in_scope)} whose bots competed in the scored tournament. For each "
            "survey question, we show how answers were spread across winners, non-winners, and the top 10, "
            "and how each feature of a bot (for example its final model, research tools, or cost per "
            "question) relates to its performance, measured by its average spot peer score. Correlations use "
            f"the {len(corr_pool)} bots that forecast at least {config.MIN_QUESTIONS_FOR_CORRELATION} scored "
            "questions.",
            "",
            f"We committed in advance to {len(precommitted.PRECOMMITTED_KEYS)} hypotheses from past surveys and "
            "treat every other feature as exploratory, adjusting both groups for the number of tests we ran. "
            f"With about {len(corr_pool)} bots, the results are suggestive rather than conclusive, and a "
            "correlation does not show that a feature caused a better score. **Appendix A** describes the "
            "methodology in full, and **Appendix B** lists the caveats.",
        ]
    )


def _precommitted_methodology() -> str:
    names = [
        f"\"{variable_spec(hypothesis.key).label}\" ({'higher' if hypothesis.predicted_sign > 0 else 'lower'} score)"
        for hypothesis in precommitted.PRECOMMITTED_HYPOTHESES
    ]
    listed = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else names[0]
    late_note = f" {season_notes.PRECOMMIT_LATE_DEFINITION_NOTE}" if season_notes.PRECOMMIT_LATE_DEFINITION_NOTE else ""
    return (
        f"We chose {len(names)} hypotheses from past surveys before downloading the responses, and confirmed "
        f"them before looking at any hypothesis result: {listed}. Each is tested only in the predicted "
        f"direction (a one-sided test), and the {len(names)} are corrected together with Holm's method "
        "instead of joining the false-discovery correction, so their q-values are Holm-adjusted p-values. "
        f"They are marked \"precommitted: yes\" in the evidence summary.{late_note}"
    )


def _methodology_appendix(in_scope, excluded, corr_pool, counts, leaderboard_size) -> str:
    total = len(in_scope) + len(excluded)
    parts = [
        "## Appendix A: Methodology",
        "",
        f"{total} bot makers answered the {season_notes.SEASON_LABEL} survey. This report covers the "
        f"{len(in_scope)} whose bots competed in the scored FutureEval tournament. It shows, for each "
        "structured question, how answers were distributed and how they relate to bot performance.",
        "",
        "We show 2 types of graphs. The first is the raw distribution of answers for each option for "
        "each question. The second compares the value of some feature of the survey with the "
        "performance of the bot in the tournament.",
        "",
        f"Performance is a bot's average spot peer score in the {season_notes.SEASON_LABEL} FutureEval "
        f"tournament. Its spot [peer score]({_PEER_SCORE_URL}) on a question compares its forecast at "
        "scoring time against the geometric mean of its peers. If it is positive, the prediction was "
        "(on average) better than others. If it is negative, it was worse than others.",
        "",
        "Three groups appear in the distribution charts:",
        "",
        f"* Non-winners: {counts['non_winner']} FutureEval participants who did not perform well enough "
        "to win a prize.",
        f"* Winners: {counts['winner']} participants who won prize money in the tournament.",
        f"* Top 10 (by peer score): {counts['top_10']} participants who filled the survey and whose bot "
        f"placed in the top 10 of the full {leaderboard_size}-bot leaderboard by sum of spot peer score. "
        "This group overlaps with winners.",
        "",
        "Charts show the share within each group, since the groups differ in size.",
        "",
        f"Correlations use a stricter set of the {len(corr_pool)} bots that forecast at least "
        f"{config.MIN_QUESTIONS_FOR_CORRELATION} scored questions, so a bot with only a few questions "
        "cannot swing a result due to lucky forecasting. The distribution charts still use all "
        f"{len(in_scope)} participants, and only the performance correlations apply the question floor.",
        "",
        "Each feature is correlated with a bot's average spot peer score using Pearson's r for yes/no "
        "traits and Spearman's rank correlation for ordered or counted ones. Because many features are "
        "tested at once, every p-value also carries a Benjamini-Hochberg q-value (its p adjusted for "
        "the false-discovery rate across all exploratory features), and a result is called \"significant\" "
        "only when q < 0.05.",
        "",
        _precommitted_methodology(),
        "",
        "For the frontier model chart, a bot is called \"frontier\" if the model it used for its final "
        "prediction is a flagship model (not a mini, flash, or fast variant) and was "
        f"{config.FRONTIER_RELEASE_CUTOFF_LABEL}. A vendor's flagship line still counts when the vendor "
        "adds a pricier tier above it (for example Claude Opus below Claude Fable, or GPT-6 Sol below "
        "GPT-6 Astra), while mid tiers such as Claude Sonnet do not.",
        "",
        "The number of research sources counts each option a bot maker ticked, plus each source they "
        "named in a write-in that is on our list of known extra sources (for example RSS feeds or "
        "OpenRouter web search). We add new write-in sources to that list each season, before looking "
        "at any scores.",
        "",
        "One feature does not come from the survey: questions missed, the number of scored questions "
        "still open after a bot's first forecast that the bot did not forecast. It is a rough measure "
        "of engineering effort, since a bot that crashes, runs out of credits, or stops early misses "
        "questions. Questions that closed before a bot started do not count against it.",
        "",
        "How to read the correlations: r runs from -1 to +1. Values near 0 mean no relationship, "
        "positive means the trait is associated with a higher average peer score, and negative with a "
        "lower one. A low p-value means the pattern is unlikely to be chance, but due to the number of "
        "features being tested, lean on the q-value for significance.",
    ]
    if excluded:
        parts.append("")
        parts.append(
            f"{len(excluded)} respondents are excluded from this report since they made no forecasts "
            f"in the scored FutureEval tournament ({_excluded_detail(excluded)})."
        )
    return "\n".join(parts)


def _market_tool_without_strategy_count(corr_pool: list[RespondentFeatures]) -> int:
    """Bots that ticked the Metaculus/Manifold search-tool research option but not the strategy."""
    return sum(
        1
        for feature in corr_pool
        if parsing.feature_present(feature.cells["research"].matched, _MARKET_SEARCH_TOOL_SUBSTRING)
        and feature.variables.get("uses_similar_qs") == 0.0
    )


def _caveats_section(features, corr_pool, counts, leaderboard_size) -> str:
    owners = [owner for owner in loading.load_prize_owners() if owner.in_futureeval]
    winning_owners = sum(1 for owner in owners if owner.won_any)
    respondent_winner_pct = round(100 * counts["winner"] / len(features)) if features else 0
    field_winner_pct = round(100 * winning_owners / len(owners)) if owners else 0
    lines = ["## Appendix B: Caveats", ""]
    lines.append(
        f"- Self-selection: Only {len(features)} of the {leaderboard_size} scored bots answered the "
        "survey, and winners were far likelier to respond since the survey was required to receive "
        f"prizes. {counts['winner']} of {len(features)} analyzed respondents won a prize "
        f"({respondent_winner_pct}%), versus {winning_owners} of the {len(owners)} bot makers who "
        f"entered the FutureEval tournament ({field_winner_pct}%). The distributions and correlations "
        "describe this self-selected group, so a habit's popularity here can differ from its "
        "popularity among all bots, and correlations can be distorted if the makers who responded "
        "differ systematically from those who did not."
    )
    lines.append(
        f"- Small samples: At {len(features)} survey responses, this analysis is generally "
        "underpowered, so results should be treated as suggestive."
    )
    lines.append(
        "- Self-reported answers: Bot makers can misunderstand questions, accidentally click options, "
        "or intentionally skip options."
    )
    lines.append(
        "- Implementation rigor: Just because someone says they did something doesn't necessitate that "
        "they did it well. If something is effective when done well, but is easy to do wrong, it will "
        "get a low correlation."
    )
    lines.append(
        "- Correlation is not causation: None of these links prove that a survey feature caused a better score. "
        "For instance, if it is only compute on a forecast that matters, teams with more money to "
        "spend on compute may also fine-tune an open source model. This may exaggerate the correlation "
        "of finetuned models with score."
    )
    lines.append(
        "- Bar chart averages: In the per-question performance charts, each bar is the mean of the "
        "individual bots' averages in that group. Consequently, a bar sits below the strongest bots in "
        "the group because it blends strong and weak bots. A bar shows only a group's average and "
        "hides the spread within the group, so a clean-looking staircase of bars can still reflect a "
        "weak overall correlation. Each bar carries a 95% confidence interval to show that spread. The "
        "r, p, and q values are a better guide."
    )
    lines.append(
        "- Questions missed: The count cannot tell a bot that stopped early from one that crashed "
        "or ran out of credits mid-season. Because correlations only use bots with at least "
        f"{config.MIN_QUESTIONS_FOR_CORRELATION} scored questions, bots that stopped very early are "
        "left out of that analysis."
    )
    market_tool_only = _market_tool_without_strategy_count(corr_pool)
    if market_tool_only:
        lines.append(
            "- Checking similar questions or markets: This counts only bots whose makers ticked that "
            f"strategy. {market_tool_only} bots in the correlation set ticked the research option for "
            "Metaculus/Manifold search tools without ticking the strategy, so they count as not "
            "checking."
        )
    if season_notes.FRONTIER_LATE_DEFINITION_CAVEAT:
        lines.append(f"- Frontier definition: {season_notes.FRONTIER_LATE_DEFINITION_CAVEAT}")
    if season_notes.POST_HOC_FEATURE_CAVEAT:
        lines.append(f"- Features added after seeing results: {season_notes.POST_HOC_FEATURE_CAVEAT}")
    return "\n".join(lines)


def write_report(features: list[RespondentFeatures], blind: bool = False) -> None:
    markdown = generate_report(features, blind=blind)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.REPORT_MD, "w") as handle:
        handle.write(markdown)
    logger.info("Wrote %s report to %s", "blinded" if blind else "unblinded", config.REPORT_MD)


# string.Template ($season) rather than an f-string, because the CSS braces
# would otherwise need escaping.
_INDEX_HTML_TEMPLATE = string.Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$season FutureEval survey analysis</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         max-width: 780px; margin: 0 auto; padding: 3rem 1.5rem; line-height: 1.55; }
  h1 { font-size: 1.9rem; margin-bottom: 0.3rem; }
  p.sub { color: #667085; margin-top: 0; }
  .card { display: block; border: 1px solid #d0d5dd; border-radius: 10px; padding: 1.1rem 1.3rem;
          margin: 1rem 0; text-decoration: none; color: inherit; transition: border-color .15s, box-shadow .15s; }
  .card:hover { border-color: #2f80ed; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
  .card h2 { margin: 0 0 .3rem 0; font-size: 1.2rem; color: #2f80ed; }
  .card p { margin: 0; color: #475467; }
  @media (prefers-color-scheme: dark) {
    body { background:#0f1115; color:#e4e7ec; }
    .card { border-color:#2a2f3a; } .card p { color:#98a2b3; } p.sub{color:#98a2b3;}
  }
</style>
</head>
<body>
  <h1>$season FutureEval bot-maker survey</h1>
  <p class="sub">Analysis v2 &middot; generated from script</p>

  <a class="card" href="$report_html">
    <h2>Analysis report &rarr;</h2>
    <p>Answer distributions per question (split by non-winners, winners, top 10) and correlations against average spot peer score.</p>
  </a>
  <a class="card" href="parsing_decisions.html">
    <h2>Parsing decisions &rarr;</h2>
    <p>Every rule: group definitions, the model registry with release dates, option vocabularies, midpoint maps, and test methods.</p>
  </a>
  <a class="card" href="parsing_review.html">
    <h2>Parsing review (internal only) &rarr;</h2>
    <p>The audit: all columns accounted for, full join table, per-bot model classification, and every write-in that was set aside.</p>
  </a>
</body>
</html>
""")


def write_index_html() -> None:
    """Write the static landing page that links the three rendered docs."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    index_path = os.path.join(config.OUTPUT_DIR, "index.html")
    with open(index_path, "w") as handle:
        handle.write(
            _INDEX_HTML_TEMPLATE.substitute(
                season=season_notes.SEASON_LABEL,
                report_html=os.path.splitext(os.path.basename(config.REPORT_MD))[0] + ".html",
            )
        )
    logger.info("Wrote index page to %s", index_path)
