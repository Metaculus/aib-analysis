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
    NUMERIC_VARIABLE_SPECS,
    RespondentFeatures,
    variable_spec,
)
from aib_analysis.survey_analysis_v2.leaderboard import get_leaderboard_rows

logger = logging.getLogger(__name__)


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
        label = _fmt_num(value_min) if value_min == value_max else f"{_fmt_num(value_min)} to {_fmt_num(value_max)}"
        if label not in merged:
            merged[label] = []
            order.append(label)
        merged[label].extend(score for _value, score in chunk)
    return [(label, merged[label]) for label in order]


def _correlation_chart(
    features: list[RespondentFeatures], key: str, index: int
) -> tuple[str | None, str | None]:
    """Return (chart_filename, None) if drawn, else (None, exclusion_reason)."""
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
        return None, "Graph excluded: this variable is not bucketed for a group chart"

    created = plots.plot_group_means(labeled, f"{spec.label}: average peer score", out_path)
    if created:
        return filename, None
    return None, "Graph excluded due to low bucket count"


def _spec_effect_size(spec, feature_results: dict[str, "stats.CorrelationResult"]) -> float:
    """A question's strongest correlation effect, for ordering. Distribution-only
    questions (no performance correlation) sort last via -1."""
    effects = [
        abs(feature_results[key].coefficient)
        for key in spec.correlations
        if key in feature_results and feature_results[key].coefficient is not None
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

    Returns (results_by_key, q_by_key). q-values are Benjamini-Hochberg adjusted
    across the whole family of tested features, so the same q is reused wherever a
    feature appears in the report.
    """
    results = {key: stats.correlate_with_score(corr_pool, key) for key, _label in _all_feature_keys()}
    scored = [(key, res) for key, res in results.items() if res.p_value is not None]
    qvalues = _benjamini_hochberg([res.p_value for _key, res in scored])
    q_by_key = {key: qvalues[i] for i, (key, _res) in enumerate(scored)}
    return results, q_by_key


def _ordered_feature_keys(ordered_specs, feature_results) -> list[str]:
    """Every measured feature key, grouped by its source question in the same
    order as the question sections, and by |r| within each question.

    A question owns its numeric correlation(s) plus every boolean habit derived
    from its column, so a question with several features keeps them together.
    """
    numeric_keys = {spec.key for spec in NUMERIC_VARIABLE_SPECS}

    def effect(key: str) -> float:
        coef = feature_results[key].coefficient
        return abs(coef) if coef is not None else -1.0

    ordered: list[str] = []
    seen: set[str] = set()
    for spec in ordered_specs:
        keys = [key for key in spec.correlations if key in numeric_keys]
        keys += [bf.key for bf in config.BOOLEAN_FEATURES if bf.column_slug == spec.slug]
        group = [key for key in dict.fromkeys(keys) if key not in seen]
        group.sort(key=lambda key: -effect(key))
        for key in group:
            seen.add(key)
            ordered.append(key)
    for key, _label in _all_feature_keys():  # safety net for any unassigned feature
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _evidence_summary_section(
    results: dict[str, "stats.CorrelationResult"], q_by_key: dict[str, float], ordered_specs
) -> str:
    tested = sum(1 for res in results.values() if res.p_value is not None)
    n_obs = max((res.n for res in results.values()), default=0)
    labels = dict(_all_feature_keys())
    ordered_keys = _ordered_feature_keys(ordered_specs, results)

    lines = ["## Evidence summary", ""]
    lines.append(
        "Every measured feature versus performance (average spot peer score), grouped by question in "
        "the same order as the sections below, with a question's several features kept together. These "
        "are the raw numbers, no recommendation; decide for yourself which are worth acting on."
    )
    lines.append("")
    lines.append(
        f"r is the correlation (its sign is the direction), p is the uncorrected p-value, and q is the "
        f"Benjamini-Hochberg p-value adjusted for testing all {tested} features. \"Significant\" means "
        f"q < {config.EVIDENCE_SIGNIFICANT_Q}."
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
        lines.append(
            f"- **{label}**: r = {res.coefficient:+.2f} · p = {_fmt_pq(res.p_value)} · "
            f"q = {_fmt_pq(q_value)} · significant: {'yes' if significant else 'no'}"
        )
    lines.append("")
    lines.append(
        f"Hypothesis-generating only. With n ≈ {n_obs} bots and {tested} features tested, nothing here "
        "is proof. Correlation is not causation, and some correlates (hours, frontier model) may track "
        "a maker's skill or budget rather than being the lever."
    )
    return "\n".join(lines)


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
    feature_results, q_by_key = _feature_stats(corr_pool)
    leaderboard_size = len(get_leaderboard_rows())

    lines: list[str] = []
    lines.append("# Spring 2026 FutureEval bot-maker survey")
    lines.append("")
    lines.append(_intro(in_scope, excluded, corr_pool, counts, leaderboard_size))
    lines.append("")
    ordered_specs = sorted(
        config.QUESTION_SPECS,
        key=lambda spec: _spec_effect_size(spec, feature_results),
        reverse=True,
    )
    lines.append(_evidence_summary_section(feature_results, q_by_key, ordered_specs))
    lines.append("")
    lines.append(
        "The question sections below are ordered by their strongest correlation with performance "
        "(largest |r| first); questions with no performance correlation come last. The evidence "
        "summary above follows the same order."
    )
    lines.append("")

    for index, spec in enumerate(ordered_specs, start=1):
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
            for key in spec.correlations:
                result = feature_results.get(key) or stats.correlate_with_score(scored, key)
                chart, reason = _correlation_chart(scored, key, index)
                sentence = _correlation_sentence(result, q_by_key.get(key))
                if chart is None:
                    sentence += f" [{reason}]"
                lines.append(sentence)
                lines.append("")
                if chart is not None:
                    lines.append(f"![]({_rel(os.path.join(config.CHARTS_DIR, chart))})")
                    lines.append("")
        lines.append("")

    lines.append(_caveats_section(in_scope, counts, leaderboard_size))
    return "\n".join(lines)


def _intro(in_scope, excluded, corr_pool, counts, leaderboard_size) -> str:
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
        "Why bars, not scatter: the performance charts use grouped bars so no individual bot can be "
        "identified from its public peer score. A bar shows only a group's average and hides the spread "
        "within the group, so a clean-looking staircase of bars can still reflect a weak overall "
        "correlation; each bar carries a 95% confidence interval to show that spread, and the r, p, and "
        "q values, which use every bot, are the better guide.",
        "",
        "Three groups appear in every distribution chart:",
        "",
        f"- Non-winners: {counts['non_winner']} FutureEval participants who did not win an AIB prize.",
        f"- Winners: {counts['winner']} participants who won an AIB prize.",
        f"- Top 10 (peer score): {counts['top_10']} participants whose bot placed in the top 10 of "
        f"the full {leaderboard_size}-bot leaderboard by total score. This group overlaps with winners.",
        "",
        "Charts show the share within each group, since the groups differ in size.",
        "",
        f"Correlations use a stricter set: the {len(corr_pool)} bots that forecast at least "
        f"{config.MIN_QUESTIONS_FOR_CORRELATION} scored questions, so a bot with only a few questions "
        "cannot swing a result with one lucky forecast. The distributions above still use all "
        f"{len(in_scope)} participants; only the performance correlations apply the question floor.",
        "",
        "Methodology: each feature is correlated with a bot's average spot peer score using Pearson's r "
        "for yes/no traits and Spearman's rank correlation for ordered or counted ones. Because many "
        "features are tested at once, every p-value also carries a Benjamini-Hochberg q-value (its p "
        "adjusted for the false-discovery rate across all tested features), and a result is called "
        "\"significant\" only when q < 0.05.",
        "",
        "How to read the correlations: r runs from -1 to +1. Values near 0 mean no relationship, "
        "positive means the trait goes with a higher average peer score, negative with a lower one. A "
        "low p-value means the pattern is unlikely to be chance, but with many features tested, lean on "
        "the q-value for significance. With this few bots, treat single results as suggestive, not proof.",
    ]
    if excluded:
        parts.append("")
        parts.append(
            f"{len(excluded)} respondents are excluded from this report: they made no forecasts in the "
            "scored FutureEval tournament (MiniBench-only participants, plus one bot with no matching "
            "tournament record). They are listed in the parsing review doc."
        )
    return "\n".join(parts)


def _caveats_section(features, counts, leaderboard_size) -> str:
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
        f"- Top 10 is the raw ranking. It uses total spot peer score over all {leaderboard_size} bots, so it can "
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


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spring 2026 FutureEval survey analysis</title>
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
  <h1>Spring 2026 FutureEval bot-maker survey</h1>
  <p class="sub">Analysis v2 &middot; generated from script</p>

  <a class="card" href="spring_survey_analysis.html">
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
"""


def write_index_html() -> None:
    """Write the static landing page that links the three rendered docs."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    index_path = os.path.join(config.OUTPUT_DIR, "index.html")
    with open(index_path, "w") as handle:
        handle.write(_INDEX_HTML)
    logger.info("Wrote index page to %s", index_path)
