"""Assemble the write-up markdown from computed results and precommitted sentences.

Season-specific prose that can't be computed is left as [===Fill me===] with a reminder of
what belongs there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from aib_analysis.pro_vs_bot_writeup import phrasing
from aib_analysis.pro_vs_bot_writeup.phrasing import (
    LATE_START_MIN_WEEKS,
    fill_me,
    format_ci,
    format_compact,
    format_date_range,
    format_duration,
    format_score,
    format_sum,
    join_names,
    late_starter_count,
    month_day,
    ordinal,
    percent,
    plural,
)
from aib_analysis.pro_vs_bot_writeup.results import (
    FIELD_CHART_MIN_QUESTIONS,
    TOP_N,
    CloseTimeDifference,
    LeaderboardRow,
    SeasonResults,
    SeasonTrend,
    bot_label,
    split_baseline_name,
)
from aib_analysis.pro_vs_bot_writeup.season_config import (
    ANALYSIS_CODE_URL,
    CONTACT_EMAIL,
    DATA_REQUEST_FORM_URL,
    FORECASTING_TOOLS_URL,
    FUTUREEVAL_URL,
    RESOURCES_URL,
    SCORING_FAQ_URL,
    BotQuestionWindow,
    SeasonConfig,
)

QUESTION_TYPE_LABELS = {
    "binary": "binary",
    "numeric": "numeric",
    "discrete": "discrete",
    "multiple_choice": "multiple-choice",
}
QUESTION_TYPE_DESCRIPTIONS = {
    "binary": "Binary",
    "numeric": "Numeric",
    "discrete": "Discrete (numeric questions with a limited set of whole-number outcomes)",
    "multiple_choice": "Multiple choice",
}
TITLE_SUGGESTIONS = {
    ("pros", "strong"): "Pros Beat Bots",
    ("pros", "weak"): "Pros Edge Out Bots",
    ("pros", "none"): "Pros Edge Out Bots, Without Statistical Significance",
    ("bots", "strong"): "Bots Beat Pros",
    ("bots", "weak"): "Bots Edge Out Pros",
    ("bots", "none"): "Bots Edge Out Pros, Without Statistical Significance",
}
INDIVIDUAL_CHART_ROWS = 20
TOP_BOTS_CHART_ROWS = 20
METAC_CHART_ROWS = 15
METAC_BOT_PROMPT = """You are a professional forecaster interviewing for a job.

Your interview question is:
{question_title}

Question background:
{paragraph_of_background_info_about_question}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{paragraph_defining_resolution_criteria}

{paragraph_defining_specifics_and_fine_print}

Your research assistant says:
{many_paragraphs_of_news_summary}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100"""


@dataclass(frozen=True)
class ChartFiles:
    season_trend: str
    team_size: str
    individuals: str
    accuracy: str
    top_bots: str
    field_rank: str
    field: str
    metac: str


def render_writeup(results: SeasonResults, config: SeasonConfig, charts: ChartFiles, field_hidden_count: int) -> str:
    sections = [
        _title_block(results, config),
        _takeaways_section(results, config),
        _introduction_section(results, config),
        _pros_vs_bots_section(results, config, charts),
        _accuracy_section(results, charts),
        _top_bots_section(results, config, charts, field_hidden_count),
        _model_comparison_section(results, charts),
        _links_section(config),
        _future_section(),
        _methodology_appendix(results, config),
        _prompt_appendix(),
        _data_notes_appendix(results, config),
    ]
    return "\n\n".join(sections) + "\n"


def _value_or_fill_me(value: str | None, reminder: str) -> str:
    return value if value is not None else fill_me(reminder)


def _number_or_fill_me(value: float | None, reminder: str) -> str:
    return phrasing.format_compact(value) if value is not None else fill_me(reminder)


def _bot_window_text(windows: tuple[BotQuestionWindow, ...]) -> str:
    """E.g. "1.5 hours", or "1.5 hours (through June 4) or 3 hours (from June 5 on)" when the window changed."""
    hours = [_number_or_fill_me(window.open_hours, "hours each bot question stays open, e.g. 1.5") for window in windows]
    if len(windows) == 1:
        return f"{hours[0]} hours"
    parts = []
    for index, (window, window_hours) in enumerate(zip(windows, hours)):
        next_start = windows[index + 1].first_open_date if index + 1 < len(windows) else None
        last_day = month_day(next_start - timedelta(days=1)) if next_start is not None else None
        if window.first_open_date is None:
            span = f"through {last_day}"
        elif last_day is None:
            span = f"from {month_day(window.first_open_date)} on"
        else:
            span = f"{month_day(window.first_open_date)} to {last_day}"
        parts.append(f"{window_hours} hours ({span})")
    return join_names(parts, conjunction="or")


def _image(path: str) -> str:
    return f"![]({path})"


def _type_mix(counts: dict[str, int], with_percentages: bool) -> str:
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if with_percentages:
        parts = [f"{QUESTION_TYPE_LABELS.get(kind, kind)} ({percent(count, total)})" for kind, count in ordered]
    else:
        parts = [f"{count} {QUESTION_TYPE_LABELS.get(kind, kind)}" for kind, count in ordered]
    return join_names(parts)


def _non_primary_sentence(non_primary_bots: tuple[str, ...]) -> str:
    if not non_primary_bots:
        return ""
    count = len(non_primary_bots)
    return (
        f" {phrasing.count_word(count, capitalize=True)} of the bot makers' bots {'was an extra bot' if count == 1 else 'were extra bots'} marked "
        "non-primary, which the official leaderboard doesn't rank; we rank every bot here, including Metac Bots."
    )


def _leaderboard_table(rows: list[LeaderboardRow], name_header: str, non_primary_bots: tuple[str, ...] = ()) -> str:
    lines = [
        f"| Rank | {name_header} | Sum of spot peer scores | Average spot peer score | 95% CI | Questions scored |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        ci_text = format_ci(row.ci) if row.ci is not None else "n/a"
        lines.append(
            f"| {row.rank} | {bot_label(row.name, non_primary_bots)} | {format_sum(row.sum_of_scores)} | {format_score(row.average_score)} | "
            f"{ci_text} | {row.question_count} |"
        )
    return "\n".join(lines)


def _season_trend_table(trend: SeasonTrend) -> str:
    lines = [
        "| Season | Bot team head-to-head score | 95% CI | Result |",
        "| --- | --- | --- | --- |",
    ]
    for point in trend.points:
        lines.append(
            f"| {point.season_label} | {phrasing.format_compact(point.mean)} | {format_ci(point.ci, compact=True)} | "
            f"{phrasing.season_result(point)} |"
        )
    return "\n".join(lines)


def _title_block(results: SeasonResults, config: SeasonConfig) -> str:
    head_to_head = results.head_to_head
    suggestion = TITLE_SUGGESTIONS[(head_to_head.leader, head_to_head.evidence)]
    return "\n\n".join(
        [
            f"# FutureEval {config.label} Results: {phrasing.FILL_ME}",
            fill_me(
                "a short headline for the top finding. Past titles: \"Pros Beat Bots, but the Gap is Nearly Gone\" "
                f"(Spring 2026) and \"Pros Maintain Clear Lead\" (Q2 2025). Precommitted suggestion: \"{suggestion}\""
            ),
        ]
    )


def _takeaways_section(results: SeasonResults, config: SeasonConfig) -> str:
    top_findings = [
        phrasing.headline_takeaway(results.head_to_head, results.head_to_head.pro_member_count, config.main_bot_team_size),
        phrasing.trend_takeaway(results.season_trend),
        phrasing.individuals_takeaway(results.individuals),
    ]
    other_takeaways = [
        phrasing.team_size_takeaway(results.team_sizes, config.main_bot_team_size),
        phrasing.top_baseline_takeaway(results.metac),
        phrasing.baseline_history_takeaway(results.metac),
        phrasing.top_10_spread_takeaway(results.field),
        phrasing.accuracy_takeaway(results.bot_field_accuracy),
        fill_me(
            "add 1-3 takeaways that don't repeat every season, each as a bold one-sentence finding followed by the "
            "numbers. Candidates: results of this season's Metac Bot experiments (search providers, reasoning "
            "effort, agentic Metac Bots; see the LLM section), who the top bot makers were (students and hobbyists "
            "vs companies, from the bot maker survey), and notable rule or question-mix changes"
        ),
    ]
    return "\n".join(
        ["## Main takeaways", "", "**Top findings:**", ""]
        + [f"* {item}" for item in top_findings]
        + ["", "**Other takeaways:**", ""]
        + [f"* {item}" for item in other_takeaways]
    )


def _introduction_section(results: SeasonResults, config: SeasonConfig) -> str:
    overview = results.overview
    head_to_head = results.head_to_head
    prize_pool = _value_or_fill_me(config.prize_pool, "the prize pool, e.g. \"$50,000\"")
    question_dates = phrasing.format_date_range(overview.first_question_close, overview.last_question_close)
    return "\n\n".join(
        [
            "## Introduction",
            f"Metaculus ran the {config.label} FutureEval tournament, with questions closing between "
            f"{question_dates}. It had two aims: 1) comparing AI bots to the best human forecasters and 2) evaluating "
            f"the best forecasting bots against one another. This is the primary analysis post for {config.label} "
            "FutureEval, and it focuses especially on comparing the rankings of Pros and bots. You can find our "
            f"other research on the [FutureEval resources page]({RESOURCES_URL}).",
            f"There were {overview.bot_count} bots ({overview.non_metac_bot_count} excluding the ones run by "
            f"Metaculus) forecasting on {overview.bot_scored_count} scored questions.{_non_primary_sentence(overview.non_primary_bots)} The non-Metaculus bot makers "
            f"competed for a prize pool of {prize_pool}. A team of {head_to_head.pro_member_count} Metaculus Pro "
            f"forecasters served as a human benchmark on {head_to_head.question_count} of these questions against "
            f"the bots that finished in the top {len(head_to_head.bot_team_members)} of the leaderboard on the "
            "bot-only questions.",
            "For how the questions, scoring, and teams work, see **Appendix A: Methodology** at the end of this "
            f"post. See a full overview of Metaculus's AI forecasting benchmark at the [FutureEval home page]({FUTUREEVAL_URL}).",
        ]
    )


def _pros_vs_bots_section(results: SeasonResults, config: SeasonConfig, charts: ChartFiles) -> str:
    head_to_head = results.head_to_head
    trend_reminder = fill_me(
        "1-2 sentences on what might explain this season's result, such as new model releases or changes in the "
        "question mix or Pro team. Don't describe a trend the confidence intervals don't support"
    )
    return "\n\n".join(
        [
            "## Are bots better than human Pros?",
            f"To compare the bots to Pros, we used only the {head_to_head.question_count} questions that were "
            "identical in the bot and Pro tournament and that were not annulled or otherwise had resolution issues. "
            "We then aggregated forecasts of the \"bot team\" and the \"Pro team\" (see Appendix A for how teams are "
            f"chosen and scored). The bot team was {join_names(list(head_to_head.bot_team_members))}.",
            phrasing.head_to_head_paragraph(head_to_head),
            "### Team performance over seasons",
            phrasing.trend_paragraph(results.season_trend),
            _image(charts.season_trend),
            _season_trend_table(results.season_trend),
            trend_reminder,
            "### Pros and bots as individuals",
            phrasing.individuals_paragraph(results.individuals, INDIVIDUAL_CHART_ROWS),
            _image(charts.individuals),
            "### How much does bot team size affect aggregation strength?",
            phrasing.team_size_paragraph(results.team_sizes, config.main_bot_team_size),
            _image(charts.team_size),
        ]
    )


def _accuracy_section(results: SeasonResults, charts: ChartFiles) -> str:
    return "\n\n".join(
        [
            "## How accurate are the forecasts?",
            phrasing.accuracy_definition_paragraph(),
            _image(charts.accuracy),
            phrasing.bot_field_accuracy_paragraph(results.bot_field_accuracy),
            phrasing.team_accuracy_paragraph(results.team_accuracy),
            phrasing.peer_score_translation_paragraph(results.bot_field_accuracy),
        ]
    )


def _top_bots_section(results: SeasonResults, config: SeasonConfig, charts: ChartFiles, field_hidden_count: int) -> str:
    field = results.field
    metac = results.metac
    standing = metac.current_standing
    if config.bot_tournament_url is not None:
        leaderboard_link = f"You can see the full leaderboard [here]({config.bot_tournament_url})."
    else:
        leaderboard_link = fill_me("link the bot tournament page with the full leaderboard")
    return "\n\n".join(
        [
            "## Which bots did best?",
            f"The bot tournament pitted all {len(field.rows)} bots (both bot makers' bots and in-house Metac Bots) "
            f"against one another on {field.scored_question_count} questions. {leaderboard_link} The chart below "
            f"shows the top {TOP_BOTS_CHART_ROWS} bots ranked by the sum of their spot peer scores, and the table "
            f"gives exact numbers for the top {TOP_N}.",
            _image(charts.top_bots),
            _leaderboard_table(field.top_10, "Bot", results.overview.non_primary_bots),
            "### The whole field",
            phrasing.field_paragraph(field, FIELD_CHART_MIN_QUESTIONS, field_hidden_count),
            _image(charts.field_rank),
            phrasing.field_band_paragraph(field, FIELD_CHART_MIN_QUESTIONS),
            _image(charts.field),
            "### The winning bot",
            fill_me(
                f"describe the winning bot ({field.rows[0].name}) in 2-4 sentences from its maker's public "
                "description or survey answers: research sources, models, and how forecasts are combined. Say "
                "whether it's open source and link it. Note that we pass the description along as reported and "
                "don't treat any part of it as a causal explanation of the bot's performance"
            ),
            "### Open-source bots",
            fill_me(
                "name the highest-placing open-source bot (rank, license, repo link, 1-2 sentences on how it works), "
                "then list other public bot repos with their ranks. The FutureEval resources page keeps the list "
                "of open-source bots"
            ),
            "### The best same-prompt Metac Bot",
            f"The top-scoring same-prompt Metac Bot was {standing.bot_name}, which placed {ordinal(standing.rank)} "
            f"of {standing.bot_count} in the bot tournament. The prompt it used is in Appendix B. "
            + fill_me("link an example question page showing its research and reasoning"),
        ]
    )


def _model_comparison_section(results: SeasonResults, charts: ChartFiles) -> str:
    metac = results.metac
    top_model, top_search, _variance_label = split_baseline_name(metac.top_baseline.name)
    other_count = len(metac.other_rows)
    other_clause = (
        f", plus {other_count} other Metac {plural(other_count, 'Bot')} that tested different setups" if other_count else ""
    )
    reminder = fill_me(
        "2-4 bullets on this season's Metac Bot experiments, each with scores and whether the difference is "
        "within the noise (e.g. \"No search provider is clearly better than another\"). Groups found among this "
        "season's bots (scores here rank Metac Bots only against each other, so they differ from the bot tournament's):"
    )
    reminder_lines = "\n".join(f"* {line}" for line in phrasing.experiment_fact_lines(results))
    return "\n\n".join(
        [
            "## How do LLMs compare?",
            f"To compare LLMs, Metaculus ran {len(metac.baseline_rows)} in-house Metac Bots that all used the same "
            f"prompt with a different model or search provider{other_clause}. We ranked all {metac.metac_bot_count} "
            f"Metac Bots against one another only (no other participants) on the {metac.scored_question_count} scored "
            f"questions; the average same-prompt bot forecast on {round(metac.average_questions_per_baseline_bot)} of "
            f"them. Each bot's name shows its model and search provider: for instance, {metac.top_baseline.name} uses "
            f"{top_model} with {top_search}. The prompt has stayed generally the same across all the benchmarks since "
            "2024 (see Appendix B).",
            _image(charts.metac),
            phrasing.metac_comparison_sentence(metac),
            f"{reminder}\n\n{reminder_lines}" if reminder_lines else reminder,
        ]
    )


def _links_section(config: SeasonConfig) -> str:
    metac_code = (
        f"[Metac Bot code]({config.metac_bot_code_url})"
        if config.metac_bot_code_url is not None
        else f"Metac Bot code {fill_me('link the exact version of the Metac Bot code run this season')}"
    )
    tournament = (
        f"[{config.label} bot tournament]({config.bot_tournament_url})"
        if config.bot_tournament_url is not None
        else f"{config.label} bot tournament {fill_me('link the bot tournament page')}"
    )
    return "\n".join(
        [
            "## Links to code and data",
            "",
            "Below are links to code and data you may be interested in:",
            "",
            f"* {metac_code} that was run during {config.label}. You can export this bot from the "
            f"[forecasting-tools]({FORECASTING_TOOLS_URL}) Python package.",
            f"* {tournament} with all bot questions, forecasts, and comments.",
            f"* [Code for this analysis]({ANALYSIS_CODE_URL}), if you want to review our math.",
            "* We often share data with researchers. If you are interested in running analyses on specific Metaculus "
            f"data, please fill out our [data request form]({DATA_REQUEST_FORM_URL}) or reach out to us at {CONTACT_EMAIL}.",
            f"* You can find a list of open-source bots and our other research on the [FutureEval resources page]({RESOURCES_URL}).",
        ]
    )


def _future_section() -> str:
    return "\n\n".join(
        [
            "## Future FutureEval tournaments",
            "We are continuing to run tournaments like the one described in this analysis. See more at "
            f"[metaculus.com/futureeval]({FUTUREEVAL_URL}). Email {CONTACT_EMAIL} if you have any questions about "
            "this analysis, how to participate, or anything else.",
        ]
    )


def _binary_resolution_sentence(yes_count: int, binary_total: int) -> str:
    no_count = binary_total - yes_count
    if yes_count == no_count:
        return "Binary questions resolved Yes and No equally often."
    more, fewer = ("No", "Yes") if no_count > yes_count else ("Yes", "No")
    more_count, fewer_count = max(yes_count, no_count), min(yes_count, no_count)
    return (
        f"More binary questions resolved {more} ({percent(more_count, binary_total)}) than {fewer} "
        f"({percent(fewer_count, binary_total)})."
    )


def _late_metac_clause(results: SeasonResults) -> str:
    late_count = late_starter_count(results.metac)
    if not late_count:
        return ""
    return f", though {late_count} of the {results.metac.metac_bot_count} started at least {LATE_START_MIN_WEEKS} weeks into it"


def _pro_weight_sentence(pro_weighted_question_count: int) -> str:
    if pro_weighted_question_count == 0:
        return "Pro questions all have a weight of 1."
    return f"{pro_weighted_question_count} Pro {plural(pro_weighted_question_count, 'question')} also had a weight below 1."


def _methodology_appendix(results: SeasonResults, config: SeasonConfig) -> str:
    overview = results.overview
    head_to_head = results.head_to_head
    metac = results.metac
    window_text = _bot_window_text(config.bot_question_windows)
    interval = config.bot_question_interval_hours
    launch_text = (
        f"one at a time, every {format_compact(interval)} hours"
        if interval is not None
        else fill_me('how bot questions launched, e.g. "one at a time, every 2 hours"')
    )
    closes_before_next = interval is not None and all(
        window.open_hours is not None and window.open_hours <= interval for window in config.bot_question_windows
    )
    pro_days = _number_or_fill_me(config.pro_forecast_window_days, "days Pros had to forecast each question, e.g. 3")
    top_model, top_search, _variance_label = split_baseline_name(metac.top_baseline.name)
    examples = [
        f"* {QUESTION_TYPE_DESCRIPTIONS.get(example.question_type, example.question_type)}, e.g., "
        f"[{example.title}]({example.url})"
        + (f" Options: {', '.join(example.options)}." if example.options else "")
        + f" (Resolved as {example.resolution_text}.)"
        for example in results.example_questions
    ]
    bot_tournament_text = (
        f"**Bot tournament:** {overview.non_metac_bot_count} bots and {overview.metac_bot_count} in-house Metac Bots "
        f"(running different models) participated in a tournament of {overview.bot_question_count} questions. "
        f"{overview.bot_annulled_count} {plural(overview.bot_annulled_count, 'question was', 'questions were')} "
        f"annulled, leaving {overview.bot_scored_count} scored questions. Questions receiving scores are those that "
        "were not annulled and didn't have resolution issues. The scored questions were "
        f"{_type_mix(overview.bot_type_counts, with_percentages=True)}. The \"bot-only\" portion of the questions "
        f"consisted of {overview.bot_only_scored_count} scored questions, while the Pro vs. bot comparison used "
        f"{head_to_head.question_count}. Each week, we launched a new set of questions {launch_text}. Each question "
        f"stayed open for {window_text}{', so it closed before the next one launched' if closes_before_next else ''}. "
        "This ensured forecasting occurred at all times of the day and night. Bots submitted their "
        "forecasts using our API. Questions spanned an extensive range of topics, including geopolitics, economics, "
        "elections, sports, tech, etc. Bot makers were allowed to update their bots over time (e.g., incorporate a "
        f"newer model, or iterate on prompts), but Metac Bots were not updated during the season{_late_metac_clause(results)}. "
        + _binary_resolution_sentence(overview.binary_yes_count, overview.binary_scored_count)
    )
    pro_annulled = overview.pro_annulled_count
    pro_annulled_clause = (
        f" (plus {pro_annulled} that {plural(pro_annulled, 'was', 'were')} annulled)" if pro_annulled else ""
    )
    if head_to_head.question_count == overview.pro_scored_count:
        shared_sentence = (
            "All of these also resolved cleanly in the bot tournament, which gave us "
            f"{head_to_head.question_count} questions with direct comparisons between bots and Pros."
        )
    else:
        shared_sentence = (
            f"{head_to_head.question_count} of these also resolved cleanly in the bot tournament and form the direct "
            "comparison between bots and Pros."
        )
    pro_tournament_text = (
        f"**Pro tournament:** {head_to_head.pro_member_count} Pro forecasters predicted "
        f"{overview.pro_scored_count} scored questions{pro_annulled_clause}. {shared_sentence} Pro vs. bot "
        f"questions consisted of {_type_mix(head_to_head.question_type_counts, with_percentages=False)} questions. "
        "Pro questions were randomly sampled from each week's upcoming bot questions (with some constraints to space "
        f"out schedules). Pros had {pro_days} days to forecast each question before it closed. The Pro tournament "
        "was private and hidden throughout the tournament, which prevented bots from copying the Pros' forecasts. "
        "Pros, however, could see each other's forecasts and could communicate in the comment section of each "
        f"question. Bots would forecast in the last {window_text} of the time that the Pros got to see the "
        "question, though the bot forecasts were hidden."
        + (" Appendix C lists shared questions where the two closed at different times." if overview.close_time_differences else "")
    )
    metac_text = (
        f"**Metac Bots:** Metaculus ran {metac.metac_bot_count} in-house Metac Bots. {len(metac.baseline_rows)} of "
        "them used the same prompt but a different LLM, and occasionally a different method for doing research; the "
        "others tested different setups. In general, the same-prompt bots do an internet search, run a ~30-line "
        "prompt (Appendix B), forecast 5 times with that prompt, and submit an aggregation of these forecasts. Each "
        f"bot's name indicates which model and research provider it used. For instance, \"{metac.top_baseline.name}\" "
        f"uses {top_model} as its model and {top_search} as its research source."
    )
    shared_weighted_count = len(overview.shared_question_weights_below_1)
    shared_clause = f", {shared_weighted_count} of them also asked of the Pros" if shared_weighted_count else ""
    weights_text = (
        "**Weights:** Some bot tournament questions were given weights to counteract possible correlated resolutions "
        f"between questions ({overview.weighted_question_count} of the {overview.bot_scored_count} scored bot "
        f"questions this season{shared_clause}). {_pro_weight_sentence(overview.pro_weighted_question_count)} The "
        "Pro vs bot comparison uses the Pro questions' weights. The weight, when applied, is noted in the "
        "\"Question Info\" section of the question's listing on the Metaculus website."
    )
    removed_count = overview.bot_scored_count - overview.bot_only_scored_count
    example_reminder = [
        fill_me(
            "these examples were picked automatically (the shortest title of each type). Swap in more "
            "representative ones with example_question_ids in season_config.py"
        )
    ] if any(not example.was_pinned for example in results.example_questions) else []
    return "\n\n".join(
        [
            "## Appendix A: Methodology",
            "To test how well bots are doing relative to each other and relative to humans, we set up a forecasting "
            "tournament for bots and one for Pros, where the Pros' tournament overlapped with "
            f"{head_to_head.question_count} of the {overview.bot_scored_count} scored bot questions. The Pro forecasts "
            "were hidden from bots to prevent them from being copied. Each tournament launched a series of questions "
            "that were resolved sometime during the season, and participants were asked to assign probabilities to "
            "outcomes. We then used our established scoring rule to evaluate participants. During the analysis, we "
            "aggregated the predictions of groups of forecasters to create \"teams.\"",
            "### Data collection",
            "\n".join(f"* {text}" for text in (bot_tournament_text, pro_tournament_text, metac_text)),
            "### Question types",
            f"The questions covered {len(results.example_questions)} question types:",
            "\n".join(examples),
            *example_reminder,
            "### Scoring",
            f"For a deeper understanding of scoring, we recommend reading our [scoring FAQ]({SCORING_FAQ_URL}). We "
            "primarily use spot peer scores and head-to-head scores in this analysis.",
            "\n".join(
                [
                    "* **Peer score:** The Peer score compares the relative performance of a forecaster (or team) vs. "
                    "all others on a question, over the whole question lifetime. Unlike Baseline or Brier scores, it "
                    "adjusts for the difficulty of a question based on the performance of others. The score is "
                    "zero-sum (i.e., the sum of all participants' peer scores is 0). A higher score indicates greater "
                    "relative accuracy. A score of 0 corresponds to average accuracy.",
                    "* **Spot peer score:** Spot peer scores are the same as peer scores, except evaluated only at a "
                    "single point in time, using the forecasters' standing predictions at that time (rather than "
                    "averaging scores over the question lifetime). For the bot and Pro tournaments, spot peer scores "
                    "were evaluated when questions closed.",
                    "* **Head-to-head score:** Head-to-head scores are the same as spot peer scores, except they "
                    "compare only two participants. For example, if comparing the bot team and Pro team, the "
                    "head-to-head score for a question that resolves Yes is 100 × ln(bot_team_prediction / "
                    "pro_team_prediction).",
                    f"* {weights_text}",
                    "* **Typical probability on the outcome:** On a binary question that resolved Yes, a forecaster's "
                    "probability on the outcome is the P(Yes) it forecast; if it resolved No, it is 1 − P(Yes). A "
                    "forecaster's typical probability on the outcome is the geometric mean of these across the binary "
                    "questions it forecast, exp(Σ wᵢ ln pᵢ / Σ wᵢ), using the same question weights wᵢ as the "
                    "leaderboards. It is the exponentiated log score, so it rewards honest forecasts, and it tracks "
                    "peer scores closely. We use it instead of the share of questions forecast on the right side of "
                    "the base rate, which rewards pushing forecasts past the base rate and ignores confidence. Only "
                    "binary questions are used, because for multiple-choice and numeric questions the probability on "
                    "the outcome depends on the number of options or the width of the numeric bins. Confidence "
                    "intervals use a t-distribution on the log probabilities.",
                    "* **Ranking by sum:** We rank participants on leaderboards by the sum of their spot peer scores "
                    "to reduce the noise caused by bots that got high averages by getting lucky on only a few "
                    "questions. Ranking by sum incentivizes forecasting on more questions, which reduces our "
                    "uncertainty (i.e., gives tighter error bars).",
                    "* **Statistical tests:** Confidence intervals are 95% intervals using a t-distribution. We test "
                    "whether one team outperformed the other with a one-sided t-test on the per-question head-to-head "
                    "scores, and call a result significant when p < 0.05. To compare a season with the previous one, "
                    "we use a two-sample z-test on the two seasons' means and confidence intervals.",
                ]
            ),
            "### Team selection and aggregation",
            "In the analyses above, we sometimes aggregate a group of forecasters to create a \"team.\" To aggregate "
            "predictions, we take the median for binary predictions, the normalized median for multiple-choice "
            "predictions, and an average (aka mixture of distributions) for numeric and discrete predictions. This is "
            "the same as the default aggregation method for the Community Prediction shown on Metaculus, except the "
            "Community Prediction also weighs predictions by recency. Note that occasionally a member of a team "
            "misses a question, and thus is not included in the team aggregate prediction for that question. Below "
            "is how the compositions of the teams were decided:",
            "\n".join(
                [
                    f"* **Bot team:** We took the {overview.bot_scored_count} scored tournament questions, then "
                    f"removed the {removed_count} questions that were also asked of the Pros, to get "
                    f"{overview.bot_only_scored_count} bot-only questions. We compute the bots' spot peer scores on "
                    "these questions, and we rank the bots on a leaderboard by the sum of their scores. The top "
                    f"{len(head_to_head.bot_team_members)} bots from this leaderboard are our bot team. We exclude the "
                    "Pro questions to avoid bias in the Pro-bot comparison. By using the bot-only questions to "
                    "determine the team, we effectively run a \"qualification tournament\" to choose bot team members "
                    "(similar to how the Pros were chosen based on their performance on past tournaments on "
                    "Metaculus).",
                    f"* **Pro team:** The Pro team includes all {head_to_head.pro_member_count} Pros who participated "
                    "in the Pro tournament. Metaculus selects Pro Forecasters based on their excellent forecasting "
                    "track record on the platform and for their history of clearly describing their rationales.",
                ]
            ),
            "### Notes on data collection",
            "* **Very bad numeric forecasts can be rejected:** Numeric questions have an upper and lower bound that "
            "forecasters on the site are generally expected to forecast within. To ensure numerical stability (and "
            "non-astronomically-negative scores), our API rejects predictions that put no or very little probability "
            "mass in bounds. Occasionally, some Metac Bots would forecast out of bounds in this way (especially those "
            "running lower-quality models), and it is expected that some participants encountered this error as "
            "well. Our forecasting user interface does not let humans make forecasts that bad either, so in this "
            "sense, they're on equal footing. However, it does mean that bots are not punished for their very worst "
            "numeric forecasts.\n"
            "* **Season-specific notes:** See Appendix C.",
            "This analysis generally follows the methodology we laid out at the beginning of the tournament series, "
            "but now uses a simplified bot team selection algorithm we started using in Q1 2025 (as described above).",
        ]
    )


def _prompt_appendix() -> str:
    return "\n\n".join(
        [
            "## Appendix B: Metac Bot prompt",
            "For reference, here is the prompt that the same-prompt Metac Bots used for binary questions. The code "
            "links above include the numeric and multiple-choice versions.",
            f"```\n{METAC_BOT_PROMPT}\n```",
        ]
    )


def _close_time_note(differences: tuple[CloseTimeDifference, ...], shared_question_count: int) -> list[str]:
    notes = []
    for differences_one_way, direction, advantaged in (
        ([difference for difference in differences if difference.minutes_bot_closed_after_pro > 0], "after", "bots"),
        ([difference for difference in differences if difference.minutes_bot_closed_after_pro < 0], "before", "Pros"),
    ):
        if not differences_one_way:
            continue
        minutes = sorted({abs(difference.minutes_bot_closed_after_pro) for difference in differences_one_way})
        gap_text = format_duration(minutes[0]) if len(minutes) == 1 else f"{format_duration(minutes[0])} to {format_duration(minutes[-1])}"
        first_day, last_day = differences_one_way[0].bot_close.date(), differences_one_way[-1].bot_close.date()
        dates = f"on {month_day(first_day)}, {first_day.year}" if first_day == last_day else f"between {format_date_range(first_day, last_day)}"
        notes.append(
            f"On {len(differences_one_way)} of the {shared_question_count} shared questions (closing {dates}), "
            f"the bot question closed {gap_text} {direction} the Pro question, so {advantaged} had extra time to "
            "forecast. The comparison scores both teams at the later close."
        )
    return notes


def _season_disclosures(results: SeasonResults, config: SeasonConfig) -> list[str]:
    overview = results.overview
    disclosures = list(config.data_notes)
    if len(config.bot_question_windows) > 1:
        disclosures.append(
            "The bot question window changed during the season: questions stayed open for "
            f"{_bot_window_text(config.bot_question_windows)}."
        )
    disclosures.extend(_close_time_note(overview.close_time_differences, results.head_to_head.question_count))
    shared_weights = overview.shared_question_weights_below_1
    if shared_weights:
        weight_range = (
            format_compact(shared_weights[0])
            if shared_weights[0] == shared_weights[-1]
            else f"{format_compact(shared_weights[0])} to {format_compact(shared_weights[-1])}"
        )
        disclosures.append(
            f"{len(shared_weights)} of the shared questions had weights below 1 in the bot tournament ({weight_range}), "
            "but the Pro vs bot comparison uses the Pro questions' weights"
            + (" of 1." if overview.pro_weighted_question_count == 0 else ".")
        )
    return disclosures


def _data_notes_appendix(results: SeasonResults, config: SeasonConfig) -> str:
    overview = results.overview
    head_to_head = results.head_to_head
    checks = [
        f"{overview.bot_annulled_count} bot questions and {overview.pro_annulled_count} Pro questions were annulled "
        "or unscored. Mention any with notable problems, and any Pro/bot questions that had to be force-matched "
        "(aib_analysis/data_structures/problem_questions.py).",
        "Numeric questions converted to discrete after launch (e.g. to stop \"porcupine\" forecasts that spike on "
        "whole numbers), and which questions.",
        "Bots (including Metac Bots) that missed questions because of provider outages or formatting errors.",
        "Whether these simulated scores match the official leaderboard (ranks and sums), and why if not.",
    ]
    if overview.non_pro_accounts_in_pro_tournament or head_to_head.non_pro_member_count:
        checks.append(
            f"The Pro tournament has {overview.non_pro_accounts_in_pro_tournament} scored non-Pro "
            f"{plural(overview.non_pro_accounts_in_pro_tournament, 'account')}, and the Pro team aggregate includes "
            f"{head_to_head.non_pro_member_count} non-Pro {plural(head_to_head.non_pro_member_count, 'member')}. "
            "Decide whether to disclose or fix this."
        )
    disclosures = _season_disclosures(results, config)
    blocks = ["## Appendix C: Season data notes"]
    if disclosures:
        blocks.append("\n".join(f"* {item}" for item in disclosures))
    blocks.append(
        fill_me(
            "add anything else unusual about this season's data in short bullets, and delete what doesn't apply. "
            "Things to check:"
        )
    )
    blocks.append("\n".join(f"* {item}" for item in checks))
    return "\n\n".join(blocks)
