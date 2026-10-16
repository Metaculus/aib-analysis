"""Precommitted sentences for the write-up.

Every sentence that reports a result is chosen by a fixed rule from the numbers, so the
framing is decided before anyone sees a season's results. Wording follows past FutureEval
posts. Each function covers both directions (Pros ahead or bots ahead).
"""

from __future__ import annotations

import itertools
import math
from datetime import date

from aib_analysis.pro_vs_bot_writeup.results import (
    PROMISING_MIN_SHARE_OF_QUESTIONS,
    SIGNIFICANCE_LEVEL,
    TOP_N,
    BotFieldAccuracy,
    FieldSummary,
    HeadToHeadResult,
    IndividualResults,
    LeaderboardRow,
    MetacResults,
    PercentCorrectResults,
    SeasonResults,
    SeasonTrend,
    SeasonTrendPoint,
    TeamAccuracy,
    TeamSizeResult,
    split_baseline_name,
)

FILL_ME = "[===Fill me===]"
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
HEAD_TO_HEAD_REMINDER = (
    "(Reminder: 0 would mean bots and Pros did equally well, and a positive number would mean bots beat Pros.)"
)
INDIVIDUAL_COMPARISON_CAVEAT = (
    "Note that though this type of comparison is interesting, unlike the Pro vs bot team comparison, the more "
    "individuals we compare, the higher the chance of one bot or Pro getting incorrectly ranked due to luck."
)


def fill_me(reminder: str) -> str:
    return f"{FILL_ME} _(Reminder: {reminder})_"


def format_score(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def format_compact(value: float) -> str:
    """Up to 2 decimals without trailing zeros, matching how past posts quote season results."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def format_ci(ci: tuple[float, float], compact: bool = False) -> str:
    formatter = format_compact if compact else format_score
    return f"[{formatter(ci[0])}, {formatter(ci[1])}]"


def format_p(p_value: float) -> str:
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def format_sum(value: float) -> str:
    return f"{value:,.1f}"


def ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def count_word(count: int, capitalize: bool = False) -> str:
    word = COUNT_WORDS[count] if 0 <= count < len(COUNT_WORDS) else str(count)
    return word[0].upper() + word[1:] if capitalize else word


def join_names(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    return singular if count == 1 else (plural_form or f"{singular}s")


def percent(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%"


def headline_takeaway(head_to_head: HeadToHeadResult, pro_count: int, bot_team_size: int) -> str:
    pro_team = f"our team of {pro_count} Metaculus Pro Forecasters"
    bot_team = f"the top-{bot_team_size} bot team"
    margin = format_score(abs(head_to_head.bot_team_mean))
    p_text = format_p(head_to_head.leader_p_value)
    lower, upper = (format_score(value) for value in head_to_head.ci)
    if head_to_head.leader == "pros":
        winner, loser, winner_label, loser_label, side = pro_team, bot_team, "Pro forecasters", "bots", "below"
        short_winner = "Pros"
    else:
        winner, loser, winner_label, loser_label, side = bot_team, pro_team, "Bots", "Pro forecasters", "above"
        short_winner = "bots"
    edge = f"{winner[0].upper()}{winner[1:]} outperformed {loser} by an average of {margin} head-to-head points per question."
    if head_to_head.evidence == "strong":
        return (
            f"**{winner_label} significantly outperformed {loser_label}:** {edge} This edge is statistically "
            f"significant (one-sided {p_text}), and the bot team's 95% confidence interval ({lower} to {upper}) "
            f"sits entirely {side} zero."
        )
    if head_to_head.evidence == "weak":
        return (
            f"**{winner_label} outperformed {loser_label}, but the evidence is weak:** {edge} A one-sided t-test "
            f"finds this edge significant ({p_text}), but the bot team's 95% confidence interval ({lower} to "
            f"{upper}) still includes zero."
        )
    return (
        f"**{winner_label} beat {loser_label} but without significance:** {edge} But this edge is small relative "
        f"to the question-to-question variation and not statistically significant (one-sided {p_text}), so this "
        f"sample doesn't give us enough evidence to confirm {short_winner} were genuinely ahead this season."
    )


def head_to_head_paragraph(head_to_head: HeadToHeadResult) -> str:
    lower, upper = (format_score(value) for value in head_to_head.ci)
    sentences = [
        "We then calculated the head-to-head scores of the two teams, along with 95% confidence intervals "
        "using a t-distribution.",
        f"The bot team's average head-to-head score was {format_score(head_to_head.bot_team_mean)} with a 95% "
        f"confidence interval of {lower} to {upper} over {head_to_head.question_count} questions.",
    ]
    if head_to_head.leader == "pros":
        sentences.append("The negative score indicates that the bot team had lower accuracy than the Pro team.")
        leader, trailer = "Pros", "bots"
    else:
        sentences.append("The positive score indicates that the bot team had higher accuracy than the Pro team.")
        leader, trailer = "bots", "Pros"
    p_text = format_p(head_to_head.leader_p_value)
    if head_to_head.evidence == "strong":
        sentences.append(
            f"A one-sided t-test demonstrates that {leader} significantly outperformed {trailer} ({p_text}), and "
            "the 95% confidence interval does not include zero."
        )
    elif head_to_head.evidence == "weak":
        sentences.append(
            f"A one-sided t-test demonstrates that {leader} outperformed {trailer} ({p_text}), though the 95% "
            "confidence interval still includes zero, so the evidence is weak."
        )
    else:
        sentences.append(
            f"A one-sided t-test did not demonstrate that {leader} outperformed {trailer} to a statistically "
            f"significant degree ({p_text}). The 95% confidence interval includes zero."
        )
    sentences.append(question_wins_sentence(head_to_head))
    return " ".join(sentences)


def question_wins_sentence(head_to_head: HeadToHeadResult) -> str:
    total = head_to_head.question_count
    ties = f", with {head_to_head.ties} {plural(head_to_head.ties, 'tie')}" if head_to_head.ties else ""
    sentence = (
        f"Pros scored better on {head_to_head.pro_wins} of the {total} questions "
        f"({percent(head_to_head.pro_wins, total)}) and bots on {head_to_head.bot_wins} "
        f"({percent(head_to_head.bot_wins, total)}){ties}."
    )
    if head_to_head.average_pro_win_margin is not None and head_to_head.average_bot_win_margin is not None:
        sentence += (
            f" The average winning margin was {format_score(head_to_head.average_pro_win_margin, 1)} points for "
            f"Pros and {format_score(head_to_head.average_bot_win_margin, 1)} for bots."
        )
    return sentence


def season_result(point: SeasonTrendPoint) -> str:
    """A season is a clear win when its 95% CI sits entirely on one side of zero."""
    if point.ci[1] < 0:
        return "Pros clearly ahead"
    if point.ci[0] > 0:
        return "Bots clearly ahead"
    return "No clear winner"


def clear_wins_sentence(trend: SeasonTrend) -> str:
    pro_wins = [point.season_label for point in trend.points if season_result(point) == "Pros clearly ahead"]
    bot_wins = [point.season_label for point in trend.points if season_result(point) == "Bots clearly ahead"]
    clauses = []
    if pro_wins:
        clauses.append(f"Pros won clearly in {join_names(pro_wins)}")
    if bot_wins:
        clauses.append(f"{'bots' if clauses else 'Bots'} won clearly in {join_names(bot_wins)}")
    if not clauses:
        return "No season had a clear winner."
    sentence = " and ".join(clauses)
    if len(pro_wins) + len(bot_wins) < len(trend.points):
        sentence += "; the other seasons had no clear winner"
    return f"{sentence}."


def season_change_clause(trend: SeasonTrend) -> str | None:
    previous = trend.previous
    if previous is None or trend.change_p_value is None:
        return None
    move = (
        f"The bot team's score went from {format_compact(previous.mean)} in {previous.season_label} to "
        f"{format_compact(trend.current.mean)} this season"
    )
    if trend.change_p_value < SIGNIFICANCE_LEVEL:
        return f"{move}, a statistically significant change ({format_p(trend.change_p_value)})."
    return f"{move}, a change too small to be statistically significant ({format_p(trend.change_p_value)})."


def format_date_range(start: date, end: date) -> str:
    if start.year == end.year:
        return f"{start:%B} {start.day} and {end:%B} {end.day}, {end.year}"
    return f"{start:%B} {start.day}, {start.year} and {end:%B} {end.day}, {end.year}"


def trend_takeaway(trend: SeasonTrend) -> str:
    current = trend.current
    previous = trend.previous
    current_text = f"{format_compact(current.mean)} this season"
    if previous is None or trend.change_p_value is None:
        return (
            f"**The bot team scored {format_compact(current.mean)} against Pros:** The bot team's head-to-head "
            f"score against Pros was {current_text}. {HEAD_TO_HEAD_REMINDER} {clear_wins_sentence(trend)}"
        )
    if trend.change_p_value >= SIGNIFICANCE_LEVEL:
        heading = f"No clear change in the bot team's standing since {previous.season_label}"
    elif current.mean > previous.mean:
        heading = f"The bot team improved clearly relative to Pros since {previous.season_label}"
    else:
        heading = f"The bot team fell back relative to Pros since {previous.season_label}"
    earlier = ", ".join(f"{format_compact(point.mean)} in {point.season_label}" for point in trend.points[:-1])
    return (
        f"**{heading}:** The bot team's head-to-head score against Pros was {earlier}, and {current_text}. "
        f"{HEAD_TO_HEAD_REMINDER} {clear_wins_sentence(trend)} {season_change_clause(trend)}"
    )


def trend_paragraph(trend: SeasonTrend) -> str:
    sentences = [
        "Below are the bot team's head-to-head scores against the Pro team in each season, with 95% confidence "
        "intervals. A score of zero means the teams' performance was equal. Negative means the Pros performed "
        "better, and positive means the bots performed better. We count a season as a clear win when its 95% "
        "confidence interval is entirely on one side of zero.",
        clear_wins_sentence(trend),
    ]
    change_clause = season_change_clause(trend)
    if change_clause:
        sentences.append(change_clause)
    if trend.skipped_seasons:
        sentences.append(
            f"{join_names(list(trend.skipped_seasons))} compared bots with the Metaculus community prediction "
            f"instead of Pros, so {'it is' if len(trend.skipped_seasons) == 1 else 'they are'} not shown."
        )
    return " ".join(sentences)


def individuals_takeaway(individuals: IndividualResults) -> str:
    pro_count = individuals.pro_count
    pros_above_all_bots = individuals.pros_above_best_bot
    pros_in_top = individuals.pros_in_top_10
    setup = "We created a leaderboard where both Pros and bots competed together as individuals."
    if pros_above_all_bots == pro_count:
        body = (
            f"**Every individual Pro beat every individual bot:** {setup} In this comparison, the top {pro_count} "
            "places would all be taken by Pros."
        )
    elif pros_above_all_bots == 0:
        body = (
            f"**A bot topped the combined leaderboard of Pros and bots:** {setup} In this comparison, "
            f"{individuals.best_bot.name} would take 1st place, and Pros would take {pros_in_top} of the top "
            f"{TOP_N} places."
        )
    else:
        body = (
            f"**{count_word(pros_above_all_bots, capitalize=True)} of {count_word(pro_count)} individual Pros beat "
            f"every individual bot:** {setup} In this comparison, {pros_in_top} of the top {TOP_N} places would be "
            "taken by Pros."
        )
    return f"{body} {INDIVIDUAL_COMPARISON_CAVEAT}"


def individuals_paragraph(individuals: IndividualResults, shown_count: int) -> str:
    sentences = [
        "Pros and bots also competed together as individuals on the "
        f"{individuals.question_count} overlapping questions. Below is this leaderboard ordered by sum of spot peer "
        f"scores. The top {shown_count} participants are shown (with human Pros anonymized).",
        f"If Pros and bots competed together, Pros would take {individuals.pros_in_top_10} of the top {TOP_N} places.",
    ]
    bots_above = individuals.bots_above_lowest_pro
    if bots_above:
        sentences.append(f"{bots_above} {plural(bots_above, 'bot')} placed above at least one Pro.")
    else:
        sentences.append("Every Pro placed above every bot.")
    sentences.append(
        "This comparison is provided for reference, with the knowledge that the Pro team vs bot team comparison "
        "is more rigorous."
    )
    return " ".join(sentences)


def _team_size_label(result: TeamSizeResult) -> str:
    return f"all {result.team_size} bots" if result.is_all_bots else f"the top {result.team_size} {plural(result.team_size, 'bot')}"


def team_size_takeaway(team_sizes: list[TeamSizeResult]) -> str:
    best = max(team_sizes, key=lambda result: result.mean)
    single_bot = next((result for result in team_sizes if result.team_size == 1), None)
    largest = team_sizes[-1]
    if best.team_size == 1:
        heading = "The top bot alone beat every larger bot team"
    elif best.is_all_bots:
        heading = "Aggregating every bot gave the best bot team"
    else:
        heading = f"Bot team aggregation peaked at {best.team_size} bots"
    comparisons = []
    if single_bot is not None and single_bot is not best:
        comparisons.append(f"{format_score(single_bot.mean)} for the top bot alone")
    if largest is not best:
        comparisons.append(f"{format_score(largest.mean)} for a team of {_team_size_label(largest)}")
    sentence = f"**{heading}:** A team of {_team_size_label(best)} scored {format_score(best.mean)} against Pros"
    sentence += f", compared with {join_names(comparisons)}." if comparisons else "."
    if largest is not best and largest.ci is not None and largest.ci[1] < best.mean:
        sentence += " Merely increasing the size of a bot team does not automatically improve the ensemble's forecasting."
    return sentence


def team_size_paragraph(team_sizes: list[TeamSizeResult]) -> str:
    best = max(team_sizes, key=lambda result: result.mean)
    single_bot = next((result for result in team_sizes if result.team_size == 1), None)
    largest = team_sizes[-1]
    best_ci = f" {format_ci(best.ci)}" if best.ci is not None else ""
    sentences = [
        "We ran a sensitivity analysis of the bot team's head-to-head score against Pros for bot teams of "
        "different sizes. The bot-only questions are used to determine which bots are included in a team of size "
        "X (i.e., the same method used to select the main bot team).",
        f"The highest point estimate came from a team of {_team_size_label(best)} ({format_score(best.mean)}{best_ci}).",
    ]
    if single_bot is not None and single_bot is not best:
        sentences.append(f"The top bot by itself scored {format_score(single_bot.mean)}.")
    if largest is not best:
        largest_ci = f" {format_ci(largest.ci)}" if largest.ci is not None else ""
        sentences.append(f"A team of {_team_size_label(largest)} scored {format_score(largest.mean)}{largest_ci}.")
    sentences.append("Neighboring team sizes share most of their members, so small differences between them are mostly noise.")
    return " ".join(sentences)


def top_baseline_takeaway(metac: MetacResults) -> str:
    top = metac.top_baseline
    standing = metac.current_standing
    sentence = (
        f"**The top-ranked same-prompt Metac Bot was {top.name}:** Metaculus ran "
        f"{len(metac.baseline_rows)} bots that used the same prompt but different LLMs or search setups. "
        f"{top.name} finished first among these, and {ordinal(standing.rank)} of {standing.bot_count} overall in "
        "the bot tournament."
    )
    close_rows = metac.close_to_top()
    if close_rows:
        sentence += f" {close_to_top_sentence(close_rows)}"
    higher_average_rows = metac.higher_average_on_fewer_questions()
    if higher_average_rows:
        standout = higher_average_rows[0]
        sentence += (
            f" {standout.name} had a higher average ({format_score(standout.average_score)}) but forecast on "
            f"{standout.question_count} questions{_late_start_clause(metac, standout)}."
        )
    return sentence


def close_to_top_sentence(close_rows: list[LeaderboardRow], max_names: int = 3) -> str:
    names = [row.name for row in close_rows[:max_names]]
    remaining = len(close_rows) - len(names)
    if remaining:
        names.append(f"{remaining} {plural(remaining, 'other')}")
    return (
        f"However, {join_names(names)} scored within the top bot's 95% confidence interval, so it's hard to declare "
        'any one setup "the best."'
    )


def _late_start_clause(metac: MetacResults, row: LeaderboardRow) -> str:
    weeks_late = round(metac.days_after_season_start(row.name) / 7)
    if weeks_late < 2:
        return ""
    return f", starting {weeks_late} weeks into the season"


def higher_average_on_fewer_questions_sentence(metac: MetacResults) -> str | None:
    rows = metac.higher_average_on_fewer_questions()
    if not rows:
        return None
    descriptions = [
        f"{row.name} ({format_score(row.average_score)} on {row.question_count} questions{_late_start_clause(metac, row)})"
        for row in rows
    ]
    return (
        f"{join_names(descriptions)} {plural(len(rows), 'had a higher average', 'had higher averages')} than "
        f"{metac.top_baseline.name} but forecast on fewer questions, so {plural(len(rows), 'it ranks', 'they rank')} "
        "lower by sum. Averages over fewer questions are less certain."
    )


def _top_share(rank: int, bot_count: int) -> int:
    return max(1, math.ceil(100 * rank / bot_count))


def baseline_history_takeaway(metac: MetacResults) -> str:
    history = metac.baseline_history
    current = history[-1]
    current_share = _top_share(current.rank, current.bot_count)
    placements = [
        f"{ordinal(point.rank)} of {point.bot_count} in {point.season_label}" for point in history[:-1]
    ]
    current_text = f"{ordinal(current.rank)} of {current.bot_count} this season ({current.bot_name})"
    heading = f"The best same-prompt Metac Bot placed in the top {current_share}% of bots"
    if len(history) >= 2:
        previous = history[-2]
        previous_share = _top_share(previous.rank, previous.bot_count)
        if current_share < previous_share:
            heading += f", up from the top {previous_share}% in {previous.season_label}"
        elif current_share > previous_share:
            heading += f", down from the top {previous_share}% in {previous.season_label}"
        else:
            heading += f", the same as in {previous.season_label}"
    return (
        f"**{heading}:** The best Metac Bot using our standard single prompt placed "
        f"{join_names(placements + [current_text])}."
    )


def _separated_top_10_pairs(field: FieldSummary) -> int:
    rows_with_ci = [row for row in field.top_10 if row.ci is not None]
    return sum(
        1
        for higher, lower in itertools.combinations(rows_with_ci, 2)
        if higher.ci[0] > lower.ci[1] or lower.ci[0] > higher.ci[1]
    )


def top_10_spread_takeaway(field: FieldSummary) -> str:
    averages = [row.average_score for row in field.top_10]
    spread = (
        f"Their average spot peer scores ranged from {format_score(min(averages), 1)} to "
        f"{format_score(max(averages), 1)}"
    )
    if field.top_10_average_ci_width is not None:
        spread += (
            f", and the average confidence interval of the top {TOP_N} bots was "
            f"~{format_score(field.top_10_average_ci_width, 1)} points wide"
        )
    separated_pairs = _separated_top_10_pairs(field)
    if separated_pairs == 0:
        return (
            f"**No one leads the pack:** None of the top {TOP_N} bots scored well enough to clearly differentiate "
            f"from each other. {spread}."
        )
    pair_count = len(field.top_10) * (len(field.top_10) - 1) // 2
    return (
        f"**Some top-{TOP_N} bots are clearly separated from others:** {spread}. {separated_pairs} of the "
        f"{pair_count} pairs of top-{TOP_N} bots have non-overlapping 95% confidence intervals."
    )


def field_paragraph(field: FieldSummary, chart_min_questions: int, hidden_count: int) -> str:
    sentences = [
        f"The first chart below shows every bot with at least {chart_min_questions} scored questions in leaderboard "
        "order, with its average spot peer score and 95% confidence interval.",
        f"Among the {field.eligible_count} bots that forecast on at least half of the "
        f"{field.scored_question_count} questions, the highest average was "
        f"{format_score(field.best_eligible.average_score)} ({field.best_eligible.name}), the median was "
        f"{format_score(field.median_eligible_average)}, and the lowest was "
        f"{format_score(field.worst_eligible.average_score)}.",
    ]
    if field.uniform_reference is not None:
        sentences.append(
            "For reference, a Metaculus bot that always forecasts a uniform distribution (50% on every binary "
            f"question) averaged {format_score(field.uniform_reference.average_score)}."
        )
    if field.promising:
        promising_count = len(field.promising)
        names = [f"{row.name} ({row.question_count} questions)" for row in field.promising]
        sentences.append(
            f"{count_word(promising_count, capitalize=True)} {plural(promising_count, 'bot')} (in green) forecast on "
            f"fewer than half of the questions but averaged more than the {ordinal(TOP_N)}-place bot "
            f"({format_score(field.tenth_place.average_score)}): {join_names(names)}. With more questions, "
            f"{'it' if promising_count == 1 else 'some of these'} might prove to be strong. Lucky streaks matter "
            f"more over fewer questions, so we don't highlight bots with fewer than {field.promising_min_questions} "
            f"questions ({format_whole_percent(PROMISING_MIN_SHARE_OF_QUESTIONS)} of the total)."
        )
    if hidden_count:
        sentences.append(f"({hidden_count} {plural(hidden_count, 'bot')} with fewer than {chart_min_questions} scored questions are not shown.)")
    return " ".join(sentences)


def field_band_paragraph() -> str:
    return (
        "The second chart plots each bot's average against how many questions it forecast. The shaded band shows "
        "how much luck alone moves an average: a bot exactly as good as the typical bot in the field (a true average "
        "of 0) would land inside the band 95% of the time, given how much bot scores vary from question to "
        "question. The band is wide on the left because an average over a few questions swings a lot, and narrow on "
        "the right because more questions even out the luck. Bots above the band did better than luck alone "
        "explains. Lines show 95% confidence intervals for the highlighted bots."
    )


def percent_correct_paragraph(results: PercentCorrectResults) -> str:
    at_50 = results.at_50
    at_yes_rate = results.at_yes_rate
    gaming = (
        "Percent correct isn't a proper scoring rule, so it can be gamed. Forecasting 1% on every question would be "
        f"right on {format_whole_percent(at_50.always_no)} of binary questions"
    )
    if at_50.share_of_bots_below_always_no > 0:
        gaming += f", better than {format_whole_percent(at_50.share_of_bots_below_always_no)} of regular bots"
    gaming += (
        ", while putting almost no probability on every question that resolved Yes. It also ignores confidence: a "
        "51% forecast and a 99% forecast count the same. Moving the cutoff to the Yes rate doesn't fix this; always "
        f"forecasting 1% still scores {format_whole_percent(at_yes_rate.always_no)}."
    )
    return " ".join(
        [
            "Many readers know percent correct: the share of binary questions where a forecast was on the right side "
            "of 50%. The chart shows it for the same forecasters, and again with the cutoff moved to the season's "
            "Yes rate.",
            gaming,
            f"By this measure, the top bot was right on {format_whole_percent(at_50.top_bot)} of binary questions, "
            f"the median bot on {format_whole_percent(at_50.median)}, and the lowest-scoring bot on "
            f"{format_whole_percent(at_50.worst_bot)}. On the shared questions, the Pro team was right on "
            f"{format_whole_percent(at_50.pro_team)} and the bot team on {format_whole_percent(at_50.bot_team)}.",
        ]
    )


def format_whole_percent(share: float) -> str:
    return f"{100 * share:.0f}%"


def format_percent(probability: float) -> str:
    return f"{100 * probability:.1f}%"


def accuracy_takeaway(accuracy: BotFieldAccuracy) -> str:
    return (
        "**On binary questions, the top bot gave the actual outcome a typical probability of "
        f"{format_percent(accuracy.top_bot.typical_probability)}:** That compares with "
        f"{format_percent(accuracy.median_probability)} for the median bot and "
        f"{format_percent(accuracy.worst_bot.typical_probability)} for the lowest-scoring bot, among the "
        f"{len(accuracy.eligible_bots)} bots that forecast on at least half of the questions. A coin flip scores 50%, "
        f"and always forecasting this season's Yes rate ({format_percent(accuracy.yes_rate)}) scores "
        f"{format_percent(accuracy.base_rate_probability)}. {round(100 * accuracy.share_below_base_rate)}% of these "
        "bots did worse than always forecasting the Yes rate."
    )


def accuracy_definition_paragraph() -> str:
    return (
        "Peer scores only say how a forecaster compares with the others. To show accuracy in absolute terms, we use "
        "each forecaster's typical probability on the outcome: on each binary question, take the probability it "
        "gave to the outcome that actually happened, then take the geometric mean across questions. The geometric "
        "mean counts confident misses heavily, just as peer scores do. A coin flip scores 50% and a perfect "
        "forecaster 100%."
    )


def bot_field_accuracy_paragraph(accuracy: BotFieldAccuracy) -> str:
    return (
        f"On the {accuracy.binary_question_count} binary questions in the bot tournament, the top bot "
        f"({accuracy.top_bot.name}) gave the outcome that happened a typical probability of "
        f"{format_percent(accuracy.top_bot.typical_probability)}. The median bot gave "
        f"{format_percent(accuracy.median_probability)} and the lowest-scoring bot "
        f"{format_percent(accuracy.worst_bot.typical_probability)} (among the {len(accuracy.eligible_bots)} bots that "
        f"forecast on at least half of the questions). Always forecasting this season's Yes rate of "
        f"{format_percent(accuracy.yes_rate)} would score {format_percent(accuracy.base_rate_probability)}, and "
        f"{round(100 * accuracy.share_below_base_rate)}% of these bots scored below that."
    )


def team_accuracy_paragraph(team: TeamAccuracy) -> str:
    sentences = [
        f"On the {team.binary_question_count} binary questions in the Pro comparison, the Pro team gave the actual "
        f"outcome a typical probability of {format_percent(team.pro_team_probability)} and the bot team "
        f"{format_percent(team.bot_team_probability)}."
    ]
    ratio_text = f"{team.bot_to_pro_ratio:.2f}×"
    if team.bot_to_pro_ratio_ci is None:
        sentences.append(f"The bot team's typical probability was {ratio_text} the Pro team's.")
    else:
        lower, upper = team.bot_to_pro_ratio_ci
        sentences.append(
            f"The bot team's typical probability was {ratio_text} the Pro team's (95% CI: {lower:.2f}× to {upper:.2f}×)."
        )
        if lower <= 1 <= upper:
            sentences.append("The interval includes 1 (equal accuracy), so this gap could be due to chance.")
        elif upper < 1:
            sentences.append(
                "The interval is entirely below 1, so the Pro team's edge on these questions is unlikely to be due to "
                "chance alone."
            )
        else:
            sentences.append(
                "The interval is entirely above 1, so the bot team's edge on these questions is unlikely to be due to "
                "chance alone."
            )
    sentences.append(
        f"These values depend on the question mix (the Yes rate on these questions was {format_percent(team.yes_rate)}), "
        "so compare them only with values on the same questions."
    )
    return " ".join(sentences)


def peer_score_translation_paragraph(accuracy: BotFieldAccuracy) -> str:
    return (
        "Peer scores translate into this scale. A binary peer score of +X means a forecaster gave the actual outcome "
        "about e^(X/100) times the typical probability the rest of the field did (+10 ≈ 1.11×, +20 ≈ 1.22×), and the "
        f"field's typical probability this season was {format_percent(accuracy.field_probability)}. Across all question "
        f"types, {accuracy.top_bot.name}'s average spot peer score of {format_score(accuracy.top_bot_average_peer_score)} "
        f"means it gave the actual outcome about {accuracy.top_bot_field_multiplier:.2f}× the probability the rest of "
        "the field did. (Numeric and discrete peer scores are halved, so they count double in this conversion.)"
    )


def experiment_fact_lines(results: SeasonResults) -> list[str]:
    """One line per Metac Bot experiment group, with each bot's average and 95% CI."""
    groups = results.experiment_groups
    lines: list[str] = []

    def describe(row: LeaderboardRow, label: str) -> str:
        ci_text = f" {format_ci(row.ci)}" if row.ci is not None else ""
        return f"{label} {format_score(row.average_score)}{ci_text}"

    for model, rows in groups.search_variants.items():
        variants = ", ".join(describe(row, split_baseline_name(row.name)[1]) for row in rows)
        lines.append(f"Same model ({model}), different search: {variants}")
    for standard, high in groups.reasoning_pairs:
        lines.append(f"Standard vs high reasoning: {describe(standard, standard.name)} vs {describe(high, high.name)}")
    for setup, rows in groups.variance_tests.items():
        runs = ", ".join(format_score(row.average_score) for row in rows)
        lines.append(f"Repeated runs of {setup} (shows run-to-run noise): {runs}")
    if results.metac.other_rows:
        others = ", ".join(
            f"{row.name} ({ordinal(row.rank)} of {results.metac.metac_bot_count}, {format_score(row.average_score)})"
            for row in results.metac.other_rows
        )
        lines.append(f"Other Metac Bots (not the same prompt): {others}")
    return lines


def metac_comparison_sentence(metac: MetacResults) -> str:
    top = metac.top_baseline
    sentence = (
        f"{top.name} ranked first by sum of spot peer scores, with an average of {format_score(top.average_score)}"
    )
    sentence += f" (95% CI: {format_ci(top.ci)})." if top.ci is not None else "."
    close_rows = metac.close_to_top()
    if close_rows:
        sentence += f" {close_to_top_sentence(close_rows)}"
    elif not metac.higher_average_on_fewer_questions():
        sentence += " Every other same-prompt bot's average falls below the top bot's 95% confidence interval."
    higher_average_sentence = higher_average_on_fewer_questions_sentence(metac)
    if higher_average_sentence:
        sentence += f" {higher_average_sentence}"
    return sentence
