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
    FieldSummary,
    HeadToHeadResult,
    IndividualResults,
    LeaderboardRow,
    MetacResults,
    RunnerUpComparison,
    SeasonResults,
    SeasonTrend,
    SeasonTrendPoint,
    TeamSizeResult,
    require_head_to_head_vs_main,
    split_baseline_name,
)
from aib_analysis.pro_vs_bot_writeup.season_config import (
    MULTIPLE_COMPARISONS_URL,
    NEAR_MISS_SEASONS,
    PUBLISHED_PROS_IN_INDIVIDUAL_TOP_10,
    seasons_before,
)

FILL_ME = "[===Fill me===]"
LATE_START_MIN_WEEKS = 2
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
HEAD_TO_HEAD_SCALE = "(0 would mean bots and Pros did equally well, and a positive number would mean bots beat Pros)"
MULTIPLE_COMPARISONS_LINK = f"[multiple comparison problems]({MULTIPLE_COMPARISONS_URL})"


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


def join_names(names: list[str], conjunction: str = "and") -> str:
    if len(names) <= 2:
        return f" {conjunction} ".join(names)
    return ", ".join(names[:-1]) + f", {conjunction} " + names[-1]


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    return singular if count == 1 else (plural_form or f"{singular}s")


def percent(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%"


def headline_takeaway(head_to_head: HeadToHeadResult, pro_count: int, bot_team_size: int) -> str:
    pro_team = f"our team of {pro_count} Metaculus Pro Forecasters"
    bot_team = f"the top-{bot_team_size} bot team"
    margin = format_score(abs(head_to_head.bot_team_mean))
    p_text = format_p(head_to_head.p_value)
    lower, upper = (format_score(value) for value in head_to_head.ci)
    if head_to_head.leader == "pros":
        winner, loser, winner_label, loser_label, side = pro_team, bot_team, "Pro forecasters", "bots", "below"
        short_winner = "Pros"
    else:
        winner, loser, winner_label, loser_label, side = bot_team, pro_team, "Bots", "Pro forecasters", "above"
        short_winner = "bots"
    edge = f"{winner[0].upper()}{winner[1:]} outperformed {loser} by an average of {margin} points per question."
    if head_to_head.is_significant:
        return (
            f"**{winner_label} significantly outperformed {loser_label}:** {edge} This edge is statistically "
            f"significant (two-sided {p_text}), and the bot team's 95% confidence interval ({lower} to {upper}) "
            f"sits entirely {side} zero."
        )
    return (
        f"**{winner_label} beat {loser_label} but without significance:** {edge} But this edge is small relative "
        f"to the question-to-question variation and not statistically significant (two-sided {p_text}), so this "
        f"sample doesn't give us enough evidence to confirm {short_winner} were genuinely ahead this season."
    )


def head_to_head_paragraph(head_to_head: HeadToHeadResult) -> str:
    lower, upper = (format_score(value) for value in head_to_head.ci)
    winner, loser = ("Pro team", "bot team") if head_to_head.leader == "pros" else ("bot team", "Pro team")
    margin_clause = "." if head_to_head.is_significant else ", but only marginally."
    sentences = [
        f"In the end, the {winner} beat the {loser}{margin_clause}",
        "We calculated the head-to-head scores of the two teams, along with 95% confidence intervals "
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
    p_text = format_p(head_to_head.p_value)
    if head_to_head.is_significant:
        sentences.append(
            f"A two-sided t-test demonstrates that {leader} significantly outperformed {trailer} ({p_text}), and "
            "the 95% confidence interval does not include zero."
        )
    else:
        sentences.append(
            f"A two-sided t-test did not demonstrate that {leader} outperformed {trailer} to a statistically "
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
    """A season is a significant win when its 95% CI sits entirely on one side of zero."""
    if point.ci[1] < 0:
        return "Pros significantly ahead"
    if point.ci[0] > 0:
        return "Bots significantly ahead"
    return "No clear winner"


def clear_wins_sentence(trend: SeasonTrend) -> str:
    pro_wins = [point.season_label for point in trend.points if season_result(point) == "Pros significantly ahead"]
    bot_wins = [point.season_label for point in trend.points if season_result(point) == "Bots significantly ahead"]
    clauses = []
    if pro_wins:
        clauses.append(f"Pros won with significance in {join_names(pro_wins)}")
    if bot_wins:
        clauses.append(f"{'bots' if clauses else 'Bots'} won with significance in {join_names(bot_wins)}")
    if not clauses:
        return "No season had a significant winner."
    return f"{' and '.join(clauses)}."


def _between_seasons(previous_label: str, current_label: str) -> str:
    """E.g. "Between Spring and Summer", keeping the years when the seasons fall in different years."""
    previous_name, previous_year = previous_label.rsplit(" ", 1)
    current_name, current_year = current_label.rsplit(" ", 1)
    if previous_year == current_year:
        return f"Between {previous_name} and {current_name}"
    return f"Between {previous_label} and {current_label}"


def season_change_clause(trend: SeasonTrend, between_seasons: bool = False) -> str | None:
    previous = trend.previous
    if previous is None or trend.change_p_value is None:
        return None
    if between_seasons:
        move = (
            f"{_between_seasons(previous.season_label, trend.current.season_label)}, the bot team's score went from "
            f"{format_compact(previous.mean)} to {format_compact(trend.current.mean)}"
        )
    else:
        move = (
            f"The bot team's score went from {format_compact(previous.mean)} in {previous.season_label} to "
            f"{format_compact(trend.current.mean)} this season"
        )
    if trend.change_p_value < SIGNIFICANCE_LEVEL:
        return f"{move}, a statistically significant change ({format_p(trend.change_p_value)})."
    return f"{move}, a change too small to be statistically significant ({format_p(trend.change_p_value)})."


def month_day(day: date) -> str:
    return f"{day:%B} {day.day}"


def format_duration(minutes: float) -> str:
    if minutes < 120:
        return f"{round(minutes)} minutes"
    return f"{format_compact(minutes / 60)} hours"


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
            f"score against Pros was {current_text} {HEAD_TO_HEAD_SCALE}. {clear_wins_sentence(trend)}"
        )
    if trend.change_p_value >= SIGNIFICANCE_LEVEL:
        heading = f"No clear change in the bot team's standing since {previous.season_label}"
    elif current.mean > previous.mean:
        heading = f"The bot team improved clearly relative to Pros since {previous.season_label}"
    else:
        heading = f"The bot team fell back relative to Pros since {previous.season_label}"
    earlier = ", ".join(f"{format_compact(point.mean)} in {point.season_label}" for point in trend.points[:-1])
    return (
        f"**{heading}:** The bot team's head-to-head score against Pros was {earlier}, and {current_text} "
        f"{HEAD_TO_HEAD_SCALE}. {clear_wins_sentence(trend)} {season_change_clause(trend)}"
    )


def _clear_wins_with_near_misses(trend: SeasonTrend) -> str:
    near_misses = [
        point.season_label
        for point in trend.points
        if point.season_label in NEAR_MISS_SEASONS and season_result(point) == "No clear winner"
    ]
    sentence = clear_wins_sentence(trend)
    if not near_misses:
        return sentence
    return f"{sentence[:-1]}, with {join_names(near_misses)} barely crossing into non-significance."


def trend_paragraph(trend: SeasonTrend) -> str:
    sentences = [
        "Below are the bot team's head-to-head scores against the Pro team in each season, with 95% confidence "
        "intervals. A score of zero means the teams' performance was equal. Negative means the Pros performed "
        "better, and positive means the bots performed better. We count a season as a clear win when its 95% "
        "confidence interval is entirely on one side of zero.",
        _clear_wins_with_near_misses(trend),
    ]
    change_clause = season_change_clause(trend, between_seasons=True)
    if change_clause:
        sentences.append(change_clause)
    if trend.skipped_seasons:
        is_single = len(trend.skipped_seasons) == 1
        sentences.append(
            f"There {'was no Pro comparison' if is_single else 'were no Pro comparisons'} in "
            f"{join_names(list(trend.skipped_seasons))}, so {'it is' if is_single else 'they are'} not shown."
        )
    return " ".join(sentences)


def _previous_pros_in_top_10_sentence(pros_in_top: int, trend: SeasonTrend) -> str:
    previous = trend.previous
    if previous is None:
        return ""
    previous_count = PUBLISHED_PROS_IN_INDIVIDUAL_TOP_10.get(previous.season_label)
    if previous_count is None:
        return " " + fill_me(
            f"compare with how many of the top {TOP_N} individual spots Pros took in {previous.season_label}, and add "
            "it to PUBLISHED_PROS_IN_INDIVIDUAL_TOP_10 in season_config.py"
        )
    is_last_season = seasons_before(trend.current.season_label)[-1] == previous.season_label
    when = "last season" if is_last_season else f"in {previous.season_label}"
    if pros_in_top == previous_count:
        return f" This is the same as {when}."
    direction = "down" if pros_in_top < previous_count else "up"
    return f" This is {direction} from {previous_count} of the top {TOP_N} spots {when}."


def individuals_takeaway(individuals: IndividualResults, trend: SeasonTrend) -> str:
    pro_count = individuals.pro_count
    pros_above_all_bots = individuals.pros_above_best_bot
    pros_in_top = individuals.pros_in_top_10
    setup = "On a leaderboard ranking bots and Pros individually,"
    if pros_above_all_bots == pro_count:
        body = (
            f"**Every individual Pro beat every individual bot:** {setup} the top {pro_count} spots are all taken by "
            "Pros."
        )
    elif pros_above_all_bots == 0:
        body = (
            f"**A bot topped the combined leaderboard of Pros and bots:** {setup} {individuals.best_bot.name} takes "
            f"1st place, and Pros take {pros_in_top} of the top {TOP_N}."
        )
    else:
        body = (
            f"**{count_word(pros_above_all_bots, capitalize=True)} of {count_word(pro_count)} individual Pros beat "
            f"every individual bot:** {setup} the top {pros_above_all_bots} spots are taken by Pros, and they take "
            f"{pros_in_top} of the top {TOP_N}."
        )
    return body + _previous_pros_in_top_10_sentence(pros_in_top, trend)


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
        f"is more rigorous and not liable to {MULTIPLE_COMPARISONS_LINK}."
    )
    return " ".join(sentences)


def _team_size_label(result: TeamSizeResult) -> str:
    return f"all {result.team_size} bots" if result.is_all_bots else f"the top {result.team_size} {plural(result.team_size, 'bot')}"


def _compared_with_main(result: TeamSizeResult) -> str:
    """"better", "worse", or "unclear": whether the 95% CI of the head-to-head score against the main bot team excludes 0."""
    ci = result.head_to_head_vs_main_ci
    if ci is not None and ci[0] > 0:
        return "better"
    if ci is not None and ci[1] < 0:
        return "worse"
    return "unclear"


def _main_team(team_sizes: list[TeamSizeResult], main_size: int) -> TeamSizeResult:
    return next(result for result in team_sizes if result.team_size == main_size and not result.is_all_bots)


def _larger_teams_verdict(team_sizes: list[TeamSizeResult], main_size: int) -> str | None:
    verdicts = {_compared_with_main(result) for result in team_sizes if result.team_size > main_size}
    return verdicts.pop() if len(verdicts) == 1 else None


def team_size_takeaway(team_sizes: list[TeamSizeResult], main_size: int) -> str:
    require_head_to_head_vs_main(team_sizes, main_size)
    main = _main_team(team_sizes, main_size)
    single_bot = next((result for result in team_sizes if result.team_size == 1), None)
    larger = [result for result in team_sizes if result.team_size > main_size]
    smaller = [result for result in team_sizes if result.team_size < main_size]
    larger_verdict = _larger_teams_verdict(team_sizes, main_size)
    if single_bot is not None and _compared_with_main(single_bot) == "better":
        heading = f"The top bot alone did better than the top-{main_size} bot team"
    elif larger and larger_verdict in ("worse", "better"):
        heading = (
            f"Bot teams of {larger[0].team_size} or more bots did {larger_verdict} than a team of the top {main_size} bots"
        )
    elif all(_compared_with_main(result) == "unclear" for result in team_sizes):
        heading = "Bot team size made no clear difference"
    else:
        heading = "Bot team size had mixed effects"
    sentence = f"**{heading}:** The top {main_size} bots scored {format_score(main.mean)} head-to-head points against Pros"
    if larger:
        largest = larger[-1]
        sentence += f", and a team of {_team_size_label(largest)} scored {format_score(largest.mean)}"
    sentence += "."
    smaller_worse = [result for result in smaller if _compared_with_main(result) == "worse"]
    if smaller:
        if smaller_worse:
            sentence += f" Of the smaller teams, {_team_group_label(smaller_worse)} did significantly worse than the top {main_size}."
        else:
            sentence += f" No smaller team did significantly worse than the top {main_size}."
        sentence += f" {_smaller_team_scores_sentence(smaller)}"
    larger_worse = [result for result in larger if _compared_with_main(result) == "worse"]
    if larger_worse:
        group = (
            f"bot teams of size {larger[0].team_size} and greater"
            if larger_worse == larger
            else _team_group_label(larger_worse)
        )
        clause = f"{group} did significantly worse, so increasing the size of a bot team does not automatically improve accuracy."
        sentence += f" However, {clause}" if smaller and not smaller_worse else f" {clause[0].upper()}{clause[1:]}"
    return sentence


def _smaller_team_scores_sentence(smaller: list[TeamSizeResult]) -> str:
    """E.g. "The top bot alone scored -3.59, and teams of 2 to 9 bots scored between -2.39 and 0.89"."""
    single_bot = next((result for result in smaller if result.team_size == 1), None)
    others = [result for result in smaller if result is not single_bot]
    parts = [f"the top bot alone scored {format_score(single_bot.mean)}"] if single_bot is not None else []
    if len(others) == 1:
        parts.append(f"a team of {_team_size_label(others[0])} scored {format_score(others[0].mean)}")
    elif others:
        low, high = min(result.mean for result in others), max(result.mean for result in others)
        sizes = [result.team_size for result in others]
        parts.append(
            f"teams of {min(sizes)} to {max(sizes)} bots scored between {format_score(low)} and {format_score(high)}"
        )
    text = ", and ".join(parts)
    return f"{text[0].upper()}{text[1:]}."


def _team_group_label(results: list[TeamSizeResult]) -> str:
    """E.g. "the top bot alone, the top 30 and 50 bots, and all 274 bots"."""
    parts = ["the top bot alone"] if any(result.team_size == 1 and not result.is_all_bots for result in results) else []
    sized = [str(result.team_size) for result in results if result.team_size > 1 and not result.is_all_bots]
    if sized:
        parts.append(f"the top {join_names(sized)} bots")
    parts += [_team_size_label(result) for result in results if result.is_all_bots]
    return join_names(parts)


def team_size_paragraph(team_sizes: list[TeamSizeResult], main_size: int) -> str:
    best = max(team_sizes, key=lambda result: result.mean)
    single_bot = next((result for result in team_sizes if result.team_size == 1), None)
    best_ci = f" {format_ci(best.ci)}" if best.ci is not None else ""
    sentences = [
        "We ran a sensitivity analysis of the bot team's head-to-head score against Pros for bot teams of "
        "different sizes. The bot-only questions are used to determine which bots are included in a team of size "
        "X (i.e., the same method used to select the main bot team).",
    ]
    all_bots = next((result for result in team_sizes if result.is_all_bots), None)
    if all_bots is not None:
        sentences.append(
            f"The largest team includes all {all_bots.team_size} bots that forecast on at least one bot-only question."
        )
    sentences += [
        "The chart below shows each team's average head-to-head score against the Pro team, with a 95% confidence band.",
        f"The highest point estimate came from a team of {_team_size_label(best)} ({format_score(best.mean)}{best_ci}).",
    ]
    if single_bot is not None and single_bot is not best:
        sentences.append(f"The top bot by itself scored {format_score(single_bot.mean)}.")
    sentences.append(
        f"Note that we do not correct for {MULTIPLE_COMPARISONS_LINK}, so we expect some aggregates to stand out by "
        "chance rather than because of a meaningful underlying driver."
    )
    return " ".join(sentences)


def team_size_vs_main_paragraph(team_sizes: list[TeamSizeResult], main_size: int) -> str:
    require_head_to_head_vs_main(team_sizes, main_size)
    sentences = [
        "The chart above shows performance vs Pros but doesn't properly indicate whether one bot team is better than "
        "another.",
        f"To compare each team with the top {main_size} directly, we also scored each team head-to-head against "
        f"the top-{main_size} team on the same questions, the same way we score bots against Pros.",
        "There is more statistical power here (smaller error bars) since there is more variation between a bot team "
        "and the Pro team than there is between two different bot teams that share most of their members.",
        f"The chart below shows each team's average head-to-head score against the top-{main_size} team, with 95% "
        f"confidence intervals. A team did significantly better or worse than the top {main_size} when its "
        "interval excludes 0.",
    ]
    verdict_groups = {
        verdict: [result for result in team_sizes if _compared_with_main(result) == verdict] for verdict in ("worse", "better")
    }
    for verdict, group in verdict_groups.items():
        if group:
            label = _team_group_label(group)
            sentences.append(f"{label[0].upper()}{label[1:]} did significantly {verdict} than the top {main_size}.")
    if not any(verdict_groups.values()):
        sentences.append(f"No team did significantly better or worse than the top {main_size}.")
    sentences.append(f"Note that we do not correct for {MULTIPLE_COMPARISONS_LINK} in this chart.")
    sentences.append(
        "Neighboring team sizes share most of their members, and we compared many sizes, so a single significant "
        "difference between small teams can easily be luck."
    )
    return " ".join(sentences)


def runner_up_sentence(comparison: RunnerUpComparison) -> str:
    runner_up = comparison.runner_up.name
    ci = comparison.difference_ci
    on_shared = f"on the {comparison.shared_question_count} questions both forecast"
    if ci is not None:
        on_shared += f" (95% CI: {format_ci(ci)})"
    gap = comparison.mean_difference
    if ci is not None and ci[0] > 0:
        return f"It scored {format_score(gap)} points higher than the runner-up, {runner_up}, {on_shared}, a statistically significant lead."
    if ci is not None and ci[1] < 0:
        return (
            f"However, the runner-up, {runner_up}, scored {format_score(-gap)} points higher than it {on_shared}, a "
            "statistically significant difference, even though it ranks lower by sum."
        )
    if gap >= 0:
        return (
            f"However, it scored only {format_score(gap)} points higher than the runner-up, {runner_up}, {on_shared}, "
            'so it\'s hard to declare any one setup "the best."'
        )
    return (
        f"However, the runner-up, {runner_up}, scored {format_score(-gap)} points higher than it {on_shared}, so "
        'it\'s hard to declare any one setup "the best."'
    )


def _weeks_late(metac: MetacResults, name: str) -> int:
    return round(metac.days_after_season_start(name) / 7)


def late_starter_count(metac: MetacResults) -> int:
    return sum(1 for name in metac.first_scored_close_by_name if _weeks_late(metac, name) >= LATE_START_MIN_WEEKS)


def _late_start_clause(metac: MetacResults, row: LeaderboardRow) -> str:
    weeks_late = _weeks_late(metac, row.name)
    if weeks_late < LATE_START_MIN_WEEKS:
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
    current_text = f"{ordinal(current.rank)} of {current.bot_count} this season"
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
        f"**{heading}:** The best standardized Metaculus Bot ({current.bot_name}), using our single prompt, placed "
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
        f"{format_score(max(averages), 1)} points"
    )
    if field.top_10_average_ci_width is not None:
        spread += f", with confidence intervals ~{format_score(field.top_10_average_ci_width, 1)} points wide"
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
        f"The chart below shows every bot with at least {chart_min_questions} scored questions in leaderboard "
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


def format_whole_percent(share: float) -> str:
    return f"{100 * share:.0f}%"


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
    for one_shot, standard_rows in groups.no_research_pairs:
        standard = ", ".join(f"{describe(row, row.name)} on {row.question_count}" for row in standard_rows)
        lines.append(
            "No research, one shot vs the standard pipeline with the same model (if you write about this, say these "
            "bots forecast on fewer, different questions, so the gaps aren't on matched questions): "
            f"{describe(one_shot, one_shot.name)} on {one_shot.question_count} vs {standard}"
        )
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
        f"{top.name} ranked first among the same-prompt bots by sum of spot peer scores, with an average of "
        f"{format_score(top.average_score)}"
    )
    sentence += f" (95% CI: {format_ci(top.ci)})." if top.ci is not None else "."
    if metac.runner_up_comparison is not None:
        sentence += f" {runner_up_sentence(metac.runner_up_comparison)}"
    higher_average_sentence = higher_average_on_fewer_questions_sentence(metac)
    if higher_average_sentence:
        sentence += f" {higher_average_sentence}"
    return sentence
