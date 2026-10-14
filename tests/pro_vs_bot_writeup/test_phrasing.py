from __future__ import annotations

import pytest

from aib_analysis.pro_vs_bot_writeup import phrasing
from aib_analysis.pro_vs_bot_writeup.results import (
    BaselineRankPoint,
    BotFieldAccuracy,
    BotTypicalProbability,
    FieldSummary,
    HeadToHeadResult,
    IndividualResults,
    LeaderboardRow,
    MetacResults,
    SeasonTrend,
    SeasonTrendPoint,
    TeamAccuracy,
    TeamSizeResult,
)


def _head_to_head(mean: float, ci: tuple[float, float], p_pros_better: float) -> HeadToHeadResult:
    return HeadToHeadResult(
        question_count=97,
        bot_team_mean=mean,
        ci=ci,
        p_pros_better=p_pros_better,
        p_bots_better=1 - p_pros_better,
        pro_wins=60,
        bot_wins=37,
        ties=0,
        average_pro_win_margin=15.4,
        average_bot_win_margin=24.4,
        bot_team_members=tuple(f"bot-{index}" for index in range(10)),
        pro_member_count=10,
        non_pro_member_count=0,
        question_type_counts={"binary": 97},
    )


def _row(rank: int, name: str, user_type: str = "bot", average: float = 10.0, ci: tuple[float, float] | None = (5.0, 15.0), questions: int = 100) -> LeaderboardRow:
    return LeaderboardRow(rank, name, user_type, average * questions, average, ci, questions)


HEADLINE_CASES = [
    (-10.0, (-14.0, -6.0), 0.0001, "**Pro forecasters significantly outperformed bots:**", "entirely below zero"),
    (-3.0, (-6.1, 0.1), 0.03, "**Pro forecasters outperformed bots, but the evidence is weak:**", "still includes zero"),
    (-0.26, (-6.47, 5.96), 0.468, "**Pro forecasters beat bots but without significance:**", "confirm Pros were genuinely ahead"),
    (10.0, (6.0, 14.0), 0.9999, "**Bots significantly outperformed Pro forecasters:**", "entirely above zero"),
    (3.0, (-0.1, 6.1), 0.97, "**Bots outperformed Pro forecasters, but the evidence is weak:**", "still includes zero"),
    (0.26, (-5.96, 6.47), 0.532, "**Bots beat Pro forecasters but without significance:**", "confirm bots were genuinely ahead"),
]


@pytest.mark.parametrize(("mean", "ci", "p_pros_better", "heading", "phrase"), HEADLINE_CASES)
def test_headline_takeaway_branches(mean: float, ci: tuple[float, float], p_pros_better: float, heading: str, phrase: str) -> None:
    text = phrasing.headline_takeaway(_head_to_head(mean, ci, p_pros_better), pro_count=10, bot_team_size=10)
    assert text.startswith(heading)
    assert phrase in text
    assert f"{abs(mean):.2f} head-to-head points per question" in text


def test_headline_names_the_winner_first() -> None:
    pros_text = phrasing.headline_takeaway(_head_to_head(-0.26, (-6.47, 5.96), 0.468), 10, 10)
    bots_text = phrasing.headline_takeaway(_head_to_head(0.26, (-5.96, 6.47), 0.532), 10, 10)
    assert "Our team of 10 Metaculus Pro Forecasters outperformed the top-10 bot team" in pros_text
    assert "The top-10 bot team outperformed our team of 10 Metaculus Pro Forecasters" in bots_text


@pytest.mark.parametrize(
    ("mean", "ci", "p_pros_better", "expected"),
    [
        (-10.0, (-14.0, -6.0), 0.0001, "demonstrates that Pros significantly outperformed bots (p < 0.001)"),
        (-3.0, (-6.1, 0.1), 0.03, "so the evidence is weak"),
        (-0.26, (-6.47, 5.96), 0.468, "did not demonstrate that Pros outperformed bots"),
        (0.26, (-5.96, 6.47), 0.532, "did not demonstrate that bots outperformed Pros"),
        (10.0, (6.0, 14.0), 0.9999, "The positive score indicates that the bot team had higher accuracy"),
    ],
)
def test_head_to_head_paragraph_branches(mean: float, ci: tuple[float, float], p_pros_better: float, expected: str) -> None:
    text = phrasing.head_to_head_paragraph(_head_to_head(mean, ci, p_pros_better))
    assert expected in text
    assert "Pros scored better on 60 of the 97 questions (62%) and bots on 37 (38%)." in text


def _trend(previous_mean: float, current_mean: float, change_p_value: float) -> SeasonTrend:
    return SeasonTrend(
        points=[
            SeasonTrendPoint("Q2 2025", -20.03, (-28.63, -11.41), 93, False),
            SeasonTrendPoint("Spring 2026", previous_mean, (previous_mean - 3, previous_mean + 3), 99, False),
            SeasonTrendPoint("Summer 2026", current_mean, (current_mean - 6, current_mean + 6), 97, True),
        ],
        change_p_value=change_p_value,
        skipped_seasons=("Fall 2025",),
    )


@pytest.mark.parametrize(
    ("previous_mean", "current_mean", "change_p_value", "heading"),
    [
        (-1.25, -0.26, 0.784, "**No clear change in the bot team's standing since Spring 2026:**"),
        (-20.0, -1.0, 0.001, "**The bot team improved clearly relative to Pros since Spring 2026:**"),
        (-1.0, -20.0, 0.001, "**The bot team fell back relative to Pros since Spring 2026:**"),
    ],
)
def test_trend_takeaway_branches(previous_mean: float, current_mean: float, change_p_value: float, heading: str) -> None:
    text = phrasing.trend_takeaway(_trend(previous_mean, current_mean, change_p_value))
    assert text.startswith(heading)
    assert "-20.03 in Q2 2025, " in text
    assert "95% CI" not in text
    assert phrasing.HEAD_TO_HEAD_REMINDER in text


def test_trend_takeaway_states_clear_winners_and_change() -> None:
    text = phrasing.trend_takeaway(_trend(-1.25, -0.26, 0.784))
    assert "Pros won clearly in Q2 2025; the other seasons had no clear winner." in text
    assert "The change from Spring 2026 (-1.25) is not statistically significant (p = 0.784)." in text


def test_trend_takeaway_without_history() -> None:
    trend = SeasonTrend(points=[SeasonTrendPoint("Q3 2024", -11.3, (-21.8, -0.7), None, True)], change_p_value=None, skipped_seasons=())
    assert phrasing.trend_takeaway(trend).startswith("**The bot team scored -11.3 against Pros:**")


def test_trend_paragraph_lists_clear_wins_and_skipped_seasons() -> None:
    text = phrasing.trend_paragraph(_trend(5.0, 12.0, 0.04))
    assert "when its 95% confidence interval is entirely on one side of zero." in text
    assert "Pros won clearly in Q2 2025 and bots won clearly in Spring 2026 and Summer 2026." in text
    assert "Fall 2025 compared bots with the Metaculus community prediction instead of Pros, so it is not shown." in text


def test_clear_wins_sentence_without_clear_winners() -> None:
    trend = SeasonTrend(
        points=[SeasonTrendPoint("Spring 2026", -1.25, (-4.87, 2.37), 99, False), SeasonTrendPoint("Summer 2026", -0.26, (-6.47, 5.96), 97, True)],
        change_p_value=0.784,
        skipped_seasons=(),
    )
    assert phrasing.clear_wins_sentence(trend) == "No season had a clear winner."
    assert [phrasing.season_result(point) for point in trend.points] == ["No clear winner", "No clear winner"]


def _individuals(order: list[str]) -> IndividualResults:
    rows = [_row(rank, f"name-{rank}", user_type) for rank, user_type in enumerate(order, start=1)]
    return IndividualResults(rows=rows, pro_aliases={}, excluded_account_count=0, question_count=97)


def test_individuals_takeaway_when_every_pro_beats_every_bot() -> None:
    text = phrasing.individuals_takeaway(_individuals(["pro"] * 3 + ["bot"] * 10))
    assert text.startswith("**Every individual Pro beat every individual bot:**")
    assert "the top 3 places would all be taken by Pros" in text


def test_individuals_takeaway_when_some_pros_beat_every_bot() -> None:
    text = phrasing.individuals_takeaway(_individuals(["pro"] * 4 + ["bot"] + ["pro"] * 6 + ["bot"] * 5))
    assert text.startswith("**Four of ten individual Pros beat every individual bot:**")
    assert "9 of the top 10 places would be taken by Pros" in text
    assert phrasing.INDIVIDUAL_COMPARISON_CAVEAT in text


def test_individuals_takeaway_when_a_bot_is_first() -> None:
    text = phrasing.individuals_takeaway(_individuals(["bot", "pro", "pro", "bot"]))
    assert text.startswith("**A bot topped the combined leaderboard of Pros and bots:**")
    assert "name-1 would take 1st place" in text


def test_individuals_paragraph_counts_bots_above_lowest_pro() -> None:
    assert "2 bots placed above at least one Pro." in phrasing.individuals_paragraph(_individuals(["pro", "bot", "bot", "pro"]), 20)
    assert "Every Pro placed above every bot." in phrasing.individuals_paragraph(_individuals(["pro", "pro", "bot"]), 20)


def _team_size(size: int, mean: float, ci: tuple[float, float] | None = None, is_all: bool = False) -> TeamSizeResult:
    return TeamSizeResult(size, is_all, mean, ci or (mean - 6, mean + 6), 97)


def test_team_size_takeaway_peak_in_the_middle() -> None:
    sizes = [_team_size(1, -3.59), _team_size(8, 0.89), _team_size(10, -0.26), _team_size(276, -12.1, (-20.75, -3.44), True)]
    text = phrasing.team_size_takeaway(sizes)
    assert text.startswith("**Bot team aggregation peaked at 8 bots:**")
    assert "compared with -3.59 for the top bot alone and -12.10 for a team of all 276 bots" in text
    assert "Merely increasing the size of a bot team" in text


def test_team_size_takeaway_when_top_bot_or_all_bots_win() -> None:
    top_bot_wins = [_team_size(1, 5.0), _team_size(10, 1.0), _team_size(50, 0.0, is_all=True)]
    assert phrasing.team_size_takeaway(top_bot_wins).startswith("**The top bot alone beat every larger bot team:**")
    all_bots_win = [_team_size(1, -5.0), _team_size(10, 1.0), _team_size(50, 2.0, is_all=True)]
    text = phrasing.team_size_takeaway(all_bots_win)
    assert text.startswith("**Aggregating every bot gave the best bot team:**")
    assert "Merely increasing" not in text


def _metac(history: list[BaselineRankPoint], close: bool) -> MetacResults:
    top_ci = (5.0, 15.0)
    second_average = 9.0 if close else 2.0
    rows = [
        _row(1, "metac-model-a+asknews", average=10.0, ci=top_ci),
        _row(2, "metac-model-b+asknews", average=second_average),
    ]
    return MetacResults(
        baseline_rows=rows,
        other_rows=[],
        metac_bot_count=2,
        scored_question_count=300,
        average_questions_per_baseline_bot=280.0,
        baseline_history=history,
    )


def test_top_baseline_takeaway_mentions_close_competitors() -> None:
    history = [BaselineRankPoint("Summer 2026", 24, 282, "metac-model-a+asknews")]
    text = phrasing.top_baseline_takeaway(_metac(history, close=True))
    assert "24th of 282 overall" in text
    assert "its lead over metac-model-b+asknews is within its 95% confidence interval" in text
    assert "its lead" not in phrasing.top_baseline_takeaway(_metac(history, close=False))


@pytest.mark.parametrize(
    ("previous", "expected_heading"),
    [
        (BaselineRankPoint("Spring 2026", 18, 173, None), "placed in the top 9% of bots, up from the top 11% in Spring 2026"),
        (BaselineRankPoint("Spring 2026", 2, 96, None), "placed in the top 9% of bots, down from the top 3% in Spring 2026"),
        (BaselineRankPoint("Spring 2026", 24, 282, None), "placed in the top 9% of bots, the same as in Spring 2026"),
    ],
)
def test_baseline_history_takeaway(previous: BaselineRankPoint, expected_heading: str) -> None:
    history = [previous, BaselineRankPoint("Summer 2026", 24, 282, "metac-model-a+asknews")]
    text = phrasing.baseline_history_takeaway(_metac(history, close=False))
    assert expected_heading in text
    assert "and 24th of 282 this season (metac-model-a+asknews)" in text


def _field(top_cis: list[tuple[float, float]]) -> FieldSummary:
    rows = [_row(rank, f"bot-{rank}", average=(ci[0] + ci[1]) / 2, ci=ci) for rank, ci in enumerate(top_cis, start=1)]
    return FieldSummary(
        rows=rows,
        scored_question_count=300,
        eligible_min_questions=150,
        eligible_count=len(rows),
        best_eligible=rows[0],
        median_eligible_average=0.0,
        worst_eligible=rows[-1],
        tenth_place=rows[min(10, len(rows)) - 1],
        promising=[],
        promising_min_questions=30,
        per_question_score_sd=40.0,
        uniform_reference=None,
        top_10_average_ci_width=8.0,
    )


def test_top_10_spread_takeaway() -> None:
    overlapping = _field([(10.0, 20.0)] * 10)
    assert phrasing.top_10_spread_takeaway(overlapping).startswith("**No one leads the pack:**")
    separated = _field([(25.0, 35.0)] + [(10.0, 20.0)] * 9)
    text = phrasing.top_10_spread_takeaway(separated)
    assert text.startswith("**Some top-10 bots are clearly separated from others:**")
    assert "9 of the 45 pairs" in text


def _team_accuracy(ratio_ci: tuple[float, float] | None) -> TeamAccuracy:
    return TeamAccuracy(
        binary_question_count=46,
        yes_rate=0.391,
        base_rate_probability=0.512,
        pro_team_probability=0.637,
        pro_team_ci=(0.538, 0.756),
        bot_team_probability=0.630,
        bot_team_ci=(0.53, 0.75),
        bot_to_pro_ratio=0.988,
        bot_to_pro_ratio_ci=ratio_ci,
    )


@pytest.mark.parametrize(
    ("ratio_ci", "expected"),
    [
        ((0.918, 1.064), "The interval includes 1 (equal accuracy), so this gap could be due to chance."),
        ((0.90, 0.97), "the Pro team's edge on these questions is unlikely to be due to chance alone."),
        ((1.02, 1.10), "the bot team's edge on these questions is unlikely to be due to chance alone."),
    ],
)
def test_team_accuracy_paragraph_branches(ratio_ci: tuple[float, float], expected: str) -> None:
    text = phrasing.team_accuracy_paragraph(_team_accuracy(ratio_ci))
    assert "the Pro team gave the actual outcome a typical probability of 63.7% and the bot team 63.0%" in text
    assert expected in text


def test_accuracy_takeaway() -> None:
    eligible = [
        BotTypicalProbability("laertes", 0.635, 160),
        BotTypicalProbability("middle", 0.556, 150),
        BotTypicalProbability("weakest", 0.417, 150),
        BotTypicalProbability("below-base", 0.50, 150),
    ]
    accuracy = BotFieldAccuracy(
        binary_question_count=167,
        yes_rate=0.383,
        base_rate_probability=0.515,
        field_probability=0.554,
        eligible_bots=eligible,
        top_bot=eligible[0],
        top_bot_average_peer_score=17.91,
        top_bot_field_multiplier=1.302,
        worst_bot=eligible[2],
    )
    text = phrasing.accuracy_takeaway(accuracy)
    assert text.startswith("**On binary questions, the top bot gave the actual outcome a typical probability of 63.5%:**")
    assert "52.8% for the median bot and 41.7% for the lowest-scoring bot" in text
    assert "always forecasting this season's Yes rate (38.3%) scores 51.5%. 50% of these bots" in text
    translation = phrasing.peer_score_translation_paragraph(accuracy)
    assert "laertes's average spot peer score of 17.91 means it gave the actual outcome about 1.30× the probability" in translation


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (102, "102nd"), (111, "111th")],
)
def test_ordinal(number: int, expected: str) -> None:
    assert phrasing.ordinal(number) == expected


def test_number_formatting() -> None:
    assert phrasing.format_compact(-11.30) == "-11.3"
    assert phrasing.format_compact(1.0) == "1"
    assert phrasing.format_compact(-0.001) == "0"
    assert phrasing.format_p(0.0004) == "p < 0.001"
    assert phrasing.format_p(0.4681) == "p = 0.468"
    assert phrasing.join_names(["a", "b", "c"]) == "a, b, and c"
    assert phrasing.count_word(4, capitalize=True) == "Four"
    assert phrasing.count_word(12) == "12"
    assert phrasing.fill_me("say X") == "[===Fill me===] _(Reminder: say X)_"
