from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import pytest

from aib_analysis.pro_vs_bot_writeup import phrasing
from aib_analysis.pro_vs_bot_writeup.results import (
    BaselineRankPoint,
    BotFieldAccuracy,
    BotGroups,
    BotGroupSummary,
    BotTypicalProbability,
    FieldSummary,
    HeadToHeadResult,
    IndividualResults,
    LeaderboardRow,
    MetacResults,
    RunnerUpComparison,
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
    assert (
        "The bot team's score went from -1.25 in Spring 2026 to -0.26 this season, a change too small to be "
        "statistically significant (p = 0.784)." in text
    )


def test_trend_takeaway_without_history() -> None:
    trend = SeasonTrend(points=[SeasonTrendPoint("Q3 2024", -11.3, (-21.8, -0.7), None, True)], change_p_value=None, skipped_seasons=())
    assert phrasing.trend_takeaway(trend).startswith("**The bot team scored -11.3 against Pros:**")


def test_trend_paragraph_lists_clear_wins_and_skipped_seasons() -> None:
    text = phrasing.trend_paragraph(_trend(5.0, 12.0, 0.04))
    assert "when its 95% confidence interval is entirely on one side of zero." in text
    assert "Pros won clearly in Q2 2025 and bots won clearly in Spring 2026 and Summer 2026." in text
    assert "There was no Pro comparison in Fall 2025, so it is not shown." in text


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


def _team_size(size: int, mean: float, vs_main: tuple[float, float] | None = (-1.0, 1.0), is_all: bool = False) -> TeamSizeResult:
    return TeamSizeResult(size, is_all, mean, (mean - 6, mean + 6), 97, None if size == 10 else vs_main)


def test_team_size_takeaway_when_larger_teams_did_worse() -> None:
    sizes = [
        _team_size(1, -3.59),
        _team_size(8, 0.89, (0.35, 1.94)),
        _team_size(10, -0.26),
        _team_size(30, -2.48, (-4.3, -0.1)),
        _team_size(274, -12.1, (-17.98, -5.7), is_all=True),
    ]
    text = phrasing.team_size_takeaway(sizes, 10)
    assert text.startswith("**Bot teams of 30 or more bots did worse than the top 10:**")
    assert "a team of all 274 bots scored -12.10" in text
    assert "Smaller teams, down to the top bot alone, scored between -3.59 and 0.89." in text
    assert "Merely increasing the size of a bot team" in text
    assert "peak" not in text
    paragraph = phrasing.team_size_paragraph(sizes, 10)
    assert "The top 8 bots did significantly better than the top 10." in paragraph
    assert "The top 30 bots and all 274 bots did significantly worse than the top 10." in paragraph
    assert "a single significant difference between small teams can easily be luck" in paragraph


def test_team_size_takeaway_other_headings() -> None:
    top_bot_better = [_team_size(1, 5.0, (1.0, 9.0)), _team_size(10, 1.0), _team_size(50, 0.0, is_all=True)]
    assert phrasing.team_size_takeaway(top_bot_better, 10).startswith("**The top bot alone did better than the top-10 bot team:**")
    no_difference = [_team_size(1, -1.0), _team_size(10, 0.0), _team_size(50, 0.5, is_all=True)]
    text = phrasing.team_size_takeaway(no_difference, 10)
    assert text.startswith("**Bot team size made no clear difference:**")
    assert "Merely increasing" not in text
    mixed = [_team_size(1, -1.0), _team_size(10, 0.0), _team_size(30, 3.0, (0.5, 5.5)), _team_size(50, -0.5, is_all=True)]
    assert phrasing.team_size_takeaway(mixed, 10).startswith("**Bot team size had mixed effects:**")


def test_field_band_paragraph_counts_bots_outside_the_band() -> None:
    field = _field([(10.0, 20.0)] * 10)
    rows = [_row(rank, f"bot-{rank}", average=average, questions=100) for rank, average in enumerate([30.0] * 6 + [0.0] * 30 + [-30.0] * 4, start=1)]
    text = phrasing.field_band_paragraph(dataclasses.replace(field, rows=rows), chart_min_questions=10)
    assert "about 95% of the time" in text
    assert "Luck alone can't explain how spread out the bots are: 6 of the 40 bots shown sit above the band and 4 below it, where we'd expect only about 1 on each side" in text
    calm = [_row(rank, f"bot-{rank}", average=0.0, questions=100) for rank in range(1, 41)]
    assert "The spread is about what luck alone would produce" in phrasing.field_band_paragraph(dataclasses.replace(field, rows=calm), 10)


def _metac(
    history: list[BaselineRankPoint],
    close: bool,
    late_starter_average: float | None = None,
    runner_up_gap: tuple[float, tuple[float, float] | None] = (0.71, (-3.14, 4.56)),
) -> MetacResults:
    top_ci = (5.0, 15.0)
    second_average = 9.0 if close else 2.0
    rows = [
        _row(1, "metac-model-a+asknews", average=10.0, ci=top_ci, questions=300),
        _row(2, "metac-model-b+asknews", average=second_average, questions=300),
    ]
    if late_starter_average is not None:
        rows.append(_row(3, "metac-model-c+asknews", average=late_starter_average, questions=150))
    season_start = date(2026, 5, 18)
    return MetacResults(
        baseline_rows=rows,
        other_rows=[],
        metac_bot_count=len(rows),
        scored_question_count=300,
        average_questions_per_baseline_bot=280.0,
        baseline_history=history,
        first_question_close=season_start,
        first_scored_close_by_name={
            "metac-model-a+asknews": season_start,
            "metac-model-b+asknews": season_start,
            "metac-model-c+asknews": season_start + timedelta(days=23),
        },
        runner_up_comparison=RunnerUpComparison(rows[1], 309, runner_up_gap[0], runner_up_gap[1]),
    )


def test_top_baseline_takeaway_compares_with_the_runner_up() -> None:
    history = [BaselineRankPoint("Summer 2026", 24, 282, "metac-model-a+asknews")]
    text = phrasing.top_baseline_takeaway(_metac(history, close=True))
    assert "24th of 282 overall" in text
    assert (
        "However, it scored only 0.71 points higher than the runner-up, metac-model-b+asknews, on the 309 questions "
        'both forecast (95% CI: [-3.14, 4.56]), so it\'s hard to declare any one setup "the best."' in text
    )


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        ((5.2, (1.1, 9.3)), "It scored 5.20 points higher than the runner-up, metac-model-b+asknews, on the 309 questions both forecast (95% CI: [1.10, 9.30]), a statistically significant lead."),
        ((-4.0, (-7.0, -1.0)), "However, the runner-up, metac-model-b+asknews, scored 4.00 points higher than it on the 309 questions both forecast (95% CI: [-7.00, -1.00]), a statistically significant difference"),
        ((-0.5, (-4.0, 3.0)), "However, the runner-up, metac-model-b+asknews, scored 0.50 points higher than it on the 309 questions both forecast (95% CI: [-4.00, 3.00]), so it's hard"),
        ((0.3, None), "However, it scored only 0.30 points higher than the runner-up, metac-model-b+asknews, on the 309 questions both forecast, so it's hard"),
    ],
)
def test_runner_up_sentence_branches(gap: tuple[float, tuple[float, float] | None], expected: str) -> None:
    history = [BaselineRankPoint("Summer 2026", 24, 282, "metac-model-a+asknews")]
    metac = _metac(history, close=True, runner_up_gap=gap)
    assert expected in phrasing.runner_up_sentence(metac.runner_up_comparison)
    assert expected in phrasing.metac_comparison_sentence(metac)


def test_higher_average_bots_that_started_late_are_called_out() -> None:
    history = [BaselineRankPoint("Summer 2026", 24, 282, "metac-model-a+asknews")]
    metac = _metac(history, close=True, late_starter_average=12.84)
    takeaway = phrasing.top_baseline_takeaway(metac)
    assert "metac-model-c+asknews had a higher average (12.84) but forecast on 150 questions, starting 3 weeks into the season." in takeaway
    section = phrasing.metac_comparison_sentence(metac)
    assert section.startswith("metac-model-a+asknews ranked first by sum of spot peer scores, with an average of 10.00")
    assert (
        "metac-model-c+asknews (12.84 on 150 questions, starting 3 weeks into the season) had a higher average than "
        "metac-model-a+asknews but forecast on fewer questions, so it ranks lower by sum." in section
    )
    assert "had a higher average" not in phrasing.metac_comparison_sentence(_metac(history, close=True))


def test_format_date_range() -> None:
    assert phrasing.format_date_range(date(2026, 5, 18), date(2026, 8, 20)) == "May 18 and August 20, 2026"
    assert phrasing.format_date_range(date(2025, 12, 1), date(2026, 3, 2)) == "December 1, 2025 and March 2, 2026"


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


def test_field_paragraph_names_promising_bots_with_their_own_question_counts() -> None:
    promising = [_row(11, "late-joiner", average=22.0, questions=119), _row(12, "part-timer", average=18.0, questions=102)]
    field = dataclasses.replace(_field([(10.0, 20.0)] * 10), promising=promising)
    text = phrasing.field_paragraph(field, chart_min_questions=10, hidden_count=0)
    assert "Two bots (in green) forecast on fewer than half of the questions" in text
    assert "late-joiner (119 questions) and part-timer (102 questions)" in text
    assert "some of these might prove to be strong" in text
    assert "we don't highlight bots with fewer than 30 questions (10% of the total)" in text
    single_text = phrasing.field_paragraph(dataclasses.replace(field, promising=promising[:1]), chart_min_questions=10, hidden_count=0)
    assert "One bot (in green)" in single_text
    assert "it might prove to be strong" in single_text


def _bot_groups(hobbyist_ranks: list[int], commercial_ranks: list[int], hobbyists_on_team: int = 8) -> tuple[BotGroups, FieldSummary]:
    names_by_rank = {rank: f"hobby-{rank}" for rank in hobbyist_ranks} | {rank: f"company-{rank}" for rank in commercial_ranks}
    rows = [_row(rank, names_by_rank[rank], average=20.0 - rank, questions=300) for rank in sorted(names_by_rank)]
    field = dataclasses.replace(_field([(10.0, 20.0)] * 10), rows=rows)
    hobbyist = [row for row in rows if row.name.startswith("hobby")]
    commercial = [row for row in rows if row.name.startswith("company")]
    groups = BotGroups(
        hobbyist=BotGroupSummary("Hobbyist", hobbyist, hobbyist),
        commercial=BotGroupSummary("Commercial", commercial, commercial) if commercial else None,
        metac=None,
        hobbyists_on_main_team=hobbyists_on_team,
    )
    return groups, field


def test_hobbyist_takeaway_when_a_hobbyist_wins() -> None:
    groups, field = _bot_groups(hobbyist_ranks=[1, 4, 5, 6, 7, 8, 9, 10, 11], commercial_ranks=[2, 3])
    text = phrasing.hobbyist_takeaway(groups, field, bot_team_size=10)
    assert text.startswith("**Hobbyist bots took 8 of the top 10 spots, including first place:**")
    assert "hobby-1 topped the leaderboard with an average spot peer score of 19.00; the best commercial bot, company-2, placed 2nd with 18.00." in text
    assert "Hobbyist bots made up 8 of the 10 bots on the team that faced the Pros." in text
    assert "the median commercial bot averaged 17.50 (2 bots) and the median hobbyist bot 13.00 (9 bots)." in text


def test_hobbyist_takeaway_other_cases() -> None:
    groups, field = _bot_groups(hobbyist_ranks=[3, 12], commercial_ranks=list(range(1, 3)) + list(range(4, 12)), hobbyists_on_team=1)
    text = phrasing.hobbyist_takeaway(groups, field, bot_team_size=10)
    assert text.startswith("**Hobbyist bots took 1 of the top 10 spots:** The best hobbyist bot, hobby-3, placed 3rd")
    groups, field = _bot_groups(hobbyist_ranks=[11], commercial_ranks=list(range(1, 11)), hobbyists_on_team=0)
    assert phrasing.hobbyist_takeaway(groups, field, bot_team_size=10).startswith("**No hobbyist bot made the top 10:**")
    groups, field = _bot_groups(hobbyist_ranks=[1, 2], commercial_ranks=[])
    text = phrasing.hobbyist_takeaway(groups, field, bot_team_size=10)
    assert "commercial" not in text


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
    assert text.startswith("**The top bot gave the actual outcome a typical probability of 63.5%, vs 52.8% for the median bot:**")
    assert "The lowest-scoring bot gave a typical probability of 41.7%." in text
    assert "always forecasting 38.3% (the share of binary questions that resolved Yes) scores 51.5%. 50% of bots did worse than that." in text
    assert text.endswith("These figures pertain to only binary questions among the 4 bots that forecast on at least half of the questions.")
    translation = phrasing.peer_score_translation_paragraph(accuracy)
    assert "there +X means about e^(2X/100) times the field's probability density" in translation
    assert "laertes's average spot peer score of 17.91 means it gave the actual outcome about 1.30× the probability (or probability density)" in translation


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
