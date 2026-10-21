from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from aib_analysis.pro_vs_bot_writeup.results import (
    LeaderboardRow,
    SeasonTrendPoint,
    anonymize_pros,
    bot_label,
    check_comparison_uses_pro_weights,
    compute_field,
    compute_head_to_head,
    compute_season_results,
    find_close_time_differences,
    find_metac_experiment_groups,
    find_simulation_file,
    format_resolution,
    is_baseline_metac_bot,
    pick_example_questions,
    season_change_p_value,
    shared_question_pairs,
    split_baseline_name,
)
from aib_analysis.pro_vs_bot_writeup.season_config import SeasonConfig
from tests.pro_vs_bot_writeup.conftest import (
    ANNULLED_BOT_QUESTION_IDS,
    BOT_QUESTION_COUNT,
    NON_PRIMARY_BOT_NAME,
    SHARED_QUESTION_COUNT,
    make_question,
    make_team_tournament,
)

PRO_MEMBERS = (("pro_user_1", "pro"), ("pro_user_2", "pro"))
BOT_MEMBERS = (("bot-a", "bot"), ("bot-b", "bot"), ("bot-c", "bot"))


def _row(name: str, average: float, questions: int = 100, ci: tuple[float, float] | None = None, rank: int = 1) -> LeaderboardRow:
    return LeaderboardRow(rank, name, "bot", average * questions, average, ci, questions)


def _scores(mean: float, spread: float, count: int = 40) -> list[float]:
    return [mean + spread * (1 if index % 2 == 0 else -1) for index in range(count)]


@pytest.mark.parametrize(
    ("bot_mean", "spread", "leader", "evidence"),
    [
        (-10.0, 5.0, "pros", "strong"),
        (-2.0, 7.0, "pros", "weak"),
        (-0.5, 10.0, "pros", "none"),
        (10.0, 5.0, "bots", "strong"),
        (2.0, 7.0, "bots", "weak"),
        (0.5, 10.0, "bots", "none"),
    ],
)
def test_head_to_head_direction_and_evidence(
    synthetic_config: SeasonConfig, bot_mean: float, spread: float, leader: str, evidence: str
) -> None:
    raw = make_team_tournament(_scores(bot_mean, spread), BOT_MEMBERS, PRO_MEMBERS)
    result = compute_head_to_head(raw, synthetic_config)
    assert result.leader == leader
    assert result.evidence == evidence
    assert result.question_count == 40
    assert result.p_pros_better + result.p_bots_better == pytest.approx(1)
    assert result.pro_member_count == 2


def test_head_to_head_counts_wins_and_margins(synthetic_config: SeasonConfig) -> None:
    raw = make_team_tournament([5.0, -1.0, -3.0, 2.0, 0.0] * 8, BOT_MEMBERS, PRO_MEMBERS)
    result = compute_head_to_head(raw, synthetic_config)
    assert (result.bot_wins, result.pro_wins, result.ties) == (16, 16, 8)
    assert result.average_bot_win_margin == pytest.approx(3.5)
    assert result.average_pro_win_margin == pytest.approx(2.0)


def test_head_to_head_rejects_unexpected_pro_team_members(synthetic_config: SeasonConfig) -> None:
    raw = make_team_tournament(_scores(-1, 5), BOT_MEMBERS, PRO_MEMBERS + (("some_bot", "bot"),))
    with pytest.raises(ValueError, match="non-Pro members"):
        compute_head_to_head(raw, synthetic_config)


def test_head_to_head_rejects_wrong_bot_team_size(synthetic_config: SeasonConfig) -> None:
    raw = make_team_tournament(_scores(-1, 5), BOT_MEMBERS[:2], PRO_MEMBERS)
    with pytest.raises(ValueError, match="3-bot team"):
        compute_head_to_head(raw, synthetic_config)


def test_head_to_head_rejects_non_bot_members_in_the_main_bot_team(synthetic_config: SeasonConfig) -> None:
    raw = make_team_tournament(_scores(-1, 5), BOT_MEMBERS[:2] + (("a_human", "pro"),), PRO_MEMBERS)
    with pytest.raises(ValueError, match="1 non-bot members"):
        compute_head_to_head(raw, synthetic_config)


def test_shared_question_pairs_reads_the_simulation_notes() -> None:
    raw = make_team_tournament(_scores(-1, 5, count=3), BOT_MEMBERS, PRO_MEMBERS)
    assert sorted(shared_question_pairs(raw)) == [(1, 1), (2, 2), (3, 3)]
    next(iter(raw["scores_cache"].values()))["forecast"]["question"]["notes"] = "Some other note"
    with pytest.raises(ValueError, match="Can't find the Pro and bot question ids"):
        shared_question_pairs(raw)


def test_find_close_time_differences() -> None:
    pro_questions = {1: {"spot_scoring_time": "2026-06-05T12:00:00Z"}, 2: {"spot_scoring_time": "2026-06-06T12:00:00Z"}}
    bot_questions = {11: {"spot_scoring_time": "2026-06-05T13:30:00Z"}, 12: {"spot_scoring_time": "2026-06-06T12:00:00Z"}}
    differences = find_close_time_differences([(1, 11), (2, 12)], pro_questions, bot_questions)
    assert [(difference.bot_question_id, difference.minutes_bot_closed_after_pro) for difference in differences] == [(11, 90.0)]


def test_check_comparison_uses_pro_weights() -> None:
    pro_questions = {1: {"weight": 1.0}, 2: {"weight": 1.0}}
    check_comparison_uses_pro_weights({1: 1.0, 2: 1.0}, pro_questions)
    with pytest.raises(ValueError, match="don't use the Pro question's weight"):
        check_comparison_uses_pro_weights({1: 1.0, 2: 0.7}, pro_questions)


def test_head_to_head_rejects_asymmetric_scores(synthetic_config: SeasonConfig) -> None:
    raw = make_team_tournament(_scores(-1, 5), BOT_MEMBERS, PRO_MEMBERS)
    pro_score = next(score for score in raw["scores_cache"].values() if score["forecast"]["user"]["name"] == "Pro Team" and score["type"] == "spot_peer")
    pro_score["score"] += 1
    with pytest.raises(ValueError, match="exact opposites"):
        compute_head_to_head(raw, synthetic_config)


def test_season_change_p_value_matches_hand_calculation() -> None:
    previous = SeasonTrendPoint("Spring 2026", -1.25, (-4.87, 2.37), 99, False)
    current = SeasonTrendPoint("Summer 2026", -0.26, (-6.47, 5.96), 97, True)
    assert season_change_p_value(previous, current) == pytest.approx(0.784, abs=0.002)


def test_season_change_uses_normal_critical_value_without_question_count() -> None:
    previous = SeasonTrendPoint("Q3 2024", 0.0, (-1.96, 1.96), None, False)
    current = SeasonTrendPoint("Q4 2024", math.sqrt(2) * 1.959964, (math.sqrt(2) * 1.959964 - 1.96, math.sqrt(2) * 1.959964 + 1.96), None, True)
    assert season_change_p_value(previous, current) == pytest.approx(0.05, abs=1e-4)


@pytest.mark.parametrize(
    ("name", "is_baseline"),
    [
        ("metac-gpt-5-5-high+asknews", True),
        ("metac-deepseek-r1+no-research", True),
        ("metac-deepseek-v3-1+asknews(variance-test-1)", True),
        ("metac-gpt-4-1+asknews[optimized-prompt]", False),
        ("metac-asknews-deepnews[research-only]", False),
        ("metac-agent", False),
        ("metac-autocast", False),
        ("metac-uniform-probability-bot", False),
        ("GreeneiBot2", False),
    ],
)
def test_baseline_metac_bot_rule(name: str, is_baseline: bool) -> None:
    assert is_baseline_metac_bot(name) is is_baseline


def test_split_baseline_name_separates_variance_tests() -> None:
    assert split_baseline_name("metac-deepseek-v3-1+asknews(variance-test-2)") == ("deepseek-v3-1", "asknews", "variance-test-2")
    assert split_baseline_name("metac-o3-high+asknews") == ("o3-high", "asknews", None)
    with pytest.raises(ValueError):
        split_baseline_name("metac-agent")


def test_experiment_groups() -> None:
    rows = [
        _row("metac-deepseek-r1+asknews", 1),
        _row("metac-deepseek-r1+exa", 2),
        _row("metac-deepseek-r1+no-research", 3),
        _row("metac-deepseek-r1+asknews(variance-test)", 4),
        _row("metac-gpt-5+asknews", 5),
        _row("metac-gpt-5-high+asknews", 6),
        _row("metac-claude+asknews", 7),
        _row("metac-claude+exa", 8),
    ]
    other_rows = [_row("metac-gpt-5-no-research-one-shot", -4), _row("metac-unknown-no-research-one-shot", -9), _row("metac-agent", 9)]
    groups = find_metac_experiment_groups(rows, other_rows)
    assert list(groups.search_variants) == ["deepseek-r1"]
    assert [row.name for row in groups.search_variants["deepseek-r1"]] == [
        "metac-deepseek-r1+asknews", "metac-deepseek-r1+exa", "metac-deepseek-r1+no-research"
    ]
    assert [(standard.name, high.name) for standard, high in groups.reasoning_pairs] == [
        ("metac-gpt-5+asknews", "metac-gpt-5-high+asknews")
    ]
    assert list(groups.variance_tests) == ["deepseek-r1+asknews"]
    assert len(groups.variance_tests["deepseek-r1+asknews"]) == 2
    assert [(one_shot.name, [row.name for row in standard]) for one_shot, standard in groups.no_research_pairs] == [
        ("metac-gpt-5-no-research-one-shot", ["metac-gpt-5+asknews", "metac-gpt-5-high+asknews"])
    ]


def test_anonymize_pros_by_rank_order() -> None:
    rows = [
        LeaderboardRow(1, "zeta_pro", "pro", 50, 5, None, 10),
        LeaderboardRow(2, "some-bot", "bot", 40, 4, None, 10),
        LeaderboardRow(3, "alpha_pro", "pro", 30, 3, None, 10),
    ]
    anonymized, aliases = anonymize_pros(rows)
    assert [row.name for row in anonymized] == ["Pro A", "some-bot", "Pro B"]
    assert aliases == {"zeta_pro": "Pro A", "alpha_pro": "Pro B"}


def test_compute_field_groups_bots() -> None:
    rows = [
        _row(f"top-{index}", 20 - index, questions=100, ci=(10.0, 30.0), rank=index + 1) for index in range(10)
    ] + [
        _row("promising", 25, questions=30, rank=11),
        _row("tiny-sample", 40, questions=5, rank=12),
        _row("metac-uniform-probability-bot", -12, questions=100, rank=13),
        _row("weak", -20, questions=90, rank=14),
    ]
    field = compute_field(rows, [1.0, -1.0, 3.0, -3.0], scored_question_count=100)
    assert field.eligible_min_questions == 50
    assert field.promising_min_questions == 10
    assert [row.name for row in field.promising] == ["promising"]
    assert field.best_eligible.name == "top-0"
    assert field.worst_eligible.name == "weak"
    assert field.uniform_reference is not None and field.uniform_reference.average_score == -12
    assert field.tenth_place.name == "top-9"
    assert field.per_question_score_sd == pytest.approx(math.sqrt(5))
    assert field.top_bot_clearly_ahead is False


def test_format_resolution() -> None:
    assert format_resolution(make_question(1, "binary", True)) == "Yes"
    assert format_resolution(make_question(1, "binary", False)) == "No"
    assert format_resolution(make_question(1, "multiple_choice", "B")) == "B"
    assert format_resolution(make_question(1, "numeric", 480126.0)) == "480,126"
    assert format_resolution(make_question(1, "numeric", 4.10300)) == "4.103"
    assert format_resolution(make_question(1, "numeric", "above_upper_bound")) == "above the upper bound"


def test_pick_example_questions_prefers_pins_then_shortest_titles() -> None:
    questions = {
        1: make_question(1, "binary", True, title="A long binary question title?"),
        2: make_question(2, "binary", False, title="Short?"),
        3: make_question(3, "numeric", 12.0, title="Numeric?"),
        4: make_question(4, "binary", None, title="No?"),
    }
    examples = pick_example_questions(questions, {})
    assert [(example.question_type, example.title) for example in examples] == [("binary", "Short?"), ("numeric", "Numeric?")]
    pinned = pick_example_questions(questions, {"binary": 1})
    assert pinned[0].title == "A long binary question title?" and pinned[0].was_pinned
    with pytest.raises(ValueError, match="not a scored question"):
        pick_example_questions(questions, {"binary": 4})


def test_find_simulation_file_needs_exactly_one_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Pro tournament"):
        find_simulation_file(tmp_path, "pro_tournament.json")
    (tmp_path / "1_pro_tournament.json").write_text("{}")
    assert find_simulation_file(tmp_path, "pro_tournament.json").name == "1_pro_tournament.json"


def test_compute_season_results_on_synthetic_folder(pros_ahead_folder: Path, synthetic_config: SeasonConfig) -> None:
    results = compute_season_results(pros_ahead_folder, synthetic_config)
    overview = results.overview
    assert overview.bot_question_count == BOT_QUESTION_COUNT
    assert overview.bot_scored_count == BOT_QUESTION_COUNT - len(ANNULLED_BOT_QUESTION_IDS)
    assert overview.bot_only_scored_count == BOT_QUESTION_COUNT - SHARED_QUESTION_COUNT - len(ANNULLED_BOT_QUESTION_IDS)
    assert overview.pro_count == 3
    assert overview.pro_annulled_count == 1
    assert results.head_to_head.leader == "pros"
    assert results.individuals.excluded_account_count == 0
    assert {row.name for row in results.individuals.rows if row.user_type == "pro"} == {"Pro A", "Pro B", "Pro C"}
    assert [result.team_size for result in results.team_sizes] == [1, 2, 3, 12]
    assert results.team_sizes[-1].is_all_bots
    assert overview.shared_question_weights_below_1 == ()
    assert overview.close_time_differences == ()
    assert overview.non_primary_bots == (NON_PRIMARY_BOT_NAME,)
    assert bot_label(NON_PRIMARY_BOT_NAME, overview.non_primary_bots) == f"{NON_PRIMARY_BOT_NAME} (non-primary)"
    assert bot_label("bot-a", overview.non_primary_bots) == "bot-a"
    assert "metac-agent" in {row.name for row in results.metac.other_rows}
    assert all(is_baseline_metac_bot(row.name) for row in results.metac.baseline_rows)
    assert results.season_trend.current.season_label == "Summer 2026"
    assert results.example_questions[0].title == "Short?"


def test_accuracy_results_on_synthetic_folder(pros_ahead_folder: Path, synthetic_config: SeasonConfig) -> None:
    results = compute_season_results(pros_ahead_folder, synthetic_config)
    field_accuracy = results.bot_field_accuracy
    assert field_accuracy.top_bot.name == results.field.rows[0].name
    assert field_accuracy.worst_bot.name == results.field.worst_eligible.name
    assert field_accuracy.top_bot.typical_probability > field_accuracy.median_probability > field_accuracy.worst_bot.typical_probability
    assert "metac-uniform-probability-bot" not in {bot.name for bot in field_accuracy.eligible_bots}
    assert 0 < field_accuracy.yes_rate < 1
    assert 0 <= field_accuracy.share_below_base_rate <= 1
    team_accuracy = results.team_accuracy
    assert team_accuracy.binary_question_count == results.head_to_head.question_type_counts["binary"]
    assert team_accuracy.bot_to_pro_ratio == pytest.approx(
        team_accuracy.bot_team_probability / team_accuracy.pro_team_probability
    )


def test_compute_season_results_requires_main_team_file(pros_ahead_folder: Path, synthetic_config: SeasonConfig) -> None:
    with pytest.raises(FileNotFoundError, match="main bot team size 7"):
        compute_season_results(pros_ahead_folder, replace(synthetic_config, main_bot_team_size=7))


def test_team_sizes_reject_human_accounts_unless_the_season_allows_them(
    pros_ahead_folder: Path, synthetic_config: SeasonConfig
) -> None:
    team_file = pros_ahead_folder / "7_pro_vs_bot_tournament__teams_size_all.json"
    team_file.write_text(json.dumps(make_team_tournament(_scores(-1, 5, count=SHARED_QUESTION_COUNT), BOT_MEMBERS + (("a_human", "pro"),), PRO_MEMBERS)))
    with pytest.raises(ValueError, match="1 human accounts"):
        compute_season_results(pros_ahead_folder, synthetic_config)
    compute_season_results(pros_ahead_folder, replace(synthetic_config, human_accounts_in_bot_teams_allowed=True))
