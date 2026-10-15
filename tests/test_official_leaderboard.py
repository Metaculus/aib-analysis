import pytest

from aib_analysis.main_logic.official_leaderboard import build_official_leaderboard


def ranks_by_user(score_by_user: dict[str, float], excluded: set[str]) -> dict[str, int]:
    return {
        entry.user_name: entry.rank
        for entry in build_official_leaderboard(score_by_user, excluded)
    }


def test_tied_scores_share_a_rank_and_the_next_rank_is_skipped() -> None:
    ranks = ranks_by_user({"first": 10.0, "tied": 10.0, "third": 5.0}, set())
    assert ranks == {"first": 1, "tied": 1, "third": 3}


def test_excluded_users_are_shown_but_do_not_use_up_a_rank() -> None:
    ranks = ranks_by_user({"bot_a": 10.0, "human": 8.0, "bot_b": 5.0}, {"human"})
    assert ranks == {"bot_a": 1, "human": 2, "bot_b": 2}


def test_take_is_squared_positive_score() -> None:
    entries = build_official_leaderboard({"winner": 3.0, "loser": -4.0}, set())
    takes = {entry.user_name: entry.take for entry in entries}
    assert takes == {"winner": 9.0, "loser": 0.0}


def test_excluding_a_user_without_a_score_raises() -> None:
    with pytest.raises(ValueError):
        build_official_leaderboard({"bot_a": 1.0}, {"missing"})
