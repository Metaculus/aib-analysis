"""Join / matching logic in loading.py.

The report's groups and scores are only as good as the survey-bot -> leaderboard
and survey-bot -> prize joins. These test the pure matching helpers directly,
with special attention to the guards that stop a short name from being joined to
the wrong bot. Runs without env vars or local data.
"""

from __future__ import annotations

import csv

import pytest

from aib_analysis.survey_analysis import config
from aib_analysis.survey_analysis.leaderboard import LeaderboardRow
from aib_analysis.survey_analysis.loading import (
    PrizeOwner,
    Respondent,
    _alnum_key,
    _match_leaderboard,
    _match_prize,
    _resolve_winner,
    _split_bot_list,
    _to_float,
    _to_int,
    load_survey,
    normalize_name,
)
from aib_analysis.survey_analysis.manual_adjustments import WinnerOverride


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_name_strips_emoji_and_collapses_space():
    assert normalize_name("\U0001F916  My   Bot ") == "my bot"


def test_alnum_key_strips_punctuation_emoji_and_case():
    assert _alnum_key("Preseen-Atlas") == "preseenatlas"
    assert _alnum_key("MWG Bot") == "mwgbot"
    assert _alnum_key("\U0001F916 5cast-v1") == "5castv1"


def test_split_bot_list_on_commas_and_semicolons():
    assert _split_bot_list("a, b; c ,, d") == ["a", "b", "c", "d"]
    assert _split_bot_list("") == []


def test_to_int_and_to_float_tolerate_junk():
    assert _to_int("3") == 3
    assert _to_int("3.0") == 3
    assert _to_int("") == 0
    assert _to_int("n/a") == 0
    assert _to_float("2.5") == 2.5
    assert _to_float("") == 0.0
    assert _to_float("junk") == 0.0


# --------------------------------------------------------------------------- #
# Leaderboard match: exact, alnum, none (never loose substring)
# --------------------------------------------------------------------------- #
def _row(name: str) -> LeaderboardRow:
    return LeaderboardRow(rank=1, bot_name=name, sum_spot_peer=1.0, question_count=100)


def test_leaderboard_exact_match():
    by_norm = {"my bot": _row("My Bot")}
    by_alnum = {"mybot": _row("My Bot")}
    row, kind = _match_leaderboard("My Bot", by_norm, by_alnum)
    assert row is not None and kind == "exact"


def test_leaderboard_alnum_match_when_punctuation_differs():
    by_norm = {"my bot": _row("My Bot")}
    by_alnum = {"mybot": _row("My Bot")}
    # "My-Bot" normalizes to "my-bot" (no exact hit) but alnum "mybot" matches.
    row, kind = _match_leaderboard("My-Bot", by_norm, by_alnum)
    assert row is not None and kind == "alnum"


def test_leaderboard_no_loose_substring_match():
    by_norm = {"mybotpro": _row("MyBotPro")}
    by_alnum = {"mybotpro": _row("MyBotPro")}
    row, kind = _match_leaderboard("MyBot", by_norm, by_alnum)
    assert row is None and kind == "none"


# --------------------------------------------------------------------------- #
# Prize match: exact vs tokenized, with the >=4-char guard
# --------------------------------------------------------------------------- #
def _owner(bots: list[str], **kw) -> PrizeOwner:
    return PrizeOwner(
        owner_username=kw.get("owner", "someone"),
        bot_usernames=bots,
        winner_count=kw.get("winner_count", 1),
        aib_prize=kw.get("aib_prize", 100.0),
        total_prize=kw.get("total_prize", 100.0),
    )


def test_prize_exact_match_against_bot_list():
    owner, kind = _match_prize("5cast-v1", [_owner(["5cast-v1", "5cast-v2"])])
    assert owner is not None and kind == "exact"


def test_prize_tokenized_substring_match():
    # Survey name "atlasforecaster" is a superstring of listed "atlasforecast"
    # (both >= 6 chars), and the candidate owner is unique.
    owner, kind = _match_prize("atlasforecaster", [_owner(["atlasforecast"])])
    assert owner is not None and kind == "tokenized"


def test_prize_tokenized_match_rejected_when_ambiguous():
    # Two different owners both plausibly contain the name -> left unmatched.
    owners = [_owner(["greeneibot"], owner="a"), _owner(["greeneibot2x"], owner="b")]
    owner, kind = _match_prize("greeneibot2", owners)
    assert owner is None and kind == "none"


def test_prize_short_names_do_not_trigger_tokenized_match():
    # "abc" (<4 alnum) must not tokenized-match, even though it's inside "abcbot".
    owner, kind = _match_prize("abc", [_owner(["abcbot"])])
    assert owner is None and kind == "none"


def test_prize_no_match_returns_none():
    owner, kind = _match_prize("totallydifferent", [_owner(["atlas"])])
    assert owner is None and kind == "none"


def test_prize_exact_beats_tokenized_priority():
    # An exact hit should win and be labeled "exact" even if a tokenized hit also exists.
    owners = [_owner(["somethingelse"]), _owner(["exactname"], owner="right")]
    owner, kind = _match_prize("exactname", owners)
    assert kind == "exact" and owner.owner_username == "right"


# --------------------------------------------------------------------------- #
# Respondent derived properties
# --------------------------------------------------------------------------- #
def test_average_spot_peer_divides_sum_by_questions():
    r = Respondent(bot_name="b", answers={}, sum_spot_peer=50.0, question_count=100)
    assert r.average_spot_peer == 0.5


def test_average_spot_peer_none_when_no_questions():
    assert Respondent("b", {}, sum_spot_peer=50.0, question_count=0).average_spot_peer is None
    assert Respondent("b", {}, sum_spot_peer=50.0, question_count=None).average_spot_peer is None
    assert Respondent("b", {}, sum_spot_peer=None, question_count=100).average_spot_peer is None


def test_meets_correlation_minimum_boundary():
    assert Respondent("b", {}, question_count=100).meets_correlation_minimum is True
    assert Respondent("b", {}, question_count=99).meets_correlation_minimum is False
    assert Respondent("b", {}, question_count=None).meets_correlation_minimum is False


def test_groups_include_top10_only_when_flagged():
    winner_top10 = Respondent("b", {}, is_winner=True, is_top_10=True)
    assert winner_top10.groups == ["winner", "top_10"]
    plain = Respondent("b", {}, is_winner=False, is_top_10=False)
    assert plain.groups == ["non_winner"]


# --------------------------------------------------------------------------- #
# Bot-level winner resolution (the prize sheet is owner-level)
# --------------------------------------------------------------------------- #
def test_single_bot_owner_status_transfers_to_bot():
    won = _owner(["solo-bot"], winner_count=1, aib_prize=50.0)
    assert _resolve_winner("solo-bot", won, None) == (True, "owner_single_bot")
    lost = _owner(["solo-bot"], winner_count=0, aib_prize=0.0)
    assert _resolve_winner("solo-bot", lost, None) == (False, "owner_single_bot")


def test_multi_bot_owner_with_no_wins_means_no_winner():
    owner = _owner(["bot-1", "bot-2"], winner_count=0, aib_prize=0.0)
    assert _resolve_winner("bot-1", owner, None) == (False, "owner_multi_bot_no_wins")


def test_multi_bot_owner_with_wins_requires_override():
    owner = _owner(["bot-1", "bot-2"], winner_count=1, aib_prize=100.0)
    with pytest.raises(ValueError, match="owner-level"):
        _resolve_winner("bot-1", owner, None)


def test_override_resolves_ambiguous_multi_bot_owner():
    owner = _owner(["bot-1", "bot-2"], winner_count=1, aib_prize=100.0)
    override = WinnerOverride(bot_name="bot-1", is_winner=True, reason="best ranked")
    assert _resolve_winner("bot-1", owner, override) == (True, "manual_override")
    loser_override = WinnerOverride(bot_name="bot-2", is_winner=False, reason="worse ranked")
    assert _resolve_winner("bot-2", owner, loser_override) == (False, "manual_override")


def test_override_applies_even_without_prize_match():
    override = WinnerOverride(bot_name="bot-x", is_winner=True, reason="known winner")
    assert _resolve_winner("bot-x", None, override) == (True, "manual_override")


def test_no_prize_match_defaults_to_non_winner():
    assert _resolve_winner("bot-x", None, None) == (False, "no_prize_match")


# --------------------------------------------------------------------------- #
# Survey loading: blank bot names must never be dropped silently
# --------------------------------------------------------------------------- #
def _write_survey_csv(path, rows: list[list[str]]) -> None:
    header = [config.COLUMNS["timestamp"], config.COLUMNS["bot_name"]]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_load_survey_skips_fully_empty_rows(tmp_path, monkeypatch):
    path = tmp_path / "survey.csv"
    _write_survey_csv(path, [["1/1/26", "bot-a"], ["", ""], ["1/2/26", "bot-b"]])
    monkeypatch.setattr(config, "SURVEY_CSV", str(path))
    records = load_survey()
    assert [record["bot_name"] for record in records] == ["bot-a", "bot-b"]


def test_load_survey_raises_on_answers_without_bot_name(tmp_path, monkeypatch):
    path = tmp_path / "survey.csv"
    _write_survey_csv(path, [["1/1/26", "bot-a"], ["1/2/26", ""]])
    monkeypatch.setattr(config, "SURVEY_CSV", str(path))
    with pytest.raises(ValueError, match="no bot name"):
        load_survey()
