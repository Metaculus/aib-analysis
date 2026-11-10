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
    _split_bot_list,
    _to_float,
    load_survey,
    normalize_name,
    winner_sources,
)


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


def test_to_float_parses_numbers_and_blanks():
    assert _to_float("2.5") == 2.5
    assert _to_float("3,374.02") == pytest.approx(3374.02)
    assert _to_float("") == 0.0


@pytest.mark.parametrize("junk", ["junk", "n/a", "$5"])
def test_to_float_raises_on_junk_instead_of_reading_zero(junk):
    # A prize read as 0 would silently turn a winner into a non-winner.
    with pytest.raises(ValueError, match="Expected a number"):
        _to_float(junk)


# --------------------------------------------------------------------------- #
# Leaderboard match: exact, alnum, none (never loose substring)
# --------------------------------------------------------------------------- #
def _row(name: str) -> LeaderboardRow:
    return LeaderboardRow(
        rank=1, bot_name=name, sum_spot_peer=1.0, question_count=100, questions_open_after_start=120
    )


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
        aib_prize=kw.get("aib_prize", 100.0),
        in_futureeval=True,
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
# Bot-level winners (the prize sheet is owner-level)
# --------------------------------------------------------------------------- #
def _scored_row(rank: int, name: str, score: float) -> LeaderboardRow:
    return LeaderboardRow(rank, name, score, 100, 100)


def _owner_named(owner: str, bots: list[str], prize: float) -> PrizeOwner:
    return _owner(bots, owner=owner, aib_prize=prize)


def test_single_bot_prize_winner_is_a_winner_and_sets_the_bar():
    rows = [_scored_row(1, "solo-a", 50.0), _scored_row(2, "solo-b", 20.0), _scored_row(3, "loser", 10.0)]
    owners = [
        _owner_named("a", ["solo-a"], 100.0),
        _owner_named("b", ["solo-b"], 50.0),
        _owner_named("c", ["loser"], 0.0),
    ]
    assert winner_sources(owners, rows) == {
        "soloa": "single-bot prize winner",
        "solob": "single-bot prize winner",
    }


def test_any_bot_above_the_lowest_single_bot_winner_is_a_winner():
    rows = [
        _scored_row(1, "multi-main", 60.0),
        _scored_row(2, "solo", 30.0),
        _scored_row(3, "multi-secondary", 25.0),
        _scored_row(4, "metac-house", 70.0),
    ]
    owners = [_owner_named("m", ["multi-main", "multi-secondary"], 200.0), _owner_named("s", ["solo"], 10.0)]
    sources = winner_sources(owners, rows)
    assert sources == {"solo": "single-bot prize winner", "multimain": "scored above a prize winner"}


def test_prize_owner_with_no_identifiable_winning_bot_raises():
    rows = [_scored_row(1, "solo", 30.0), _scored_row(2, "multi-1", 20.0), _scored_row(3, "multi-2", 10.0)]
    owners = [_owner_named("s", ["solo"], 100.0), _owner_named("m", ["multi-1", "multi-2"], 5.0)]
    with pytest.raises(ValueError, match="cannot be identified"):
        winner_sources(owners, rows)


def test_single_bot_winner_missing_from_leaderboard_raises():
    with pytest.raises(ValueError, match="not on the leaderboard"):
        winner_sources([_owner_named("s", ["ghost"], 100.0)], [_scored_row(1, "other", 1.0)])


# --------------------------------------------------------------------------- #
# Survey loading: blank bot names must never be dropped silently
# --------------------------------------------------------------------------- #
def _write_survey_csv(path, rows: list[list[str]], header: list[str] | None = None) -> None:
    # Timestamp and bot name first, so each test row can give just those two cells.
    header = header or [config.COLUMNS["timestamp"], config.COLUMNS["bot_name"]] + [
        text for slug, text in config.COLUMNS.items() if slug not in ("timestamp", "bot_name")
    ]
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


def test_load_survey_raises_on_a_column_missing_from_the_export(tmp_path, monkeypatch):
    path = tmp_path / "survey.csv"
    header = [text for slug, text in config.COLUMNS.items() if slug != "hours"]
    _write_survey_csv(path, [["1/1/26", "bot-a"]], header=header)
    monkeypatch.setattr(config, "SURVEY_CSV", str(path))
    with pytest.raises(ValueError, match="Missing from the CSV"):
        load_survey()


def test_load_survey_raises_on_a_question_the_config_does_not_know(tmp_path, monkeypatch):
    path = tmp_path / "survey.csv"
    header = list(config.COLUMNS.values()) + ["A brand new question?"]
    _write_survey_csv(path, [["1/1/26", "bot-a"]], header=header)
    monkeypatch.setattr(config, "SURVEY_CSV", str(path))
    with pytest.raises(ValueError, match="A brand new question"):
        load_survey()
