from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aib_analysis.bot_maker_advice.answers import (
    assign_answer_ids,
    first_url,
    load_answers,
)
from aib_analysis.bot_maker_advice.season_config import AdviceSeasonConfig
from tests.bot_maker_advice.conftest import (
    ABANDONED_TEXT,
    NONSTANDARD_CONSENT_ROW,
    PRIVATE_ABANDONED_ROW,
    PUBLIC_ABANDONED_ROW,
    THEMES_PER_ANSWER,
)


def test_load_answers_applies_the_sharing_rules(synthetic_config: AdviceSeasonConfig) -> None:
    survey = load_answers(synthetic_config.survey_csv, id_seed="test-ids")
    assert survey.respondent_count == len(THEMES_PER_ANSWER) + 2
    assert len(survey.answers) == len(THEMES_PER_ANSWER)
    assert (survey.blank_lessons_count, survey.null_lessons_count) == (1, 1)
    by_bot = {answer.bot_name: answer for answer in survey.answers}
    assert by_bot[f"synthetic-bot-{PUBLIC_ABANDONED_ROW}"].abandoned == ABANDONED_TEXT
    assert by_bot[f"synthetic-bot-{PRIVATE_ABANDONED_ROW}"].abandoned is None
    nonstandard = by_bot[f"synthetic-bot-{NONSTANDARD_CONSENT_ROW}"]
    assert nonstandard.abandoned is None and nonstandard.nonstandard_consent == "yes, but anonymously"
    assert survey.withheld_abandoned_count == 2
    assert [answer.answer_id for answer in survey.answers] == sorted(answer.answer_id for answer in survey.answers)
    assert [(bot.bot_name, bot.url) for bot in survey.open_source_bots] == [
        ("synthetic-bot-0", "https://github.com/example/bot-0"),
        ("synthetic-bot-1", "https://github.com/example/bot-1"),
        ("synthetic-bot-2", None),
    ]


def test_answer_ids_are_stable_unique_and_not_in_answer_order() -> None:
    ids = assign_answer_ids(46, "summer_2026-answer-ids")
    assert ids == assign_answer_ids(46, "summer_2026-answer-ids")
    assert len(set(ids)) == 46 and all(10 <= answer_id <= 99 for answer_id in ids)
    assert ids != sorted(ids)
    assert all(100 <= answer_id <= 999 for answer_id in assign_answer_ids(91, "seed"))


def test_first_url_prefers_github_and_strips_trailing_punctuation() -> None:
    assert first_url("Docs at https://example.com/a, code at https://github.com/org/repo).") == "https://github.com/org/repo"
    assert first_url("Traces: https://example.com/data.") == "https://example.com/data"
    assert first_url("No link here") is None


def test_load_answers_rejects_a_survey_missing_a_column(tmp_path: Path) -> None:
    path = tmp_path / "survey.csv"
    with open(path, "w", newline="") as file_handle:
        csv.writer(file_handle).writerows([["Timestamp", "What is your bot's name?"], ["10/1/2026", "bot"]])
    with pytest.raises(ValueError, match="In summary, what should other bot makers learn"):
        load_answers(path, id_seed="test-ids")
