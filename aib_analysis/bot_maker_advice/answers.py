"""Load a season's survey answers, anonymize them, and apply the sharing rules.

Every non-blank answer to the lessons question is published verbatim under a random ID, as the survey
told respondents it would be. An answer to the "tried and abandoned" question is published, and so
analyzed, only when the respondent chose "Yes, publicly" for sharing their individual response.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

COLUMN_PREFIXES: dict[str, str] = {
    "timestamp": "Timestamp",
    "bot_name": "What is your bot's name",
    "lessons": "In summary, what should other bot makers learn",
    "abandoned": "Tell us about any approaches you tried and abandoned",
    "response_consent": "Can we share your individual survey response",
    "code_link": "If you are a prize winner, provide a way to review your code",
    "code_consent": "If you included a link to your code or a description",
}
LESSONS_QUESTION = "In summary, what should other bot makers learn from your experience?"
ABANDONED_QUESTION = "Tell us about any approaches you tried and abandoned, and why"
PUBLIC_RESPONSE_CONSENT = "Yes, publicly"
NO_RESPONSE_CONSENT = "No"
RESEARCHERS_ONLY_CONSENT_PREFIX = "Yes, but only to"
OPEN_SOURCE_CODE_CONSENT = "Yes everything (my bot is open source and the repo is public)"
NULL_ANSWERS = frozenset({"n/a", "na", "none", "no", "nothing", "nope", "-"})
URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]\"']+")


@dataclass(frozen=True)
class Answer:
    answer_id: int
    bot_name: str
    lessons: str
    abandoned: str | None
    nonstandard_consent: str | None = None

    @property
    def texts(self) -> tuple[str, ...]:
        return (self.lessons,) if self.abandoned is None else (self.lessons, self.abandoned)


@dataclass(frozen=True)
class OpenSourceBot:
    bot_name: str
    url: str | None


@dataclass(frozen=True)
class SurveyAnswers:
    answers: tuple[Answer, ...]
    respondent_count: int
    blank_lessons_count: int
    null_lessons_count: int
    withheld_abandoned_count: int
    open_source_bots: tuple[OpenSourceBot, ...]

    def by_id(self) -> dict[int, Answer]:
        return {answer.answer_id: answer for answer in self.answers}


def assign_answer_ids(count: int, seed: str) -> list[int]:
    """Random two-digit IDs (three digits past 90 answers), so an ID says nothing about when someone answered."""
    id_range = range(10, 100) if count <= 90 else range(100, 1000)
    return random.Random(seed).sample(id_range, count)


def first_url(text: str) -> str | None:
    urls = [url.rstrip(".,;:!?") for url in URL_PATTERN.findall(text)]
    github_urls = [url for url in urls if "github.com" in url]
    return (github_urls or urls or [None])[0]


def _is_null_answer(text: str) -> bool:
    return text.lower().strip(" .!") in NULL_ANSWERS


def _nonstandard_consent(consent: str) -> str | None:
    if not consent or consent in (PUBLIC_RESPONSE_CONSENT, NO_RESPONSE_CONSENT) or consent.startswith(RESEARCHERS_ONLY_CONSENT_PREFIX):
        return None
    return consent


def _column_indexes(header: list[str]) -> dict[str, int]:
    indexes = {}
    for slug, prefix in COLUMN_PREFIXES.items():
        matches = [index for index, name in enumerate(header) if name.strip().startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one survey column starting with {prefix!r}, found {len(matches)}")
        indexes[slug] = matches[0]
    return indexes


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def load_answers(survey_csv: Path, id_seed: str) -> SurveyAnswers:
    with open(survey_csv, newline="") as file_handle:
        rows = list(csv.reader(file_handle))
    indexes = _column_indexes(rows[0])
    records = [
        {slug: _clean(row[index]) if index < len(row) else "" for slug, index in indexes.items()}
        for row in rows[1:]
        if any(cell.strip() for cell in row)
    ]
    answered = [record for record in records if record["lessons"] and not _is_null_answer(record["lessons"])]
    answer_ids = assign_answer_ids(len(answered), id_seed)
    answers = []
    withheld_abandoned_count = 0
    for answer_id, record in zip(answer_ids, answered):
        shareable = record["response_consent"] == PUBLIC_RESPONSE_CONSENT
        if record["abandoned"] and not shareable:
            withheld_abandoned_count += 1
        answers.append(
            Answer(
                answer_id=answer_id,
                bot_name=record["bot_name"],
                lessons=record["lessons"],
                abandoned=record["abandoned"] if shareable and record["abandoned"] else None,
                nonstandard_consent=_nonstandard_consent(record["response_consent"]) if record["abandoned"] else None,
            )
        )
    open_source_bots = tuple(
        OpenSourceBot(bot_name=record["bot_name"], url=first_url(record["code_link"]))
        for record in records
        if record["code_consent"] == OPEN_SOURCE_CODE_CONSENT
    )
    survey_answers = SurveyAnswers(
        answers=tuple(sorted(answers, key=lambda answer: answer.answer_id)),
        respondent_count=len(records),
        blank_lessons_count=sum(1 for record in records if not record["lessons"]),
        null_lessons_count=sum(1 for record in records if record["lessons"] and _is_null_answer(record["lessons"])),
        withheld_abandoned_count=withheld_abandoned_count,
        open_source_bots=open_source_bots,
    )
    logger.info(
        f"Loaded {survey_answers.respondent_count} respondents: {len(answers)} answered the lessons question "
        f"({survey_answers.blank_lessons_count} blank, {survey_answers.null_lessons_count} null-like); "
        f"{sum(answer.abandoned is not None for answer in answers)} 'tried and abandoned' answers are shareable "
        f"and {withheld_abandoned_count} are withheld"
    )
    return survey_answers
