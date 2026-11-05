"""A synthetic survey and a scripted stand-in for the models, so the pipeline runs without private data or network."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from aib_analysis.bot_maker_advice import prompts
from aib_analysis.bot_maker_advice.answers import (
    COLUMN_PREFIXES,
    OPEN_SOURCE_CODE_CONSENT,
    PUBLIC_RESPONSE_CONSENT,
)
from aib_analysis.bot_maker_advice.method import ModelRoster
from aib_analysis.bot_maker_advice.season_config import AdviceSeasonConfig

ReplyType = TypeVar("ReplyType", bound=BaseModel)

THEME_SENTENCES: dict[int, tuple[str, str]] = {
    1: ("Keep the pipeline simple", "Keep the pipeline simple because extra agents made our scores worse."),
    2: ("Invest in research quality", "Spend most of your effort on research quality since news retrieval mattered most."),
    3: ("Backtest changes on resolved questions", "Backtest every change on resolved questions before you ship it."),
    4: ("Avoid overconfident forecasts", "Avoid overconfidence by capping extreme probabilities near the tails."),
    5: ("Track costs per question", "Track API costs per question so the budget does not run out."),
}
OPPOSES_THEME_1 = "Complex multi-agent pipelines worked better than simple ones for us."
ONE_OFF_SENTENCE = "Use a separate model to read the resolution criteria line by line."
REJECT_MARKER = "Honestly I am not sure about any of this."
SIGNIFICANT_EVIDENCE = "Total development hours (midpoint)"
NOISY_EVIDENCE = "Uses explicit base rates"

THEMES_PER_ANSWER: dict[int, tuple[int, ...]] = {
    0: (1, 2),
    1: (1, 3),
    2: (1, 4),
    3: (1, 2, 5),
    4: (2, 3),
    5: (2, 3),
    6: (2,),
    7: (3, 4),
    8: (4,),
    9: (5,),
    10: (1, 5),
    11: (3,),
    12: (2, 4),
    13: (1, 3),
}
PUBLIC_ABANDONED_ROW = 4
PRIVATE_ABANDONED_ROW = 5
NONSTANDARD_CONSENT_ROW = 8
ABANDONED_TEXT = "We tried an ensemble of five prompts and dropped it because it cost too much."

FAKE_MODELS = ModelRoster(
    theme_generators=("fake/generator-a", "fake/generator-b"),
    codebook="fake/codebook",
    coders=("fake/coder-1", "fake/coder-2"),
    tie_breaker="fake/tie-breaker",
    verifier="fake/verifier",
    one_offs="fake/one-offs",
    data_check="fake/data-check",
    data_check_confirmer="fake/confirmer",
)

SURVEY_REPORT = f"""# Survey analysis

## Evidence summary

Some explanation.

- **{SIGNIFICANT_EVIDENCE}**: r = -0.51 · p = <0.001 · q = 0.012 · significant: yes
- **{NOISY_EVIDENCE}**: r = +0.10 · p = 0.245 · q = 0.245 · significant: no · precommitted: yes

## Next section
"""


def _lessons_text(row: int) -> str:
    sentences = [THEME_SENTENCES[number][1] for number in THEMES_PER_ANSWER[row]]
    if row == 13:
        sentences.append(OPPOSES_THEME_1)
    if row == 9:
        sentences.append(ONE_OFF_SENTENCE)
    if row == 10:
        sentences.append(REJECT_MARKER)
    return " ".join(sentences)


def write_survey_csv(path: Path) -> None:
    header = [
        "Timestamp",
        "What is your bot's name as listed in the Synthetic 2026 Leaderboard?",
        "Which LLM model(s) did you use to make your final prediction/answer?",
        "Tell us about any approaches you tried and abandoned, and why ",
        "Can we share your individual survey response in association with your bot outside our aggregated results?",
        "If you are a prize winner, provide a way to review your code or a general overview of your bot",
        "If you included a link to your code or a description, can we share it publicly?",
        "In summary, what should other bot makers learn from your experience? ",
    ]
    assert len(header) == len(COLUMN_PREFIXES) + 1
    rows = []
    for row in range(len(THEMES_PER_ANSWER)):
        consent = PUBLIC_RESPONSE_CONSENT if row % 2 == 0 else "No"
        if row == NONSTANDARD_CONSENT_ROW:
            consent = "yes, but anonymously"
        abandoned = ABANDONED_TEXT if row in (PUBLIC_ABANDONED_ROW, PRIVATE_ABANDONED_ROW, NONSTANDARD_CONSENT_ROW) else ""
        code_consent = OPEN_SOURCE_CODE_CONSENT if row in (0, 1) else "Don't share anything"
        code_link = "See https://github.com/example/bot-0." if row == 0 else ""
        rows.append([f"10/{row + 1}/2026 10:00:00", f"synthetic-bot-{row}", "GPT", abandoned, consent, code_link, code_consent, _lessons_text(row)])
    rows.append(["10/20/2026 10:00:00", "synthetic-bot-blank", "GPT", "", "No", "", "", "  "])
    rows.append(["10/21/2026 10:00:00", "synthetic-bot-null", "GPT", "", "No", "", "", "N/A"])
    with open(path, "w", newline="") as file_handle:
        csv.writer(file_handle).writerows([header, *rows])


@pytest.fixture
def synthetic_config(tmp_path: Path) -> AdviceSeasonConfig:
    survey_csv = tmp_path / "survey.csv"
    write_survey_csv(survey_csv)
    report = tmp_path / "survey_report.md"
    report.write_text(SURVEY_REPORT)
    return AdviceSeasonConfig(
        key="synthetic_2026",
        label="Synthetic 2026",
        survey_csv=survey_csv,
        tournament_url="https://example.com/tournament",
        pro_vs_bot_post_url=None,
        survey_post_url="https://example.com/survey",
        survey_report_markdown=report,
    )


ANSWER_PATTERN = re.compile(r"Bot maker #(?P<answer_id>\d+)\nWhat others should learn: (?P<lessons>[^\n]*)")


class ScriptedLlm:
    """Answers each stage from the synthetic sentences; coder 2 misses theme 4 so tie-breaks and alpha get exercised."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def usage_note(self) -> str:
        return f"{len(self.calls)} scripted calls"

    async def ask(self, stage: str, model: str, prompt: str, reply_type: type[ReplyType]) -> ReplyType:
        self.calls.append((stage, model))
        answers = {int(match["answer_id"]): match["lessons"] for match in ANSWER_PATTERN.finditer(prompt)}
        if reply_type is prompts.CandidateThemes:
            return reply_type(themes=[self._candidate(number, answers) for number in THEME_SENTENCES])
        if reply_type is prompts.Codebook:
            return reply_type(
                themes=[
                    prompts.CodebookEntry(
                        label=label,
                        definition=f"The answer says to {label.lower()}.",
                        include_when="It gives this advice.",
                        exclude_when="It only mentions the topic.",
                        source_candidates=[f"A{number}", f"B{number}"] if number != 5 else ["A5"],
                    )
                    for number, (label, _sentence) in THEME_SENTENCES.items()
                ]
            )
        if reply_type is prompts.AnswerCoding:
            return reply_type(codes=self._codes(model, prompt, answers))
        if reply_type is prompts.CitationCheckReply:
            rejected = REJECT_MARKER in prompt
            return reply_type(verdict="not_supported" if rejected else "supported", excerpt_shows_claim=not rejected, reason="Scripted.")
        if reply_type is prompts.OneOffCandidates:
            return reply_type(
                one_offs=[
                    prompts.OneOffCandidate(answer_id=answer_id, label="Read resolution criteria with a separate model", excerpt=ONE_OFF_SENTENCE)
                    for answer_id, lessons in answers.items()
                    if ONE_OFF_SENTENCE in lessons
                ]
            )
        if reply_type is prompts.OneOffCheckReply:
            return reply_type(verdict="supported", covered_by_theme=False, reason="Scripted.")
        if reply_type is prompts.EvidenceLinks:
            return reply_type(
                links=[
                    prompts.EvidenceLink(theme_number=1, evidence_label=SIGNIFICANT_EVIDENCE, relation="contradicts", reason="Scripted."),
                    prompts.EvidenceLink(theme_number=2, evidence_label=NOISY_EVIDENCE, relation="agrees", reason="Scripted."),
                    prompts.EvidenceLink(theme_number=2, evidence_label="A feature that does not exist", relation="agrees", reason="Scripted."),
                ]
            )
        if reply_type is prompts.EvidenceLinkCheck:
            relation = "contradicts" if "this result contradicts the advice" in prompt else "agrees"
            return reply_type(direct_measure=True, relation=relation, reason="Scripted.")
        raise AssertionError(f"Unexpected reply type {reply_type}")

    @staticmethod
    def _candidate(number: int, answers: dict[int, str]) -> prompts.CandidateTheme:
        label, sentence = THEME_SENTENCES[number]
        return prompts.CandidateTheme(
            label=label,
            definition=f"The answer says to {label.lower()}.",
            include_when="It gives this advice.",
            exclude_when="It only mentions the topic.",
            supporting_ids=[answer_id for answer_id, lessons in answers.items() if sentence in lessons],
        )

    @staticmethod
    def _codes(model: str, prompt: str, answers: dict[int, str]) -> list[prompts.ThemeCode]:
        (lessons,) = answers.values()
        scope = re.search(r"For only themes? ([\d, ]+), give", prompt)
        numbers = [int(number) for number in scope.group(1).split(", ")] if scope else list(THEME_SENTENCES)
        codes = []
        for number in numbers:
            sentence = THEME_SENTENCES[number][1]
            if number == 1 and OPPOSES_THEME_1 in lessons:
                codes.append(prompts.ThemeCode(theme_number=number, stance="opposes", excerpt=OPPOSES_THEME_1, concreteness=2))
            elif sentence in lessons and not (model == "fake/coder-2" and number == 4):
                excerpt = sentence.replace("'", "’").upper() if number == 3 else sentence
                codes.append(prompts.ThemeCode(theme_number=number, stance="supports", excerpt=excerpt, concreteness=3 if number == 2 else 2))
            else:
                codes.append(prompts.ThemeCode(theme_number=number, stance="absent", excerpt="", concreteness=0))
        return codes


@pytest.fixture
def scripted_llm() -> ScriptedLlm:
    return ScriptedLlm()
