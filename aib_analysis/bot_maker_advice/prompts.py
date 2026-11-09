"""Prompts for each stage of the thematic analysis and the JSON each must return."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from aib_analysis.bot_maker_advice import method
from aib_analysis.bot_maker_advice.answers import (
    ABANDONED_QUESTION,
    LESSONS_QUESTION,
    Answer,
)
from aib_analysis.bot_maker_advice.quality import Stance
from aib_analysis.bot_maker_advice.results import Theme, Verdict
from aib_analysis.bot_maker_advice.survey_evidence import SurveyEvidence


class CandidateTheme(BaseModel):
    label: str
    definition: str
    include_when: str
    exclude_when: str
    supporting_ids: list[int]


class CandidateThemes(BaseModel):
    themes: list[CandidateTheme]


class CodebookEntry(BaseModel):
    label: str
    definition: str
    include_when: str
    exclude_when: str
    source_candidates: list[str]


class Codebook(BaseModel):
    themes: list[CodebookEntry]


class ThemeCode(BaseModel):
    theme_number: int
    stance: Stance
    excerpt: str
    concreteness: int


class AnswerCoding(BaseModel):
    codes: list[ThemeCode]


class CitationCheckReply(BaseModel):
    verdict: Verdict
    excerpt_shows_claim: bool
    reason: str


class OneOffCandidate(BaseModel):
    answer_id: int
    label: str
    excerpt: str


class OneOffCandidates(BaseModel):
    one_offs: list[OneOffCandidate]


class OneOffCheckReply(BaseModel):
    verdict: Verdict
    covered_by_theme: bool
    reason: str


class EvidenceLink(BaseModel):
    theme_number: int
    evidence_label: str
    relation: Literal["agrees", "contradicts"]
    reason: str


class EvidenceLinks(BaseModel):
    links: list[EvidenceLink]


class EvidenceLinkCheck(BaseModel):
    direct_measure: bool
    relation: Literal["agrees", "contradicts", "unclear"]
    reason: str


LABEL_RULE = "the advice as an imperative sentence of 3 to 8 words, in sentence case, with no final period"
EXCERPT_RULE = (
    f"{method.MIN_EXCERPT_WORDS} to {method.MAX_EXCERPT_WORDS} words copied character for character from the answer, "
    "with no ellipses, brackets, or other edits"
)


def _context(season_label: str) -> str:
    return (
        f"The answers come from a survey of bot makers who entered AI forecasting bots in Metaculus's {season_label} "
        f'FutureEval tournament. Each answered "{LESSONS_QUESTION}" Some also answered "{ABANDONED_QUESTION}".'
    )


def format_answer(answer: Answer) -> str:
    lines = [f"Bot maker #{answer.answer_id}", f"What others should learn: {answer.lessons}"]
    if answer.abandoned is not None:
        lines.append(f"Tried and abandoned: {answer.abandoned}")
    return "\n".join(lines)


def _format_answers(answers: list[Answer]) -> str:
    return "\n\n".join(format_answer(answer) for answer in answers)


def _format_theme(theme: Theme, prefix: str) -> str:
    return "\n".join(
        [
            f"{prefix}: {theme.label}",
            f"Definition: {theme.definition}",
            f"Include when: {theme.include_when}",
            f"Exclude when: {theme.exclude_when}",
        ]
    )


def _theme_list(themes: list[Theme]) -> str:
    return "\n".join(f"{theme.number}. {theme.label}: {theme.definition}" for theme in themes)


def theme_generation_prompt(season_label: str, answers: list[Answer]) -> str:
    return "\n\n".join(
        [
            "You are an experienced qualitative researcher doing codebook thematic analysis.",
            _context(season_label),
            f"Propose {method.MIN_CODEBOOK_THEMES} to {method.MAX_CODEBOOK_THEMES} themes for a codebook. A theme is one "
            "piece of advice or one lesson that several bot makers share, stated as a single central idea rather than a "
            'topic heading (write "Test changes on resolved questions", not "Testing").',
            "\n".join(
                [
                    "Rules:",
                    f"- Each theme must be supported by at least {method.MIN_SUPPORTING_ANSWERS_PER_CANDIDATE} different bot "
                    "makers. List their ID numbers in supporting_ids.",
                    "- Themes must not overlap. Each idea belongs to exactly one theme.",
                    "- Use only what the answers say. Do not add advice of your own.",
                    "- Leave out comments about the survey or the tournament itself unless they are advice for other bot makers.",
                    f"- label: {LABEL_RULE}.",
                    "- definition: one sentence saying what an answer must express to count.",
                    "- include_when: concrete signs that an answer belongs to the theme.",
                    "- exclude_when: near misses that do not count, especially ones that belong to another theme.",
                ]
            ),
            f"The {len(answers)} answers:",
            _format_answers(answers),
            "Reply with only a JSON object in this form:\n"
            '{"themes": [{"label": "...", "definition": "...", "include_when": "...", "exclude_when": "...", '
            '"supporting_ids": [12, 34, 56]}]}',
        ]
    )


def codebook_prompt(season_label: str, answers: list[Answer], candidates: tuple[list[Theme], list[Theme]]) -> str:
    candidate_blocks = []
    for analyst, themes in zip(("A", "B"), candidates):
        candidate_blocks.append(
            f"Candidate themes from analyst {analyst}:\n\n"
            + "\n\n".join(
                _format_theme(theme, f"{analyst}{theme.number}")
                + f"\nSupported by: {', '.join(f'#{answer_id}' for answer_id in theme.supporting_ids)}"
                for theme in themes
            )
        )
    return "\n\n".join(
        [
            "You are the lead researcher in a codebook thematic analysis. Two analysts read the same survey answers "
            "separately, and each proposed candidate themes. Merge their candidates into one final codebook.",
            _context(season_label),
            "\n".join(
                [
                    "Rules:",
                    f"- The final codebook has {method.MIN_CODEBOOK_THEMES} to {method.MAX_CODEBOOK_THEMES} themes.",
                    "- Merge candidates that express the same idea. Split a candidate that mixes two ideas. Drop a candidate "
                    f"that fewer than {method.MIN_SUPPORTING_ANSWERS_PER_CANDIDATE} bot makers support (check the answers yourself).",
                    "- Every final theme must come from at least one candidate. List the candidate codes it came from in "
                    'source_candidates, e.g. ["A2", "B5"].',
                    "- Themes must not overlap, so that a coder can tell which theme an idea belongs to.",
                    f"- label: {LABEL_RULE}.",
                    "- definition, include_when, exclude_when: rewritten so that two coders working separately would make the "
                    "same calls. Where confusion is likely, name the nearest other theme in exclude_when.",
                ]
            ),
            *candidate_blocks,
            f"The {len(answers)} answers:",
            _format_answers(answers),
            "Reply with only a JSON object in this form:\n"
            '{"themes": [{"label": "...", "definition": "...", "include_when": "...", "exclude_when": "...", '
            '"source_candidates": ["A1", "B2"]}]}',
        ]
    )


def coding_prompt(season_label: str, codebook: list[Theme], answer: Answer, theme_numbers: list[int]) -> str:
    scope = (
        "every theme"
        if len(theme_numbers) == len(codebook)
        else f"only theme{'s' if len(theme_numbers) > 1 else ''} {', '.join(str(number) for number in theme_numbers)}"
    )
    return "\n\n".join(
        [
            "You are coding one survey answer against a fixed codebook for a qualitative analysis.",
            _context(season_label),
            "Codebook:\n\n" + "\n\n".join(_format_theme(theme, f"Theme {theme.number}") for theme in codebook),
            "The answer to code:\n\n" + format_answer(answer),
            "\n".join(
                [
                    f"For {scope}, give the answer's stance:",
                    '- "supports": the answer gives this advice, or reports an experience that clearly backs it.',
                    '- "opposes": the answer argues against this advice, or reports an experience that clearly contradicts it.',
                    '- "absent": anything else, including answers that only touch on the topic. When unsure, choose "absent".',
                ]
            ),
            f'For "supports" or "opposes", give the one excerpt that best shows the stance: {EXCERPT_RULE}. Then rate the '
            "excerpt's concreteness: 3 if it names a specific method, tool, model, number, or measured result; 2 if it is "
            'specific advice without such detail; 1 if it is a general statement. For "absent", use an empty excerpt and '
            "concreteness 0.",
            "Reply with only a JSON object with one entry per theme you were asked about, in codebook order:\n"
            '{"codes": [{"theme_number": 1, "stance": "absent", "excerpt": "", "concreteness": 0}]}',
        ]
    )


def citation_check_prompt(season_label: str, theme: Theme, stance: Stance, excerpt: str | None, answer: Answer) -> str:
    if stance == "supports":
        claim, meets_claim = "gives this advice", "gives this advice or reports an experience that backs it"
    else:
        claim, meets_claim = "argues against this advice", "argues against this advice or reports an experience that contradicts it"
    quoted = f'The post quotes this excerpt from the answer: "{excerpt}"' if excerpt else "The post quotes no excerpt."
    return "\n\n".join(
        [
            "You are fact-checking a draft blog post that summarizes survey answers from AI forecasting bot makers. An "
            "automated process wrote the claim below about one answer, and it may be wrong. Read the full answer and "
            "judge the claim skeptically.",
            _context(season_label),
            f'Claim: Bot maker #{answer.answer_id} {claim}: "{theme.label}"\nWhat the advice means: {theme.definition}\n{quoted}',
            "The full answer:\n\n" + format_answer(answer),
            "\n".join(
                [
                    "Questions:",
                    f'1. verdict: "supported" if the answer clearly {meets_claim}; "partial" if it only touches the topic or '
                    'makes a weaker or different point; "not_supported" otherwise.',
                    "2. excerpt_shows_claim: true if the quoted excerpt, read on its own, shows the point; false if it doesn't "
                    "or there is no excerpt.",
                    "3. reason: one sentence.",
                ]
            ),
            'Reply with only a JSON object in this form:\n{"verdict": "supported", "excerpt_shows_claim": true, "reason": "..."}',
        ]
    )


def one_off_prompt(season_label: str, answers: list[Answer], published_themes: list[Theme]) -> str:
    return "\n\n".join(
        [
            "You are a qualitative researcher picking notable one-off ideas for a blog post that summarizes survey answers.",
            _context(season_label),
            "The post already covers these themes:\n" + _theme_list(published_themes),
            f"List up to {2 * method.MAX_ONE_OFFS} notable one-off ideas: specific, actionable advice or findings that one bot "
            "maker raised and that the themes above do not cover. Prefer concrete ideas (a named method, tool, result, or "
            "failure). Use each bot maker at most once, and spread the ideas across bot makers.",
            "\n".join(
                [
                    "For each idea give:",
                    "- answer_id: the bot maker's ID number",
                    "- label: the idea in 3 to 8 words, in sentence case, with no final period",
                    f"- excerpt: {EXCERPT_RULE}",
                ]
            ),
            f"The {len(answers)} answers:",
            _format_answers(answers),
            'Reply with only a JSON object in this form:\n{"one_offs": [{"answer_id": 12, "label": "...", "excerpt": "..."}]}',
        ]
    )


def one_off_check_prompt(season_label: str, label: str, excerpt: str, answer: Answer, published_themes: list[Theme]) -> str:
    return "\n\n".join(
        [
            "You are fact-checking a draft blog post that summarizes survey answers from AI forecasting bot makers. An "
            "automated process wrote the claim below, and it may be wrong.",
            _context(season_label),
            f'Claim: Bot maker #{answer.answer_id} raised this idea: "{label}"\nThe post quotes this excerpt from the answer: "{excerpt}"',
            "The full answer:\n\n" + format_answer(answer),
            "The post's themes, which this idea should not repeat:\n" + _theme_list(published_themes),
            "\n".join(
                [
                    "Questions:",
                    '1. verdict: "supported" if the answer clearly raises this idea and the excerpt shows it; "partial" if it '
                    'only touches it; "not_supported" otherwise.',
                    "2. covered_by_theme: true if the idea is essentially one of the themes listed.",
                    "3. reason: one sentence.",
                ]
            ),
            'Reply with only a JSON object in this form:\n{"verdict": "supported", "covered_by_theme": false, "reason": "..."}',
        ]
    )


def _direction(evidence: SurveyEvidence) -> str:
    return "bots with more of this feature scored higher" if evidence.correlation > 0 else "bots with more of this feature scored lower"


def data_check_prompt(published_themes: list[Theme], evidence: list[SurveyEvidence]) -> str:
    return "\n\n".join(
        [
            "You are comparing a blog post of advice from AI forecasting bot makers with a separate survey analysis that "
            "correlated features of their bots with performance (average spot peer score).",
            "The post's themes:\n" + _theme_list(published_themes),
            "The survey analysis results (feature name: correlation r with score; positive r means bots with more of the "
            "feature scored higher):\n" + "\n".join(f"- {item.label}: r = {item.correlation:+.2f}" for item in evidence),
            f"For each theme, list at most {method.MAX_DATA_CHECKS_PER_THEME} survey results that directly measure the practice "
            "the advice recommends or warns against. Skip loose topical matches: a result counts only if a bot that follows "
            "the advice would clearly differ on that feature. For each link, say whether the result's direction agrees with "
            "the advice (bots following it scored higher) or contradicts it (they scored lower). A theme may have no links.",
            "Reply with only a JSON object in this form, copying each feature name exactly:\n"
            '{"links": [{"theme_number": 1, "evidence_label": "...", "relation": "agrees", "reason": "one sentence"}]}',
        ]
    )


def data_check_confirmation_prompt(theme: Theme, evidence: SurveyEvidence, relation: str) -> str:
    return "\n\n".join(
        [
            "You are reviewing a proposed link between a piece of advice from AI forecasting bot makers and a survey "
            "result that correlated a feature of their bots with performance (average spot peer score).",
            f'Advice: "{theme.label}" ({theme.definition})',
            f'Survey result: "{evidence.label}" had r = {evidence.correlation:+.2f} with score, so {_direction(evidence)}.',
            f"Proposed reading: this result {'agrees with' if relation == 'agrees' else 'contradicts'} the advice.",
            "\n".join(
                [
                    "Questions:",
                    "1. direct_measure: true only if the survey feature directly measures the practice the advice recommends "
                    "or warns against, so that a bot following the advice would clearly differ on it.",
                    '2. relation: given the sign of r, does the result "agrees" with the advice, "contradicts" it, or is it '
                    '"unclear"?',
                    "3. reason: one sentence.",
                ]
            ),
            'Reply with only a JSON object in this form:\n{"direct_measure": true, "relation": "agrees", "reason": "..."}',
        ]
    )
