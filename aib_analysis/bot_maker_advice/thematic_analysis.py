"""Run the precommitted thematic analysis: propose themes, build a codebook, double-code, break ties, and check citations.

Then pick one-off ideas and compare the published themes with the season's survey analysis.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Coroutine
from dataclasses import replace
from typing import Any, TypeVar

from aib_analysis.bot_maker_advice import method, prompts
from aib_analysis.bot_maker_advice.answers import Answer, SurveyAnswers
from aib_analysis.bot_maker_advice.llm import StructuredLlm, ask_until_valid
from aib_analysis.bot_maker_advice.quality import (
    locate_excerpt,
    majority_stance,
    word_count,
)
from aib_analysis.bot_maker_advice.results import (
    AnalysisResult,
    CitationCheck,
    CodedPair,
    CoderVote,
    DataCheck,
    OneOffReview,
    PublishedTheme,
    Theme,
    assign_statuses,
    choose_citations,
    summarize_theme,
)
from aib_analysis.bot_maker_advice.survey_evidence import SurveyEvidence

logger = logging.getLogger(__name__)

ResultType = TypeVar("ResultType")
STAGE_COUNT = 7


async def _gather_logged(coroutines: list[Coroutine[Any, Any, ResultType]], description: str) -> list[ResultType]:
    total = len(coroutines)
    finished = 0

    async def tracked(coroutine: Coroutine[Any, Any, ResultType]) -> ResultType:
        nonlocal finished
        result = await coroutine
        finished += 1
        if finished % 10 == 0 or finished == total:
            logger.info(f"  {description}: {finished}/{total} done")
        return result

    return list(await asyncio.gather(*(tracked(coroutine) for coroutine in coroutines)))


def _stage(number: int, description: str) -> None:
    logger.info(f"Stage {number}/{STAGE_COUNT}: {description}")


def _shuffled(answers: list[Answer], seed: str) -> list[Answer]:
    shuffled = list(answers)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def excerpt_problem(excerpt: str, answer: Answer) -> str | None:
    located = locate_excerpt(excerpt, answer.texts)
    if located is None:
        return f'the excerpt "{excerpt}" is not copied exactly from the answer'
    words = word_count(located)
    if not method.MIN_EXCERPT_WORDS <= words <= method.MAX_EXCERPT_WORDS:
        return f"the excerpt has {words} words; use {method.MIN_EXCERPT_WORDS} to {method.MAX_EXCERPT_WORDS}"
    return None


async def propose_candidate_themes(
    llm: StructuredLlm, models: method.ModelRoster, season_key: str, season_label: str, answers: list[Answer]
) -> tuple[tuple[list[Theme], list[Theme]], list[str]]:
    """Each generator's candidates; a candidate still invalid after the retries is dropped and noted, not fatal."""
    valid_ids = {answer.answer_id for answer in answers}

    def theme_problem(theme: prompts.CandidateTheme) -> str | None:
        unknown_ids = sorted(set(theme.supporting_ids) - valid_ids)
        if unknown_ids:
            return f'Theme "{theme.label}" cites IDs that are not in the answers: {unknown_ids}.'
        if len(set(theme.supporting_ids)) < method.MIN_SUPPORTING_ANSWERS_PER_CANDIDATE:
            return f'Theme "{theme.label}" needs at least {method.MIN_SUPPORTING_ANSWERS_PER_CANDIDATE} different supporting IDs.'
        return None

    def find_problems(reply: prompts.CandidateThemes) -> list[str]:
        problems = [problem for theme in reply.themes if (problem := theme_problem(theme)) is not None]
        if not method.MIN_CODEBOOK_THEMES <= len(reply.themes) <= method.MAX_CODEBOOK_THEMES:
            problems.append(f"Give {method.MIN_CODEBOOK_THEMES} to {method.MAX_CODEBOOK_THEMES} themes, not {len(reply.themes)}.")
        return problems

    async def propose(generator_index: int, model: str) -> tuple[list[Theme], list[str]]:
        ordered_answers = _shuffled(answers, f"{season_key}-theme-generator-{generator_index}")
        prompt = prompts.theme_generation_prompt(season_label, ordered_answers)
        reply, problems = await ask_until_valid(llm, "theme_generation", model, prompt, prompts.CandidateThemes, find_problems)
        valid_themes = [theme for theme in reply.themes if theme_problem(theme) is None]
        if not valid_themes:
            raise ValueError(f"{model} proposed no valid candidate themes: {problems}")
        notes = [f"Analyst {'AB'[generator_index]} ({method.model_label(model)}) after retries: {problem}" for problem in problems]
        themes = [
            Theme(
                number=number,
                label=theme.label,
                definition=theme.definition,
                include_when=theme.include_when,
                exclude_when=theme.exclude_when,
                supporting_ids=tuple(sorted(set(theme.supporting_ids))),
            )
            for number, theme in enumerate(valid_themes, start=1)
        ]
        return themes, notes

    (first, first_notes), (second, second_notes) = await asyncio.gather(
        *(propose(index, model) for index, model in enumerate(models.theme_generators))
    )
    for note in first_notes + second_notes:
        logger.warning(note)
    return (first, second), first_notes + second_notes


async def build_codebook(
    llm: StructuredLlm, models: method.ModelRoster, season_label: str, answers: list[Answer], candidates: tuple[list[Theme], list[Theme]]
) -> list[Theme]:
    candidate_codes = {f"{analyst}{theme.number}" for analyst, themes in zip("AB", candidates) for theme in themes}

    def find_problems(reply: prompts.Codebook) -> list[str]:
        problems = []
        if not method.MIN_CODEBOOK_THEMES <= len(reply.themes) <= method.MAX_CODEBOOK_THEMES:
            problems.append(f"Give {method.MIN_CODEBOOK_THEMES} to {method.MAX_CODEBOOK_THEMES} themes, not {len(reply.themes)}.")
        for entry in reply.themes:
            unknown_codes = sorted(set(entry.source_candidates) - candidate_codes)
            if not entry.source_candidates or unknown_codes:
                problems.append(f'Theme "{entry.label}" must list source candidates from {sorted(candidate_codes)}; got {entry.source_candidates}.')
        return problems

    prompt = prompts.codebook_prompt(season_label, answers, candidates)
    reply, problems = await ask_until_valid(llm, "codebook", models.codebook, prompt, prompts.Codebook, find_problems)
    if problems:
        raise ValueError(f"{models.codebook} did not return a valid codebook: {problems}")
    return [
        Theme(
            number=number,
            label=entry.label,
            definition=entry.definition,
            include_when=entry.include_when,
            exclude_when=entry.exclude_when,
            source_candidates=tuple(entry.source_candidates),
        )
        for number, entry in enumerate(reply.themes, start=1)
    ]


async def code_answer(
    llm: StructuredLlm, stage: str, model: str, season_label: str, codebook: list[Theme], answer: Answer, theme_numbers: list[int]
) -> dict[int, CoderVote]:
    def find_problems(reply: prompts.AnswerCoding) -> list[str]:
        if sorted(code.theme_number for code in reply.codes) != sorted(theme_numbers):
            return [f"Give exactly one entry for each of themes {theme_numbers}."]
        return [
            f"Theme {code.theme_number}: {problem}"
            for code in reply.codes
            if code.stance != "absent" and (problem := excerpt_problem(code.excerpt, answer)) is not None
        ]

    prompt = prompts.coding_prompt(season_label, codebook, answer, theme_numbers)
    reply, problems = await ask_until_valid(llm, stage, model, prompt, prompts.AnswerCoding, find_problems)
    if sorted(code.theme_number for code in reply.codes) != sorted(theme_numbers):
        raise ValueError(f"{model} did not code themes {theme_numbers} for answer #{answer.answer_id}: {problems}")
    votes = {}
    for code in reply.codes:
        if code.stance == "absent":
            votes[code.theme_number] = CoderVote(model=model, stance="absent", excerpt=None, concreteness=0)
            continue
        problem = excerpt_problem(code.excerpt, answer)
        votes[code.theme_number] = CoderVote(
            model=model,
            stance=code.stance,
            excerpt=None if problem else locate_excerpt(code.excerpt, answer.texts),
            concreteness=min(max(code.concreteness, 1), 3),
            excerpt_problem=problem,
        )
    return votes


async def check_citation(
    llm: StructuredLlm, model: str, season_label: str, theme: Theme, pair: CodedPair, answer: Answer
) -> CitationCheck:
    prompt = prompts.citation_check_prompt(season_label, theme, pair.final_stance, pair.excerpt, answer)
    reply = await llm.ask("citation_check", model, prompt, prompts.CitationCheckReply)
    return CitationCheck(verdict=reply.verdict, excerpt_shows_claim=reply.excerpt_shows_claim and pair.excerpt is not None, reason=reply.reason)


async def code_and_check(
    llm: StructuredLlm, models: method.ModelRoster, season_label: str, codebook: list[Theme], answers: list[Answer]
) -> list[CodedPair]:
    all_numbers = [theme.number for theme in codebook]
    _stage(3, f"double-coding {len(answers)} answers against {len(codebook)} themes ({2 * len(answers)} calls)")
    coder_votes = await _gather_logged(
        [
            code_answer(llm, f"coding_{coder_index + 1}", model, season_label, codebook, answer, all_numbers)
            for answer in answers
            for coder_index, model in enumerate(models.coders)
        ],
        "coding",
    )
    votes_by_answer = {
        answer.answer_id: (coder_votes[2 * index], coder_votes[2 * index + 1]) for index, answer in enumerate(answers)
    }
    disputed: dict[int, list[int]] = {}
    for answer_id, (first_votes, second_votes) in votes_by_answer.items():
        numbers = [number for number in all_numbers if first_votes[number].stance != second_votes[number].stance]
        if numbers:
            disputed[answer_id] = numbers
    answers_by_id = {answer.answer_id: answer for answer in answers}
    _stage(4, f"breaking {sum(map(len, disputed.values()))} disagreements over {len(disputed)} answers ({method.model_label(models.tie_breaker)}, blind)")
    tie_votes = await _gather_logged(
        [
            code_answer(llm, "tie_break", models.tie_breaker, season_label, codebook, answers_by_id[answer_id], numbers)
            for answer_id, numbers in disputed.items()
        ],
        "tie-breaks",
    )
    tie_votes_by_answer = dict(zip(disputed, tie_votes))
    unchecked_pairs = []
    for answer in answers:
        first_votes, second_votes = votes_by_answer[answer.answer_id]
        for number in all_numbers:
            tie_break = tie_votes_by_answer.get(answer.answer_id, {}).get(number)
            stance, has_majority = majority_stance(first_votes[number].stance, second_votes[number].stance, tie_break.stance if tie_break else None)
            unchecked_pairs.append(
                CodedPair(
                    answer_id=answer.answer_id,
                    theme_number=number,
                    votes=(first_votes[number], second_votes[number]),
                    tie_break=tie_break,
                    final_stance=stance,
                    has_majority=has_majority,
                    check=None,
                )
            )
    to_check = [pair for pair in unchecked_pairs if pair.final_stance != "absent"]
    themes_by_number = {theme.number: theme for theme in codebook}
    _stage(5, f"checking {len(to_check)} citations ({method.model_label(models.verifier)})")
    checks = await _gather_logged(
        [
            check_citation(llm, models.verifier, season_label, themes_by_number[pair.theme_number], pair, answers_by_id[pair.answer_id])
            for pair in to_check
        ],
        "citation checks",
    )
    checks_by_key = {(pair.answer_id, pair.theme_number): check for pair, check in zip(to_check, checks)}
    return [replace(pair, check=checks_by_key.get((pair.answer_id, pair.theme_number))) for pair in unchecked_pairs]


async def pick_one_offs(
    llm: StructuredLlm,
    models: method.ModelRoster,
    season_label: str,
    answers: list[Answer],
    published: list[PublishedTheme],
) -> list[OneOffReview]:
    published_themes = [theme.stats.theme for theme in published]
    quoted_ids = {citation.answer_id for theme in published for citation in theme.exemplars}
    answers_by_id = {answer.answer_id: answer for answer in answers}
    reply = await llm.ask("one_offs", models.one_offs, prompts.one_off_prompt(season_label, answers, published_themes), prompts.OneOffCandidates)

    async def review(candidate: prompts.OneOffCandidate) -> OneOffReview:
        answer = answers_by_id.get(candidate.answer_id)
        problem = "unknown answer ID" if answer is None else excerpt_problem(candidate.excerpt, answer)
        if answer is None or problem is not None:
            return OneOffReview(candidate.answer_id, candidate.label, None, None, None, problem or "", selected=False)
        excerpt = locate_excerpt(candidate.excerpt, answer.texts) or ""
        prompt = prompts.one_off_check_prompt(season_label, candidate.label, excerpt, answer, published_themes)
        check = await llm.ask("one_off_check", models.verifier, prompt, prompts.OneOffCheckReply)
        return OneOffReview(candidate.answer_id, candidate.label, excerpt, check.verdict, check.covered_by_theme, check.reason, selected=False)

    reviews = await _gather_logged([review(candidate) for candidate in reply.one_offs], "one-off checks")
    first_per_answer: list[OneOffReview] = []
    for item in reviews:
        if item.verdict == "supported" and not item.covered_by_theme and item.answer_id not in {chosen.answer_id for chosen in first_per_answer}:
            first_per_answer.append(item)
    chosen = sorted(first_per_answer, key=lambda item: item.answer_id in quoted_ids)[: method.MAX_ONE_OFFS]
    chosen_keys = {(item.answer_id, item.label) for item in chosen}
    return [replace(item, selected=(item.answer_id, item.label) in chosen_keys) for item in reviews]


async def check_against_survey(
    llm: StructuredLlm, models: method.ModelRoster, published: list[PublishedTheme], evidence: list[SurveyEvidence]
) -> tuple[list[DataCheck], list[str]]:
    themes_by_number = {theme.stats.theme.number: theme.stats.theme for theme in published}
    evidence_by_label = {item.label: item for item in evidence}
    prompt = prompts.data_check_prompt(list(themes_by_number.values()), evidence)
    reply = await llm.ask("data_check", models.data_check, prompt, prompts.EvidenceLinks)
    links = []
    notes = []
    for link in reply.links:
        if link.theme_number not in themes_by_number or link.evidence_label not in evidence_by_label:
            notes.append(f"Dropped a data-check link to an unknown theme or survey feature: theme {link.theme_number}, {link.evidence_label!r}")
            logger.warning(notes[-1])
            continue
        already_linked = [existing for existing in links if existing.theme_number == link.theme_number]
        if len(already_linked) < method.MAX_DATA_CHECKS_PER_THEME and link.evidence_label not in {existing.evidence_label for existing in already_linked}:
            links.append(link)
    confirmations = await _gather_logged(
        [
            llm.ask(
                "data_check_confirmation",
                models.data_check_confirmer,
                prompts.data_check_confirmation_prompt(themes_by_number[link.theme_number], evidence_by_label[link.evidence_label], link.relation),
                prompts.EvidenceLinkCheck,
            )
            for link in links
        ],
        "data-check confirmations",
    )
    checks = [
        DataCheck(
            theme_number=link.theme_number,
            evidence=evidence_by_label[link.evidence_label],
            relation=link.relation,
            reason=link.reason,
            confirmed=confirmation.direct_measure and confirmation.relation == link.relation,
            confirmer_reason=confirmation.reason,
        )
        for link, confirmation in zip(links, confirmations)
    ]
    return checks, notes


async def run_thematic_analysis(
    llm: StructuredLlm,
    models: method.ModelRoster,
    season_key: str,
    season_label: str,
    survey: SurveyAnswers,
    survey_evidence: list[SurveyEvidence] | None,
) -> AnalysisResult:
    answers = list(survey.answers)
    _stage(1, f"proposing candidate themes from {len(answers)} answers ({', '.join(map(method.model_label, models.theme_generators))})")
    candidates, run_notes = await propose_candidate_themes(llm, models, season_key, season_label, answers)
    logger.info(f"  {len(candidates[0])} and {len(candidates[1])} candidate themes; {llm.usage_note()}")

    _stage(2, f"merging candidates into a codebook ({method.model_label(models.codebook)})")
    codebook = await build_codebook(llm, models, season_label, answers, candidates)
    logger.info(f"  {len(codebook)} codebook themes: {'; '.join(theme.label for theme in codebook)}")

    pairs = await code_and_check(llm, models, season_label, codebook, answers)
    logger.info(f"  {llm.usage_note()}")
    theme_stats = assign_statuses([summarize_theme(theme, [pair for pair in pairs if pair.theme_number == theme.number]) for theme in codebook], len(answers))
    published = choose_citations([stats for stats in theme_stats if stats.status == "published"], pairs)
    logger.info(f"  Published {len(published)} of {len(codebook)} themes: {'; '.join(theme.stats.theme.label for theme in published)}")

    _stage(6, f"picking up to {method.MAX_ONE_OFFS} one-offs ({method.model_label(models.one_offs)}, checked by {method.model_label(models.verifier)})")
    one_off_reviews = await pick_one_offs(llm, models, season_label, answers, published)

    data_checks: list[DataCheck] = []
    if survey_evidence is None:
        _stage(7, "skipped the survey data check (no survey analysis report)")
    else:
        _stage(7, f"comparing themes with {len(survey_evidence)} survey analysis results ({method.model_label(models.data_check)}, confirmed by {method.model_label(models.data_check_confirmer)})")
        data_checks, data_check_notes = await check_against_survey(llm, models, published, survey_evidence)
        run_notes += data_check_notes
        published = [
            replace(theme, data_checks=tuple(check for check in data_checks if check.theme_number == theme.stats.theme.number and check.published))
            for theme in published
        ]
    logger.info(f"  {llm.usage_note()}")
    return AnalysisResult(
        season_label=season_label,
        survey=survey,
        models=models,
        candidate_themes=(tuple(candidates[0]), tuple(candidates[1])),
        codebook=tuple(codebook),
        pairs=tuple(pairs),
        theme_stats=tuple(theme_stats),
        published_themes=tuple(published),
        one_off_reviews=tuple(one_off_reviews),
        data_checks=tuple(data_checks),
        survey_evidence_available=survey_evidence is not None,
        run_notes=tuple(run_notes),
    )
