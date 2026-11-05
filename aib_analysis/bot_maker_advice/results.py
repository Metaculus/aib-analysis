"""Coding results and the precommitted rules that turn them into published themes and quotes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from aib_analysis.bot_maker_advice import method
from aib_analysis.bot_maker_advice.answers import SurveyAnswers
from aib_analysis.bot_maker_advice.quality import (
    Stance,
    gwet_ac1,
    krippendorff_alpha,
    percent_agreement,
)
from aib_analysis.bot_maker_advice.survey_evidence import SurveyEvidence

Verdict = Literal["supported", "partial", "not_supported"]
ThemeStatus = Literal["published", "held_low_agreement", "below_minimum", "cut_by_cap"]


@dataclass(frozen=True)
class Theme:
    """A codebook theme (with the candidates it merged) or a candidate theme (with the answers its proposer cited)."""

    number: int
    label: str
    definition: str
    include_when: str
    exclude_when: str
    source_candidates: tuple[str, ...] = ()
    supporting_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CoderVote:
    model: str
    stance: Stance
    excerpt: str | None
    concreteness: int
    excerpt_problem: str | None = None


@dataclass(frozen=True)
class CitationCheck:
    verdict: Verdict
    excerpt_shows_claim: bool
    reason: str


@dataclass(frozen=True)
class CodedPair:
    """One answer coded against one theme by both coders, the tie-breaker if needed, and the verifier."""

    answer_id: int
    theme_number: int
    votes: tuple[CoderVote, CoderVote]
    tie_break: CoderVote | None
    final_stance: Stance
    has_majority: bool
    check: CitationCheck | None

    @property
    def deciding_vote(self) -> CoderVote | None:
        """The first vote, in coder order then the tie-breaker, that holds the final stance with a usable excerpt."""
        candidates = [*self.votes, *([self.tie_break] if self.tie_break else [])]
        matching = [vote for vote in candidates if vote.stance == self.final_stance and vote.excerpt is not None]
        return matching[0] if matching else None

    @property
    def excerpt(self) -> str | None:
        vote = self.deciding_vote
        return vote.excerpt if vote else None

    @property
    def concreteness(self) -> int:
        vote = self.deciding_vote
        return vote.concreteness if vote else 0

    @property
    def counted(self) -> bool:
        return self.final_stance != "absent" and self.check is not None and self.check.verdict == "supported"

    @property
    def quotable(self) -> bool:
        return self.counted and self.excerpt is not None and self.check is not None and self.check.excerpt_shows_claim


@dataclass(frozen=True)
class ThemeStats:
    theme: Theme
    coder_supporter_counts: tuple[int, int]
    alpha: float | None
    ac1: float
    percent_agreement: float
    disputed_count: int
    no_majority_count: int
    supporters: tuple[int, ...]
    opposers: tuple[int, ...]
    rejected_count: int
    checked_count: int
    status: ThemeStatus = "below_minimum"


@dataclass(frozen=True)
class Citation:
    answer_id: int
    excerpt: str


@dataclass(frozen=True)
class DataCheck:
    theme_number: int
    evidence: SurveyEvidence
    relation: Literal["agrees", "contradicts"]
    reason: str
    confirmed: bool
    confirmer_reason: str

    @property
    def published(self) -> bool:
        return self.confirmed and self.evidence.significant


@dataclass(frozen=True)
class PublishedTheme:
    rank: int
    stats: ThemeStats
    exemplars: tuple[Citation, ...]
    also_raised_by: tuple[int, ...]
    counterpoint: Citation | None
    other_opposers: tuple[int, ...]
    data_checks: tuple[DataCheck, ...] = ()


@dataclass(frozen=True)
class OneOffReview:
    answer_id: int
    label: str
    excerpt: str | None
    verdict: Verdict | None
    covered_by_theme: bool | None
    reason: str
    selected: bool


@dataclass(frozen=True)
class AnalysisResult:
    season_label: str
    survey: SurveyAnswers
    models: method.ModelRoster
    candidate_themes: tuple[tuple[Theme, ...], tuple[Theme, ...]]
    codebook: tuple[Theme, ...]
    pairs: tuple[CodedPair, ...]
    theme_stats: tuple[ThemeStats, ...]
    published_themes: tuple[PublishedTheme, ...]
    one_off_reviews: tuple[OneOffReview, ...]
    data_checks: tuple[DataCheck, ...]
    survey_evidence_available: bool
    run_notes: tuple[str, ...] = ()

    @property
    def answer_count(self) -> int:
        return len(self.survey.answers)

    @property
    def one_offs(self) -> tuple[OneOffReview, ...]:
        return tuple(review for review in self.one_off_reviews if review.selected)

    def stats_with_status(self, status: ThemeStatus) -> list[ThemeStats]:
        return [stats for stats in self.theme_stats if stats.status == status]


def summarize_theme(theme: Theme, pairs: list[CodedPair]) -> ThemeStats:
    first_supports = [pair.votes[0].stance == "supports" for pair in pairs]
    second_supports = [pair.votes[1].stance == "supports" for pair in pairs]
    checked = [pair for pair in pairs if pair.final_stance != "absent"]
    return ThemeStats(
        theme=theme,
        coder_supporter_counts=(sum(first_supports), sum(second_supports)),
        alpha=krippendorff_alpha(first_supports, second_supports),
        ac1=gwet_ac1(first_supports, second_supports),
        percent_agreement=percent_agreement(first_supports, second_supports),
        disputed_count=sum(pair.tie_break is not None for pair in pairs),
        no_majority_count=sum(not pair.has_majority for pair in pairs),
        supporters=tuple(sorted(pair.answer_id for pair in checked if pair.counted and pair.final_stance == "supports")),
        opposers=tuple(sorted(pair.answer_id for pair in checked if pair.counted and pair.final_stance == "opposes")),
        rejected_count=sum(not pair.counted for pair in checked),
        checked_count=len(checked),
    )


def assign_statuses(theme_stats: list[ThemeStats], answer_count: int) -> list[ThemeStats]:
    """Publish themes with enough verified supporters and coder agreement, most supporters first, up to the cap."""
    minimum = method.min_supporters(answer_count)
    eligible = [stats for stats in theme_stats if len(stats.supporters) >= minimum]
    agreeing = [stats for stats in eligible if stats.alpha is not None and stats.alpha >= method.MIN_CODER_ALPHA]
    ranked = sorted(agreeing, key=lambda stats: (-len(stats.supporters), -(stats.alpha or 0), stats.theme.number))
    statuses: dict[int, ThemeStatus] = {stats.theme.number: "held_low_agreement" for stats in eligible}
    for index, stats in enumerate(ranked):
        statuses[stats.theme.number] = "published" if index < method.MAX_PUBLISHED_THEMES else "cut_by_cap"
    rank_order = {stats.theme.number: index for index, stats in enumerate(ranked)}
    with_status = [replace(stats, status=statuses.get(stats.theme.number, "below_minimum")) for stats in theme_stats]
    return sorted(with_status, key=lambda stats: (rank_order.get(stats.theme.number, len(ranked)), stats.theme.number))


def _best_first(pairs: list[CodedPair]) -> list[CodedPair]:
    return sorted(pairs, key=lambda pair: (-pair.concreteness, pair.answer_id))


def choose_citations(published_stats: list[ThemeStats], pairs: list[CodedPair]) -> list[PublishedTheme]:
    """Up to EXEMPLARS_PER_THEME quotes per theme, most concrete first; nobody is quoted twice unless a theme runs out."""
    quoted_answer_ids: set[int] = set()
    published = []
    for rank, stats in enumerate(published_stats, start=1):
        theme_pairs = [pair for pair in pairs if pair.theme_number == stats.theme.number]
        quotable_supporters = _best_first([pair for pair in theme_pairs if pair.final_stance == "supports" and pair.quotable])
        fresh = [pair for pair in quotable_supporters if pair.answer_id not in quoted_answer_ids]
        repeats = [pair for pair in quotable_supporters if pair.answer_id in quoted_answer_ids]
        exemplar_pairs = (fresh + repeats)[: method.EXEMPLARS_PER_THEME]
        exemplar_ids = {pair.answer_id for pair in exemplar_pairs}
        quoted_answer_ids |= exemplar_ids
        quotable_opposers = _best_first([pair for pair in theme_pairs if pair.final_stance == "opposes" and pair.quotable])
        counterpoint = quotable_opposers[0] if quotable_opposers else None
        published.append(
            PublishedTheme(
                rank=rank,
                stats=stats,
                exemplars=tuple(Citation(pair.answer_id, pair.excerpt or "") for pair in exemplar_pairs),
                also_raised_by=tuple(answer_id for answer_id in stats.supporters if answer_id not in exemplar_ids),
                counterpoint=Citation(counterpoint.answer_id, counterpoint.excerpt or "") if counterpoint else None,
                other_opposers=tuple(
                    answer_id for answer_id in stats.opposers if counterpoint is None or answer_id != counterpoint.answer_id
                ),
            )
        )
    return published
