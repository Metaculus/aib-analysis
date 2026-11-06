from __future__ import annotations

from aib_analysis.bot_maker_advice import method
from aib_analysis.bot_maker_advice.quality import Stance
from aib_analysis.bot_maker_advice.results import (
    CitationCheck,
    CodedPair,
    CoderVote,
    Theme,
    ThemeStats,
    Verdict,
    assign_statuses,
    choose_citations,
    summarize_theme,
)

ANSWER_COUNT = 40


def _theme(number: int) -> Theme:
    return Theme(number=number, label=f"Theme {number}", definition="", include_when="", exclude_when="")


def _stats(number: int, supporters: int, alpha: float | None) -> ThemeStats:
    return ThemeStats(
        theme=_theme(number),
        coder_supporter_counts=(supporters, supporters),
        alpha=alpha,
        ac1=0.9,
        percent_agreement=0.95,
        disputed_count=0,
        no_majority_count=0,
        supporters=tuple(range(supporters)),
        opposers=(),
        rejected_count=0,
        checked_count=supporters,
    )


def test_minimum_supporters_is_ten_percent_with_a_floor_of_three() -> None:
    assert method.min_supporters(46) == 5
    assert method.min_supporters(20) == 3


def test_assign_statuses_gates_on_supporters_and_agreement_then_caps_by_supporters() -> None:
    theme_stats = [
        _stats(1, 3, 0.9),
        _stats(2, 9, 0.6),
        _stats(3, 8, 0.7),
        _stats(4, 8, 0.95),
        _stats(5, 12, 0.8),
        _stats(6, 6, 0.8),
        _stats(7, 5, 0.9),
        _stats(8, 5, None),
        _stats(9, 7, 1.0),
    ]
    statuses = {stats.theme.number: stats.status for stats in assign_statuses(theme_stats, ANSWER_COUNT)}
    assert statuses == {
        1: "below_minimum",
        2: "held_low_agreement",
        3: "published",
        4: "published",
        5: "published",
        6: "published",
        7: "cut_by_cap",
        8: "held_low_agreement",
        9: "published",
    }
    ordered = [stats.theme.number for stats in assign_statuses(theme_stats, ANSWER_COUNT)]
    assert ordered[:6] == [5, 4, 3, 9, 6, 7]


def _pair(answer_id: int, theme_number: int, stance: Stance, concreteness: int, verdict: Verdict = "supported") -> CodedPair:
    vote = CoderVote(model="coder", stance=stance, excerpt=f"quote {answer_id}-{theme_number}", concreteness=concreteness)
    return CodedPair(
        answer_id=answer_id,
        theme_number=theme_number,
        votes=(vote, vote),
        tie_break=None,
        final_stance=stance,
        has_majority=True,
        check=CitationCheck(verdict=verdict, excerpt_shows_claim=True, reason=""),
    )


def test_choose_citations_quotes_concrete_answers_once_and_falls_back_to_repeats() -> None:
    pairs = [
        _pair(11, 1, "supports", 1),
        _pair(12, 1, "supports", 3),
        _pair(13, 1, "supports", 2),
        _pair(14, 1, "supports", 3),
        _pair(15, 1, "opposes", 2),
        _pair(16, 1, "supports", 3, verdict="partial"),
        _pair(12, 2, "supports", 3),
        _pair(14, 2, "supports", 3),
        _pair(17, 2, "supports", 1),
    ]
    theme_stats = [summarize_theme(_theme(number), [pair for pair in pairs if pair.theme_number == number]) for number in (1, 2)]
    first, second = choose_citations(theme_stats, pairs)
    assert [citation.answer_id for citation in first.exemplars] == [12, 14, 13]
    assert first.also_raised_by == (11,)
    assert first.counterpoint is not None and first.counterpoint.answer_id == 15
    assert first.stats.rejected_count == 1 and 16 not in first.stats.supporters
    assert [citation.answer_id for citation in second.exemplars] == [17, 12, 14]
    assert second.also_raised_by == ()
