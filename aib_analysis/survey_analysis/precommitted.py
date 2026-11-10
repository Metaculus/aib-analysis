"""Hypotheses committed to before the Summer 2026 survey responses were downloaded.

They are the strongest past survey findings that measure different things. Each is
tested one-sided in its committed direction and the family is corrected with Holm's
method instead of joining the exploratory false-discovery correction. Do not edit a
hypothesis after seeing results; new ideas belong in the exploratory list.

The picks were confirmed on 2026-10-30, after the responses were downloaded but
before anyone looked at a hypothesis result. At the same time, the research-source
count was defined to also count write-ins naming a source in
config.RESEARCH_WRITE_IN_SOURCES, and checking similar questions/markets was kept
to the strategy checkbox alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from aib_analysis.survey_analysis import stats
from aib_analysis.survey_analysis.features import RespondentFeatures


@dataclass(frozen=True)
class PrecommittedHypothesis:
    key: str
    predicted_sign: int


PRECOMMITTED_HYPOTHESES: list[PrecommittedHypothesis] = [
    # Bots that check forecasts on similar Metaculus questions or prediction markets score higher.
    PrecommittedHypothesis(key="uses_similar_qs", predicted_sign=1),
    # Bots that use more distinct research sources score higher.
    PrecommittedHypothesis(key="n_research_sources", predicted_sign=1),
    # Bots that mathematically extremize their predictions score lower.
    PrecommittedHypothesis(key="uses_extremizing", predicted_sign=-1),
]

PRECOMMITTED_KEYS: frozenset[str] = frozenset(hypothesis.key for hypothesis in PRECOMMITTED_HYPOTHESES)


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis: PrecommittedHypothesis
    one_sided_p: float | None
    holm_p: float | None


def one_sided_p(result: stats.CorrelationResult, predicted_sign: int) -> float | None:
    """One-sided p in the predicted direction, from the symmetric two-sided test."""
    if result.coefficient is None or result.p_value is None:
        return None
    if result.coefficient * predicted_sign > 0:
        return result.p_value / 2
    return 1 - result.p_value / 2


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: pvalues[index])
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (count - rank) * pvalues[index]))
        adjusted[index] = running_max
    return adjusted


def evaluate_precommitted(pool: list[RespondentFeatures]) -> list[HypothesisResult]:
    """Test every precommitted hypothesis on the correlation pool.

    An untestable hypothesis enters the Holm family with p = 1, so dropping it can
    never make the others easier to pass.
    """
    correlations = [stats.correlate_with_score(pool, h.key) for h in PRECOMMITTED_HYPOTHESES]
    one_sided = [
        one_sided_p(correlation, hypothesis.predicted_sign)
        for hypothesis, correlation in zip(PRECOMMITTED_HYPOTHESES, correlations)
    ]
    holm = holm_adjust([p if p is not None else 1.0 for p in one_sided])
    return [
        HypothesisResult(
            hypothesis=hypothesis,
            one_sided_p=p_value,
            holm_p=holm_p if p_value is not None else None,
        )
        for hypothesis, p_value, holm_p in zip(PRECOMMITTED_HYPOTHESES, one_sided, holm)
    ]
