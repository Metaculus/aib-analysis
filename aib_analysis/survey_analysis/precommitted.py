"""Hypotheses committed to before the Summer 2026 survey responses were downloaded.

They are the strongest Fall 2025 and Spring 2026 survey findings that measure
different things (pairwise Spearman |rho| <= 0.18 across the Spring 2026 bots).
Each is tested one-sided in its committed direction and the family is corrected
with Holm's method, so these few tests avoid most of the multiple-comparison
penalty the exploratory feature list pays. Do not edit a hypothesis after seeing
Summer results; new ideas belong in the exploratory list.
"""

from __future__ import annotations

from dataclasses import dataclass

from aib_analysis.survey_analysis import stats
from aib_analysis.survey_analysis.features import RespondentFeatures

FAMILY_ALPHA = 0.05


@dataclass(frozen=True)
class PrecommittedHypothesis:
    key: str
    statement: str
    predicted_sign: int
    prior_evidence: str


PRECOMMITTED_HYPOTHESES: list[PrecommittedHypothesis] = [
    PrecommittedHypothesis(
        key="uses_similar_qs",
        statement="Bots that check forecasts on similar Metaculus questions or prediction markets score higher.",
        predicted_sign=1,
        prior_evidence=(
            "Fall 2025: 34% of winners vs 0% of non-winners did this (p = 0.04). Spring 2026: "
            "r = +0.34 (p = 0.031, n = 41), the strongest link of any feature that is not a model choice."
        ),
    ),
    PrecommittedHypothesis(
        key="n_research_sources",
        statement="Bots that use more distinct research sources score higher.",
        predicted_sign=1,
        prior_evidence=(
            "Fall 2025: the strongest result of that survey (r = +0.42 with total score, p = 0.006). "
            "Spring 2026: Spearman r = +0.19 (p = 0.23, n = 42)."
        ),
    ),
    PrecommittedHypothesis(
        key="uses_extremizing",
        statement="Bots that mathematically extremize their predictions score lower.",
        predicted_sign=-1,
        prior_evidence=(
            "Spring 2026: r = -0.30 (p = 0.055, n = 41). Fall 2025 did not report it. In Spring it was "
            "nearly unrelated to the other two hypotheses (|rho| <= 0.07), so it is the most independent test."
        ),
    ),
]

# Considered and set aside. They stay in the exploratory list; while results are
# blinded they are withheld too, so any of them can still replace a hypothesis above.
ALTERNATIVE_HYPOTHESES: list[PrecommittedHypothesis] = [
    PrecommittedHypothesis(
        key="questions_missed",
        statement="Bots that miss fewer of the season's questions score higher.",
        predicted_sign=-1,
        prior_evidence=(
            "Not one of last year's hypotheses (it was first measured for Summer 2026). A retroactive "
            "Spring 2026 check found Spearman r = -0.46 (p = 0.002, n = 42), taking the most-forecast "
            "bot's question count as the season total."
        ),
    ),
    PrecommittedHypothesis(
        key="llm_calls_mid",
        statement="Bots that make more LLM calls per question score higher.",
        predicted_sign=1,
        prior_evidence=(
            "Fall 2025: winners made a median of about 28 calls per question vs 7 for non-winners "
            "(p = 0.022). Spring 2026: Spearman r = +0.20 (p = 0.21). Set aside because in Spring it "
            "overlapped with checking similar questions (rho = +0.38) and research breadth (rho = +0.37)."
        ),
    ),
    PrecommittedHypothesis(
        key="hours_mid",
        statement="Bots with more total development hours score higher.",
        predicted_sign=1,
        prior_evidence=(
            "Spring 2026: Spearman r = +0.32 (p = 0.039), but Fall 2025 found no link (r = 0.08). Set "
            "aside because the two seasons disagree."
        ),
    ),
    PrecommittedHypothesis(
        key="uses_scraping",
        statement="Bots that scrape web pages during research score higher.",
        predicted_sign=1,
        prior_evidence=(
            "Spring 2026: r = +0.33 (p = 0.032). Set aside because in Spring it overlapped with research "
            "breadth (rho = +0.57) and checking similar questions (rho = +0.41)."
        ),
    ),
]

BLINDED_KEYS: frozenset[str] = frozenset(
    hypothesis.key for hypothesis in PRECOMMITTED_HYPOTHESES + ALTERNATIVE_HYPOTHESES
)


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis: PrecommittedHypothesis
    correlation: stats.CorrelationResult
    one_sided_p: float | None
    holm_p: float | None

    @property
    def supported(self) -> bool:
        return self.holm_p is not None and self.holm_p < FAMILY_ALPHA

    @property
    def verdict(self) -> str:
        coefficient = self.correlation.coefficient
        if coefficient is None:
            return "Not testable (too few answers)"
        if self.supported:
            return "Supported"
        if coefficient * self.hypothesis.predicted_sign < 0:
            return "Not supported (the estimate pointed the opposite way)"
        return "Not supported"


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
            correlation=correlation,
            one_sided_p=p_value,
            holm_p=holm_p if p_value is not None else None,
        )
        for hypothesis, correlation, p_value, holm_p in zip(
            PRECOMMITTED_HYPOTHESES, correlations, one_sided, holm
        )
    ]
