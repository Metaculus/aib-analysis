"""Hypothesis tests and multiple-comparison correction.

Correction is applied per *family* of tests. A family is the full set of tests
you were willing to run to answer one question, decided before looking at the
results. Correcting within a family you defined after seeing which features
looked interesting does not control anything, so `TestFamily` requires you to
register every test, including the ones that came back null.

Two corrections are reported side by side:

- Bonferroni: controls the family-wise error rate, the probability of even one
  false positive. Very conservative at these family sizes.
- Benjamini-Hochberg: controls the false discovery rate, the expected share of
  claimed findings that are false. Less conservative, and the more sensible
  default when the analysis is exploratory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class TestResult:
    """One hypothesis test, before and after multiplicity correction."""

    feature: str
    label: str
    test: str
    p_raw: float | None
    n: int
    effect: float | None = None
    effect_label: str = ""
    group_a: float | None = None
    group_b: float | None = None
    detail: str = ""

    # Filled in by TestFamily.finalize().
    family: str = ""
    family_size: int = 0
    p_bonferroni: float | None = None
    bonferroni_alpha: float | None = None
    q_value: float | None = None

    @property
    def significant_raw(self) -> bool:
        return self.p_raw is not None and self.p_raw < 0.05

    @property
    def significant_bonferroni(self) -> bool:
        return (
            self.p_bonferroni is not None
            and self.bonferroni_alpha is not None
            and self.p_raw is not None
            and self.p_raw < self.bonferroni_alpha
        )

    @property
    def significant_fdr(self) -> bool:
        return self.q_value is not None and self.q_value < 0.05

    @property
    def stars(self) -> str:
        """Compact significance marker used in tables and chart labels."""
        if self.significant_bonferroni:
            return "**"
        if self.significant_fdr:
            return "*"
        if self.significant_raw:
            return "."
        return ""


@dataclass
class TestFamily:
    """A set of tests corrected together.

    `alpha` is the family-wise error rate for Bonferroni and the FDR target for
    Benjamini-Hochberg.
    """

    name: str
    description: str
    alpha: float = 0.05
    results: list[TestResult] = field(default_factory=list)
    _finalized: bool = False

    def add(self, result: TestResult) -> TestResult:
        if self._finalized:
            raise RuntimeError(
                f"Cannot add tests to family '{self.name}' after finalize(); "
                "the family size is already baked into the correction."
            )
        result.family = self.name
        self.results.append(result)
        return result

    @property
    def size(self) -> int:
        """Number of tests that produced a usable p-value.

        Tests that could not run (an empty group, zero variance) carry no
        evidence and are excluded from the divisor. Every test that *did* run
        is counted, whether or not it looked promising.
        """
        return sum(1 for r in self.results if r.p_raw is not None)

    def finalize(self) -> list[TestResult]:
        """Apply Bonferroni and Benjamini-Hochberg across the family."""
        m = self.size
        if m == 0:
            self._finalized = True
            return self.results

        bonferroni_alpha = self.alpha / m
        for result in self.results:
            result.family_size = m
            if result.p_raw is None:
                continue
            result.p_bonferroni = min(1.0, result.p_raw * m)
            result.bonferroni_alpha = bonferroni_alpha

        # Benjamini-Hochberg. Sort ascending, q_i = p_i * m / i, then enforce
        # monotonicity by sweeping back down so a small later q cannot be
        # undone by a large earlier one.
        testable = sorted(
            (r for r in self.results if r.p_raw is not None),
            key=lambda r: r.p_raw,  # type: ignore[arg-type,return-value]
        )
        running_min = 1.0
        for rank in range(len(testable), 0, -1):
            result = testable[rank - 1]
            raw_q = result.p_raw * m / rank  # type: ignore[operator]
            running_min = min(running_min, raw_q)
            result.q_value = min(1.0, running_min)

        self._finalized = True
        return self.results

    def sorted_results(self) -> list[TestResult]:
        """Results ordered by raw p-value, untestable ones last."""
        return sorted(
            self.results,
            key=lambda r: (r.p_raw is None, r.p_raw if r.p_raw is not None else 1.0),
        )

    def survivors(self, level: str = "bonferroni") -> list[TestResult]:
        if level == "bonferroni":
            return [r for r in self.results if r.significant_bonferroni]
        if level == "fdr":
            return [r for r in self.results if r.significant_fdr]
        return [r for r in self.results if r.significant_raw]


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def fisher_binary(
    feature: str,
    label: str,
    flags: list[bool | None],
    is_winner: list[bool],
) -> TestResult:
    """Fisher exact test of a binary feature against winner status.

    Fisher rather than chi-square because several cells here are 0 or 1, where
    the chi-square approximation is not trustworthy.
    """
    pairs = [(f, w) for f, w in zip(flags, is_winner) if f is not None]
    if not pairs:
        return TestResult(feature, label, "fisher", None, 0, effect_label="pp gap")

    winner_vals = [f for f, w in pairs if w]
    loser_vals = [f for f, w in pairs if not w]
    if not winner_vals or not loser_vals:
        return TestResult(
            feature, label, "fisher", None, len(pairs), effect_label="pp gap"
        )

    table = [
        [sum(winner_vals), len(winner_vals) - sum(winner_vals)],
        [sum(loser_vals), len(loser_vals) - sum(loser_vals)],
    ]
    try:
        _, p = scipy_stats.fisher_exact(table)
    except ValueError:
        p = None

    winner_rate = 100 * sum(winner_vals) / len(winner_vals)
    loser_rate = 100 * sum(loser_vals) / len(loser_vals)
    return TestResult(
        feature=feature,
        label=label,
        test="fisher",
        p_raw=float(p) if p is not None else None,
        n=len(pairs),
        effect=winner_rate - loser_rate,
        effect_label="pp gap",
        group_a=winner_rate,
        group_b=loser_rate,
        detail=f"{sum(winner_vals)}/{len(winner_vals)} vs {sum(loser_vals)}/{len(loser_vals)}",
    )


def mannwhitney_continuous(
    feature: str,
    label: str,
    values: list[float | None],
    is_winner: list[bool],
) -> TestResult:
    """Mann-Whitney U on a continuous feature against winner status.

    Used instead of Welch's t-test because these features are bucket midpoints
    of ordinal survey answers. They are heavily skewed (cost per question spans
    three orders of magnitude) and the midpoints are arbitrary within a bucket,
    so a rank-based test is the honest choice. Medians are reported alongside.
    """
    pairs = [(v, w) for v, w in zip(values, is_winner) if v is not None]
    winner_vals = [v for v, w in pairs if w]
    loser_vals = [v for v, w in pairs if not w]
    if len(winner_vals) < 2 or len(loser_vals) < 2:
        return TestResult(
            feature, label, "mannwhitney", None, len(pairs), effect_label="median diff"
        )

    try:
        _, p = scipy_stats.mannwhitneyu(winner_vals, loser_vals, alternative="two-sided")
    except ValueError:
        p = None

    winner_median = float(np.median(winner_vals))
    loser_median = float(np.median(loser_vals))
    return TestResult(
        feature=feature,
        label=label,
        test="mannwhitney",
        p_raw=float(p) if p is not None else None,
        n=len(pairs),
        effect=winner_median - loser_median,
        effect_label="median diff",
        group_a=winner_median,
        group_b=loser_median,
        detail=f"n={len(winner_vals)} vs {len(loser_vals)}",
    )


def pearson_vs_score(
    feature: str,
    label: str,
    values: list[float | None],
    scores: list[float | None],
) -> TestResult:
    """Pearson correlation between a feature and total score.

    The p-value comes from the t-distribution with n-2 degrees of freedom,
    which is the exact small-sample test. (The earlier R implementation used a
    normal approximation, which is anti-conservative at n around 30.)
    """
    pairs = [
        (v, s)
        for v, s in zip(values, scores)
        if v is not None and s is not None and not math.isnan(s)
    ]
    n = len(pairs)
    if n < 4:
        return TestResult(feature, label, "pearson", None, n, effect_label="r")

    xs = np.array([float(v) for v, _ in pairs])
    ys = np.array([float(s) for _, s in pairs])
    if xs.std() == 0 or ys.std() == 0:
        return TestResult(feature, label, "pearson", None, n, effect_label="r")

    r, p = scipy_stats.pearsonr(xs, ys)
    return TestResult(
        feature=feature,
        label=label,
        test="pearson",
        p_raw=float(p),
        n=n,
        effect=float(r),
        effect_label="r",
        detail=f"n={n}",
    )


def fisher_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson score interval for a proportion, as percentages."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denom
    )
    return (100 * max(0.0, center - margin), 100 * min(1.0, center + margin))


def pearson_ci(r: float, n: int) -> tuple[float, float]:
    """Fisher z-transform confidence interval for a correlation."""
    if n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return (math.tanh(lo), math.tanh(hi))
