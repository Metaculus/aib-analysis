"""Correlation helpers, scaled to a general audience.

Tests used:
- binary feature vs peer score : point-biserial correlation (Pearson on 0/1)
- ordinal / count vs peer score : Spearman rank correlation
- continuous vs peer score      : Pearson correlation

Every result carries the coefficient, p-value, sample size, and a plain-language
direction so the report can describe it without jargon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from aib_analysis.survey_analysis_v2.features import RespondentFeatures, variable_spec

MIN_N_FOR_CORRELATION = 8


@dataclass(frozen=True)
class CorrelationResult:
    key: str
    label: str
    method: str
    coefficient: float | None
    p_value: float | None
    n: int
    note: str = ""

    def direction_phrase(self) -> str:
        if self.coefficient is None:
            return "no estimate"
        if abs(self.coefficient) < 0.1:
            strength = "no"
        elif abs(self.coefficient) < 0.3:
            strength = "a weak"
        elif abs(self.coefficient) < 0.5:
            strength = "a moderate"
        else:
            strength = "a strong"
        if strength == "no":
            return "no clear relationship with peer score"
        sign = "higher" if self.coefficient > 0 else "lower"
        return f"{strength} link to {sign} peer score"


def _paired_values(
    features: list[RespondentFeatures], key: str
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        value = feature.variables.get(key)
        score = feature.score
        if value is None or score is None:
            continue
        xs.append(float(value))
        ys.append(float(score))
    return xs, ys


def correlate_with_score(
    features: list[RespondentFeatures], key: str
) -> CorrelationResult:
    spec = variable_spec(key)
    xs, ys = _paired_values(features, key)
    n = len(xs)
    if n < MIN_N_FOR_CORRELATION or len(set(xs)) < 2:
        return CorrelationResult(
            key=key,
            label=spec.label,
            method="(insufficient data)",
            coefficient=None,
            p_value=None,
            n=n,
            note="too few paired, varying observations",
        )

    if spec.kind in ("binary", "continuous"):
        # Binary traits use Pearson (identical to point-biserial for a 0/1 trait).
        # "continuous" is not used by any current variable (team size is ordinal),
        # but is kept so a genuinely continuous trait added later gets Pearson, not
        # rank correlation, by default.
        method = "Pearson"
        coef, p_value = stats.pearsonr(xs, ys)
    else:
        method = "Spearman"
        coef, p_value = stats.spearmanr(xs, ys)

    coef_value = None if (coef is None or math.isnan(coef)) else float(coef)
    p_value_value = None if (p_value is None or math.isnan(p_value)) else float(p_value)
    return CorrelationResult(
        key=key,
        label=spec.label,
        method=method,
        coefficient=coef_value,
        p_value=p_value_value,
        n=n,
    )
