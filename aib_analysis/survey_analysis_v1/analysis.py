"""Feature derivation and the test families that make up the analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from aib_analysis.survey_analysis_v1.config import SeasonConfig
from aib_analysis.survey_analysis_v1.loading import Dataset, Respondent
from aib_analysis.survey_analysis_v1.statistics import (
    TestFamily,
    TestResult,
    fisher_binary,
    mannwhitney_continuous,
    pearson_vs_score,
)

logger = logging.getLogger(__name__)


def derive_features(dataset: Dataset) -> None:
    """Populate `respondent.features` in place for every respondent."""
    config = dataset.config
    for respondent in dataset.respondents:
        features: dict[str, object] = {}

        for spec in config.binary_features:
            features[spec.name] = spec.evaluate(respondent.answer(spec.source))
        for spec in config.ordinal_features:
            features[spec.name] = spec.evaluate(respondent.answer(spec.source))
        for spec in config.count_features:
            features[spec.name] = spec.evaluate(respondent.answer(spec.source))
        for spec in config.categorical_features:
            raw = (respondent.answer(spec.source) or "").strip()
            features[spec.name] = raw or None
            if spec.ordinal_scores:
                features[f"{spec.name}_score"] = spec.ordinal_scores.get(raw)

        features["model_tier"] = config.model_tiering.classify(
            respondent.answer("final_model")
        )
        features["team_size"] = _parse_team_size(respondent.answer("team_size"))

        respondent.features = features

    _log_unparsed(dataset)


def _parse_team_size(raw: str | None) -> float | None:
    if not raw:
        return None
    text = raw.strip().lower()
    if text in ("not in a team", "solo"):
        return 1.0
    try:
        return float(text)
    except ValueError:
        return None


def _log_unparsed(dataset: Dataset) -> None:
    """Warn about ordinal answers that fell through the midpoint maps.

    Free-text "Other" responses are common and silently dropping them would
    shrink n without anyone noticing.
    """
    for spec in dataset.config.ordinal_features:
        unparsed = [
            r.answer(spec.source)
            for r in dataset.respondents
            if r.features.get(spec.name) is None and (r.answer(spec.source) or "").strip()
        ]
        if unparsed:
            logger.warning(
                "%s: %d answer(s) not mapped to a midpoint: %s",
                spec.name,
                len(unparsed),
                unparsed,
            )


# ---------------------------------------------------------------------------
# Test families
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResults:
    dataset: Dataset
    winner_family: TestFamily
    within_winner_family: TestFamily
    score_family: TestFamily
    notes: list[str] = field(default_factory=list)

    @property
    def families(self) -> list[TestFamily]:
        return [self.winner_family, self.within_winner_family, self.score_family]


def run_analysis(dataset: Dataset) -> AnalysisResults:
    config = dataset.config
    compared = dataset.compared
    winners = dataset.winners

    winner_family = _winner_vs_non_winner(config, compared)
    score_family = _score_correlations(config, compared, "All tournament respondents")
    within_family = _score_correlations(
        config, winners, "Prize winners only", family_name="within-winners"
    )

    for family in (winner_family, score_family, within_family):
        family.finalize()

    notes = [
        f"{len(dataset.respondents)} survey responses total.",
        f"{len(winners)} prize winners, {len(dataset.non_winners)} non-winners "
        f"who competed in the main tournament.",
        f"{len(dataset.minibench_only)} respondents ran only in MiniBench and are "
        f"excluded from the winner comparison.",
    ]
    unknown = dataset.cohort("unknown")
    if unknown:
        notes.append(
            f"{len(unknown)} respondent(s) had no participation record: "
            + ", ".join(r.bot_name for r in unknown)
        )

    return AnalysisResults(
        dataset=dataset,
        winner_family=winner_family,
        within_winner_family=within_family,
        score_family=score_family,
        notes=notes,
    )


def _winner_vs_non_winner(
    config: SeasonConfig, respondents: list[Respondent]
) -> TestFamily:
    """Every winner-vs-non-winner test, corrected as one family.

    The family includes all configured features, registered before any results
    are inspected. That is what makes the Bonferroni divisor honest.
    """
    family = TestFamily(
        name="winner-vs-non-winner",
        description=(
            "Does this feature differ between prize winners and non-winners? "
            "Fisher exact for binary features, Mann-Whitney U for ordinal ones."
        ),
        alpha=config.alpha,
    )
    is_winner = [r.is_winner for r in respondents]

    for spec in config.binary_features:
        flags = [r.features.get(spec.name) for r in respondents]
        family.add(fisher_binary(spec.name, spec.label, flags, is_winner))  # type: ignore[arg-type]

    for spec in config.ordinal_features:
        values = [r.features.get(spec.name) for r in respondents]
        family.add(
            mannwhitney_continuous(spec.name, spec.label, values, is_winner)  # type: ignore[arg-type]
        )

    for spec in config.count_features:
        values = [r.features.get(spec.name) for r in respondents]
        family.add(
            mannwhitney_continuous(spec.name, spec.label, values, is_winner)  # type: ignore[arg-type]
        )

    for spec in config.categorical_features:
        if not spec.ordinal_scores:
            continue
        values = [r.features.get(f"{spec.name}_score") for r in respondents]
        family.add(
            mannwhitney_continuous(spec.name, spec.label, values, is_winner)  # type: ignore[arg-type]
        )

    return family


def _score_correlations(
    config: SeasonConfig,
    respondents: list[Respondent],
    population: str,
    family_name: str = "score-correlation",
) -> TestFamily:
    """Pearson r between each feature and total score, corrected as one family."""
    family = TestFamily(
        name=family_name,
        description=f"Pearson correlation with total spot-peer score. {population}.",
        alpha=config.alpha,
    )
    scores = [r.total_score for r in respondents]

    for spec in config.binary_features:
        values = [
            None if r.features.get(spec.name) is None else float(bool(r.features[spec.name]))
            for r in respondents
        ]
        family.add(pearson_vs_score(spec.name, spec.label, values, scores))

    for spec in config.ordinal_features:
        values = [r.features.get(spec.name) for r in respondents]
        family.add(pearson_vs_score(spec.name, spec.label, values, scores))  # type: ignore[arg-type]

    for spec in config.count_features:
        values = [
            None if r.features.get(spec.name) is None else float(r.features[spec.name])  # type: ignore[arg-type]
            for r in respondents
        ]
        family.add(pearson_vs_score(spec.name, spec.label, values, scores))

    for spec in config.categorical_features:
        if not spec.ordinal_scores:
            continue
        values = [r.features.get(f"{spec.name}_score") for r in respondents]
        family.add(pearson_vs_score(spec.name, spec.label, values, scores))  # type: ignore[arg-type]

    return family


# ---------------------------------------------------------------------------
# Descriptive helpers used by the report
# ---------------------------------------------------------------------------


def rate(respondents: list[Respondent], feature: str) -> tuple[float, int, int]:
    """Percentage of respondents with the feature, plus successes and total."""
    flags = [r.features.get(feature) for r in respondents]
    valid = [bool(f) for f in flags if f is not None]
    if not valid:
        return (0.0, 0, 0)
    return (100 * sum(valid) / len(valid), sum(valid), len(valid))


def median_of(respondents: list[Respondent], feature: str) -> float | None:
    values = [
        r.features.get(feature) for r in respondents if r.features.get(feature) is not None
    ]
    if not values:
        return None
    return float(np.median([float(v) for v in values]))  # type: ignore[arg-type]


def category_counts(
    respondents: list[Respondent], feature: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in respondents:
        value = r.features.get(feature)
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def split_top_bottom(
    winners: list[Respondent], top_n: int
) -> tuple[list[Respondent], list[Respondent]]:
    """Split winners into top-N and the rest by total score."""
    ranked = sorted(
        [w for w in winners if w.total_score is not None],
        key=lambda r: r.total_score,  # type: ignore[arg-type,return-value]
        reverse=True,
    )
    return ranked[:top_n], ranked[top_n:]
