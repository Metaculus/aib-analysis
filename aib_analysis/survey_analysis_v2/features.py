"""Turn raw respondents into parsed cells, booleans, and numeric variables.

Produces two views of the same data:
- per-cell parse results (raw -> matched -> leftover) for the review doc
- per-respondent numeric/boolean variables for correlation and charts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aib_analysis.survey_analysis_v2 import config, parsing
from aib_analysis.survey_analysis_v2.config import ModelInfo
from aib_analysis.survey_analysis_v2.loading import Respondent

logger = logging.getLogger(__name__)


@dataclass
class ParsedCell:
    raw: str
    matched: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    numeric: float | None = None


@dataclass
class RespondentFeatures:
    respondent: Respondent
    cells: dict[str, ParsedCell]
    booleans: dict[str, bool]
    variables: dict[str, float | None]
    final_models: list[ModelInfo]
    final_ignored: list[str]
    final_unmatched: list[str]
    support_models: list[ModelInfo]
    support_ignored: list[str]
    support_unmatched: list[str]
    frontier: bool

    @property
    def bot_name(self) -> str:
        return self.respondent.bot_name

    @property
    def score(self) -> float | None:
        """Performance metric used in the analysis: average spot peer score."""
        return self.respondent.average_spot_peer


# Numeric variables that are not simple booleans: (label, kind).
# kind drives the correlation test: "binary" -> point-biserial (Pearson on 0/1),
# "ordinal"/"count" -> Spearman, "continuous" -> Pearson.
@dataclass(frozen=True)
class VariableSpec:
    key: str
    label: str
    kind: str


NUMERIC_VARIABLE_SPECS: list[VariableSpec] = [
    VariableSpec("frontier", "Frontier final model", "binary"),
    VariableSpec("n_research_sources", "Number of research sources", "count"),
    VariableSpec("team_size", "Team size", "ordinal"),
    VariableSpec("iterations_mid", "Iterations forecast live (midpoint)", "ordinal"),
    VariableSpec("hours_mid", "Total active hours (midpoint)", "ordinal"),
    VariableSpec("llm_calls_mid", "LLM calls per question (midpoint)", "ordinal"),
    VariableSpec("cost_mid", "Cost per question (midpoint)", "ordinal"),
    VariableSpec("research_vs_reasoning_ord", "Research vs reasoning (0=research..4=reasoning)", "ordinal"),
    VariableSpec("writeup_rating_ord", "Write-up usefulness rating (0..2)", "ordinal"),
]


def variable_spec(key: str) -> VariableSpec:
    for spec in NUMERIC_VARIABLE_SPECS:
        if spec.key == key:
            return spec
    for feature in config.BOOLEAN_FEATURES:
        if feature.key == key:
            return VariableSpec(feature.key, feature.label, "binary")
    raise KeyError(f"Unknown variable key: {key}")


def _ordinal_index(value: str | None, order: list[str]) -> float | None:
    if value is None:
        return None
    try:
        return float(order.index(value))
    except ValueError:
        return None


def build_features(respondents: list[Respondent]) -> list[RespondentFeatures]:
    features: list[RespondentFeatures] = []
    for respondent in respondents:
        answers = respondent.answers
        cells: dict[str, ParsedCell] = {}

        # Multi-select columns
        for slug, vocab in config.MULTISELECT_VOCAB.items():
            matched, other = parsing.parse_multiselect(answers.get(slug, ""), vocab)
            cells[slug] = ParsedCell(raw=answers.get(slug, ""), matched=matched, other=other)

        # Single-select columns
        for slug, vocab in config.SINGLE_SELECT_VOCAB.items():
            canonical, other = parsing.parse_single_select(answers.get(slug, ""), vocab)
            cells[slug] = ParsedCell(
                raw=answers.get(slug, ""),
                matched=[canonical] if canonical else [],
                other=[other] if other else [],
            )

        # Numeric midpoints attached to their ordinal cells
        cells["iterations"].numeric = parsing.bucket_to_midpoint(
            _canon(cells["iterations"]), config.ITERATIONS_MIDPOINT
        )
        cells["hours"].numeric = parsing.bucket_to_midpoint(
            _canon(cells["hours"]), config.HOURS_MIDPOINT
        )
        cells["llm_calls"].numeric = parsing.bucket_to_midpoint(
            _canon(cells["llm_calls"]), config.LLM_CALLS_MIDPOINT
        )
        cells["cost_per_q"].numeric = parsing.bucket_to_midpoint(
            _canon(cells["cost_per_q"]), config.COST_MIDPOINT
        )

        # Team size
        team_size, team_other = parsing.parse_team_size(answers.get("team_size", ""))
        cells["team_size"] = ParsedCell(
            raw=answers.get("team_size", ""),
            matched=[str(team_size)] if team_size is not None else [],
            other=[team_other] if team_other else [],
            numeric=float(team_size) if team_size is not None else None,
        )

        # Models
        final_models, final_ignored, final_unmatched = parsing.classify_models(
            answers.get("final_model", "")
        )
        support_models, support_ignored, support_unmatched = parsing.classify_models(
            answers.get("support_model", "")
        )
        frontier = parsing.is_frontier_final(answers.get("final_model", ""))
        cells["final_model"] = ParsedCell(
            raw=answers.get("final_model", ""),
            matched=[m.display for m in final_models],
            other=final_unmatched + final_ignored,
        )
        cells["support_model"] = ParsedCell(
            raw=answers.get("support_model", ""),
            matched=[m.display for m in support_models],
            other=support_unmatched + support_ignored,
        )

        # Boolean habit features, matched against parsed options (not raw text)
        # so a write-in cannot spuriously trip a flag.
        booleans: dict[str, bool] = {}
        for feature in config.BOOLEAN_FEATURES:
            matched_options = cells[feature.column_slug].matched
            booleans[feature.key] = parsing.feature_present(matched_options, feature.match_substring)

        n_research = parsing.count_research_sources(
            answers.get("research", ""), config.RESEARCH_SOURCE_OPTIONS
        )

        # Item non-response: a bot that left the source question blank is coded
        # None (missing), not a definitive "no", so it is dropped from that
        # correlation exactly like a blank numeric answer. A non-blank answer with
        # only a write-in still counts as "no" for the canonical habit.
        variables: dict[str, float | None] = {}
        for feature in config.BOOLEAN_FEATURES:
            answered = bool(cells[feature.column_slug].raw.strip())
            variables[feature.key] = (1.0 if booleans[feature.key] else 0.0) if answered else None
        variables["frontier"] = (
            (1.0 if frontier else 0.0) if answers.get("final_model", "").strip() else None
        )
        variables["n_research_sources"] = float(n_research) if answers.get("research") else None
        variables["team_size"] = cells["team_size"].numeric
        variables["iterations_mid"] = cells["iterations"].numeric
        variables["hours_mid"] = cells["hours"].numeric
        variables["llm_calls_mid"] = cells["llm_calls"].numeric
        variables["cost_mid"] = cells["cost_per_q"].numeric
        variables["research_vs_reasoning_ord"] = _ordinal_index(
            _canon(cells["research_vs_reasoning"]),
            config.ORDINAL_ORDER["research_vs_reasoning"],
        )
        variables["writeup_rating_ord"] = _ordinal_index(
            _canon(cells["writeup_rating"]), config.ORDINAL_ORDER["writeup_rating"]
        )

        features.append(
            RespondentFeatures(
                respondent=respondent,
                cells=cells,
                booleans=booleans,
                variables=variables,
                final_models=final_models,
                final_ignored=final_ignored,
                final_unmatched=final_unmatched,
                support_models=support_models,
                support_ignored=support_ignored,
                support_unmatched=support_unmatched,
                frontier=frontier,
            )
        )
    logger.info("Built features for %d respondents", len(features))
    return features


def _canon(cell: ParsedCell) -> str | None:
    return cell.matched[0] if cell.matched else None
