"""Turn raw respondents into parsed cells, booleans, and numeric variables.

Produces two views of the same data:
- per-cell parse results (raw -> matched -> leftover) for the review doc
- per-respondent numeric/boolean variables for correlation and charts

Human-reviewed manual adjustments (manual_adjustments.py) are applied after
parsing and before any derived value is computed, so booleans, midpoints, and
counts all reflect the adjusted answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aib_analysis.survey_analysis_v2 import config, manual_adjustments, parsing
from aib_analysis.survey_analysis_v2.config import ModelInfo
from aib_analysis.survey_analysis_v2.loading import Respondent, normalize_name
from aib_analysis.survey_analysis_v2.manual_adjustments import AnswerAdjustment

logger = logging.getLogger(__name__)

_MODEL_BY_DISPLAY: dict[str, ModelInfo] = {
    model.display: model for model in config.MODEL_REGISTRY
}


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
    adjustments_applied: list[AnswerAdjustment] = field(default_factory=list)

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
    VariableSpec("iterations_mid", "Iterations that went live (midpoint)", "ordinal"),
    VariableSpec("hours_mid", "Total development hours (midpoint)", "ordinal"),
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


def _find_other_index(others: list[str], write_in: str) -> int | None:
    target = write_in.strip().lower()
    for index, token in enumerate(others):
        if token.strip().lower() == target:
            return index
    return None


def _remove_token(tokens: list[str], removed: str) -> None:
    for index, token in enumerate(tokens):
        if token.strip().lower() == removed.strip().lower():
            tokens.pop(index)
            return


def _apply_answer_adjustment(
    cell: ParsedCell,
    adjustment: AnswerAdjustment,
    models: list[ModelInfo] | None = None,
    ignored: list[str] | None = None,
    unmatched: list[str] | None = None,
) -> bool:
    """Apply one reviewed adjustment to a parsed cell.

    Returns False when the write-in is no longer present (a stale row), so the
    caller can flag it. `models`/`ignored`/`unmatched` are only passed for the
    model columns, where a mapped write-in also joins the classified model list.
    """
    index = _find_other_index(cell.other, adjustment.write_in)
    if index is None:
        return False
    if adjustment.action == "leave":
        return True
    removed = cell.other.pop(index)
    if models is not None:
        _remove_token(ignored or [], removed)
        _remove_token(unmatched or [], removed)
    if adjustment.action == "ignore":
        return True
    if models is not None:
        model = _MODEL_BY_DISPLAY[adjustment.canonical_option]
        if model.display not in cell.matched:
            models.append(model)
            cell.matched.append(model.display)
        return True
    slug = adjustment.column_slug
    if slug in config.MULTISELECT_VOCAB:
        vocab = config.MULTISELECT_VOCAB[slug]
        if adjustment.canonical_option not in cell.matched:
            cell.matched.append(adjustment.canonical_option)
            cell.matched.sort(key=vocab.index)
    else:
        cell.matched = [adjustment.canonical_option]
    return True


def _adjustment_key(adjustment: AnswerAdjustment) -> tuple[str, str, str]:
    return (
        normalize_name(adjustment.bot_name),
        adjustment.column_slug,
        adjustment.write_in.strip().lower(),
    )


def build_features(respondents: list[Respondent]) -> list[RespondentFeatures]:
    all_adjustments = manual_adjustments.load_answer_adjustments()
    adjustments_by_bot: dict[str, list[AnswerAdjustment]] = {}
    for adjustment in all_adjustments:
        adjustments_by_bot.setdefault(
            normalize_name(adjustment.bot_name), []
        ).append(adjustment)
    applied_keys: set[tuple[str, str, str]] = set()

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

        # Manual adjustments, before anything is derived from the cells.
        models_by_slug = {"final_model": final_models, "support_model": support_models}
        ignored_by_slug = {"final_model": final_ignored, "support_model": support_ignored}
        unmatched_by_slug = {"final_model": final_unmatched, "support_model": support_unmatched}
        applied: list[AnswerAdjustment] = []
        for adjustment in adjustments_by_bot.get(normalize_name(respondent.bot_name), []):
            slug = adjustment.column_slug
            is_model_column = slug in manual_adjustments.MODEL_COLUMN_SLUGS
            found = _apply_answer_adjustment(
                cells[slug],
                adjustment,
                models=models_by_slug[slug] if is_model_column else None,
                ignored=ignored_by_slug[slug] if is_model_column else None,
                unmatched=unmatched_by_slug[slug] if is_model_column else None,
            )
            if found:
                applied.append(adjustment)
                applied_keys.add(_adjustment_key(adjustment))
                logger.info(
                    "Manual adjustment applied for bot %r: %s",
                    respondent.bot_name,
                    adjustment.description,
                )

        # Numeric midpoints attached to their ordinal cells (post-adjustment)
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

        frontier = any(model.is_frontier for model in final_models)

        # Boolean habit features, matched against parsed options (not raw text)
        # so a write-in cannot spuriously trip a flag.
        booleans: dict[str, bool] = {}
        for feature in config.BOOLEAN_FEATURES:
            matched_options = cells[feature.column_slug].matched
            booleans[feature.key] = parsing.feature_present(matched_options, feature.match_substring)

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
        variables["n_research_sources"] = (
            float(len(cells["research"].matched)) if answers.get("research") else None
        )
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
                adjustments_applied=applied,
            )
        )

    for adjustment in all_adjustments:
        if _adjustment_key(adjustment) not in applied_keys:
            logger.warning(
                "Manual adjustment did NOT apply (stale write_in or unknown bot): "
                "bot=%r, %s",
                adjustment.bot_name,
                adjustment.description,
            )
    logger.info("Built features for %d respondents", len(features))
    return features


def _canon(cell: ParsedCell) -> str | None:
    return cell.matched[0] if cell.matched else None
