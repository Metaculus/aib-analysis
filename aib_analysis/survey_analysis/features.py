"""Turn raw respondents into parsed cells, booleans, and numeric variables.

Produces two views of the same data:
- per-cell parse results (raw -> matched -> leftover) for the review doc
- per-respondent numeric/boolean variables for correlation and charts

Write-ins are never corrected by hand; they stay "Other" unless a committed rule
(such as config.RESEARCH_WRITE_IN_SOURCES) counts them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from aib_analysis.survey_analysis import config, parsing
from aib_analysis.survey_analysis.config import ModelInfo
from aib_analysis.survey_analysis.loading import Respondent

logger = logging.getLogger(__name__)

_MODERATE_SOURCES_MIN, _MODERATE_SOURCES_MAX = config.MODERATE_RESEARCH_SOURCE_RANGE


@dataclass
class ParsedCell:
    raw: str
    matched: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    numeric: float | None = None
    non_answers: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return bool(self.matched or self.other)


@dataclass
class RespondentFeatures:
    respondent: Respondent
    cells: dict[str, ParsedCell]
    booleans: dict[str, bool]
    variables: dict[str, float | None]
    final_models: list[ModelInfo]
    final_ignored: list[str]
    final_unmatched: list[str]
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
    VariableSpec("frontier_support", "Frontier supporting-role model", "binary"),
    VariableSpec("final_model_release", "Final model release date (flagship models only)", "continuous"),
    VariableSpec("n_research_sources", "Number of research sources", "count"),
    VariableSpec(
        "moderate_research_sources",
        f"Uses {_MODERATE_SOURCES_MIN} to {_MODERATE_SOURCES_MAX} research sources (vs fewer or more)",
        "binary",
    ),
    VariableSpec("team_size", "Team size", "ordinal"),
    VariableSpec("iterations_mid", "Iterations that went live (midpoint)", "ordinal"),
    VariableSpec("hours_mid", "Total development hours (midpoint)", "ordinal"),
    VariableSpec("llm_calls_mid", "LLM calls per question (midpoint)", "ordinal"),
    VariableSpec("cost_mid", "Cost per question (midpoint)", "ordinal"),
    VariableSpec("research_vs_reasoning_ord", "Research vs reasoning (0=research..4=reasoning)", "ordinal"),
    VariableSpec("questions_missed", "Questions missed after the bot's first forecast", "count"),
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
    uncovered_research_write_ins: list[str] = []
    for respondent in respondents:
        answers = respondent.answers
        cells: dict[str, ParsedCell] = {}

        # Multi-select columns
        for slug, vocab in config.MULTISELECT_VOCAB.items():
            matched, other = parsing.parse_multiselect(answers.get(slug, ""), vocab)
            non_answers = [write_in for write_in in other if re.search(config.NON_ANSWER_WRITE_IN, write_in, re.IGNORECASE)]
            cells[slug] = ParsedCell(
                raw=answers.get(slug, ""),
                matched=matched,
                other=[write_in for write_in in other if write_in not in non_answers],
                non_answers=non_answers,
            )
        research_write_in_sources, uncovered = _apply_research_write_ins(cells["research"])
        uncovered_research_write_ins.extend(uncovered)

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

        frontier = any(model.is_frontier for model in final_models)

        # Boolean habit features, matched against parsed options (not raw text)
        # so a write-in cannot spuriously trip a flag.
        booleans: dict[str, bool] = {}
        for feature in config.BOOLEAN_FEATURES:
            matched_options = cells[feature.column_slug].matched
            booleans[feature.key] = parsing.feature_present(matched_options, feature.match_substring)

        # Item non-response: a bot that left the source question blank is coded
        # None (missing), not a definitive "no", so it is dropped from that
        # correlation exactly like a blank numeric answer. An answer with only a
        # write-in (other than a non-answer) still counts as "no" for the habit.
        variables: dict[str, float | None] = {}
        for feature in config.BOOLEAN_FEATURES:
            answered = cells[feature.column_slug].answered
            variables[feature.key] = (1.0 if booleans[feature.key] else 0.0) if answered else None
        variables["frontier"] = _frontier_variable(final_models)
        variables["frontier_support"] = _frontier_variable(support_models)
        variables["final_model_release"] = _high_power_release_ordinal(final_models)
        variables["n_research_sources"] = (
            float(len(cells["research"].matched) + len(research_write_in_sources))
            if cells["research"].answered
            else None
        )
        n_research_sources = variables["n_research_sources"]
        variables["moderate_research_sources"] = (
            float(_MODERATE_SOURCES_MIN <= n_research_sources <= _MODERATE_SOURCES_MAX)
            if n_research_sources is not None
            else None
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
        variables["questions_missed"] = (
            float(respondent.questions_missed) if respondent.questions_missed is not None else None
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
                support_unmatched=support_unmatched,
                frontier=frontier,
            )
        )

    if uncovered_research_write_ins:
        raise ValueError(
            "Research write-ins that no rule in config.RESEARCH_WRITE_IN_OPTIONS, RESEARCH_WRITE_IN_SOURCES "
            f"or RESEARCH_WRITE_INS_NOT_COUNTED covers: {uncovered_research_write_ins}. Add each to one of them."
        )
    logger.info("Built features for %d respondents", len(features))
    return features


def _apply_research_write_ins(cell: ParsedCell) -> tuple[list[str], list[str]]:
    """Apply the research write-in rules to a parsed research cell.

    Form options the write-ins name join the cell's matched options, and write-ins
    that only name form options leave its "Other" list. Returns the extra sources
    the write-ins name and the write-ins no rule covers.
    """
    vocab = config.MULTISELECT_VOCAB["research"]
    named_options: set[str] = set()
    named_sources: set[str] = set()
    still_other: list[str] = []
    uncovered: list[str] = []
    for write_in in cell.other:
        options = parsing.options_named_in_write_in(write_in, vocab, config.RESEARCH_WRITE_IN_OPTIONS)
        sources = parsing.match_write_in_sources([write_in], config.RESEARCH_WRITE_IN_SOURCES)
        not_counted = parsing.match_write_in_sources([write_in], config.RESEARCH_WRITE_INS_NOT_COUNTED)
        named_options.update(options)
        named_sources.update(sources)
        if sources or not options:
            still_other.append(write_in)
        if not (options or sources or not_counted):
            uncovered.append(write_in)
    cell.matched = [option for option in vocab if option in cell.matched or option in named_options]
    cell.other = still_other
    return [source for source in config.RESEARCH_WRITE_IN_SOURCES if source in named_sources], uncovered


def _frontier_variable(models: list[ModelInfo]) -> float | None:
    """1 if any model is frontier, 0 if every model is known not to be, else unknown.

    Text that names no recognizable model, or a model with no release date (an
    unspecified fine-tune, a family without a version), could be frontier, so it
    leaves the answer unknown unless another listed model is frontier.
    """
    if any(model.is_frontier for model in models):
        return 1.0
    if not models or any(model.release_date is None for model in models):
        return None
    return 0.0


def _high_power_release_ordinal(models: list[ModelInfo]) -> float | None:
    """Newest release date among high-powered final models, as a day ordinal.

    Only high-powered models with a known release date count, so a bot whose
    final model is a small variant or an undated generic bucket has no value and
    drops out of this correlation.
    """
    dates = [m.release_date for m in models if m.high_power and m.release_date is not None]
    return float(max(dates).toordinal()) if dates else None


def _canon(cell: ParsedCell) -> str | None:
    return cell.matched[0] if cell.matched else None
