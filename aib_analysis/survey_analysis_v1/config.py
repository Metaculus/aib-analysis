"""Configuration for the bot-maker survey analysis pipeline.

A survey season is described entirely by a `SeasonConfig`. To analyse a new
season you write a new config; you should not need to touch any other module.

Google Forms changes its column wording between seasons ("Fall 2025
Leaderboard" vs "Spring 2026 Leaderboard"), so columns are resolved by regex
against the header row rather than by position. Multi-select answers are
parsed by substring matching against an option catalog, because several of the
option labels contain commas and a naive split on ", " tears them apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Locates one survey column by matching regexes against the header text."""

    name: str
    patterns: tuple[str, ...]
    required: bool = False

    def matches(self, header: str) -> bool:
        flat = " ".join(header.split()).lower()
        return any(re.search(p, flat, re.IGNORECASE) for p in self.patterns)


# Canonical survey fields. Patterns are deliberately loose so they survive
# rewording between seasons.
SURVEY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("timestamp", (r"^timestamp$",)),
    ColumnSpec("bot_name", (r"bot'?s name as listed", r"bot'?s name"), required=True),
    ColumnSpec("final_model", (r"model.*final prediction",)),
    ColumnSpec("support_model", (r"model.*supporting roles",)),
    ColumnSpec("iterations", (r"how many iterations",)),
    ColumnSpec("research", (r"how did your bot research",)),
    ColumnSpec("forecasting_strategies", (r"forecasting strategies",)),
    ColumnSpec("development", (r"went into the development",)),
    ColumnSpec("abandoned", (r"tried and abandoned",)),
    ColumnSpec("verification_env", (r"verification environment",)),
    ColumnSpec("aggregation_method", (r"how did you aggregate",)),
    ColumnSpec("ensemble_combination", (r"combine ensemble outputs",)),
    ColumnSpec("respondent_type", (r"what best describes you",)),
    ColumnSpec("team_size", (r"how many people are on your team",)),
    ColumnSpec("hours", (r"total active hours",)),
    ColumnSpec("llm_calls", (r"number of llm calls",)),
    ColumnSpec("cost_per_q", (r"cost per question",)),
    ColumnSpec("research_vs_reasoning", (r"optimized more for research",)),
    ColumnSpec("changed_since_last", (r"did you change how your bot predicted",)),
    ColumnSpec("code_link", (r"provide a way to review your code",)),
    ColumnSpec("share_code_publicly", (r"can we share it publicly",)),
    ColumnSpec("share_individual", (r"share your individual survey response",)),
    ColumnSpec("minibench_opinion", (r"continue running minibench",)),
    ColumnSpec("writeup_rating", (r"quality and usefulness of metaculus",)),
    ColumnSpec("lessons", (r"what should other bot makers learn",)),
    ColumnSpec("other_comments", (r"anything else you want to share",)),
    ColumnSpec("self_reported_rank", (r"what rank did your bot get",)),
)


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryFeature:
    """A yes/no feature derived from a multi-select answer.

    `needles` are matched case-insensitively as substrings against the raw cell
    text. A row is True if any needle appears.
    """

    name: str
    source: str
    needles: tuple[str, ...]
    label: str
    group: str = "other"
    negate: bool = False

    def evaluate(self, cell: str | None) -> bool | None:
        if cell is None:
            return None
        text = cell.strip().lower()
        if not text:
            return None
        hit = any(n.lower() in text for n in self.needles)
        return not hit if self.negate else hit


@dataclass(frozen=True)
class OrdinalFeature:
    """A continuous feature built by mapping bucket labels to midpoints.

    `mapping` keys are matched exactly (after whitespace/case normalisation);
    `fallback_patterns` catch free-text answers that Google Forms allows via
    the "Other" option.
    """

    name: str
    source: str
    mapping: dict[str, float]
    label: str
    unit: str = ""
    fallback_patterns: tuple[tuple[str, float], ...] = ()
    log_scale: bool = False

    def evaluate(self, cell: str | None) -> float | None:
        if cell is None:
            return None
        text = " ".join(cell.split())
        if not text:
            return None
        key = text.strip().lower()
        for raw_key, value in self.mapping.items():
            if key == raw_key.strip().lower():
                return value
        for pattern, value in self.fallback_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return value
        return None


@dataclass(frozen=True)
class CountFeature:
    """Counts how many distinct options from a catalog appear in an answer."""

    name: str
    source: str
    catalog: tuple[str, ...]
    label: str

    def evaluate(self, cell: str | None) -> int | None:
        if cell is None:
            return None
        text = cell.strip().lower()
        if not text:
            return None
        return sum(1 for option in self.catalog if option.lower() in text)


@dataclass(frozen=True)
class CategoricalFeature:
    """A single-select answer kept as a category, for distribution charts."""

    name: str
    source: str
    label: str
    order: tuple[str, ...] = ()
    ordinal_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model tiering
# ---------------------------------------------------------------------------

ModelTier = Literal["frontier", "mid", "legacy", "unknown"]


@dataclass(frozen=True)
class ModelTiering:
    """Ordered regex patterns that classify a reported model string.

    Checked frontier first, so a respondent listing both a frontier and a
    legacy model is counted as frontier (they reached for the best thing they
    had).
    """

    frontier: tuple[str, ...]
    mid: tuple[str, ...]
    legacy: tuple[str, ...]

    def classify(self, raw: str | None) -> ModelTier:
        if not raw or not raw.strip():
            return "unknown"
        text = raw.lower()
        for tier, patterns in (
            ("frontier", self.frontier),
            ("mid", self.mid),
            ("legacy", self.legacy),
        ):
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return tier  # type: ignore[return-value]
        return "unknown"


# ---------------------------------------------------------------------------
# Season config
# ---------------------------------------------------------------------------


@dataclass
class SeasonConfig:
    """Everything the pipeline needs to analyse one season's survey."""

    season: str
    survey_csv: Path
    participation_csv: Path
    leaderboard_csv: Path
    output_dir: Path

    binary_features: tuple[BinaryFeature, ...]
    ordinal_features: tuple[OrdinalFeature, ...]
    count_features: tuple[CountFeature, ...]
    categorical_features: tuple[CategoricalFeature, ...]
    model_tiering: ModelTiering

    # Participation-sheet column names.
    participation_bot_column: str = "bot_usernames"
    participation_owner_column: str = "owner_username"
    prize_column: str = "aib_prize"
    in_tournament_column: str = "in_aib"
    secondary_prize_column: str = "minibench_prize"
    secondary_forecasts_column: str = "minibench_forecasts"
    primary_forecasts_column: str = "futureeval_forecasts"

    # Bots excluded from the participant leaderboard ranking. Only the
    # Metaculus-run reference bots, which exist as a baseline and cannot win a
    # prize. Commercial entries such as the Preseen- bots are real competitors
    # and stay in the ranking.
    excluded_bot_prefixes: tuple[str, ...] = ("metac-",)

    # Entries removed from the participant ranking because they were
    # disqualified. They keep their leaderboard row but do not occupy a
    # participant rank slot, so the prize cutoff stays contiguous.
    disqualified_bots: tuple[str, ...] = ()

    # Survey bot name -> leaderboard/participation name, for the handful of
    # respondents who type something other than their registered bot name.
    name_aliases: dict[str, str] = field(default_factory=dict)

    alpha: float = 0.05
    top_group_size: int = 15
    min_group_for_test: int = 3

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def data_dir(self) -> Path:
        return self.output_dir / "data"
