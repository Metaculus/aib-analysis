"""Human-reviewed manual adjustments layered on top of automatic parsing.

Two CSVs in the non-committed private input directory drive this module. They
live next to the raw survey data because, like it, they reference individual
bots; keeping them there means they persist for reruns without entering git.

- manual_winner_overrides.csv (bot_name, is_winner, reason)
    Bot-level winner status where the prize sheet cannot provide it: the sheet
    aggregates per owner, so for an owner with several bots and at least one
    prize it records how many of the owner's bots won, not which ones. Each
    such respondent needs an explicit row here; loading raises otherwise.

- manual_answer_adjustments.csv (bot_name, column_slug, write_in, action,
  canonical_option, reason)
    One row per reviewed survey write-in. `action` is one of:
      map    -> count the write-in as canonical_option (a vocabulary option,
                or a model-registry display name for model columns)
      ignore -> drop the write-in from "Other" (a parsing fragment, or text
                whose meaning is already covered by another row's map)
      leave  -> keep as "Other"; records that a human reviewed the write-in
                and no canonical option fits

Every applied adjustment is logged and listed in parsing_review.md; an
adjustment whose write_in no longer appears in the target cell logs a warning
so stale rows are never silent.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass

from aib_analysis.survey_analysis_v2 import config

logger = logging.getLogger(__name__)

VALID_ACTIONS = ("map", "ignore", "leave")
MODEL_COLUMN_SLUGS = ("final_model", "support_model")


@dataclass(frozen=True)
class WinnerOverride:
    bot_name: str
    is_winner: bool
    reason: str


@dataclass(frozen=True)
class AnswerAdjustment:
    bot_name: str
    column_slug: str
    write_in: str
    action: str
    canonical_option: str
    reason: str

    @property
    def description(self) -> str:
        if self.action == "map":
            return f"{self.column_slug}: {self.write_in!r} -> {self.canonical_option!r}"
        return f"{self.column_slug}: {self.write_in!r} ({self.action})"


def load_winner_overrides() -> list[WinnerOverride]:
    path = config.MANUAL_WINNER_OVERRIDES_CSV
    if not os.path.exists(path):
        logger.info("No winner override file at %s; none applied", path)
        return []
    overrides: list[WinnerOverride] = []
    with open(path, newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            bot_name = (row.get("bot_name") or "").strip()
            flag_raw = (row.get("is_winner") or "").strip().lower()
            if not bot_name or flag_raw not in ("true", "false"):
                raise ValueError(
                    f"{path} line {line_number}: needs a bot_name and is_winner of true/false, "
                    f"got bot_name={bot_name!r}, is_winner={flag_raw!r}"
                )
            overrides.append(
                WinnerOverride(
                    bot_name=bot_name,
                    is_winner=flag_raw == "true",
                    reason=(row.get("reason") or "").strip(),
                )
            )
    logger.info("Loaded %d winner overrides from %s", len(overrides), path)
    return overrides


def load_answer_adjustments() -> list[AnswerAdjustment]:
    path = config.MANUAL_ANSWER_ADJUSTMENTS_CSV
    if not os.path.exists(path):
        logger.info("No answer adjustment file at %s; none applied", path)
        return []
    adjustments: list[AnswerAdjustment] = []
    with open(path, newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            adjustment = AnswerAdjustment(
                bot_name=(row.get("bot_name") or "").strip(),
                column_slug=(row.get("column_slug") or "").strip(),
                write_in=(row.get("write_in") or "").strip(),
                action=(row.get("action") or "").strip().lower(),
                canonical_option=(row.get("canonical_option") or "").strip(),
                reason=(row.get("reason") or "").strip(),
            )
            _validate_adjustment(adjustment, path, line_number)
            adjustments.append(adjustment)
    by_action = {action: sum(1 for a in adjustments if a.action == action) for action in VALID_ACTIONS}
    logger.info(
        "Loaded %d answer adjustments from %s (%d map, %d ignore, %d leave)",
        len(adjustments), path, by_action["map"], by_action["ignore"], by_action["leave"],
    )
    return adjustments


def _validate_adjustment(adjustment: AnswerAdjustment, path: str, line_number: int) -> None:
    where = f"{path} line {line_number}"
    if adjustment.action not in VALID_ACTIONS:
        raise ValueError(f"{where}: action must be one of {VALID_ACTIONS}, got {adjustment.action!r}")
    if not adjustment.bot_name or not adjustment.write_in:
        raise ValueError(f"{where}: bot_name and write_in are required")
    slug = adjustment.column_slug
    known_column = (
        slug in config.MULTISELECT_VOCAB
        or slug in config.SINGLE_SELECT_VOCAB
        or slug in MODEL_COLUMN_SLUGS
    )
    if not known_column:
        raise ValueError(f"{where}: unknown column_slug {slug!r}")
    if adjustment.action != "map":
        if adjustment.canonical_option:
            raise ValueError(
                f"{where}: canonical_option must be empty for action {adjustment.action!r}"
            )
        return
    if slug in MODEL_COLUMN_SLUGS:
        displays = {model.display for model in config.MODEL_REGISTRY}
        if adjustment.canonical_option not in displays:
            raise ValueError(
                f"{where}: canonical_option {adjustment.canonical_option!r} is not a model "
                "registry display name"
            )
        return
    vocab = config.MULTISELECT_VOCAB.get(slug) or config.SINGLE_SELECT_VOCAB[slug]
    if adjustment.canonical_option not in vocab:
        raise ValueError(
            f"{where}: canonical_option {adjustment.canonical_option!r} is not in the "
            f"{slug!r} vocabulary"
        )
