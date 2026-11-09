"""Human-reviewed join decisions that the data cannot make on its own.

Two CSVs in the non-committed private input directory drive this module. They
live next to the raw survey data because, like it, they reference individual
bots; keeping them there means they persist for reruns without entering git.

- manual_bot_name_aliases.csv (survey_bot_name, tournament_bot_name, reason)
    Joins a survey bot name to its tournament entry when the automatic matcher
    cannot safely bridge the two (e.g. a name too short to match without
    risking a wrong join). Verified by hand against the prize sheet.

- manual_winner_overrides.csv (bot_name, is_winner, reason)
    Bot-level winner status where the prize sheet cannot provide it: the sheet
    aggregates per owner, so for an owner with several bots and at least one
    prize it records how many of the owner's bots won, not which ones. Each
    such respondent needs an explicit row here; loading raises otherwise.

Survey answers themselves are never edited by hand: write-ins are handled only by
committed rules such as config.RESEARCH_WRITE_IN_SOURCES.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass

from aib_analysis.survey_analysis import config

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class WinnerOverride:
    bot_name: str
    is_winner: bool
    reason: str


def load_bot_name_aliases() -> dict[str, str]:
    """Survey bot name -> tournament bot name, from the reviewed alias CSV."""
    path = config.MANUAL_BOT_NAME_ALIASES_CSV
    if not os.path.exists(path):
        logger.info("No bot name alias file at %s; none applied", path)
        return {}
    aliases: dict[str, str] = {}
    with open(path, newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            survey_name = (row.get("survey_bot_name") or "").strip()
            tournament_name = (row.get("tournament_bot_name") or "").strip()
            if not survey_name or not tournament_name:
                raise ValueError(
                    f"{path} line {line_number}: needs survey_bot_name and tournament_bot_name"
                )
            if survey_name in aliases:
                raise ValueError(f"{path} line {line_number}: duplicate alias for {survey_name!r}")
            aliases[survey_name] = tournament_name
    logger.info("Loaded %d bot name aliases from %s", len(aliases), path)
    return aliases


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
