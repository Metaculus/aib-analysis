"""Load the survey and prize CSVs and join respondents to peer scores + groups.

Join keys:
- survey bot name -> leaderboard bot name (for sum of spot peer score, rank, top 10)
- survey bot name -> prize-stats owner row (for winner / prize status)

The prize sheet's `bot_usernames` is itself a delimited list of bots, so both an
exact match and a tokenized (substring) match are attempted. Every respondent's
match outcome is recorded so the review doc can show nothing was dropped.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass

from aib_analysis.survey_analysis_v2 import config, manual_adjustments, season_notes
from aib_analysis.survey_analysis_v2.leaderboard import (
    LeaderboardRow,
    get_leaderboard_rows,
)
from aib_analysis.survey_analysis_v2.manual_adjustments import WinnerOverride

logger = logging.getLogger(__name__)

_ROBOT_EMOJI = "\U0001F916"


def normalize_name(raw: str) -> str:
    text = raw.replace(_ROBOT_EMOJI, "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _alnum_key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]", "", raw.replace(_ROBOT_EMOJI, "").lower())


def _split_bot_list(raw: str) -> list[str]:
    return [token.strip() for token in re.split(r"[,;]", raw) if token.strip()]


@dataclass
class PrizeOwner:
    owner_username: str
    bot_usernames: list[str]
    winner_count: int
    aib_prize: float
    total_prize: float

    @property
    def won_any(self) -> bool:
        return self.winner_count > 0 or self.aib_prize > 0


@dataclass
class Respondent:
    bot_name: str
    answers: dict[str, str]
    matched_leaderboard_name: str | None = None
    rank: int | None = None
    sum_spot_peer: float | None = None
    question_count: int | None = None
    is_winner: bool = False
    # Prize amounts are owner-level totals from the prize sheet (an owner with
    # several bots reports one combined figure); only is_winner is bot-level.
    aib_prize: float = 0.0
    total_prize: float = 0.0
    is_top_10: bool = False
    leaderboard_match_kind: str = "none"
    prize_match_kind: str = "none"
    matched_owner: str | None = None
    winner_source: str = "no_prize_match"

    @property
    def average_spot_peer(self) -> float | None:
        """Per-question mean spot peer score (sum divided by questions forecast)."""
        if self.sum_spot_peer is None or not self.question_count:
            return None
        return self.sum_spot_peer / self.question_count

    @property
    def meets_correlation_minimum(self) -> bool:
        return (self.question_count or 0) >= config.MIN_QUESTIONS_FOR_CORRELATION

    @property
    def group(self) -> str:
        return "winner" if self.is_winner else "non_winner"

    @property
    def groups(self) -> list[str]:
        memberships = [self.group]
        if self.is_top_10:
            memberships.append("top_10")
        return memberships


def load_survey() -> list[dict[str, str]]:
    with open(config.SURVEY_CSV, newline="") as handle:
        reader = csv.reader(handle)
        header = [cell.strip() for cell in next(reader)]
        raw_rows = [row for row in reader]

    header_to_slug = {name: slug for slug, name in config.COLUMNS.items()}
    index_to_slug: dict[int, str] = {}
    for index, name in enumerate(header):
        if name in header_to_slug:
            index_to_slug[index] = header_to_slug[name]

    records: list[dict[str, str]] = []
    skipped_empty_rows = 0
    for row_number, row in enumerate(raw_rows, start=2):
        if not any(cell.strip() for cell in row):
            skipped_empty_rows += 1
            continue
        record = {slug: "" for slug in config.COLUMNS}
        for index, value in enumerate(row):
            slug = index_to_slug.get(index)
            if slug is not None:
                record[slug] = value.strip()
        if not record["bot_name"]:
            raise ValueError(
                f"Survey row {row_number} has answers but no bot name. Every response "
                "must name its bot; a blank name means the export is broken or someone "
                "skipped the identifying question, so this needs a manual look."
            )
        records.append(record)
    if skipped_empty_rows:
        logger.info("Skipped %d fully empty survey rows", skipped_empty_rows)
    logger.info("Loaded %d survey respondents", len(records))
    return records


def load_prize_owners() -> list[PrizeOwner]:
    with open(config.PRIZE_STATS_CSV, newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [
            (name or "").lstrip("﻿").strip() for name in reader.fieldnames or []
        ]
        owners: list[PrizeOwner] = []
        for row in reader:
            username = (row.get("owner_username") or "").strip()
            bots_raw = (row.get("bot_usernames") or "").strip()
            if not username and not bots_raw:
                continue  # aggregate/summary row
            owners.append(
                PrizeOwner(
                    owner_username=username,
                    bot_usernames=_split_bot_list(bots_raw),
                    winner_count=_to_int(row.get("winner_count")),
                    aib_prize=_to_float(row.get("aib_prize")),
                    total_prize=_to_float(row.get("total_prize")),
                )
            )
    logger.info("Loaded %d prize-stats owner rows", len(owners))
    return owners


def _to_int(value: str | None) -> int:
    try:
        return int(float((value or "").strip()))
    except ValueError:
        return 0


def _to_float(value: str | None) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return 0.0


def _match_leaderboard(
    bot_name: str, by_norm: dict[str, LeaderboardRow], by_alnum: dict[str, LeaderboardRow]
) -> tuple[LeaderboardRow | None, str]:
    """Match a survey bot to a leaderboard row.

    Only exact and alphanumeric-equal matches are allowed; loose substring
    matching is deliberately avoided so a short leaderboard name can never be
    silently joined to the wrong respondent.
    """
    key = normalize_name(bot_name)
    if key in by_norm:
        return by_norm[key], "exact"
    alnum = _alnum_key(bot_name)
    if alnum and alnum in by_alnum:
        return by_alnum[alnum], "alnum"
    return None, "none"


def _match_prize(
    bot_name: str, owners: list[PrizeOwner]
) -> tuple[PrizeOwner | None, str]:
    """Match a survey bot to a prize owner via its (delimited) bot list."""
    alnum = _alnum_key(bot_name)
    for owner in owners:
        if any(_alnum_key(bot) == alnum for bot in owner.bot_usernames):
            return owner, "exact"
    # Tokenized fallback: only when a single, unambiguous owner contains (or is
    # contained by) the name, and only for names long enough to be distinctive.
    # This feeds is_winner, so an ambiguous match is left unmatched rather than
    # risk moving a respondent into the wrong group.
    if len(alnum) >= 6:
        candidates = {
            owner.owner_username: owner
            for owner in owners
            for bot in owner.bot_usernames
            if len(_alnum_key(bot)) >= 6
            and (alnum in _alnum_key(bot) or _alnum_key(bot) in alnum)
        }
        if len(candidates) == 1:
            owner = next(iter(candidates.values()))
            logger.warning(
                "Tokenized prize match for bot %r -> owner %r", bot_name, owner.owner_username
            )
            return owner, "tokenized"
        if len(candidates) > 1:
            logger.warning("Ambiguous tokenized prize match for bot %r; left unmatched", bot_name)
    return None, "none"


def _resolve_winner(
    bot_name: str, owner: PrizeOwner | None, override: WinnerOverride | None
) -> tuple[bool, str]:
    """Bot-level winner status, with the source of the decision for the audit.

    The prize sheet is owner-level: for an owner with several bots it records
    how many bots won, not which ones. A single-bot owner's status transfers to
    the bot directly; a multi-bot owner with no wins means no bot won; a
    multi-bot owner WITH wins is ambiguous and requires a manual override row.
    """
    if override is not None:
        logger.info(
            "Winner override applied for bot %r -> %s (%s)",
            bot_name, override.is_winner, override.reason or "no reason given",
        )
        return override.is_winner, "manual_override"
    if owner is None:
        return False, "no_prize_match"
    if len(owner.bot_usernames) <= 1:
        return owner.won_any, "owner_single_bot"
    if not owner.won_any:
        return False, "owner_multi_bot_no_wins"
    raise ValueError(
        f"Cannot determine bot-level winner status for {bot_name!r}: owner "
        f"{owner.owner_username!r} has {len(owner.bot_usernames)} bots and the prize sheet "
        f"only records owner-level wins (winner_count={owner.winner_count}, "
        f"aib_prize={owner.aib_prize}). Add a row for this bot to "
        f"{config.MANUAL_WINNER_OVERRIDES_CSV}."
    )


def build_respondents(refresh: bool = False) -> list[Respondent]:
    survey_records = load_survey()
    owners = load_prize_owners()
    leaderboard_rows = get_leaderboard_rows(refresh=refresh)
    overrides_by_name = {
        normalize_name(override.bot_name): override
        for override in manual_adjustments.load_winner_overrides()
    }

    by_norm = {normalize_name(row.bot_name): row for row in leaderboard_rows}
    by_alnum = {_alnum_key(row.bot_name): row for row in leaderboard_rows}
    top_10_names = {
        normalize_name(row.bot_name)
        for row in leaderboard_rows[: config.TOP_N_FOR_TOP_GROUP]
    }

    respondents: list[Respondent] = []
    used_overrides: set[str] = set()
    for record in survey_records:
        bot_name = record["bot_name"]
        respondent = Respondent(bot_name=bot_name, answers=record)
        # A reviewed alias lets a survey name join to its tournament entry when
        # the automatic matcher cannot safely bridge the two; the respondent's
        # displayed name stays exactly what they typed.
        match_name = season_notes.BOT_NAME_ALIASES.get(bot_name, bot_name)

        lb_row, lb_kind = _match_leaderboard(match_name, by_norm, by_alnum)
        respondent.leaderboard_match_kind = lb_kind
        if lb_row is not None:
            respondent.matched_leaderboard_name = lb_row.bot_name
            respondent.rank = lb_row.rank
            respondent.sum_spot_peer = lb_row.sum_spot_peer
            respondent.question_count = lb_row.question_count
            respondent.is_top_10 = normalize_name(lb_row.bot_name) in top_10_names

        owner, prize_kind = _match_prize(match_name, owners)
        respondent.prize_match_kind = prize_kind
        if owner is not None:
            respondent.matched_owner = owner.owner_username
            respondent.aib_prize = owner.aib_prize
            respondent.total_prize = owner.total_prize

        override = overrides_by_name.get(normalize_name(bot_name))
        if override is not None:
            used_overrides.add(normalize_name(bot_name))
        respondent.is_winner, respondent.winner_source = _resolve_winner(
            bot_name, owner, override
        )

        respondents.append(respondent)

    for name, override in overrides_by_name.items():
        if name not in used_overrides:
            logger.warning(
                "Winner override for %r matched no survey respondent; stale entry?",
                override.bot_name,
            )
    _log_join_summary(respondents)
    return respondents


def _log_join_summary(respondents: list[Respondent]) -> None:
    matched_lb = sum(1 for r in respondents if r.matched_leaderboard_name)
    winners = sum(1 for r in respondents if r.is_winner)
    top10 = sum(1 for r in respondents if r.is_top_10)
    logger.info(
        "Join summary: %d respondents | %d matched to leaderboard | %d winners | %d top-10 responders",
        len(respondents),
        matched_lb,
        winners,
        top10,
    )
    for respondent in respondents:
        if not respondent.matched_leaderboard_name:
            logger.warning("No leaderboard match for bot %r", respondent.bot_name)
        if respondent.prize_match_kind == "none":
            logger.warning("No prize-stats match for bot %r", respondent.bot_name)
