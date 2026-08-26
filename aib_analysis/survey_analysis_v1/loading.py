"""Load and join the three inputs: survey, participation sheet, leaderboard."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from aib_analysis.survey_analysis.config import SURVEY_COLUMNS, SeasonConfig

logger = logging.getLogger(__name__)

_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]", re.UNICODE
)


def normalize_name(raw: str | None) -> str:
    """Key used to join bot names across the three files.

    Strips emoji (several respondents decorate their bot name), lowercases,
    and drops anything that is not alphanumeric or a common separator.
    """
    if not raw:
        return ""
    text = _EMOJI.sub("", raw).strip().lower()
    return re.sub(r"[^a-z0-9._+-]", "", text)


@dataclass
class Respondent:
    """One survey response joined to leaderboard and prize data."""

    bot_name: str
    key: str
    answers: dict[str, str]

    # From the participation sheet.
    in_tournament: bool = False
    prize: float = 0.0
    secondary_prize: float = 0.0
    primary_forecasts: int = 0
    secondary_forecasts: int = 0
    participation_found: bool = False

    # From the leaderboard.
    rank: int | None = None
    participant_rank: int | None = None
    total_score: float | None = None
    average_score: float | None = None
    question_count: int | None = None

    # Derived features.
    features: dict[str, object] = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.prize > 0

    @property
    def cohort(self) -> str:
        """Which analysis group this respondent belongs to.

        `minibench_only` respondents never forecast in the main tournament, so
        they cannot be winners or non-winners of it. Folding them into the
        non-winner group would compare people who competed and lost against
        people who never entered.
        """
        if not self.participation_found:
            return "unknown"
        if not self.in_tournament:
            return "minibench_only"
        return "winner" if self.is_winner else "non_winner"

    def answer(self, field_name: str) -> str | None:
        return self.answers.get(field_name)


@dataclass
class Dataset:
    respondents: list[Respondent]
    config: SeasonConfig
    resolved_columns: dict[str, str]
    missing_columns: list[str]
    leaderboard: list[dict[str, str]]

    def cohort(self, *names: str) -> list[Respondent]:
        return [r for r in self.respondents if r.cohort in names]

    @property
    def compared(self) -> list[Respondent]:
        """Winners plus non-winners: the population for the main comparison."""
        return self.cohort("winner", "non_winner")

    @property
    def winners(self) -> list[Respondent]:
        return self.cohort("winner")

    @property
    def non_winners(self) -> list[Respondent]:
        return self.cohort("non_winner")

    @property
    def minibench_only(self) -> list[Respondent]:
        return self.cohort("minibench_only")


def _resolve_columns(header: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map canonical field names to the actual header text of this season."""
    resolved: dict[str, str] = {}
    for spec in SURVEY_COLUMNS:
        for column in header:
            if column in resolved.values():
                continue
            if spec.matches(column):
                resolved[spec.name] = column
                break

    missing = [s.name for s in SURVEY_COLUMNS if s.name not in resolved]
    for spec in SURVEY_COLUMNS:
        if spec.required and spec.name not in resolved:
            raise ValueError(
                f"Required survey column '{spec.name}' not found. "
                f"Headers seen: {header}"
            )
    return resolved, missing


def load_dataset(config: SeasonConfig) -> Dataset:
    survey_rows, resolved, missing = _load_survey(config.survey_csv)
    participation = _load_participation(config)
    leaderboard, participant_ranks = _load_leaderboard(config)

    respondents: list[Respondent] = []
    for answers in survey_rows:
        bot_name = (answers.get("bot_name") or "").strip()
        if not bot_name:
            continue
        key = normalize_name(bot_name)
        key = config.name_aliases.get(key, config.name_aliases.get(bot_name, key))
        key = normalize_name(key)

        respondent = Respondent(bot_name=bot_name, key=key, answers=answers)

        record = participation.get(key)
        if record is not None:
            respondent.participation_found = True
            respondent.in_tournament = record["in_tournament"]
            respondent.prize = record["prize"]
            respondent.secondary_prize = record["secondary_prize"]
            respondent.primary_forecasts = record["primary_forecasts"]
            respondent.secondary_forecasts = record["secondary_forecasts"]
        else:
            logger.warning(
                "No participation row for survey respondent %r (key=%r)",
                bot_name,
                key,
            )

        entry = leaderboard.get(key)
        if entry is not None:
            respondent.rank = int(entry["rank"])
            respondent.total_score = float(entry["total_score"])
            respondent.average_score = float(entry["average_score"])
            respondent.question_count = int(entry["question_count"])
            respondent.participant_rank = participant_ranks.get(key)

        respondents.append(respondent)

    logger.info(
        "Loaded %d respondents (%d winners, %d non-winners, %d minibench-only, %d unknown)",
        len(respondents),
        sum(1 for r in respondents if r.cohort == "winner"),
        sum(1 for r in respondents if r.cohort == "non_winner"),
        sum(1 for r in respondents if r.cohort == "minibench_only"),
        sum(1 for r in respondents if r.cohort == "unknown"),
    )

    return Dataset(
        respondents=respondents,
        config=config,
        resolved_columns=resolved,
        missing_columns=missing,
        leaderboard=list(leaderboard.values()),
    )


def _load_survey(path: Path) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        raw_rows = [row for row in reader if any(cell.strip() for cell in row)]

    resolved, missing = _resolve_columns(header)
    index = {column: i for i, column in enumerate(header)}

    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        answers: dict[str, str] = {}
        for field_name, column in resolved.items():
            i = index[column]
            answers[field_name] = raw[i].strip() if i < len(raw) else ""
        rows.append(answers)

    if missing:
        logger.info("Survey columns not present this season: %s", ", ".join(missing))
    return rows, resolved, missing


def _load_participation(config: SeasonConfig) -> dict[str, dict]:
    """Index the participation sheet by every name that identifies a row.

    Rows key off `bot_usernames`, which may hold several comma-separated bots
    for one owner, and additionally off `owner_username` so respondents who
    give their account name instead of their bot name still match. Some rows
    carry only a bot name with no owner, so neither column can be required.
    """
    records: dict[str, dict] = {}

    def to_float(value: str | None) -> float:
        try:
            return float((value or "").replace(",", "").replace("$", "").strip() or 0)
        except ValueError:
            return 0.0

    def to_int(value: str | None) -> int:
        return int(to_float(value))

    with open(config.participation_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            bots_raw = (row.get(config.participation_bot_column) or "").strip()
            owner_raw = (row.get(config.participation_owner_column) or "").strip()
            if not bots_raw and not owner_raw:
                continue  # summary/blank row

            record = {
                "in_tournament": (row.get(config.in_tournament_column) or "").strip().upper()
                == "TRUE",
                "prize": to_float(row.get(config.prize_column)),
                "secondary_prize": to_float(row.get(config.secondary_prize_column)),
                "primary_forecasts": to_int(row.get(config.primary_forecasts_column)),
                "secondary_forecasts": to_int(row.get(config.secondary_forecasts_column)),
                "raw": row,
            }

            names = [n.strip() for n in re.split(r"[,;]", bots_raw) if n.strip()]
            for name in names:
                records.setdefault(normalize_name(name), record)
            if owner_raw:
                records.setdefault(normalize_name(owner_raw), record)

    logger.info("Indexed %d participation keys", len(records))
    return records


def _load_leaderboard(
    config: SeasonConfig,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Load the leaderboard and compute participant-only ranks.

    The raw rank includes Metaculus reference bots and pre-seen entries, which
    are not competing for prizes. Participant rank re-numbers after removing
    them, so "rank 8 among participants" lines up with the prize list.
    """
    entries: dict[str, dict[str, str]] = {}
    ordered: list[tuple[str, str]] = []

    with open(config.leaderboard_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row["bot_name"].strip()
            key = normalize_name(name)
            entries[key] = row
            ordered.append((key, name))

    disqualified = {normalize_name(n) for n in config.disqualified_bots}
    participant_ranks: dict[str, int] = {}
    counter = 0
    for key, name in ordered:
        if any(name.startswith(p) for p in config.excluded_bot_prefixes):
            continue
        if key in disqualified:
            logger.info("Excluding disqualified entry %r from participant ranking", name)
            continue
        counter += 1
        participant_ranks[key] = counter

    logger.info(
        "Loaded %d leaderboard entries (%d participants after excluding %s)",
        len(entries),
        counter,
        ", ".join(config.excluded_bot_prefixes),
    )
    return entries, participant_ranks
