"""Official ranks from a season's bot tournament leaderboard, fetched once from the Metaculus API and cached locally.

The official rank is the participant rank with Metac Bots removed: entries with a nonzero exclusion status
(Metac Bots, extra non-primary bots) are shown on the board but don't take a rank.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path

import requests

from aib_analysis.bot_maker_advice.answers import OpenSourceBot

logger = logging.getLogger(__name__)

LEADERBOARD_URL = "https://www.metaculus.com/api/leaderboards/project/{project_id}/"


def name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_leaderboard_entries(project_id: int, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if "METACULUS_TOKEN" not in os.environ:
        raise RuntimeError("METACULUS_TOKEN is not set; run this through envh to fetch the leaderboard for bot ranks (see README)")
    response = requests.get(
        LEADERBOARD_URL.format(project_id=project_id),
        headers={"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"},
        timeout=60,
    )
    response.raise_for_status()
    primary = [leaderboard for leaderboard in response.json() if leaderboard.get("is_primary_leaderboard")]
    if len(primary) != 1:
        raise ValueError(f"Expected one primary leaderboard for project {project_id}, found {len(primary)}")
    entries = [entry for entry in primary[0]["entries"] if entry.get("user")]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(entries))
    logger.info(f"Fetched {len(entries)} leaderboard entries for project {project_id}")
    return entries


def attach_ranks(bots: tuple[OpenSourceBot, ...], entries: list[dict]) -> tuple[OpenSourceBot, ...]:
    """Match survey bot names to leaderboard usernames, ignoring case and punctuation."""
    by_name = {name_key(entry["user"]["username"]): entry for entry in entries}
    ranked_bots = []
    for bot in bots:
        entry = by_name.get(name_key(bot.bot_name))
        if entry is None:
            ranked_bots.append(replace(bot, leaderboard_status="not_found"))
        elif entry["exclusion_status"] != 0:
            ranked_bots.append(replace(bot, leaderboard_status="unranked"))
        else:
            ranked_bots.append(replace(bot, leaderboard_status="ranked", rank=entry["rank"]))
    return tuple(ranked_bots)
