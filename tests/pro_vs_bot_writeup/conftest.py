"""Synthetic simulation folders shaped like the real season outputs, for tests without private data."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aib_analysis.pro_vs_bot_writeup.season_config import SEASONS, SeasonConfig

QUESTION_TYPE_CYCLE = ("binary", "binary", "numeric", "multiple_choice", "discrete")
PRO_NAMES = ("pro_user_1", "pro_user_2", "pro_user_3")
NON_METAC_BOT_NAMES = tuple(f"bot-{letter}" for letter in "abcdefghijkl")
METAC_BOT_NAMES = (
    "metac-gpt-5+asknews",
    "metac-gpt-5-high+asknews",
    "metac-deepseek-r1+asknews",
    "metac-deepseek-r1+exa",
    "metac-deepseek-r1+no-research",
    "metac-deepseek-r1+asknews(variance-test)",
    "metac-agent",
    "metac-uniform-probability-bot",
)
NON_PRIMARY_BOT_NAME = "bot-l"
BOT_QUESTION_COUNT = 80
SHARED_QUESTION_COUNT = 36
ANNULLED_BOT_QUESTION_IDS = (79, 80)
ANNULLED_PRO_QUESTION_ID = 500
TEAM_SIZES = (1, 2, 3, "all")


def make_question(question_id: int, question_type: str = "binary", resolution: Any = True, title: str | None = None) -> dict[str, Any]:
    is_continuous = question_type in ("numeric", "discrete")
    return {
        "question_id": question_id,
        "type": question_type,
        "question_text": title or f"Will synthetic event {question_id} happen?",
        "resolution": resolution,
        "options": ["A", "B", "C"] if question_type == "multiple_choice" else None,
        "range_max": 100.0 if is_continuous else None,
        "range_min": 0.0 if is_continuous else None,
        "open_upper_bound": False if is_continuous else None,
        "open_lower_bound": False if is_continuous else None,
        "zero_point": None,
        "inbound_outcome_count": 100 if question_type == "discrete" else None,
        "weight": 1.0,
        "post_id": 1000 + question_id,
        "created_at": "2026-05-01T00:00:00Z",
        "spot_scoring_time": "2026-05-02T00:00:00Z",
        "project": None,
        "notes": None,
        "unscored_resolution_reason": None if resolution is not None else "annulled",
    }


def make_user(name: str, user_type: str, members: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    return {
        "name": name,
        "type": user_type,
        "aggregated_users": [make_user(member_name, member_type) for member_name, member_type in members],
        "is_primary_bot": None,
        "exclude_from_aggregations": None,
    }


def binary_prediction(question: dict[str, Any], score: float | None) -> list[float]:
    """A P(Yes) that puts more probability on the outcome the higher the score, so accuracy tracks peer score."""
    if question["type"] != "binary" or question["resolution"] is None or score is None:
        return [0.5, 0.5]
    probability_on_outcome = 0.5 + 0.4 * math.tanh(score / 30)
    probability_yes = probability_on_outcome if question["resolution"] else 1 - probability_on_outcome
    return [probability_yes, 1 - probability_yes]


def make_tournament(entries: list[tuple[dict[str, Any], dict[str, Any], float | None]]) -> dict[str, Any]:
    """Build a tournament JSON from (user, question, spot peer score) entries; None means unscored."""
    forecasts: list[dict[str, Any]] = []
    scores: dict[str, dict[str, Any]] = {}
    for index, (user, question, score) in enumerate(entries):
        forecast = {
            "question": question,
            "user": user,
            "prediction": binary_prediction(question, score),
            "prediction_time": "2026-05-01T12:00:00Z",
            "end_time": None,
            "forecasters_at_time": None,
        }
        forecasts.append(forecast)
        if score is not None:
            scores[f"{index}_peer"] = {"score": score, "type": "spot_peer", "forecast": forecast}
            scores[f"{index}_baseline"] = {"score": 0.0, "type": "spot_baseline", "forecast": forecast}
    return {"name": "synthetic", "forecasts": forecasts, "scores_cache": scores}


def combined_question_notes(pro_question_id: int, bot_question_id: int) -> str:
    """Notes in the format the simulation writes when it pairs a Pro question with its bot question."""
    return (
        f"Combined https://www.metaculus.com/questions/{pro_question_id}/ (QID:{pro_question_id}) and "
        f"https://www.metaculus.com/questions/{bot_question_id}/ (QID:{bot_question_id})\nQ1 Notes: None\nQ2 Notes: None"
    )


def make_team_tournament(bot_scores: list[float], bot_members: tuple[tuple[str, str], ...], pro_members: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    bot_team = make_user("Bot Team", "aggregate", bot_members)
    pro_team = make_user("Pro Team", "aggregate", pro_members)
    entries: list[tuple[dict[str, Any], dict[str, Any], float | None]] = []
    for question_index, bot_score in enumerate(bot_scores, start=1):
        question = make_question(question_index, QUESTION_TYPE_CYCLE[question_index % len(QUESTION_TYPE_CYCLE)])
        question["notes"] = combined_question_notes(question_index, question_index)
        entries.append((bot_team, question, bot_score))
        entries.append((pro_team, question, -bot_score))
    return make_tournament(entries)


def write_simulation_folder(folder: Path, bot_team_advantage: float, seed: int = 7) -> Path:
    """Write a small season simulation folder; bot_team_advantage shifts the head-to-head scores."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    bot_questions = [
        make_question(
            question_id,
            QUESTION_TYPE_CYCLE[question_id % len(QUESTION_TYPE_CYCLE)],
            resolution=None if question_id in ANNULLED_BOT_QUESTION_IDS else rng.random() < 0.35,
            title="Short?" if question_id == 5 else None,
        )
        for question_id in range(1, BOT_QUESTION_COUNT + 1)
    ]
    for question in bot_questions:
        if question["resolution"] is not None and question["type"] != "binary":
            question["resolution"] = "B" if question["type"] == "multiple_choice" else 42.0
    shared_questions = bot_questions[:SHARED_QUESTION_COUNT]
    bot_only_questions = bot_questions[SHARED_QUESTION_COUNT:]
    pros = [make_user(name, "pro") for name in PRO_NAMES]
    bot_names = NON_METAC_BOT_NAMES + METAC_BOT_NAMES
    bots = [make_user(name, "bot") for name in bot_names]
    next(bot for bot in bots if bot["name"] == NON_PRIMARY_BOT_NAME)["is_primary_bot"] = False
    human_in_bot_tournament = make_user("human_account", "pro")
    skills = {name: rng.uniform(-6, 12) for name in bot_names + PRO_NAMES + ("human_account",)}
    skills["metac-uniform-probability-bot"] = -12.0

    def spot_score(name: str, question: dict[str, Any]) -> float | None:
        if question["resolution"] is None:
            return None
        return skills[name] + rng.gauss(0, 12)

    def entries_for(users: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], float | None]]:
        return [(user, question, spot_score(user["name"], question)) for user in users for question in questions]

    annulled_pro_question = make_question(ANNULLED_PRO_QUESTION_ID, resolution=None)
    pro_entries = entries_for(pros, shared_questions) + [(pros[0], annulled_pro_question, None)]
    files = {
        "1_pro_tournament.json": make_tournament(pro_entries),
        "2_bot_tournament.json": make_tournament(entries_for(bots + [human_in_bot_tournament], bot_questions)),
        "3_bot_tournament_without_pro_questions.json": make_tournament(entries_for(bots, bot_only_questions)),
        "4_metac_bot_tournament.json": make_tournament(
            entries_for([bot for bot in bots if bot["name"].startswith("metac-")], bot_questions)
        ),
        "6_pro_with_bot_tourn__no_teams.json": make_tournament(
            entries_for(pros + bots + [human_in_bot_tournament], shared_questions)
        ),
    }
    pro_members = tuple((name, "pro") for name in PRO_NAMES)
    for team_size in TEAM_SIZES:
        member_count = len(NON_METAC_BOT_NAMES) if team_size == "all" else team_size
        bot_members = tuple((name, "bot") for name in NON_METAC_BOT_NAMES[:member_count])
        if team_size == "all":
            bot_members += (("human_account", "pro"),)
        size_effect = -3.0 if team_size == "all" else 0.0
        bot_scores = [bot_team_advantage + size_effect + rng.gauss(0, 6) for _ in range(SHARED_QUESTION_COUNT)]
        files[f"7_pro_vs_bot_tournament__teams_size_{team_size}.json"] = make_team_tournament(
            bot_scores, bot_members, pro_members
        )
    for file_name, content in files.items():
        (folder / file_name).write_text(json.dumps(content))
    return folder


@pytest.fixture
def synthetic_config() -> SeasonConfig:
    return replace(SEASONS["summer_2026"], key="synthetic", main_bot_team_size=3)


@pytest.fixture
def pros_ahead_folder(tmp_path: Path) -> Path:
    return write_simulation_folder(tmp_path / "pros_ahead_simulations", bot_team_advantage=-8.0)


@pytest.fixture
def bots_ahead_folder(tmp_path: Path) -> Path:
    return write_simulation_folder(tmp_path / "bots_ahead_simulations", bot_team_advantage=8.0)
