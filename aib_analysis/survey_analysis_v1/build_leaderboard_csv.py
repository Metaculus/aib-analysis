"""Dump a simulated tournament's spot-peer leaderboard to CSV.

The survey analysis needs a flat bot_name -> total_score table. This produces
it from the tournament JSON written by the simulation scripts.

    poetry run python aib_analysis/survey_analysis_v1/build_leaderboard_csv.py \
        local/spring_2026_simulations/2_bot_tournament.json \
        local/spring_survey_analysis/data/spring_2026_leaderboard.csv

Loading the JSON takes a few minutes; the files run to hundreds of MB.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)

from aib_analysis.data_structures.custom_types import ScoreType
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.main_logic.process_tournament import get_leaderboard

logger = logging.getLogger(__name__)


def main(tournament_json: Path, output_csv: Path, score_type: ScoreType) -> None:
    logger.info("Loading %s", tournament_json)
    with open(tournament_json) as handle:
        data = json.load(handle)

    tournament = SimulatedTournament(**data)
    leaderboard = get_leaderboard(tournament, score_type)
    entries = leaderboard.entries_via_sum_of_scores()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "bot_name", "total_score", "average_score", "question_count"]
        )
        for rank, entry in enumerate(entries, start=1):
            writer.writerow(
                [
                    rank,
                    entry.user.name,
                    round(entry.sum_of_scores, 4),
                    round(entry.average_score, 4),
                    entry.question_count,
                ]
            )

    logger.info("Wrote %d entries to %s", len(entries), output_csv)
    print(f"Wrote {len(entries)} entries to {output_csv}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tournament_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--score-type",
        default="spot_peer",
        choices=[s.value for s in ScoreType],
    )
    args = parser.parse_args()
    main(args.tournament_json, args.output_csv, ScoreType(args.score_type))
