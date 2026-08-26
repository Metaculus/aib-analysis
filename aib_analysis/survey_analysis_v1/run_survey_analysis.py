"""Run the bot-maker survey analysis for one season.

    poetry run python aib_analysis/survey_analysis_v1/run_survey_analysis.py --season spring-2026

Outputs into the season's `output_dir`:
    survey_report.html          reviewable report with every chart and table
    charts/*.png                each figure on its own
    data/tests_*.csv            full test results including corrected p-values
    data/respondent_features.csv per-respondent derived feature matrix

The leaderboard CSV is built separately by `build_leaderboard_csv.py`, which
reads the simulated tournament JSON.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)

from aib_analysis.survey_analysis_v1.analysis import derive_features, run_analysis
from aib_analysis.survey_analysis_v1.loading import load_dataset
from aib_analysis.survey_analysis_v1.seasons import SEASONS
from aib_analysis.survey_analysis_v1.utils.plots import build_all_charts
from aib_analysis.survey_analysis_v1.utils.report import write_reports
from aib_analysis.survey_analysis_v1.utils.review_report import build_review_report

logger = logging.getLogger(__name__)


def main(season_key: str) -> None:
    config = SEASONS.get(season_key)
    if config is None:
        raise SystemExit(
            f"Unknown season '{season_key}'. Available: {', '.join(sorted(SEASONS))}"
        )

    for path, label in (
        (config.survey_csv, "survey"),
        (config.participation_csv, "participation"),
        (config.leaderboard_csv, "leaderboard"),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {label} file: {path}")

    logger.info("Analysing %s", config.season)
    dataset = load_dataset(config)
    derive_features(dataset)
    results = run_analysis(dataset)

    for family in results.families:
        survivors = family.survivors("bonferroni")
        logger.info(
            "Family '%s': %d tests, Bonferroni alpha %.5f, %d survivor(s): %s",
            family.name,
            family.size,
            family.alpha / family.size if family.size else float("nan"),
            len(survivors),
            ", ".join(s.label for s in survivors) or "none",
        )

    charts = build_all_charts(results)
    report_path = write_reports(results, charts)
    review_path = build_review_report(results)
    review_html = review_path.with_suffix(".html")

    print(f"\nSummary report:  {report_path}")
    print(f"Review report:   {review_path} (Markdown, opens in VS Code)")
    if review_html.exists():
        print(f"                 {review_html} (self-contained, opens in any browser)")
    print(f"Charts:          {config.charts_dir} ({len(charts)} summary figures)")
    print(f"Data:            {config.data_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        default="spring-2026",
        choices=sorted(SEASONS),
        help="which season config to run",
    )
    args = parser.parse_args()
    main(args.season)
