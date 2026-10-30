"""Run the full survey analysis for the season set in config.py.

Generates, under config.OUTPUT_DIR:
- the report (config.REPORT_MD) + charts/
- parsing_decisions.md       (the rules)
- parsing_review.md          (the audit + data/respondent_audit.csv)

Usage:
    poetry run python -m aib_analysis.survey_analysis.run [--refresh] [--blind]

--refresh recomputes the leaderboard from the bot tournament JSON instead of
using the cached CSV. --blind withholds every precommitted and alternative
hypothesis result from the report (see precommitted.py).
"""

from __future__ import annotations

import argparse
import logging

from aib_analysis.survey_analysis import (
    config,
    features as features_module,
    loading,
    parsing_decisions,
    report,
    review_report,
)

logger = logging.getLogger(__name__)


def main(refresh: bool = False, blind: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting survey analysis v2")

    logger.info("Step 1/5: loading respondents and joining to leaderboard + prizes")
    respondents = loading.build_respondents(refresh=refresh)

    logger.info("Step 2/5: building features")
    features = features_module.build_features(respondents)

    logger.info("Step 3/5: writing analysis report, charts, and index page")
    report.write_report(features, blind=blind)
    report.write_index_html()

    logger.info("Step 4/5: writing parsing decisions doc")
    parsing_decisions.write_parsing_decisions()

    logger.info("Step 5/5: writing parsing review doc and audit CSV")
    review_report.write_review_report(features)

    logger.info("Done. Outputs are in %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FutureEval bot-maker survey analysis")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute the leaderboard from the bot tournament JSON instead of the cache.",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Withhold precommitted and alternative hypothesis results from the report.",
    )
    args = parser.parse_args()
    main(refresh=args.refresh, blind=args.blind)
