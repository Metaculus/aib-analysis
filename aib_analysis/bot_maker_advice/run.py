"""Generate the "Advice from Bot Makers to Bot Makers" post for one season.

Usage (the OpenRouter key and a Metaculus token for bot ranks come from envh):
    envh run --with OPENROUTER_API_KEY=<secret>,METACULUS_TOKEN=<secret> --reason "..." -- \
        poetry run python -m aib_analysis.bot_maker_advice.run --season summer_2026 [--smoke-test] [--max-cost 10]

Writes, under local/<season>_bot_maker_advice/: writeup.md (the post, with [===Fill me===] reminders),
qa_review.md (the internal audit), data/codes.csv (every vote), respondent_key.csv (maps IDs to bot
names; keep it local), and llm_calls/ (every prompt and reply, reused on reruns so they cost nothing).
--smoke-test runs cheap models on the first SMOKE_TEST_ANSWER_COUNT answers into a separate folder.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import subprocess
from dataclasses import replace
from pathlib import Path

from forecasting_tools import MonetaryCostManager

from aib_analysis.bot_maker_advice import method
from aib_analysis.bot_maker_advice.answers import load_answers
from aib_analysis.bot_maker_advice.leaderboard import (
    attach_ranks,
    load_leaderboard_entries,
)
from aib_analysis.bot_maker_advice.llm import CachedOpenRouterLlm, StructuredLlm
from aib_analysis.bot_maker_advice.results import AnalysisResult
from aib_analysis.bot_maker_advice.review import CodeVersion, render_qa_review
from aib_analysis.bot_maker_advice.season_config import (
    AdviceSeasonConfig,
    get_season_config,
)
from aib_analysis.bot_maker_advice.survey_evidence import (
    SurveyEvidence,
    parse_survey_evidence,
)
from aib_analysis.bot_maker_advice.template import render_writeup
from aib_analysis.bot_maker_advice.thematic_analysis import run_thematic_analysis
from aib_analysis.pro_vs_bot_writeup.phrasing import FILL_ME

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_FOLDER = "aib_analysis/bot_maker_advice"


def current_code_version() -> CodeVersion:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", METHOD_FOLDER], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if commit.returncode != 0 or status.returncode != 0:
        logger.warning(f"Could not read the git version: {commit.stderr.strip()} {status.stderr.strip()}")
        return CodeVersion(commit=None, method_has_uncommitted_changes=True)
    return CodeVersion(commit=commit.stdout.strip(), method_has_uncommitted_changes=bool(status.stdout.strip()))


def load_survey_evidence(config: AdviceSeasonConfig) -> list[SurveyEvidence] | None:
    if config.survey_report_markdown is None:
        return None
    path = REPO_ROOT / config.survey_report_markdown
    if not path.exists():
        raise FileNotFoundError(
            f"No survey analysis report at {path}. Run the survey analysis first, or set survey_report_markdown=None in "
            "season_config.py to skip the data check."
        )
    evidence = parse_survey_evidence(path.read_text())
    logger.info(f"Read {len(evidence)} survey analysis results ({sum(item.significant for item in evidence)} significant) from {path}")
    return evidence


def write_data_files(result: AnalysisResult, output_folder: Path) -> None:
    data_folder = output_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    with open(data_folder / "codes.csv", "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["answer_id", "theme_number", "coder_1", "coder_2", "tie_break", "final_stance", "has_majority", "check_verdict", "counted", "excerpt"])
        for pair in result.pairs:
            writer.writerow(
                [
                    pair.answer_id,
                    pair.theme_number,
                    pair.votes[0].stance,
                    pair.votes[1].stance,
                    pair.tie_break.stance if pair.tie_break else "",
                    pair.final_stance,
                    pair.has_majority,
                    pair.check.verdict if pair.check else "",
                    pair.counted,
                    pair.excerpt or "",
                ]
            )
    with open(output_folder / "respondent_key.csv", "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["answer_id", "bot_name"])
        writer.writerows((answer.answer_id, answer.bot_name) for answer in result.survey.answers)


async def analyze(
    llm: StructuredLlm, models: method.ModelRoster, config: AdviceSeasonConfig, output_folder: Path, smoke_test: bool
) -> AnalysisResult:
    survey = load_answers(REPO_ROOT / config.survey_csv, id_seed=f"{config.key}-answer-ids")
    if config.bot_tournament_project_id is not None:
        entries = load_leaderboard_entries(config.bot_tournament_project_id, output_folder / "data" / "bot_leaderboard.json")
        survey = replace(survey, open_source_bots=attach_ranks(survey.open_source_bots, entries))
    if smoke_test:
        survey = replace(survey, answers=survey.answers[: method.SMOKE_TEST_ANSWER_COUNT])
        logger.info(f"Smoke test: using the first {len(survey.answers)} answers by ID")
    return await run_thematic_analysis(llm, models, config.key, config.label, survey, load_survey_evidence(config))


def write_outputs(result: AnalysisResult, config: AdviceSeasonConfig, output_folder: Path, usage_note: str) -> None:
    code_version = current_code_version()
    if code_version.method_has_uncommitted_changes:
        logger.warning("The method files have uncommitted changes, so this run isn't precommitted (noted in qa_review.md)")
    writeup = render_writeup(result, config)
    (output_folder / "writeup.md").write_text(writeup)
    (output_folder / "qa_review.md").write_text(render_qa_review(result, code_version, usage_note))
    write_data_files(result, output_folder)
    logger.info(f"Wrote writeup.md ({writeup.count(FILL_ME)} Fill me blocks), qa_review.md, data/codes.csv, and respondent_key.csv")


def main(season_key: str, output_folder: Path | None, smoke_test: bool, max_cost: float) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = get_season_config(season_key)
    suffix = "_smoke_test" if smoke_test else ""
    output_folder = output_folder or REPO_ROOT / "local" / f"{season_key}_bot_maker_advice{suffix}"
    output_folder.mkdir(parents=True, exist_ok=True)
    models = method.SMOKE_TEST_MODELS if smoke_test else method.PRECOMMITTED_MODELS
    logger.info(f"Analyzing {config.label} bot maker advice into {output_folder} (cost cap ${max_cost:.2f})")
    with MonetaryCostManager(max_cost) as cost_manager:
        llm = CachedOpenRouterLlm(output_folder / "llm_calls", cost_manager)
        result = asyncio.run(analyze(llm, models, config, output_folder, smoke_test))
        usage_note = llm.usage_note()
    write_outputs(result, config, output_folder, usage_note)
    logger.info(f"Done ({usage_note}). Review qa_review.md before publishing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the 'Advice from Bot Makers to Bot Makers' post for one season")
    parser.add_argument("--season", required=True, help="Season key in season_config.SEASONS, e.g. summer_2026")
    parser.add_argument("--output-folder", type=Path, default=None, help="Defaults to local/<season>_bot_maker_advice")
    parser.add_argument("--smoke-test", action="store_true", help="Cheap models on a few answers, into a separate folder")
    parser.add_argument("--max-cost", type=float, default=method.DEFAULT_COST_CAP_USD, help="Hard cap on OpenRouter spend in USD")
    arguments = parser.parse_args()
    main(arguments.season, arguments.output_folder, arguments.smoke_test, arguments.max_cost)
