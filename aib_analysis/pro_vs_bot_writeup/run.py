"""Generate the FutureEval Pros vs bots write-up for one season.

Usage:
    poetry run python -m aib_analysis.pro_vs_bot_writeup.run --season summer_2026
        [--simulations-folder PATH] [--output-folder PATH] [--html]

Writes writeup.md (links to charts/*.png, for pasting into a post), writeup_preview.md (the
same text with the charts embedded, for markdown viewers that won't load local images),
data/*.csv, and pro_name_key.csv (maps Pro aliases to usernames; keep it local). --html also
renders writeup.html with Quarto.
"""

from __future__ import annotations

import argparse
import base64
import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from aib_analysis.pro_vs_bot_writeup import charts, phrasing
from aib_analysis.pro_vs_bot_writeup.results import (
    FIELD_CHART_MIN_QUESTIONS,
    LeaderboardRow,
    SeasonResults,
    compute_season_results,
)
from aib_analysis.pro_vs_bot_writeup.season_config import (
    SeasonConfig,
    get_season_config,
)
from aib_analysis.pro_vs_bot_writeup.template import (
    INDIVIDUAL_CHART_ROWS,
    METAC_CHART_ROWS,
    TOP_BOTS_CHART_ROWS,
    ChartFiles,
    render_writeup,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_FILES = ChartFiles(
    season_trend="charts/season_trend.png",
    team_size="charts/team_size.png",
    individuals="charts/individual_leaderboard.png",
    accuracy="charts/accuracy.png",
    top_bots="charts/top_bots.png",
    field="charts/bot_field.png",
    metac="charts/metac_bots.png",
)
CHART_LINK_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>charts/[^)]+\.png)\)")


def write_charts(results: SeasonResults, config: SeasonConfig, output_folder: Path) -> int:
    """Draw every chart and return how many bots the field chart leaves out."""
    charts_folder = output_folder / "charts"
    charts_folder.mkdir(parents=True, exist_ok=True)
    for stale_chart in charts_folder.glob("*.png"):
        stale_chart.unlink()
    charts.season_trend_chart(results.season_trend, output_folder / CHART_FILES.season_trend)
    charts.team_size_chart(results.team_sizes, config.main_bot_team_size, output_folder / CHART_FILES.team_size)
    charts.individual_leaderboard_chart(results.individuals, INDIVIDUAL_CHART_ROWS, output_folder / CHART_FILES.individuals)
    charts.accuracy_chart(results.bot_field_accuracy, results.team_accuracy, output_folder / CHART_FILES.accuracy)
    charts.top_bots_chart(results.field, TOP_BOTS_CHART_ROWS, output_folder / CHART_FILES.top_bots)
    charts.metac_chart(results.metac, METAC_CHART_ROWS, output_folder / CHART_FILES.metac)
    hidden_bots = charts.field_chart(results.field, FIELD_CHART_MIN_QUESTIONS, output_folder / CHART_FILES.field)
    return len(hidden_bots)


def write_leaderboard_csv(rows: list[LeaderboardRow], path: Path) -> None:
    with open(path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["rank", "name", "user_type", "sum_of_scores", "average_score", "ci_lower", "ci_upper", "questions_scored"])
        for row in rows:
            writer.writerow(
                [
                    row.rank,
                    row.name,
                    row.user_type,
                    f"{row.sum_of_scores:.4f}",
                    f"{row.average_score:.4f}",
                    "" if row.ci is None else f"{row.ci[0]:.4f}",
                    "" if row.ci is None else f"{row.ci[1]:.4f}",
                    row.question_count,
                ]
            )


def write_data_files(results: SeasonResults, output_folder: Path) -> None:
    data_folder = output_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    write_leaderboard_csv(results.field.rows, data_folder / "bot_leaderboard.csv")
    write_leaderboard_csv(results.individuals.rows, data_folder / "individual_leaderboard.csv")
    write_leaderboard_csv(results.metac.baseline_rows, data_folder / "metac_same_prompt_leaderboard.csv")
    with open(data_folder / "team_sizes.csv", "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["team_size", "is_all_bots", "mean", "ci_lower", "ci_upper", "questions"])
        for team_size in results.team_sizes:
            writer.writerow(
                [
                    team_size.team_size,
                    team_size.is_all_bots,
                    f"{team_size.mean:.4f}",
                    "" if team_size.ci is None else f"{team_size.ci[0]:.4f}",
                    "" if team_size.ci is None else f"{team_size.ci[1]:.4f}",
                    team_size.question_count,
                ]
            )
    with open(output_folder / "pro_name_key.csv", "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["alias", "username"])
        writer.writerows(sorted((alias, name) for name, alias in results.individuals.pro_aliases.items()))


def embed_charts(writeup: str, output_folder: Path) -> str:
    """Replace each chart link with a base64 data URI of the PNG."""

    def to_data_uri(match: re.Match[str]) -> str:
        encoded = base64.b64encode((output_folder / match.group("path")).read_bytes()).decode("ascii")
        return f"![{match.group('alt')}](data:image/png;base64,{encoded})"

    return CHART_LINK_PATTERN.sub(to_data_uri, writeup)


def render_html(output_folder: Path) -> None:
    if shutil.which("quarto") is None:
        raise FileNotFoundError("--html needs Quarto on PATH (https://quarto.org)")
    cache_base = Path(tempfile.gettempdir()) / "quarto_pro_vs_bot_writeup"
    environment = dict(os.environ)
    for variable, folder_name in (("XDG_CACHE_HOME", "cache"), ("DENO_DIR", "deno"), ("XDG_DATA_HOME", "data")):
        (cache_base / folder_name).mkdir(parents=True, exist_ok=True)
        environment[variable] = str(cache_base / folder_name)
    logger.info("Rendering writeup.html with Quarto")
    subprocess.run(
        ["quarto", "render", "writeup.md", "--to", "html", "--embed-resources", "-M", "toc=true", "-M", "toc-location=left"],
        cwd=output_folder,
        env=environment,
        check=True,
    )


def main(season_key: str, simulations_folder: Path | None, output_folder: Path | None, html: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = get_season_config(season_key)
    simulations_folder = simulations_folder or REPO_ROOT / "local" / f"{season_key}_simulations"
    output_folder = output_folder or REPO_ROOT / "local" / f"{season_key}_pro_vs_bot_writeup"
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step 1/4: computing {config.label} results")
    results = compute_season_results(simulations_folder, config)

    logger.info("Step 2/4: drawing charts and writing data files")
    field_hidden_count = write_charts(results, config, output_folder)
    write_data_files(results, output_folder)

    logger.info("Step 3/4: writing the write-up")
    writeup = render_writeup(results, config, CHART_FILES, field_hidden_count)
    writeup_path = output_folder / "writeup.md"
    writeup_path.write_text(writeup)
    logger.info(f"Wrote {writeup_path} ({writeup.count(phrasing.FILL_ME)} Fill me blocks)")
    preview_path = output_folder / "writeup_preview.md"
    preview_path.write_text(embed_charts(writeup, output_folder))
    logger.info(f"Wrote {preview_path} (charts embedded)")

    logger.info("Step 4/4: HTML preview" if html else "Step 4/4: skipped HTML preview (pass --html)")
    if html:
        render_html(output_folder)
    logger.info(f"Done. Outputs are in {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the FutureEval Pros vs bots write-up for one season")
    parser.add_argument("--season", required=True, help="Season key in season_config.SEASONS, e.g. summer_2026")
    parser.add_argument("--simulations-folder", type=Path, default=None, help="Defaults to local/<season>_simulations")
    parser.add_argument("--output-folder", type=Path, default=None, help="Defaults to local/<season>_pro_vs_bot_writeup")
    parser.add_argument("--html", action="store_true", help="Also render writeup.html with Quarto")
    arguments = parser.parse_args()
    main(arguments.season, arguments.simulations_folder, arguments.output_folder, arguments.html)
