"""Facts about each season that the survey answers cannot provide.

A None tournament URL renders as a [===Fill me===] block in the write-up. Data-check notes appear in the
post only once survey_post_url is set, since they link to that post. Paths are relative to the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdviceSeasonConfig:
    key: str
    label: str
    survey_csv: Path
    tournament_url: str | None
    bot_tournament_project_id: int | None
    survey_post_url: str | None
    survey_report_markdown: Path | None


SEASONS: dict[str, AdviceSeasonConfig] = {
    "spring_2026": AdviceSeasonConfig(
        key="spring_2026",
        label="Spring 2026",
        survey_csv=Path(
            "local/private_input_data/fall_and_spring_survey_data/"
            "Spring 2026 FutureEval Participant Survey (Responses) - Form Responses 1.csv"
        ),
        tournament_url="https://www.metaculus.com/tournament/spring-aib-2026/",
        bot_tournament_project_id=32916,
        survey_post_url="https://www.metaculus.com/notebooks/45382/bot-survey-spring-2026/",
        survey_report_markdown=Path("local/spring_survey_analysis/spring_survey_analysis.md"),
    ),
    "summer_2026": AdviceSeasonConfig(
        key="summer_2026",
        label="Summer 2026",
        survey_csv=Path(
            "local/private_input_data/summer_2026_survey_data/"
            "Summer 2026 FutureEval Bot Maker Survey (Responses) - Form Responses 1.csv"
        ),
        tournament_url="https://www.metaculus.com/tournament/summer-futureeval-2026/",
        bot_tournament_project_id=33022,
        survey_post_url=None,
        survey_report_markdown=Path("local/summer_survey_analysis/summer_survey_analysis.md"),
    ),
}

FUTUREEVAL_RESOURCES_OPEN_SOURCE_URL = "https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots"
ANALYSIS_CODE_URL = "https://github.com/Metaculus/aib-analysis"


def get_season_config(season_key: str) -> AdviceSeasonConfig:
    if season_key not in SEASONS:
        raise KeyError(f"Unknown season {season_key!r}. Known seasons: {sorted(SEASONS)}")
    return SEASONS[season_key]
