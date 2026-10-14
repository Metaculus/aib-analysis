"""Facts about each season that the simulation outputs cannot provide.

Everything else in the write-up is computed from the simulation JSONs. When adding a
season, add it to SEASON_ORDER and SEASONS and re-check every value. A None value
renders as a [===Fill me===] block in the write-up. After a season's write-up is
published, add its results to the PUBLISHED_* tuples so later seasons can chart them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PublishedTeamResult:
    """Bot team head-to-head score against the Pro team, as published for a past season."""

    season_label: str
    mean: float
    ci_lower: float
    ci_upper: float
    question_count: int | None


@dataclass(frozen=True)
class PublishedBaselineBotRank:
    """Where the best same-prompt Metac Bot placed in a past season's bot tournament."""

    season_label: str
    rank: int
    bot_count: int


@dataclass(frozen=True)
class SeasonConfig:
    key: str
    label: str
    period_description: str | None
    series_ordinal: str | None
    prize_pool: str | None
    bot_tournament_url: str | None
    companion_posts: tuple[tuple[str, str], ...]
    bot_question_interval_hours: float | None
    bot_question_open_hours: float | None
    pro_forecast_window_days: int | None
    metac_bot_code_url: str | None
    example_question_ids: dict[str, int] = field(default_factory=dict)
    expected_non_pro_members_in_pro_team: int = 0
    main_bot_team_size: int = 10


SEASON_ORDER: tuple[str, ...] = (
    "Q3 2024",
    "Q4 2024",
    "Q1 2025",
    "Q2 2025",
    "Fall 2025",
    "Spring 2026",
    "Summer 2026",
)

# Fall 2025 compared bots with the Metaculus community prediction instead of Pros.
SEASONS_WITHOUT_PRO_COMPARISON: tuple[str, ...] = ("Fall 2025",)

PUBLISHED_PRO_VS_BOT_RESULTS: tuple[PublishedTeamResult, ...] = (
    PublishedTeamResult("Q3 2024", -11.3, -21.8, -0.7, None),
    PublishedTeamResult("Q4 2024", -8.9, -18.8, 1.0, None),
    PublishedTeamResult("Q1 2025", -17.7, -28.3, -7.0, None),
    PublishedTeamResult("Q2 2025", -20.03, -28.63, -11.41, 93),
    PublishedTeamResult("Spring 2026", -1.25, -4.87, 2.37, 99),
)

PUBLISHED_BASELINE_BOT_RANKS: tuple[PublishedBaselineBotRank, ...] = (
    PublishedBaselineBotRank("Q3 2024", 4, 55),
    PublishedBaselineBotRank("Q4 2024", 6, 44),
    PublishedBaselineBotRank("Q1 2025", 1, 45),
    PublishedBaselineBotRank("Q2 2025", 2, 96),
    PublishedBaselineBotRank("Fall 2025", 14, 134),
    PublishedBaselineBotRank("Spring 2026", 18, 173),
)

SEASONS: dict[str, SeasonConfig] = {
    "spring_2026": SeasonConfig(
        key="spring_2026",
        label="Spring 2026",
        period_description="the first third of 2026 (January 7th through April 15th)",
        series_ordinal="fifth",
        prize_pool="$50,000",
        bot_tournament_url="https://www.metaculus.com/tournament/spring-aib-2026/",
        companion_posts=(),
        bot_question_interval_hours=2,
        bot_question_open_hours=1.5,
        pro_forecast_window_days=3,
        metac_bot_code_url=None,
        # Bot-type accounts that forecast in the Spring Pro tournament were included in
        # the Pro Team aggregate. Kept as-is so this season reproduces the published post.
        expected_non_pro_members_in_pro_team=20,
    ),
    "summer_2026": SeasonConfig(
        key="summer_2026",
        label="Summer 2026",
        period_description=None,
        series_ordinal=None,
        prize_pool=None,
        bot_tournament_url=None,
        companion_posts=(),
        bot_question_interval_hours=None,
        bot_question_open_hours=None,
        pro_forecast_window_days=None,
        metac_bot_code_url=None,
    ),
}

FUTUREEVAL_URL = "https://www.metaculus.com/futureeval/"
SCORING_FAQ_URL = "https://www.metaculus.com/help/scores-faq/"
RESOURCES_URL = "https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/"
ANALYSIS_CODE_URL = "https://github.com/Metaculus/aib-analysis"
FORECASTING_TOOLS_URL = "https://github.com/Metaculus/forecasting-tools"
DATA_REQUEST_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLSeJhtZzHl5qMvBjbXbatyaqoS4IU7RE0GGw_vlhs6I9syqn1g/"
    "viewform?usp=pp_url&entry.192763438=https://www.metaculus.com/api/"
)
CONTACT_EMAIL = "support@metaculus.com"


def get_season_config(season_key: str) -> SeasonConfig:
    if season_key not in SEASONS:
        raise KeyError(f"Unknown season {season_key!r}. Known seasons: {sorted(SEASONS)}")
    config = SEASONS[season_key]
    if config.label not in SEASON_ORDER:
        raise ValueError(f"Season label {config.label!r} is missing from SEASON_ORDER")
    return config


def seasons_before(season_label: str) -> tuple[str, ...]:
    return SEASON_ORDER[: SEASON_ORDER.index(season_label)]
