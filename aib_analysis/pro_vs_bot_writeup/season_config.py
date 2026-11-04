"""Facts about each season that the simulation outputs cannot provide.

Everything else in the write-up is computed from the simulation JSONs. When adding a
season, add it to SEASON_ORDER and SEASONS and re-check every value. A None value
renders as a [===Fill me===] block in the write-up. After a season's write-up is
published, add its results to the PUBLISHED_* tuples so later seasons can chart them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


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
class BotQuestionWindow:
    """How long bot questions stayed open from first_open_date (None: the season start) until the next window."""

    first_open_date: date | None
    open_hours: float | None


@dataclass(frozen=True)
class SeasonConfig:
    key: str
    label: str
    prize_pool: str | None
    bot_tournament_url: str | None
    bot_question_interval_hours: float | None
    bot_question_windows: tuple[BotQuestionWindow, ...]
    pro_forecast_window_days: int | None
    metac_bot_code_url: str | None
    example_question_ids: dict[str, int] = field(default_factory=dict)
    expected_non_pro_members_in_pro_team: int = 0
    main_bot_team_size: int = 10
    data_notes: tuple[str, ...] = ()
    human_accounts_in_bot_teams_allowed: bool = False
    has_bot_team_vs_main_simulations: bool = True
    bot_question_launch_text: str | None = None
    window_change_reason: str | None = None
    close_time_difference_reason: str | None = None
    metac_bots_pinned_openrouter_hosts: bool = True

    def __post_init__(self) -> None:
        if not self.bot_question_windows or self.bot_question_windows[0].first_open_date is not None:
            raise ValueError(f"{self.label}: the first bot question window must start with the season (first_open_date=None)")
        later_dates = [window.first_open_date for window in self.bot_question_windows[1:]]
        if None in later_dates or later_dates != sorted(later_dates):
            raise ValueError(f"{self.label}: later bot question windows need increasing first_open_date values")


SEASON_ORDER: tuple[str, ...] = (
    "Q3 2024",
    "Q4 2024",
    "Q1 2025",
    "Q2 2025",
    "Fall 2025",
    "Spring 2026",
    "Summer 2026",
)

# Fall 2025 had no Pro comparison.
SEASONS_WITHOUT_PRO_COMPARISON: tuple[str, ...] = ("Fall 2025",)

PUBLISHED_PRO_VS_BOT_RESULTS: tuple[PublishedTeamResult, ...] = (
    PublishedTeamResult("Q3 2024", -11.3, -21.8, -0.7, None),
    PublishedTeamResult("Q4 2024", -8.9, -18.8, 1.0, None),
    PublishedTeamResult("Q1 2025", -17.7, -28.3, -7.0, None),
    PublishedTeamResult("Q2 2025", -20.03, -28.63, -11.41, 93),
    PublishedTeamResult("Spring 2026", -1.25, -4.87, 2.37, 99),
)

# Seasons with no clear winner whose 95% CI only just crossed zero; the trend paragraph says so.
NEAR_MISS_SEASONS: tuple[str, ...] = ("Q4 2024",)

# How many of the top 10 spots Pros took on the leaderboard ranking bots and Pros individually.
PUBLISHED_PROS_IN_INDIVIDUAL_TOP_10: dict[str, int] = {
    "Spring 2026": 9,
}

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
        prize_pool="$50,000",
        bot_tournament_url="https://www.metaculus.com/tournament/spring-aib-2026/",
        bot_question_interval_hours=2,
        bot_question_windows=(BotQuestionWindow(first_open_date=None, open_hours=1.5),),
        pro_forecast_window_days=3,
        metac_bot_code_url=None,
        # Bot-type accounts that forecast in the Spring Pro tournament were included in
        # the Pro Team aggregate. Kept as-is so this season reproduces the published post.
        expected_non_pro_members_in_pro_team=20,
        # Spring's simulations predate removing human accounts after the bot leaderboard, so
        # its largest team-size teams include a few. Kept so this season reproduces the published post.
        human_accounts_in_bot_teams_allowed=True,
        # Spring's simulations predate the bot team vs top-10 bot team files, so its results still compute
        # but its team-size section can't be rendered.
        has_bot_team_vs_main_simulations=False,
        metac_bots_pinned_openrouter_hosts=False,
    ),
    "summer_2026": SeasonConfig(
        key="summer_2026",
        label="Summer 2026",
        prize_pool="$50,000",
        bot_tournament_url="https://www.metaculus.com/tournament/summer-futureeval-2026/",
        bot_question_interval_hours=None,
        bot_question_windows=(
            BotQuestionWindow(first_open_date=None, open_hours=1.5),
            BotQuestionWindow(first_open_date=date(2026, 6, 5), open_hours=3),
        ),
        pro_forecast_window_days=3,
        metac_bot_code_url=(
            "https://github.com/Metaculus/forecasting-tools/blob/7c288e921b66d040c07c84a211428378d7a58111/"
            "forecasting_tools/forecast_bots/official_bots/template_bot_2026_summer.py"
        ),
        data_notes=(
            "metac-autocast was named metac-agent until October 15, 2026, so posts and pages earlier than this may "
            "still use the \"metac-agent\" name. In Fall 2026, metac-agent switched from a simple agentic harness to "
            "the harness used in autocast.",
            "Some questions were mistakenly launched as numeric and then converted to discrete after launch.",
            "Bots (including Metac Bots) missed some questions because of provider outages, GitHub Actions issues, or "
            "formatting errors. Many bot makers have mentioned that some of their biggest point losses were from bugs "
            "in code rather than their bot making a bad forecast.",
        ),
        bot_question_launch_text="at random times spread throughout the week",
        window_change_reason="Due to reliability issues in GitHub Actions, where most bot makers run their bots",
        close_time_difference_reason="because we spread out questions that bots had to forecast in parallel",
        # forecasting-tools started pinning OpenRouter hosts on Sept 25, 2026 (PR #351), after this season.
        metac_bots_pinned_openrouter_hosts=False,
    ),
}

FUTUREEVAL_URL = "https://www.metaculus.com/futureeval/"
SCORING_FAQ_URL = "https://www.metaculus.com/help/scores-faq/"
RESOURCES_URL = "https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/"
RESEARCH_SECTION_URL = f"{RESOURCES_URL}#research-reports-and-overview-of-the-field"
OPEN_SOURCE_BOTS_URL = f"{RESOURCES_URL}#open-source-bots"
DATA_ACCESS_URL = f"{RESOURCES_URL}#what-data-do-i-have-access-to-via-api-how-can-i-get-access-to-more"
CUSTOM_EVALS_URL = "https://www.metaculus.com/futureeval/partner/"
MULTIPLE_COMPARISONS_URL = "https://en.wikipedia.org/wiki/Multiple_comparisons_problem"
SERIES_METHODOLOGY_URL = (
    "https://www.metaculus.com/notebooks/28552/comparing-forecasting-track-records-for-ai-benchmarking-and-beyond/"
)
ANALYSIS_CODE_URL = "https://github.com/Metaculus/aib-analysis"
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
