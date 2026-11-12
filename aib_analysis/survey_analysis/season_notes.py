"""Season-specific facts and judgment calls that cannot be derived from the input data.

The rest of the pipeline must not hardcode claims about the data: anything
that would go stale when the inputs change is either computed at run time or
lives here, where it is reviewed once per season. When re-running for a new
season, every constant in this file must be re-checked.
"""

from datetime import date

SEASON_LABEL = "Summer 2026"
SEASON_LENGTH_LABEL = "4-month"
TOURNAMENT_NAME = "Summer 2026 FutureEval Bot Tournament"
TOURNAMENT_URL = "https://www.metaculus.com/tournament/summer-futureeval-2026/"

# Day the season's first bot question opened (Spring 2026: 2026-01-05). The
# frontier release-date cutoff in config.py is derived from it, and refreshing
# the leaderboard checks it against the tournament's first forecast.
SEASON_START_DATE = date(2026, 5, 18)

# The season start every item below was last reviewed for. The run stops while it
# differs from SEASON_START_DATE, so a new season cannot run until each item has
# been checked and this date updated.
REVIEWED_FOR_SEASON_START = date(2026, 5, 18)
SEASON_REVIEW_CHECKLIST: list[str] = [
    "config.py input paths: survey CSV, prize sheet, blank form PDF and bot tournament JSON for the new season.",
    "This file: label, season length, tournament name and URL, start date, RANK_VS_OFFICIAL_NOTE, "
    "EXCLUDED_MINIBENCH_LABEL and POST_HOC_FEATURE_KEYS.",
    "precommitted.py: hypotheses chosen and committed before the responses are downloaded, then a --blind run.",
    "config.py option lists: the run checks them against the form PDF; midpoints for new buckets and the "
    "BOOLEAN_FEATURES substrings are choices to review.",
    "config.MODEL_REGISTRY: release date (checked against OpenRouter's model list) and tier for every new "
    "model (the form check forces an entry for each), and FRONTIER_EXCEPTIONS below, reviewed together "
    "each season.",
    "config.RESEARCH_WRITE_IN_*: the run stops on any research write-in no rule covers; decide for each new "
    "one whether it names a form option, counts as an extra source, or is not counted.",
    "BOT_NAME_ALIASES below: survey names the matcher cannot join (see the join warnings in the log).",
    "PRECOMMIT_LATE_DEFINITION_NOTE and LATE_DEFINITION_CAVEATS below: replace with this season's changes.",
    "MAIN_TAKEAWAYS below: write after reading the results.",
    "Comments and docstrings in aib_analysis/survey_analysis that name the previous season or its dates.",
]

# Models counted as frontier even though they were released before the cutoff,
# because they were still their vendor's newest flagship for most of the season.
# Model -> reason, rendered in the report as "except for <model>, which was <reason>".
# Gemini 3.5 Pro was announced but not released during Summer 2026.
FRONTIER_EXCEPTIONS: dict[str, str] = {
    "Gemini 3.1 Pro": "Google's newest flagship for most of the season",
}

# Survey bot name -> tournament bot name, for names the automatic matcher cannot
# safely join (the leaderboard and prize sheet use the tournament name).
BOT_NAME_ALIASES: dict[str, str] = {
    "FJMForecastBot2026": "FJMForecastBot2026-bot",
    "neme": "Chineme",
}

# Definition changes made after the responses were downloaded, stated in the
# report for transparency. Empty when nothing changed after the download.
PRECOMMIT_LATE_DEFINITION_NOTE = (
    "Two definitions were settled after the responses were downloaded but before anyone looked at a "
    "hypothesis result: the research-source count also counts sources named in write-ins (see "
    "below), and checking similar questions or markets counts only the strategy checkbox. After the "
    "results had been computed, and while reviewing code, LLM-generated write-in categorization was "
    "dropped in favor of regular expression keyword categorization. Write-ins that decline to answer "
    "(such as \"some of the above\") were treated as blank. Across these versions, the precommitted "
    "research-source correlation ranged from +0.10 to +0.16 and was never significant."
)
# Features defined only after the results had been seen. The evidence summary and
# the feature's chart note point readers to their caveat in LATE_DEFINITION_CAVEATS.
POST_HOC_FEATURE_KEYS: frozenset[str] = frozenset({"moderate_research_sources"})
POST_HOC_FEATURE_NOTE = (
    "However, this feature was analyzed because (conditional on) finding the original results strange "
    "and thus may prove misleading (see caveats)."
)
# Caveat bullets for definitions changed or features added after the download,
# rendered verbatim in Appendix B.
LATE_DEFINITION_CAVEATS: list[str] = [
    "Questions missed definition: This was redefined after the responses were downloaded to count only questions "
    "still open after a bot's first forecast, so a bot that started late is not penalized for questions "
    "it never saw. Counting every season question instead, its link to score is weaker and does not "
    "pass the correction.",
    "LLM calls per question: Bots whose makers answered \"Highly variable across questions\" or wrote in "
    "an answer have no number, so they are left out of this correlation, and they include some of the "
    "best-scoring bots.",
    "Features added after seeing results: The \"Uses 2 to 5 research sources\" feature was added after "
    "the Summer results had been seen, since it seemed that beyond 5 sources people might be adding "
    "lower-quality or poorly configured data providers. Thus its p-value and q-value may be inflated or "
    "otherwise misleading.",
]

# Human-authored headline findings shown at the top of the report. Filled in
# once per season after reading the generated charts; the report renders each
# entry as a bullet verbatim.
MAIN_TAKEAWAYS: list[str] = [
    "**Better compute correlates strongly with better forecasting performance:** After adjusting for the "
    "number of tests, 8 of the 31 tested features proved to be significant this season. Many of these "
    "related to compute. Using frontier models in supporting roles had the highest correlation (r=0.54), "
    "along with using a frontier model for the final prediction (r=0.37). Bots with a higher cost per "
    "question (r=0.51) also did better.",
    "**Web scraping proved significant:** Bots that scraped web pages during research scored higher, with "
    "a statistically significant correlation of 0.43 (0.33 in Spring). It remains a reasonable feature to "
    "add to your bot.",
    "**Engineering effort matters:** Bots whose makers put more total hours into development scored higher "
    "(r=0.33, similar to r=0.32 in Spring), though this did not prove to be significant (q=0.09). "
    "Additionally, bot makers have previously mentioned that bugs in their code are the biggest cause of "
    "point losses. Bots that missed more questions from the tournament (after adjusting for late entry) "
    "scored lower with statistical significance (r=-0.40). These measures could be confounded with the "
    "amount of time spent on prompt iteration or other similar causes.",
    "**Our precommitted hypotheses were not confirmed:** We precommitted to three hypotheses we wanted to "
    "better test from previous seasons. We measured how performance changed when bots checked similar "
    "questions on Metaculus or prediction markets, when bots used more research sources, and when bots "
    "extremized predictions. All three pointed in the precommitted direction (correlations of 0.27, 0.16, "
    "and -0.26), but none ended up being significant despite the precommitment. With about 45 bots, a "
    "correlation needs to be roughly 0.3 to pass, so these remain open questions.",
    "**Other recommendations**: We recommend a few practices that didn't end up being significant, but we "
    "predict would prove to be significant with a larger sample size. These include using the native "
    "search provider of the model you are using, aggregating multiple forecasts (ideally with variation "
    "between them), using 2 to 5 data/search sources, and cross-referencing questions on forecasting "
    "platforms (e.g. Metaculus) or prediction markets (e.g. Polymarket).",
]

# How the excluded (no scored-leaderboard row) respondents are characterized.
# The count of any respondents with no tournament record at all is computed at
# run time, so this only needs the shared label. Verified against the join
# audit in parsing_review.md.
EXCLUDED_MINIBENCH_LABEL = "MiniBench-only participants"

# The top-10 group uses the raw sum-of-spot-peer-score ranking, which differs
# from the official prize board because that board excludes prize-ineligible
# bots. {leaderboard_size} is filled in at run time. Phrased for the join audit
# in the parsing review doc.
RANK_VS_OFFICIAL_NOTE = (
    "Note: rank is the raw sum-of-spot-peer-score position over all {leaderboard_size} bots. This "
    "can differ from the official prize leaderboard, which excludes prize-ineligible bots. In Summer "
    "2026 the top 15 match the official board exactly; below that they drift apart. The top-10 group "
    "uses these raw ranks by design."
)
