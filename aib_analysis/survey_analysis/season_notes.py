"""Season-specific facts and judgment calls that cannot be derived from the input data.

The rest of the pipeline must not hardcode claims about the data: anything
that would go stale when the inputs change is either computed at run time or
lives here, where it is reviewed once per season. When re-running for a new
season, every constant in this file must be re-checked.
"""

from datetime import date

SEASON_LABEL = "Summer 2026"
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
    "This file: label, tournament name and URL, start date, RANK_VS_OFFICIAL_NOTE and EXCLUDED_MINIBENCH_LABEL.",
    "precommitted.py: hypotheses chosen and committed before the responses are downloaded, then a --blind run.",
    "config.py option lists: the run checks them against the form PDF; midpoints for new buckets and the "
    "BOOLEAN_FEATURES substrings are choices to review.",
    "config.MODEL_REGISTRY: release date and flagship tier for every new model (the form check forces an "
    "entry for each), and FRONTIER_EXCEPTIONS below, reviewed together each season.",
    "config.RESEARCH_WRITE_IN_*: the run stops on any research write-in no rule covers; decide for each new "
    "one whether it names a form option, counts as an extra source, or is not counted.",
    "BOT_NAME_ALIASES below: survey names the matcher cannot join (see the join warnings in the log).",
    "PRECOMMIT_LATE_DEFINITION_NOTE and LATE_DEFINITION_CAVEATS below: replace with this season's changes.",
    "MAIN_TAKEAWAYS below: write after reading the results.",
    "Comments and docstrings in aib_analysis/survey_analysis that name the previous season or its dates.",
]

# Models counted as frontier even though they were released before the cutoff,
# because they were still their vendor's newest flagship for most of the season.
# Model -> reason.
FRONTIER_EXCEPTIONS: dict[str, str] = {
    "Gemini 3.1 Pro": "Google's newest Pro model all season; Gemini 3.5 Pro was announced but not released",
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
    "results had been seen, hand corrections of write-ins were dropped in favor of the keyword list "
    "alone, the list was extended, a write-in naming a listed option was counted as ticking it, and an "
    "answer made only of write-ins that decline to answer (such as \"some of the above\") was treated "
    "as blank. Across these versions the research-source correlation ranged from +0.10 to +0.16 and was "
    "never significant."
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
    "Rules revised after seeing results: After the Summer results had been seen, we revised the "
    "write-in rules (Appendix A) and counted Gemini 3.1 Pro as frontier, since it was Google's newest Pro "
    "model all season. With the revised write-in rules web scraping's correlation is +0.43; counting "
    "only ticked boxes it is +0.35.",
    "Features added after seeing results: The \"Uses 2 to 5 research sources\" feature was added after "
    "the Summer results had been seen, since it seemed that at past 5 sources people might be adding "
    "lower quality or poorly configured data providers. Thus its p-value and q-value may be inflated.",
]

# Human-authored headline findings shown at the top of the report. Filled in
# once per season after reading the generated charts; the report renders each
# entry as a bullet verbatim.
MAIN_TAKEAWAYS: list[str] = [
    "[===Fill me: takeaway 1===]",
    "[===Fill me: takeaway 2===]",
    "[===Fill me: takeaway 3===]",
    "[===Fill me: takeaway 4===]",
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
