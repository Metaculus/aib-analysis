"""Season-specific facts that cannot be derived from the input data.

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

# Definition changes made after the responses were downloaded, stated in the
# report for transparency. Empty when nothing changed after the download.
PRECOMMIT_LATE_DEFINITION_NOTE = (
    "Two definitions were settled after the responses were downloaded but before anyone looked at a "
    "hypothesis result: the research-source count also counts sources named in write-ins (see "
    "below), and checking similar questions or markets counts only the strategy checkbox. The manual "
    "answer corrections were also reviewed after the download."
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
