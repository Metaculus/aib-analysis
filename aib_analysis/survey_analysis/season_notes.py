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

# Specific final-prediction models with their own feature: the most common
# OpenAI final model among respondents, and the Claude Opus release closest to
# it in time as a like-for-like comparison (Spring 2026: GPT-5.4 and Claude
# Opus 4.6). Both must be MODEL_REGISTRY display names.
SPOTLIGHT_OPENAI_MODEL = "GPT-5.6 Sol"
SPOTLIGHT_OPUS_MODEL = "Claude Opus 5"

# Definition changes made after the responses were downloaded, stated in the
# report for transparency. Empty when nothing changed after the download.
PRECOMMIT_LATE_DEFINITION_NOTE = (
    "One definition was settled after the responses were downloaded but before anyone looked at a "
    "hypothesis result: the research-source count also counts sources named in write-ins (see "
    "Methodology)."
)
FRONTIER_LATE_DEFINITION_CAVEAT = (
    "The Summer release-date cutoff and the rule that a flagship line still counts below a pricier "
    "tier were set after an earlier draft of these results had been seen. The frontier results are "
    "exploratory either way."
)
# Exploratory features added after this season's results had been seen. Empty
# when none.
POST_HOC_FEATURE_CAVEAT = (
    "The \"Uses 2 to 5 research sources\" feature was added after the Summer results had been "
    "seen, to follow up a pattern in them, so its p-value and q-value overstate the evidence."
)

# Human-authored headline findings shown at the top of the report. Filled in
# once per season after reading the generated charts; the report renders each
# entry as a bullet verbatim.
MAIN_TAKEAWAYS: list[str] = [
    "[===Fill me: takeaway 1===]",
    "[===Fill me: takeaway 2===]",
    "[===Fill me: takeaway 3===]",
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
