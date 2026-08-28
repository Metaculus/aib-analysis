"""Season-specific facts that cannot be derived from the input data.

The rest of the pipeline must not hardcode claims about the data: anything
that would go stale when the inputs change is either computed at run time or
lives here, where it is reviewed once per season. When re-running for a new
season, every constant in this file must be re-checked.
"""

SEASON_LABEL = "Spring 2026"
TOURNAMENT_NAME = "Spring 2026 FutureEval Bot Tournament"
TOURNAMENT_URL = "https://www.metaculus.com/tournament/spring-aib-2026/"

# Human-authored headline findings shown at the top of the report. Filled in
# once per season after reading the generated charts; the report renders each
# entry as a bullet verbatim.
MAIN_TAKEAWAYS: list[str] = [
    "<add takeaway>",
    "<add takeaway>",
    "<add takeaway>",
]

# Survey bot names that refer to a tournament entry under a slightly different
# name, where the automatic join cannot safely bridge the gap on its own (e.g.
# a name too short to match without risking a wrong join). Key is the exact
# survey bot_name; value is the tournament/prize-sheet name it denotes. Verified
# by hand against the prize sheet and re-checked per season.
BOT_NAME_ALIASES: dict[str, str] = {
    "MWG Bot": "MWG",  # prize sheet lists this entry as "MWG" (owner MWGHuman)
}

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
    "can differ from the official prize leaderboard, which excludes prize-ineligible bots. For "
    "example Preseen-Chestnut ranks 5th here but is excluded on the official board, so nostreambot "
    "shows as 11th here versus 10th there. The top-10 group uses these raw ranks by design."
)
