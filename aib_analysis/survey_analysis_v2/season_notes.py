"""Season-specific facts that cannot be derived from the input data.

The rest of the pipeline must not hardcode claims about the data: anything
that would go stale when the inputs change is either computed at run time or
lives here, where it is reviewed once per season. When re-running for a new
season, every constant in this file must be re-checked.
"""

SEASON_LABEL = "Spring 2026"

# Why some survey respondents have no scored-leaderboard match. Verified
# against the join audit in parsing_review.md.
EXCLUDED_RESPONDENTS_DETAIL = (
    "MiniBench-only participants, plus one bot with no matching tournament record"
)

# The top-10 group uses the raw sum-of-spot-peer-score ranking, which differs
# from the official prize board because that board excludes prize-ineligible
# bots. {leaderboard_size} is filled in at run time.
TOP10_ELIGIBILITY_CAVEAT = (
    "Top 10 is the raw ranking. It uses total spot peer score over all {leaderboard_size} bots, "
    "so it can include bots the official prize board marks ineligible: Preseen-Chestnut ranks 5th "
    "here but is excluded there, which puts nostreambot at 11th (just outside) where the official "
    "board shows it 10th. This shifts one bot in or out of the top-10 group."
)

# Same fact, phrased for the join audit in the parsing review doc.
RANK_VS_OFFICIAL_NOTE = (
    "Note: rank is the raw sum-of-spot-peer-score position over all {leaderboard_size} bots. This "
    "can differ from the official prize leaderboard, which excludes prize-ineligible bots. For "
    "example Preseen-Chestnut ranks 5th here but is excluded on the official board, so nostreambot "
    "shows as 11th here versus 10th there. The top-10 group uses these raw ranks by design."
)
