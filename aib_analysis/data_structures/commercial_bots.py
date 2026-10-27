"""Bot accounts run by companies that build forecasting products. Every other bot not run by Metaculus counts as a hobbyist bot."""

COMMERCIAL_BOT_NAMES: frozenset[str] = frozenset(
    {
        "Preseen-Atlas",
        "Preseen-Chestnut",
        "preseen",  # Summer 2026 account of the same company as the two Preseen bots above
        "manticAI",
        "cassi",
        "futuresearch",
        "lightningrod",
        "Upskillbot",
    }
)
