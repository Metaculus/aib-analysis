"""Metaculus tournament leaderboard rules (take, ranks with ties, ranking exclusions) applied to score totals.

Prizes, medals, confidence intervals, and Metaculus's exclusion records are not modeled: they
depend on leaderboard settings that only exist on the site.
"""

import numpy as np
from pydantic import BaseModel


class OfficialLeaderboardEntry(BaseModel):
    user_name: str
    score: float
    take: float
    is_excluded_from_ranking: bool
    rank: int


def build_official_leaderboard(
    score_by_user: dict[str, float], excluded_user_names: set[str]
) -> list[OfficialLeaderboardEntry]:
    """Excluded users still get a rank (shared with whoever is next) but do not use one up.
    Scores within np.isclose of the previous entry share its rank."""
    unknown_exclusions = excluded_user_names - set(score_by_user)
    if unknown_exclusions:
        raise ValueError(
            f"Excluded users have no score: {sorted(unknown_exclusions)}"
        )

    entries: list[OfficialLeaderboardEntry] = []
    next_rank = 1
    previous_entry: OfficialLeaderboardEntry | None = None
    for user_name, score in sorted(
        score_by_user.items(), key=lambda item: item[1], reverse=True
    ):
        rank = next_rank
        if previous_entry is not None and np.isclose(score, previous_entry.score):
            rank = previous_entry.rank
        is_excluded = user_name in excluded_user_names
        entry = OfficialLeaderboardEntry(
            user_name=user_name,
            score=score,
            take=max(score, 0) ** 2,
            is_excluded_from_ranking=is_excluded,
            rank=rank,
        )
        entries.append(entry)
        previous_entry = entry
        if not is_excluded:
            next_rank += 1
    return entries
