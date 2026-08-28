"""Shared helpers for survey_analysis tests.

These build lightweight stand-ins for RespondentFeatures so the statistics and
report helpers can be exercised without loading the 327MB tournament JSON, the
private survey CSVs, or any environment variables.
"""

from __future__ import annotations

from types import SimpleNamespace


def make_feature(
    variables: dict[str, float | None] | None = None,
    score: float | None = None,
    booleans: dict[str, bool] | None = None,
    is_winner: bool = False,
    groups: list[str] | None = None,
    cells: dict[str, object] | None = None,
    frontier: bool = False,
) -> SimpleNamespace:
    """A duck-typed RespondentFeatures good enough for stats/report helpers.

    stats.py reads .variables, .score, .booleans, .respondent.is_winner and
    .respondent.groups; report.py additionally reads .cells and .frontier.
    """
    respondent = SimpleNamespace(
        is_winner=is_winner,
        groups=groups if groups is not None else (["winner"] if is_winner else ["non_winner"]),
    )
    return SimpleNamespace(
        variables=variables or {},
        score=score,
        booleans=booleans or {},
        respondent=respondent,
        cells=cells or {},
        frontier=frontier,
    )


def make_cell(matched: list[str] | None = None, other: list[str] | None = None) -> SimpleNamespace:
    """Stand-in for features.ParsedCell (report helpers read .matched/.other)."""
    return SimpleNamespace(matched=matched or [], other=other or [], raw="", numeric=None)
