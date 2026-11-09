"""Checks that don't need a model: verbatim quote matching, coder agreement, and majority votes."""

from __future__ import annotations

from typing import Literal

Stance = Literal["supports", "opposes", "absent"]
STANCES: tuple[Stance, ...] = ("supports", "opposes", "absent")

CHARACTER_FOLDS = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)
EXCERPT_EDGE_CHARACTERS = " \t\n\"'`.…"


def _fold_with_positions(text: str) -> tuple[str, list[int]]:
    """Fold quotes, dashes, whitespace runs, and case, remembering each folded character's position in text."""
    folded_characters: list[str] = []
    positions: list[int] = []
    previous_was_space = False
    for position, character in enumerate(text.translate(CHARACTER_FOLDS)):
        if character.isspace():
            if previous_was_space:
                continue
            character = " "
        previous_was_space = character == " "
        for lowered in character.lower():
            folded_characters.append(lowered)
            positions.append(position)
    return "".join(folded_characters), positions


def locate_excerpt(excerpt: str, sources: tuple[str, ...]) -> str | None:
    """Return the source's own text for an excerpt that appears in one source, ignoring quote style, whitespace, and case."""
    folded_excerpt, _positions = _fold_with_positions(excerpt.strip(EXCERPT_EDGE_CHARACTERS))
    if not folded_excerpt.strip():
        return None
    for source in sources:
        folded_source, positions = _fold_with_positions(source)
        start = folded_source.find(folded_excerpt)
        if start != -1:
            return source[positions[start] : positions[start + len(folded_excerpt) - 1] + 1]
    return None


def word_count(text: str) -> int:
    return len(text.split())


def percent_agreement(first: list[bool], second: list[bool]) -> float:
    return sum(left == right for left, right in zip(first, second, strict=True)) / len(first)


def krippendorff_alpha(first: list[bool], second: list[bool]) -> float | None:
    """Nominal alpha for two coders and no missing values; None when neither coder ever varies."""
    value_count = 2 * len(first)
    positives = sum(first) + sum(second)
    negatives = value_count - positives
    if positives == 0 or negatives == 0:
        return None
    disagreements = sum(left != right for left, right in zip(first, second, strict=True))
    return 1 - (value_count - 1) * disagreements / (positives * negatives)


def gwet_ac1(first: list[bool], second: list[bool]) -> float:
    """Gwet's AC1 for two coders and two categories; unlike alpha it stays stable when a code is rare."""
    positive_share = (sum(first) + sum(second)) / (2 * len(first))
    chance_agreement = 2 * positive_share * (1 - positive_share)
    return (percent_agreement(first, second) - chance_agreement) / (1 - chance_agreement)


def majority_stance(first: Stance, second: Stance, tie_break: Stance | None) -> tuple[Stance, bool]:
    """The coders' shared stance, else the stance the tie-breaker shares with one coder; ('absent', False) without a majority."""
    if first == second:
        return first, True
    if tie_break is None:
        raise ValueError("A disagreement needs a tie-break vote")
    if tie_break in (first, second):
        return tie_break, True
    return "absent", False
