"""Pure parsing functions for the Spring 2026 survey.

Every function returns both the normalized value AND whatever it could not map,
so the review doc can show exactly what was matched and what was set aside. No
function silently drops content.
"""

from __future__ import annotations

import re

from aib_analysis.survey_analysis_v2.config import (
    MODEL_REGISTRY,
    MODEL_TOKENS_IGNORED,
    ModelInfo,
    _norm_model,
)

_ALNUM = re.compile(r"[a-z0-9]", re.IGNORECASE)


def _has_alnum(text: str) -> bool:
    return bool(_ALNUM.search(text))


def parse_multiselect(
    raw: str, options: list[str]
) -> tuple[list[str], list[str]]:
    """Match known options as substrings (longest first) and return leftovers.

    Returns (matched_options_in_vocab_order, other_write_in_tokens). Embedded
    commas inside option text never break parsing because full option strings
    are removed before the remainder is split on commas.
    """
    text = raw.strip()
    if not text:
        return [], []

    working = text
    matched: set[str] = set()
    for option in sorted(options, key=len, reverse=True):
        idx = working.lower().find(option.lower())
        if idx != -1:
            matched.add(option)
            working = working[:idx] + " " + working[idx + len(option) :]

    others = [
        token.strip(" ,;.-")
        for token in re.split(r"[,;]", working)
    ]
    others = [token for token in others if token and _has_alnum(token)]

    matched_ordered = [option for option in options if option in matched]
    return matched_ordered, others


def parse_single_select(
    raw: str, options: list[str]
) -> tuple[str | None, str | None]:
    """Exact (case-insensitive) match against canonical options.

    Returns (canonical_option, None) on a match, else (None, raw_value).
    """
    text = raw.strip()
    if not text:
        return None, None
    lowered = text.lower()
    for option in options:
        if lowered == option.lower():
            return option, None
    return None, text


def bucket_to_midpoint(value: str | None, midpoint_map: dict[str, float]) -> float | None:
    if value is None:
        return None
    return midpoint_map.get(value)


def parse_team_size(raw: str) -> tuple[int | None, str | None]:
    """Team size as an int. 'Not in a team' -> 1 (solo). Other free text -> flagged."""
    text = raw.strip()
    if not text:
        return None, None
    if text.lower() == "not in a team":
        return 1, None
    if text.isdigit():
        return int(text), None
    match = re.match(r"^\d+", text)
    if match:
        return int(match.group()), text
    return None, text


def feature_present(raw: str, match_substring: str) -> bool:
    return match_substring.lower() in raw.lower()


def count_research_sources(raw: str, research_options: list[str]) -> int:
    matched, _others = parse_multiselect(raw, research_options)
    return len(matched)


# --------------------------------------------------------------------------- #
# Model parsing / frontier classification
# --------------------------------------------------------------------------- #
def tokenize_models(raw: str) -> list[str]:
    """Split a model cell into individual model tokens."""
    text = raw.strip()
    if not text:
        return []
    tokens = [token.strip() for token in re.split(r"[,;]", text)]
    return [token for token in tokens if token]


def match_model_token(token: str) -> list[ModelInfo]:
    """Return the maximal registry matches contained in a single token.

    A token like 'GPT-5.4 mini' matches only 'GPT-5.4 mini' (not the shorter
    'GPT-5.4'), and a token joining two models with 'and' can match both.
    """
    normalized = _norm_model(token)
    if not normalized:
        return []
    contained = [
        model for model in MODEL_REGISTRY if model.normalized_key in normalized
    ]
    maximal: list[ModelInfo] = []
    for model in contained:
        if any(
            other is not model and model.normalized_key in other.normalized_key
            for other in contained
        ):
            continue
        maximal.append(model)
    return maximal


def classify_models(
    raw: str,
) -> tuple[list[ModelInfo], list[str], list[str]]:
    """Classify every model token in a cell.

    Returns (matched_models, ignored_tokens, unmatched_tokens). Ignored tokens
    are known parsing artifacts / vague labels; unmatched tokens are anything
    unrecognized and are flagged for review.
    """
    matched: list[ModelInfo] = []
    ignored: list[str] = []
    unmatched: list[str] = []
    for token in tokenize_models(raw):
        hits = match_model_token(token)
        if hits:
            matched.extend(hits)
            continue
        if _norm_model(token) in MODEL_TOKENS_IGNORED:
            ignored.append(token)
        else:
            unmatched.append(token)
    return matched, ignored, unmatched


def is_frontier_final(raw: str) -> bool:
    """A final-model cell is frontier if any recognized model in it is frontier."""
    matched, _ignored, _unmatched = classify_models(raw)
    return any(model.is_frontier for model in matched)
