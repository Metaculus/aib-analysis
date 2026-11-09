"""The precommitted thematic-analysis method: models, thresholds, and selection rules.

The method is codebook thematic analysis (Guest, MacQueen & Namey, Applied Thematic Analysis, 2012):
two models propose themes, a third merges them into a codebook with inclusion and exclusion rules
(MacQueen et al. 1998), two coders apply it to every answer independently, a fourth model breaks ties
blind, and a fifth checks every citation skeptically. Reflexive thematic analysis (Braun & Clarke) is
not a fit: it rejects codebooks and agreement statistics. Coder agreement is Krippendorff's alpha
(Krippendorff 2011); 0.667 is the lowest value he accepts for tentative conclusions, and the 0.80 he
recommends for firm ones would drop a 5-supporter theme over two disagreements at this sample size.
Counts are computed in code, every quote must match the answer word for word, and every published
citation is checked, because fabricated and misattributed quotes are the best-documented failure of
LLM-assisted coding.

Change these values only before a season's answers are analyzed; a change made after seeing results
must be disclosed in the write-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoster:
    theme_generators: tuple[str, str]
    codebook: str
    coders: tuple[str, str]
    tie_breaker: str
    verifier: str
    one_offs: str
    data_check: str
    data_check_confirmer: str


PRECOMMITTED_MODELS = ModelRoster(
    theme_generators=("openrouter/anthropic/claude-opus-5.5", "openrouter/openai/gpt-6.1-sol"),
    codebook="openrouter/google/gemini-4-argon",
    coders=("openrouter/anthropic/claude-opus-5.5", "openrouter/openai/gpt-6.1-sol"),
    tie_breaker="openrouter/x-ai/grok-4.7",
    verifier="openrouter/google/gemini-4-argon",
    one_offs="openrouter/anthropic/claude-opus-5.5",
    data_check="openrouter/google/gemini-4-argon",
    data_check_confirmer="openrouter/anthropic/claude-opus-5.5",
)
SMOKE_TEST_MODELS = ModelRoster(
    theme_generators=("openrouter/anthropic/claude-haiku-5.5", "openrouter/openai/gpt-6-luna"),
    codebook="openrouter/google/gemini-3.8-flash",
    coders=("openrouter/anthropic/claude-haiku-5.5", "openrouter/openai/gpt-6-luna"),
    tie_breaker="openrouter/deepseek/deepseek-v4.1-flash",
    verifier="openrouter/google/gemini-3.8-flash",
    one_offs="openrouter/anthropic/claude-haiku-5.5",
    data_check="openrouter/google/gemini-3.8-flash",
    data_check_confirmer="openrouter/anthropic/claude-haiku-5.5",
)

MIN_CODEBOOK_THEMES = 5
MAX_CODEBOOK_THEMES = 8
MIN_SUPPORTING_ANSWERS_PER_CANDIDATE = 3
MAX_PUBLISHED_THEMES = 5
MIN_SUPPORTER_SHARE = 0.10
MIN_SUPPORTERS_FLOOR = 3
MIN_CODER_ALPHA = 0.667
FIRM_CODER_ALPHA = 0.80
EXEMPLARS_PER_THEME = 3
MIN_EXCERPT_WORDS = 3
MAX_EXCERPT_WORDS = 30
MAX_ONE_OFFS = 5
MAX_DATA_CHECKS_PER_THEME = 2
MAX_FORMAT_RETRIES = 2
MAX_CONCURRENT_CALLS = 8
LLM_TIMEOUT_SECONDS = 300
DEFAULT_COST_CAP_USD = 10.0
SMOKE_TEST_ANSWER_COUNT = 15


def min_supporters(answer_count: int) -> int:
    return max(MIN_SUPPORTERS_FLOOR, math.ceil(MIN_SUPPORTER_SHARE * answer_count))


def model_label(model: str) -> str:
    """'openrouter/anthropic/claude-opus-5.5' -> 'claude-opus-5.5'."""
    return model.rsplit("/", 1)[-1]
