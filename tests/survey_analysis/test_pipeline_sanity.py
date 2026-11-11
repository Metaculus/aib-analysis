"""Whole-pipeline sanity checks against the real local survey data.

Unlike the pure unit tests, these load the actual season's survey + cached
leaderboard and assert invariants on the joined result: no two respondents share
a leaderboard row, every score is finite and matches sum/questions, the
correlation pool respects the question floor, and every final-model token that
names a model family and version is classified. They SKIP cleanly when
the private data or leaderboard cache is absent, so they never break a data-less
CI run. No network and no 327MB JSON (uses the cached CSV only).
"""

from __future__ import annotations

import math
import os
import re

import pytest

from aib_analysis.survey_analysis import config

_DATA_PRESENT = os.path.exists(config.SURVEY_CSV) and os.path.exists(
    config.LEADERBOARD_CACHE_CSV
)

pytestmark = pytest.mark.skipif(
    not _DATA_PRESENT,
    reason="local survey CSV / leaderboard cache not present; pipeline sanity checks skipped",
)


@pytest.fixture(scope="module")
def features():
    import logging

    logging.disable(logging.CRITICAL)
    from aib_analysis.survey_analysis import features as fmod, loading

    respondents = loading.build_respondents(refresh=False)
    return fmod.build_features(respondents)


@pytest.fixture(scope="module")
def in_scope(features):
    return [f for f in features if f.respondent.matched_leaderboard_name]


def test_respondents_loaded(features):
    assert len(features) > 20  # sanity: the survey actually loaded


def test_no_two_respondents_share_a_leaderboard_row(in_scope):
    names = [f.respondent.matched_leaderboard_name for f in in_scope]
    assert len(names) == len(set(names)), "a leaderboard row joined to >1 respondent"


def test_all_in_scope_scores_are_finite(in_scope):
    for f in in_scope:
        assert f.score is not None and math.isfinite(f.score), f.bot_name


def test_score_equals_sum_over_questions(in_scope):
    for f in in_scope:
        r = f.respondent
        assert f.score == pytest.approx(r.sum_spot_peer / r.question_count)


def test_correlation_pool_respects_question_floor(in_scope):
    pool = [f for f in in_scope if f.respondent.meets_correlation_minimum]
    assert len(pool) >= 8  # enough for the MIN_N correlation gate
    for f in pool:
        assert f.respondent.question_count >= config.MIN_QUESTIONS_FOR_CORRELATION


def test_group_sizes_are_sane(in_scope):
    winners = sum(1 for f in in_scope if f.respondent.is_winner)
    top10 = sum(1 for f in in_scope if f.respondent.is_top_10)
    assert 0 < winners <= len(in_scope)
    assert 0 <= top10 <= config.TOP_N_FOR_TOP_GROUP


_MODEL_LIKE_TOKEN = re.compile(
    r"\b(gpt|claude|opus|sonnet|haiku|fable|gemini|gemma|grok|llama|qwen|deepseek|kimi|glm|minimax|mimo|mistral)\b[\s-]*v?\d",
    re.IGNORECASE,
)


def test_every_model_like_final_token_is_classified(features):
    # Write-ins are never corrected by hand, so non-model text stays unmatched, but a
    # token naming a model family and version must match the registry. A hit here
    # means the registry drifted behind what bot makers used.
    unmatched = sorted({t for f in features for t in f.final_unmatched if _MODEL_LIKE_TOKEN.search(t)})
    assert unmatched == [], f"unmatched model-like final-model tokens: {unmatched}"


def test_frontier_flag_only_true_with_a_frontier_model(features):
    from aib_analysis.survey_analysis import parsing

    for f in features:
        if f.frontier:
            models, _i, _u = parsing.classify_models(f.cells["final_model"].raw)
            assert any(m.is_frontier for m in models), f.bot_name


def test_prize_matches_are_never_wrongly_loose(in_scope):
    # Every prize match should be exact or tokenized (>=4-char guard), never a
    # blank "none" that still set winner status.
    for f in in_scope:
        if f.respondent.is_winner:
            assert f.respondent.prize_match_kind in {"exact", "tokenized"}
