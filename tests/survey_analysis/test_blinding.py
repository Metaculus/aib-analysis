"""Blinded report sections.

While the precommitted picks are under review, no result for a precommitted or
alternative hypothesis may reach the report, including through section order.
Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import precommitted, report
from aib_analysis.survey_analysis.config import QuestionSpec
from aib_analysis.survey_analysis.stats import CorrelationResult


HIDDEN_KEY = precommitted.PRECOMMITTED_HYPOTHESES[0].key
SHOWN_KEY = "uses_capping"


def _results() -> dict[str, CorrelationResult]:
    return {
        HIDDEN_KEY: CorrelationResult(HIDDEN_KEY, "Hidden trait", "Pearson", 0.9, 0.001, 40),
        SHOWN_KEY: CorrelationResult(SHOWN_KEY, "Shown trait", "Pearson", 0.1, 0.5, 40),
    }


def test_evidence_summary_lists_hidden_features_without_numbers(monkeypatch):
    monkeypatch.setattr(report, "_all_feature_keys", lambda: [(HIDDEN_KEY, "Hidden trait"), (SHOWN_KEY, "Shown trait")])
    summary = report._evidence_summary_section(_results(), {HIDDEN_KEY: 0.002, SHOWN_KEY: 0.5}, frozenset({HIDDEN_KEY}))
    assert "**Shown trait**: r = +0.10" in summary
    assert "Hidden trait" in summary
    assert "+0.90" not in summary


def test_evidence_summary_marks_only_precommitted_features(monkeypatch):
    monkeypatch.setattr(report, "_all_feature_keys", lambda: [(HIDDEN_KEY, "Hidden trait"), (SHOWN_KEY, "Shown trait")])
    summary = report._evidence_summary_section(_results(), {HIDDEN_KEY: 0.002, SHOWN_KEY: 0.5}, frozenset())
    assert "**Hidden trait**: r = +0.90 · p = 0.001 · q = 0.002 · significant: yes · precommitted: yes" in summary
    assert "**Shown trait**: r = +0.10 · p = 0.500 · q = 0.500 · significant: no\n" in summary + "\n"


def test_evidence_summary_points_post_hoc_features_to_caveats(monkeypatch):
    monkeypatch.setattr(report, "_all_feature_keys", lambda: [(HIDDEN_KEY, "Hidden trait"), (SHOWN_KEY, "Shown trait")])
    monkeypatch.setattr(report.season_notes, "POST_HOC_FEATURE_KEYS", frozenset({SHOWN_KEY}))
    summary = report._evidence_summary_section(_results(), {HIDDEN_KEY: 0.002, SHOWN_KEY: 0.5}, frozenset())
    assert "**Shown trait**: r = +0.10 · p = 0.500 · q = 0.500 · significant: no* (see caveats)\n" in summary + "\n"
    assert "significant: yes · precommitted: yes" in summary


def test_evidence_summary_unblinded_shows_everything(monkeypatch):
    monkeypatch.setattr(report, "_all_feature_keys", lambda: [(HIDDEN_KEY, "Hidden trait"), (SHOWN_KEY, "Shown trait")])
    summary = report._evidence_summary_section(_results(), {HIDDEN_KEY: 0.002, SHOWN_KEY: 0.5}, frozenset())
    assert "**Hidden trait**: r = +0.90" in summary
    assert "withheld" not in summary


def test_hidden_results_cannot_move_a_question_up_the_page():
    spec = QuestionSpec("strategies", "Strategies", "multiselect", [HIDDEN_KEY, SHOWN_KEY])
    assert report._spec_effect_size(spec, _results(), frozenset()) == pytest.approx(0.9)
    assert report._spec_effect_size(spec, _results(), frozenset({HIDDEN_KEY})) == pytest.approx(0.1)
    only_hidden = QuestionSpec("x", "X", "multiselect", [HIDDEN_KEY])
    assert report._spec_effect_size(only_hidden, _results(), frozenset({HIDDEN_KEY})) == -1.0

