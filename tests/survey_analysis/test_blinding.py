"""Blinded report sections.

While the precommitted picks are under review, no result for a precommitted or
alternative hypothesis may reach the report, including through section order.
Runs without env vars or local data.
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import config, precommitted, report
from aib_analysis.survey_analysis.config import QuestionSpec
from aib_analysis.survey_analysis.stats import CorrelationResult

from tests.survey_analysis.conftest import make_feature

HIDDEN_KEY = precommitted.PRECOMMITTED_HYPOTHESES[0].key
SHOWN_KEY = "uses_capping"


def _pool(size: int = 30) -> list:
    """Bots whose score tracks every precommitted feature, so each would test 'supported'."""
    pool = []
    for index in range(size):
        variables = {
            hypothesis.key: float(index % 2 if hypothesis.predicted_sign > 0 else 1 - index % 2)
            for hypothesis in precommitted.PRECOMMITTED_HYPOTHESES
        }
        pool.append(make_feature(variables=variables, score=float(index % 2) + 0.01 * index))
    return pool


def test_blinded_precommitted_section_shows_no_statistics():
    section = report._precommitted_section(_pool(), exploratory_tested=30, blind=True)
    assert section.count("withheld until the precommitted hypotheses are reviewed") == len(
        precommitted.PRECOMMITTED_HYPOTHESES
    )
    # Prior-season r values are allowed (they are the reasons for the picks); Summer results are not.
    for leaked in ("one-sided p =", "Holm-adjusted p =", "Supported", "Result: **"):
        assert leaked not in section
    for hypothesis in precommitted.ALTERNATIVE_HYPOTHESES:
        assert hypothesis.statement in section


def test_unblinded_precommitted_section_reports_each_test(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHARTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    section = report._precommitted_section(_pool(), exploratory_tested=30, blind=False)
    assert section.count("Holm-adjusted p =") == len(precommitted.PRECOMMITTED_HYPOTHESES)
    assert "**Supported**" in section


def test_precommitted_section_states_the_false_positive_math():
    section = report._precommitted_section(_pool(), exploratory_tested=30, blind=True)
    family_size = len(precommitted.PRECOMMITTED_HYPOTHESES)
    assert f"{1 - 0.95 ** family_size:.0%} chance" in section
    assert f"{1 - 0.95 ** 30:.0%} without a correction" in section


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


def test_blinded_keys_cover_primaries_and_alternatives():
    expected = {h.key for h in precommitted.PRECOMMITTED_HYPOTHESES + precommitted.ALTERNATIVE_HYPOTHESES}
    assert precommitted.BLINDED_KEYS == expected
