"""Manual adjustment loading, validation, and application.

Covers the two override CSVs (winner overrides, answer adjustments): loader
validation failures, and that applied adjustments flow through to matched
options, booleans, midpoints, counts, and model lists. Runs without env vars
or local data (CSVs are written to tmp_path).
"""

from __future__ import annotations

import logging

import pytest

from aib_analysis.survey_analysis import config, manual_adjustments
from aib_analysis.survey_analysis.features import build_features
from aib_analysis.survey_analysis.loading import Respondent

HEADER = "bot_name,column_slug,write_in,action,canonical_option,reason"
SCRAPING_OPTION = "Static web scraping (Only HTML, possibly converted to markdown)"


def _patch_adjustments_csv(tmp_path, monkeypatch, rows: list[str]) -> None:
    path = tmp_path / "manual_answer_adjustments.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    monkeypatch.setattr(config, "MANUAL_ANSWER_ADJUSTMENTS_CSV", str(path))


def _patch_missing_csvs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        config, "MANUAL_ANSWER_ADJUSTMENTS_CSV", str(tmp_path / "missing_adjustments.csv")
    )
    monkeypatch.setattr(
        config, "MANUAL_WINNER_OVERRIDES_CSV", str(tmp_path / "missing_overrides.csv")
    )


def _respondent(bot_name: str, answers: dict[str, str]) -> Respondent:
    full = {slug: "" for slug in config.COLUMNS}
    full.update(answers)
    full["bot_name"] = bot_name
    return Respondent(bot_name=bot_name, answers=full)


# --------------------------------------------------------------------------- #
# Loader behavior and validation
# --------------------------------------------------------------------------- #
def test_missing_files_mean_no_adjustments(tmp_path, monkeypatch):
    _patch_missing_csvs(tmp_path, monkeypatch)
    assert manual_adjustments.load_answer_adjustments() == []
    assert manual_adjustments.load_winner_overrides() == []


def test_unknown_action_rejected(tmp_path, monkeypatch):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['bot,research,"Exa API",remap,"Exa",r'])
    with pytest.raises(ValueError, match="action"):
        manual_adjustments.load_answer_adjustments()


def test_unknown_column_rejected(tmp_path, monkeypatch):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['bot,nonsense,"x",leave,,r'])
    with pytest.raises(ValueError, match="column_slug"):
        manual_adjustments.load_answer_adjustments()


def test_map_requires_canonical_in_vocab(tmp_path, monkeypatch):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['bot,research,"x",map,"Not An Option",r'])
    with pytest.raises(ValueError, match="vocabulary"):
        manual_adjustments.load_answer_adjustments()


def test_non_map_actions_must_not_carry_canonical(tmp_path, monkeypatch):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['bot,research,"x",leave,"Exa",r'])
    with pytest.raises(ValueError, match="canonical_option must be empty"):
        manual_adjustments.load_answer_adjustments()


def test_model_map_requires_registry_display_name(tmp_path, monkeypatch):
    _patch_adjustments_csv(
        tmp_path, monkeypatch, ['bot,support_model,"gpt-40-mini",map,"GPT-40-mini",r']
    )
    with pytest.raises(ValueError, match="registry"):
        manual_adjustments.load_answer_adjustments()


def test_winner_override_requires_true_or_false(tmp_path, monkeypatch):
    path = tmp_path / "manual_winner_overrides.csv"
    path.write_text("bot_name,is_winner,reason\nbot,maybe,r\n")
    monkeypatch.setattr(config, "MANUAL_WINNER_OVERRIDES_CSV", str(path))
    with pytest.raises(ValueError, match="is_winner"):
        manual_adjustments.load_winner_overrides()


# --------------------------------------------------------------------------- #
# Application through build_features
# --------------------------------------------------------------------------- #
def test_multiselect_map_updates_matched_boolean_and_count(tmp_path, monkeypatch):
    _patch_adjustments_csv(
        tmp_path,
        monkeypatch,
        [f'bot-a,research,"Firecrawl",map,"{SCRAPING_OPTION}","scraper"'],
    )
    features = build_features([_respondent("bot-a", {"research": "Exa, Firecrawl"})])
    cell = features[0].cells["research"]
    assert cell.matched == ["Exa", SCRAPING_OPTION]
    assert cell.other == []
    assert features[0].booleans["uses_scraping"] is True
    assert features[0].variables["n_research_sources"] == 2.0
    assert len(features[0].adjustments_applied) == 1


def test_single_select_map_enables_midpoint(tmp_path, monkeypatch):
    _patch_adjustments_csv(
        tmp_path,
        monkeypatch,
        ['bot-a,llm_calls,"Probably 5-10?",map,"5-10","their own estimate"'],
    )
    features = build_features([_respondent("bot-a", {"llm_calls": "Probably 5-10?"})])
    assert features[0].cells["llm_calls"].matched == ["5-10"]
    assert features[0].variables["llm_calls_mid"] == config.LLM_CALLS_MIDPOINT["5-10"]


def test_ignore_drops_write_in_without_mapping(tmp_path, monkeypatch):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['bot-a,strategies,"etc",ignore,,"fragment"'])
    features = build_features([_respondent("bot-a", {"strategies": "Use skills, etc"})])
    cell = features[0].cells["strategies"]
    assert cell.matched == ["Use skills"]
    assert cell.other == []


def test_leave_keeps_write_in_as_other(tmp_path, monkeypatch):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['bot-a,research,"RSS feeds",leave,,"no fit"'])
    features = build_features([_respondent("bot-a", {"research": "Exa, RSS feeds"})])
    cell = features[0].cells["research"]
    assert cell.other == ["RSS feeds"]
    assert len(features[0].adjustments_applied) == 1


def test_model_map_fixes_typo_token(tmp_path, monkeypatch):
    _patch_adjustments_csv(
        tmp_path,
        monkeypatch,
        ['bot-a,support_model,"gpt-40-mini (summarizer)",map,"GPT-4o-mini","typo"'],
    )
    features = build_features(
        [_respondent("bot-a", {"support_model": "GPT-5.4, gpt-40-mini (summarizer)"})]
    )
    feature = features[0]
    assert "GPT-4o-mini" in [model.display for model in feature.support_models]
    assert feature.cells["support_model"].other == []
    assert feature.support_unmatched == []


def test_final_model_map_updates_frontier(tmp_path, monkeypatch):
    _patch_adjustments_csv(
        tmp_path,
        monkeypatch,
        ['bot-a,final_model,"the big claude",map,"Claude Opus 4.8","description of Opus 4.8"'],
    )
    features = build_features([_respondent("bot-a", {"final_model": "the big claude"})])
    assert features[0].frontier is True
    assert features[0].variables["frontier"] == 1.0


def test_stale_adjustment_warns_and_does_not_apply(tmp_path, monkeypatch, caplog):
    _patch_adjustments_csv(
        tmp_path, monkeypatch, ['bot-a,research,"No Longer There",map,"Exa","stale"']
    )
    with caplog.at_level(logging.WARNING):
        features = build_features([_respondent("bot-a", {"research": "Exa"})])
    assert features[0].adjustments_applied == []
    assert features[0].cells["research"].matched == ["Exa"]
    assert any("did NOT apply" in record.message for record in caplog.records)


def test_adjustment_for_unknown_bot_warns(tmp_path, monkeypatch, caplog):
    _patch_adjustments_csv(tmp_path, monkeypatch, ['ghost-bot,research,"Exa API",leave,,"r"'])
    with caplog.at_level(logging.WARNING):
        build_features([_respondent("bot-a", {"research": "Exa"})])
    assert any("did NOT apply" in record.message for record in caplog.records)
