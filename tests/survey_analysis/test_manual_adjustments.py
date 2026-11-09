"""Loading and validation of the manual join files (bot name aliases, winner overrides).

Runs without env vars or local data (CSVs are written to tmp_path).
"""

from __future__ import annotations

import pytest

from aib_analysis.survey_analysis import config, manual_adjustments


def test_missing_winner_override_file_means_no_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MANUAL_WINNER_OVERRIDES_CSV", str(tmp_path / "missing_overrides.csv"))
    assert manual_adjustments.load_winner_overrides() == []


def test_winner_override_requires_true_or_false(tmp_path, monkeypatch):
    path = tmp_path / "manual_winner_overrides.csv"
    path.write_text("bot_name,is_winner,reason\nbot,maybe,r\n")
    monkeypatch.setattr(config, "MANUAL_WINNER_OVERRIDES_CSV", str(path))
    with pytest.raises(ValueError, match="is_winner"):
        manual_adjustments.load_winner_overrides()


def _patch_aliases_csv(tmp_path, monkeypatch, rows: list[str]) -> None:
    path = tmp_path / "manual_bot_name_aliases.csv"
    path.write_text("\n".join(["survey_bot_name,tournament_bot_name,reason", *rows]) + "\n")
    monkeypatch.setattr(config, "MANUAL_BOT_NAME_ALIASES_CSV", str(path))


def test_bot_name_aliases_load_from_csv(tmp_path, monkeypatch):
    _patch_aliases_csv(tmp_path, monkeypatch, ["short,short-bot,leaderboard adds -bot"])
    assert manual_adjustments.load_bot_name_aliases() == {"short": "short-bot"}


def test_missing_alias_file_means_no_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MANUAL_BOT_NAME_ALIASES_CSV", str(tmp_path / "missing.csv"))
    assert manual_adjustments.load_bot_name_aliases() == {}


@pytest.mark.parametrize(
    ("rows", "message"),
    [(["short,,r"], "needs survey_bot_name"), (["short,a,r", "short,b,r"], "duplicate alias")],
)
def test_bad_alias_rows_are_rejected(tmp_path, monkeypatch, rows, message):
    _patch_aliases_csv(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match=message):
        manual_adjustments.load_bot_name_aliases()
