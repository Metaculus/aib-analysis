from __future__ import annotations

import dataclasses
import json
import re
from datetime import date
from pathlib import Path

import pytest

from aib_analysis.pro_vs_bot_writeup import run, season_config
from aib_analysis.pro_vs_bot_writeup.phrasing import FILL_ME
from aib_analysis.pro_vs_bot_writeup.results import compute_season_results
from aib_analysis.pro_vs_bot_writeup.season_config import BotQuestionWindow, SeasonConfig
from aib_analysis.pro_vs_bot_writeup.template import ChartFiles, render_writeup
from tests.pro_vs_bot_writeup.conftest import PRO_NAMES

EXPECTED_SECTION_ORDER = [
    "## Main takeaways",
    "## Introduction",
    "## Are bots better than human Pros?",
    "## How accurate are the forecasts?",
    "## Which bots did best?",
    "## How do LLMs compare?",
    "## Links to code and data",
    "## Future FutureEval tournaments",
    "## Appendix A: Methodology",
    "## Appendix B: Metac Bot prompt",
    "## Appendix C: Season data notes",
]


def _render(folder: Path, config: SeasonConfig, output_folder: Path) -> str:
    results = compute_season_results(folder, config)
    hidden_count = run.write_charts(results, config, output_folder)
    return render_writeup(results, config, run.CHART_FILES, hidden_count)


def test_writeup_sections_charts_and_fill_mes(pros_ahead_folder: Path, synthetic_config: SeasonConfig, tmp_path: Path) -> None:
    output_folder = tmp_path / "writeup"
    writeup = _render(pros_ahead_folder, synthetic_config, output_folder)
    positions = [writeup.index(heading) for heading in EXPECTED_SECTION_ORDER]
    assert positions == sorted(positions)
    for chart_path in re.findall(r"!\[\]\((charts/[^)]+)\)", writeup):
        assert (output_folder / chart_path).exists()
    assert len(re.findall(r"!\[\]\(", writeup)) == len(dataclasses.fields(ChartFiles))
    assert writeup.count(FILL_ME) == writeup.count(f"{FILL_ME} _(Reminder:") + writeup.count(f"Results: {FILL_ME}")
    assert "Appendix A: Methodology** at the end of this post" in writeup
    assert "with questions closing between May 2 and May 2, 2026" in writeup
    assert "linkpost" not in writeup
    assert "**Pro forecasters" in writeup
    bot_groups_section = writeup[writeup.index("### Hobbyist and commercial bots") : writeup.index("### The whole field")]
    assert "| Group | Bots | On at least half the questions |" in bot_groups_section
    assert "| Hobbyist | 12 |" in bot_groups_section
    assert "| Metac Bots | 8 |" in bot_groups_section
    assert "**Hobbyist bots took" in writeup
    assert re.search(r"\bNone\b(?! of)|\bnan\b", writeup) is None
    assert "—" not in writeup
    for real_pro_name in PRO_NAMES:
        assert real_pro_name not in writeup


def test_writeup_uses_bot_phrasing_when_bots_win(bots_ahead_folder: Path, synthetic_config: SeasonConfig, tmp_path: Path) -> None:
    writeup = _render(bots_ahead_folder, synthetic_config, tmp_path / "writeup")
    assert "**Bots significantly outperformed Pro forecasters:**" in writeup
    assert "The positive score indicates that the bot team had higher accuracy than the Pro team." in writeup
    assert 'Precommitted suggestion: "Bots Beat Pros"' in writeup


def test_run_writes_all_outputs(pros_ahead_folder: Path, synthetic_config: SeasonConfig, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(season_config.SEASONS, "synthetic", synthetic_config)
    output_folder = tmp_path / "output"
    run.main("synthetic", pros_ahead_folder, output_folder, html=False)
    writeup = (output_folder / "writeup.md").read_text()
    preview = (output_folder / "writeup_preview.md").read_text()
    chart_count = len(dataclasses.fields(ChartFiles))
    assert writeup.count("](charts/") == chart_count
    assert preview.count("](data:image/png;base64,") == chart_count
    assert "](charts/" not in preview
    assert (output_folder / "pro_name_key.csv").read_text().startswith("alias,username")
    for data_file in ("bot_leaderboard.csv", "individual_leaderboard.csv", "metac_same_prompt_leaderboard.csv", "team_sizes.csv"):
        assert (output_folder / "data" / data_file).exists()
    individual_csv = (output_folder / "data" / "individual_leaderboard.csv").read_text()
    assert all(real_pro_name not in individual_csv for real_pro_name in PRO_NAMES)


def _appendix_c(writeup: str) -> str:
    return writeup[writeup.index("## Appendix C: Season data notes") :]


def test_writeup_describes_a_mid_season_window_change(pros_ahead_folder: Path, synthetic_config: SeasonConfig, tmp_path: Path) -> None:
    writeup = _render(pros_ahead_folder, synthetic_config, tmp_path / "writeup")
    assert "Each question stayed open for 1.5 hours (through June 4) or 3 hours (from June 5 on)." in writeup
    assert "Bots would forecast in the last 1.5 hours (through June 4) or 3 hours (from June 5 on) of the time" in writeup
    assert "Metac Bots were not updated during the season" in writeup
    assert "One of the bot makers' bots was an extra bot marked non-primary, which the official leaderboard doesn't rank" in writeup
    assert "Some bot tournament questions were given weights" in writeup
    assert "Pro questions all have a weight of 1. The Pro vs bot comparison uses the Pro questions' weights." in writeup
    appendix = _appendix_c(writeup)
    assert "metac-autocast was named metac-agent until October 15, 2026" in appendix
    assert "The bot question window changed during the season" in appendix
    assert "human account" not in appendix
    assert appendix.index("metac-autocast was named") < appendix.index(FILL_ME)


def test_writeup_with_one_window_and_no_season_notes(pros_ahead_folder: Path, synthetic_config: SeasonConfig, tmp_path: Path) -> None:
    config = dataclasses.replace(
        synthetic_config,
        bot_question_interval_hours=2,
        bot_question_windows=(BotQuestionWindow(first_open_date=None, open_hours=1.5),),
        data_notes=(),
    )
    writeup = _render(pros_ahead_folder, config, tmp_path / "writeup")
    assert "launched a new set of questions one at a time, every 2 hours. Each question stayed open for 1.5 hours, so it closed before the next one launched." in writeup
    assert "window changed" not in writeup


def _edit_bot_question(folder: Path, question_id: int, changes: dict[str, object]) -> None:
    path = folder / "2_bot_tournament.json"
    raw = json.loads(path.read_text())
    for entry in [*raw["forecasts"], *(score["forecast"] for score in raw["scores_cache"].values())]:
        if entry["question"]["question_id"] == question_id:
            entry["question"].update(changes)
    path.write_text(json.dumps(raw))


def test_appendix_c_discloses_shared_weights_and_close_time_differences(
    pros_ahead_folder: Path, synthetic_config: SeasonConfig, tmp_path: Path
) -> None:
    _edit_bot_question(pros_ahead_folder, 1, {"weight": 0.7, "spot_scoring_time": "2026-05-02T01:30:00Z"})
    writeup = _render(pros_ahead_folder, synthetic_config, tmp_path / "writeup")
    assert "1 of them also asked of the Pros" in writeup
    assert "Appendix C lists shared questions where the two closed at different times." in writeup
    appendix = _appendix_c(writeup)
    assert "1 of the shared questions had weights below 1 in the bot tournament (0.7), but the Pro vs bot comparison uses the Pro questions' weights of 1." in appendix
    assert "the bot question closed 90 minutes after the Pro question, so bots had extra time to forecast" in appendix


def test_season_config_rejects_unordered_windows(synthetic_config: SeasonConfig) -> None:
    with pytest.raises(ValueError, match="must start with the season"):
        dataclasses.replace(synthetic_config, bot_question_windows=(BotQuestionWindow(date(2026, 6, 5), 3),))
    with pytest.raises(ValueError, match="increasing first_open_date"):
        dataclasses.replace(
            synthetic_config,
            bot_question_windows=(
                BotQuestionWindow(None, 1.5),
                BotQuestionWindow(date(2026, 7, 1), 3),
                BotQuestionWindow(date(2026, 6, 1), 2),
            ),
        )
