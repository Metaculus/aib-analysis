from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from aib_analysis.pro_vs_bot_writeup import run, season_config
from aib_analysis.pro_vs_bot_writeup.phrasing import FILL_ME
from aib_analysis.pro_vs_bot_writeup.results import compute_season_results
from aib_analysis.pro_vs_bot_writeup.season_config import SeasonConfig
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
    percent_correct_section = writeup[writeup.index("### Percent correct (not a proper score)") : writeup.index("## Which bots did best?")]
    assert "should we keep this section?" in percent_correct_section
    assert "![](charts/percent_correct.png)" in percent_correct_section
    assert "**Pro forecasters" in writeup
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
