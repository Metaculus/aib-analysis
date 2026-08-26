"""Build the reviewable HTML report and the machine-readable CSV exports."""

from __future__ import annotations

import csv
import html
import logging
from datetime import date
from pathlib import Path

from aib_analysis.survey_analysis.analysis import (
    AnalysisResults,
    category_counts,
    median_of,
    rate,
    split_top_bottom,
)
from aib_analysis.survey_analysis.plots import Chart
from aib_analysis.survey_analysis.statistics import TestFamily

logger = logging.getLogger(__name__)


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_reports(results: AnalysisResults, charts: list[Chart]) -> Path:
    config = results.dataset.config
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    _write_csvs(results)
    path = _write_html(results, charts)
    return path


# ---------------------------------------------------------------------------


def _write_csvs(results: AnalysisResults) -> None:
    config = results.dataset.config

    for family in results.families:
        path = config.data_dir / f"tests_{family.name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "feature",
                    "label",
                    "test",
                    "n",
                    "effect",
                    "effect_label",
                    "group_a",
                    "group_b",
                    "p_raw",
                    "p_bonferroni",
                    "bonferroni_alpha",
                    "q_value_bh",
                    "family_size",
                    "survives_bonferroni",
                    "survives_fdr",
                ]
            )
            for r in family.sorted_results():
                writer.writerow(
                    [
                        r.feature,
                        r.label,
                        r.test,
                        r.n,
                        _fmt(r.effect, 4),
                        r.effect_label,
                        _fmt(r.group_a, 2),
                        _fmt(r.group_b, 2),
                        _fmt_p(r.p_raw),
                        _fmt_p(r.p_bonferroni),
                        _fmt(r.bonferroni_alpha, 5),
                        _fmt_p(r.q_value),
                        r.family_size,
                        r.significant_bonferroni,
                        r.significant_fdr,
                    ]
                )
        logger.info("Wrote %s", path)

    # Per-respondent feature matrix, for anyone who wants to re-cut the data.
    path = config.data_dir / "respondent_features.csv"
    feature_names = sorted(
        {k for r in results.dataset.respondents for k in r.features}
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bot_name",
                "cohort",
                "prize",
                "participant_rank",
                "total_score",
                "question_count",
                *feature_names,
            ]
        )
        for r in results.dataset.respondents:
            writer.writerow(
                [
                    r.bot_name,
                    r.cohort,
                    r.prize,
                    r.participant_rank if r.participant_rank else "",
                    _fmt(r.total_score, 2),
                    r.question_count if r.question_count else "",
                    *[_cell(r.features.get(name)) for name in feature_names],
                ]
            )
    logger.info("Wrote %s", path)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


# ---------------------------------------------------------------------------

CSS = """
:root {
  --ink: #1f2933; --muted: #61707d; --line: #dfe4ea; --bg: #ffffff;
  --win: #2f6f4e; --lose: #b5651d; --accent: #7b4fa8; --soft: #f6f8fa;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.4px; }
h2 {
  font-size: 21px; margin: 52px 0 14px; padding-bottom: 8px;
  border-bottom: 2px solid var(--line); letter-spacing: -0.2px;
}
h3 { font-size: 16px; margin: 30px 0 10px; }
p { margin: 0 0 14px; }
.sub { color: var(--muted); font-size: 14px; margin-bottom: 28px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 22px 0 30px; }
.card {
  flex: 1 1 150px; background: var(--soft); border: 1px solid var(--line);
  border-radius: 8px; padding: 14px 16px;
}
.card .n { font-size: 26px; font-weight: 700; line-height: 1.1; }
.card .l { font-size: 12px; color: var(--muted); margin-top: 3px; }
figure { margin: 26px 0; }
figure img {
  width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px;
  background: #fff;
}
figcaption { font-size: 13px; color: var(--muted); margin-top: 9px; }
.tablewrap { overflow-x: auto; margin: 18px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--line); }
th { background: var(--soft); font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.hit { background: #eef6f1; }
tr.hit td:first-child { border-left: 3px solid var(--win); font-weight: 600; }
tr.fdr { background: #f6f1fa; }
tr.fdr td:first-child { border-left: 3px solid var(--accent); }
.note {
  background: var(--soft); border-left: 3px solid var(--muted);
  padding: 12px 16px; margin: 18px 0; font-size: 14px; border-radius: 0 6px 6px 0;
}
.note strong { color: var(--ink); }
code { background: var(--soft); padding: 1px 5px; border-radius: 3px; font-size: 13px; }
ul { margin: 0 0 14px; padding-left: 22px; }
li { margin-bottom: 5px; }
.legend { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e6eaef; --muted: #9aa5b1; --line: #2d3540; --bg: #14181d; --soft: #1b2027;
    --win: #6fbf94; --lose: #e0a068; --accent: #b48ae0;
  }
  figure img { background: #fff; }
  tr.hit { background: #1a2a22; }
  tr.fdr { background: #241d2e; }
}
"""


def _write_html(results: AnalysisResults, charts: list[Chart]) -> Path:
    dataset = results.dataset
    config = dataset.config
    by_slug = {c.slug: c for c in charts}
    parts: list[str] = []

    def add(text: str) -> None:
        parts.append(text)

    def figure(slug: str) -> None:
        chart = by_slug.get(slug)
        if chart is None:
            return
        add(
            f'<figure><img src="charts/{html.escape(chart.filename)}" '
            f'alt="{html.escape(chart.title)}">'
            f"<figcaption>{html.escape(chart.caption)}</figcaption></figure>"
        )

    add(f"<h1>{html.escape(config.season)} bot-maker survey</h1>")
    add(
        f'<p class="sub">Generated {date.today().isoformat()} from '
        f"{len(dataset.respondents)} survey responses joined to the prize sheet and "
        f"the spot-peer leaderboard.</p>"
    )

    # --- Summary cards ---
    winners = dataset.winners
    non_winners = dataset.non_winners
    add('<div class="cards">')
    for value, label in (
        (len(dataset.respondents), "survey responses"),
        (len(winners), "prize winners"),
        (len(non_winners), "non-winners (competed)"),
        (len(dataset.minibench_only), "MiniBench only"),
        (sum(1 for r in dataset.compared if r.total_score is not None), "matched to leaderboard"),
    ):
        add(f'<div class="card"><div class="n">{value}</div><div class="l">{label}</div></div>')
    add("</div>")

    add("<h2>Sample</h2>")
    add("<ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in results.notes) + "</ul>")
    figure("01_cohort_composition")
    figure("02_model_tier")

    # --- Method note on correction ---
    add("<h2>How significance is judged</h2>")
    wf = results.winner_family
    add(
        '<div class="note">'
        f"<strong>Bonferroni.</strong> The winner-vs-non-winner family contains "
        f"<strong>{wf.size}</strong> tests, registered before any result was inspected. "
        f"The corrected threshold is alpha / m = {config.alpha} / {wf.size} = "
        f"<code>{config.alpha / wf.size:.5f}</code>. A feature is only called significant "
        f"if its raw p falls below that. Reported <code>p_bonferroni</code> is the "
        f"equivalent adjusted p-value, min(1, p x {wf.size}).</div>"
    )
    add(
        '<div class="note">'
        "<strong>Benjamini-Hochberg.</strong> Reported alongside as a q-value. It "
        "controls the expected share of false positives among claimed findings rather "
        "than the chance of any false positive, and it is the more reasonable bar for "
        "exploratory work at this sample size. Both are shown so you can see which "
        "findings depend on which standard.</div>"
    )
    add(
        '<div class="note">'
        "<strong>Families are corrected separately.</strong> The winner comparison, the "
        "all-respondent score correlations, and the within-winner correlations answer "
        "different questions, so each carries its own divisor. Correcting them as one "
        "pool would be more conservative than the questions warrant.</div>"
    )
    figure("05_correction_ladder")

    # --- Main comparison ---
    add("<h2>Winners vs non-winners</h2>")
    add(
        f"<p>Every configured feature tested across {len(dataset.compared)} respondents "
        f"who competed in the main tournament. Fisher exact for yes/no practices, "
        f"Mann-Whitney U for bucketed numeric answers.</p>"
    )
    figure("03_practice_adoption")
    figure("04_effect_vs_evidence")
    add(_family_table(wf, "Winners", "Non-winners"))

    add("<h2>Effort, spend, and research breadth</h2>")
    figure("06_effort_distributions")
    figure("07_research_breadth")
    add(_medians_table(results))

    add("<h2>Leaderboard position</h2>")
    figure("08_leaderboard_position")

    add("<h2>Correlation with total score</h2>")
    add(
        "<p>Pearson r between each feature and total spot-peer score, across all "
        "tournament respondents. This uses the score directly rather than the "
        "winner/non-winner split, so it picks up gradients the binary comparison "
        "cannot.</p>"
    )
    add(_family_table(results.score_family, "", "", correlation=True))

    add("<h2>Within the winners</h2>")
    add(
        f"<p>The same correlations computed on the {len(winners)} prize winners alone. "
        f"With n={len(winners)} this is descriptive. It points at directions to check "
        f"next season rather than establishing anything.</p>"
    )
    figure("09_within_winner_forest")
    figure("10_top_vs_bottom_winners")
    add(_family_table(results.within_winner_family, "", "", correlation=True))

    add("<h2>Ensembling and other single-select questions</h2>")
    figure("11_ensemble_methods")
    figure("12_categorical_breakdown")
    add(_categorical_tables(results))

    add("<h2>Files</h2>")
    add(
        "<ul>"
        "<li><code>data/respondent_features.csv</code> — one row per respondent with "
        "every derived feature</li>"
        + "".join(
            f"<li><code>data/tests_{f.name}.csv</code> — full results for the "
            f"{html.escape(f.name)} family ({f.size} tests)</li>"
            for f in results.families
        )
        + "<li><code>charts/</code> — every figure as PNG</li>"
        "</ul>"
    )

    body = "\n".join(parts)
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(config.season)} bot-maker survey</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>{body}</div></body></html>"
    )

    path = config.output_dir / "survey_report.html"
    path.write_text(doc, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _family_table(
    family: TestFamily, label_a: str, label_b: str, correlation: bool = False
) -> str:
    head = (
        "<tr><th>Feature</th><th class='num'>r</th><th class='num'>n</th>"
        "<th class='num'>raw p</th><th class='num'>Bonferroni p</th>"
        "<th class='num'>BH q</th></tr>"
        if correlation
        else f"<tr><th>Feature</th><th class='num'>{html.escape(label_a)}</th>"
        f"<th class='num'>{html.escape(label_b)}</th><th class='num'>gap</th>"
        f"<th class='num'>n</th><th class='num'>raw p</th>"
        f"<th class='num'>Bonferroni p</th><th class='num'>BH q</th></tr>"
    )

    rows: list[str] = []
    for r in family.sorted_results():
        css = "hit" if r.significant_bonferroni else ("fdr" if r.significant_fdr else "")
        if correlation:
            cells = (
                f"<td>{html.escape(r.label)}</td>"
                f"<td class='num'>{_fmt(r.effect, 3)}</td>"
                f"<td class='num'>{r.n}</td>"
                f"<td class='num'>{_fmt_p(r.p_raw)}</td>"
                f"<td class='num'>{_fmt_p(r.p_bonferroni)}</td>"
                f"<td class='num'>{_fmt_p(r.q_value)}</td>"
            )
        else:
            unit = "%" if r.effect_label == "pp gap" else ""
            cells = (
                f"<td>{html.escape(r.label)}</td>"
                f"<td class='num'>{_fmt(r.group_a, 1)}{unit}</td>"
                f"<td class='num'>{_fmt(r.group_b, 1)}{unit}</td>"
                f"<td class='num'>{_fmt(r.effect, 1)}</td>"
                f"<td class='num'>{r.n}</td>"
                f"<td class='num'>{_fmt_p(r.p_raw)}</td>"
                f"<td class='num'>{_fmt_p(r.p_bonferroni)}</td>"
                f"<td class='num'>{_fmt_p(r.q_value)}</td>"
            )
        rows.append(f"<tr class='{css}'>{cells}</tr>")

    legend = (
        f"<p class='legend'>Family of {family.size} tests. "
        f"Bonferroni threshold {family.alpha}/{family.size} = "
        f"{family.alpha / family.size:.5f} on the raw p. Green rows survive Bonferroni; "
        f"purple rows survive BH only.</p>"
        if family.size
        else ""
    )
    return f"<div class='tablewrap'><table>{head}{''.join(rows)}</table></div>{legend}"


def _medians_table(results: AnalysisResults) -> str:
    dataset = results.dataset
    specs = list(dataset.config.ordinal_features) + list(dataset.config.count_features)
    rows = []
    for spec in specs:
        w = median_of(dataset.winners, spec.name)
        nw = median_of(dataset.non_winners, spec.name)
        mb = median_of(dataset.minibench_only, spec.name)
        rows.append(
            f"<tr><td>{html.escape(spec.label)}</td>"
            f"<td class='num'>{_fmt(w, 2)}</td>"
            f"<td class='num'>{_fmt(nw, 2)}</td>"
            f"<td class='num'>{_fmt(mb, 2)}</td></tr>"
        )
    return (
        "<div class='tablewrap'><table>"
        "<tr><th>Median</th><th class='num'>Winners</th>"
        "<th class='num'>Non-winners</th><th class='num'>MiniBench only</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _categorical_tables(results: AnalysisResults) -> str:
    dataset = results.dataset
    blocks: list[str] = []
    for spec in dataset.config.categorical_features:
        counts = category_counts(dataset.compared, spec.name)
        if not counts:
            continue
        total = sum(counts.values())
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td class='num'>{v}</td>"
            f"<td class='num'>{100 * v / total:.0f}%</td></tr>"
            for k, v in ordered
        )
        blocks.append(
            f"<h3>{html.escape(spec.label)}</h3>"
            f"<div class='tablewrap'><table>"
            f"<tr><th>Answer</th><th class='num'>n</th><th class='num'>share</th></tr>"
            f"{rows}</table></div>"
        )
    return "".join(blocks)
