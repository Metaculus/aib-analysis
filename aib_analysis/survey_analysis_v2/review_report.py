"""Generate parsing_review.md and respondent_audit.csv: the detailed audit.

Purpose: let a human confirm that every raw response was parsed, every write-in
that was set aside is visible, and every respondent joined to the right bot. No
silent failures. This doc is tied 1:1 to the code that produced it.
"""

from __future__ import annotations

import csv
import logging
import os

from aib_analysis.survey_analysis_v2 import config
from aib_analysis.survey_analysis_v2.features import RespondentFeatures

logger = logging.getLogger(__name__)

RESPONDENT_AUDIT_CSV = os.path.join(config.DATA_DIR, "respondent_audit.csv")


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _column_inventory(features: list[RespondentFeatures]) -> list[str]:
    charted = {spec.slug: spec for spec in config.QUESTION_SPECS}
    lines = [
        "## Column inventory",
        "",
        "Every one of the 27 survey columns is accounted for below: either charted or excluded with a "
        "reason. Nothing is dropped silently.",
        "",
        "| Column | Slug | Handling |",
        "| --- | --- | --- |",
    ]
    for slug, header in config.COLUMNS.items():
        if slug in charted:
            handling = f"charted ({charted[slug].kind})"
        elif slug in config.EXCLUDED_COLUMNS:
            handling = f"excluded: {config.EXCLUDED_COLUMNS[slug]}"
        else:
            handling = "used for joining / features only"
        lines.append(f"| {_md_escape(header)} | `{slug}` | {handling} |")
    lines.append("")
    return lines


def _join_audit(features: list[RespondentFeatures]) -> list[str]:
    lines = [
        "## Join audit (all respondents)",
        "",
        "Each survey bot is matched to a leaderboard bot (for peer score) and to a prize owner (for "
        "winner status). `match` columns show how the match was made; `none` means no match.",
        "",
        "| Bot name | LB match | Match kind | Rank | Peer score | Prize owner | Prize kind | Winner | Top 10 | Frontier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for feature in sorted(
        features, key=lambda f: (f.respondent.rank is None, f.respondent.rank or 0)
    ):
        respondent = feature.respondent
        rank = respondent.rank if respondent.rank is not None else "—"
        score = f"{respondent.sum_spot_peer:.0f}" if respondent.sum_spot_peer is not None else "—"
        lines.append(
            f"| {_md_escape(respondent.bot_name)} "
            f"| {_md_escape(respondent.matched_leaderboard_name or '—')} "
            f"| {respondent.leaderboard_match_kind} | {rank} | {score} "
            f"| {_md_escape(respondent.matched_owner or '—')} | {respondent.prize_match_kind} "
            f"| {_yn(respondent.is_winner)} | {_yn(respondent.is_top_10)} | {_yn(feature.frontier)} |"
        )
    lines.append("")
    unmatched = [f.bot_name for f in features if not f.respondent.matched_leaderboard_name]
    lines.append(
        "Note: rank is the raw sum-of-spot-peer-score position over all ~180 bots. This can differ from "
        "the official prize leaderboard, which excludes prize-ineligible bots. For example "
        "Preseen-Chestnut ranks 5th here but is excluded on the official board, so nostreambot shows as "
        "11th here versus 10th there. The top-10 group uses these raw ranks by design."
    )
    lines.append("")
    lines.append(
        f"Respondents with no scored-leaderboard match ({len(unmatched)}): "
        + (", ".join(unmatched) if unmatched else "none")
        + ". These made no forecasts in the scored FutureEval tournament (MiniBench-only participants, "
        "plus any bot with no matching record) and are excluded from the analysis report entirely "
        "(both distributions and correlations). They remain in this audit and the respondent CSV."
    )
    lines.append("")
    return lines


def _model_audit(features: list[RespondentFeatures]) -> list[str]:
    lines = [
        "## Model classification audit",
        "",
        "Per respondent: the raw final-model text, the models it was matched to, the frontier verdict, "
        "and any token that was ignored (artifact/vague) or unmatched (flagged). Check the unmatched "
        "column especially.",
        "",
        "| Bot name | Raw final model(s) | Matched | Frontier | Ignored | Unmatched (flag) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    all_unmatched: list[str] = []
    for feature in features:
        matched = "; ".join(model.display for model in feature.final_models) or "—"
        ignored = "; ".join(feature.final_ignored) or "—"
        unmatched = "; ".join(feature.final_unmatched) or "—"
        all_unmatched.extend(feature.final_unmatched)
        raw = feature.cells["final_model"].raw or "(blank)"
        lines.append(
            f"| {_md_escape(feature.bot_name)} | {_md_escape(raw)} | {_md_escape(matched)} "
            f"| {_yn(feature.frontier)} | {_md_escape(ignored)} | {_md_escape(unmatched)} |"
        )
    lines.append("")
    support_unmatched = sorted({t for f in features for t in f.support_unmatched})
    lines.append(
        "Unmatched final-model tokens (should be empty): "
        + (", ".join(sorted(set(all_unmatched))) if all_unmatched else "none")
        + "."
    )
    lines.append(
        "Unmatched supporting-role tokens (informational, do not affect frontier): "
        + (", ".join(support_unmatched) if support_unmatched else "none")
        + "."
    )
    lines.append("")
    return lines


def _multiselect_audit(features: list[RespondentFeatures]) -> list[str]:
    lines = ["## Multi-select parsing audit", ""]
    for slug, vocab in config.MULTISELECT_VOCAB.items():
        lines.append(f"### {config.COLUMNS[slug]}")
        lines.append("")
        counts = {option: 0 for option in vocab}
        answered = 0
        write_ins: list[str] = []
        for feature in features:
            cell = feature.cells[slug]
            if cell.raw.strip():
                answered += 1
            for option in cell.matched:
                counts[option] += 1
            if cell.other:
                write_ins.append(f"{feature.bot_name}: " + "; ".join(cell.other))
        lines.append(f"Answered by {answered} of {len(features)} respondents.")
        lines.append("")
        lines.append("| Option | Times selected |")
        lines.append("| --- | --- |")
        for option in vocab:
            lines.append(f"| {_md_escape(option)} | {counts[option]} |")
        lines.append("")
        lines.append(f"Write-ins set aside as 'Other' ({len(write_ins)}):")
        lines.append("")
        if write_ins:
            for entry in write_ins:
                lines.append(f"- {_md_escape(entry)}")
        else:
            lines.append("- none")
        lines.append("")
    return lines


def _single_select_audit(features: list[RespondentFeatures]) -> list[str]:
    lines = ["## Single-select parsing audit", ""]
    for slug, vocab in config.SINGLE_SELECT_VOCAB.items():
        lines.append(f"### {config.COLUMNS[slug]}")
        lines.append("")
        counts = {option: 0 for option in vocab}
        others: list[str] = []
        for feature in features:
            cell = feature.cells[slug]
            if cell.matched:
                counts[cell.matched[0]] += 1
            for other in cell.other:
                others.append(f"{feature.bot_name}: {other}")
        lines.append("| Option | Count |")
        lines.append("| --- | --- |")
        for option in vocab:
            lines.append(f"| {_md_escape(option)} | {counts[option]} |")
        lines.append("")
        lines.append(f"Values bucketed as 'Other (excluded)' ({len(others)}):")
        lines.append("")
        if others:
            for entry in others:
                lines.append(f"- {_md_escape(entry)}")
        else:
            lines.append("- none")
        lines.append("")
    return lines


def _excluded_columns_section() -> list[str]:
    lines = ["## Excluded columns", "", "These columns are intentionally not charted.", ""]
    for slug, reason in config.EXCLUDED_COLUMNS.items():
        lines.append(f"- **{config.COLUMNS[slug]}** (`{slug}`): {reason}")
    lines.append("")
    return lines


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def generate_review_report(features: list[RespondentFeatures]) -> str:
    lines: list[str] = ["# Parsing review", ""]
    lines.append(
        "> **Internal only. Do not publish or share externally.** This doc and "
        "`data/respondent_audit.csv` link each bot name to its exact peer score and its individual "
        "survey answers. Because peer scores are public on the leaderboard, that combination "
        "de-anonymizes respondents. It exists so you can spot-check the parsing; the publishable "
        "output is `spring_survey_analysis.md`, which reports only aggregates."
    )
    lines.append("")
    lines.append(
        "This is the spot-check doc. It shows how every response type was parsed, every write-in that "
        "was set aside, and how each respondent joined to the leaderboard and prize data. It is "
        "generated by `aib_analysis/survey_analysis_v2/review_report.py` alongside the analysis, so it "
        "moves with the code. A full row-by-row dump is in `data/respondent_audit.csv`."
    )
    lines.append("")
    lines.extend(_column_inventory(features))
    lines.extend(_join_audit(features))
    lines.extend(_model_audit(features))
    lines.extend(_multiselect_audit(features))
    lines.extend(_single_select_audit(features))
    lines.extend(_excluded_columns_section())
    return "\n".join(lines)


def write_respondent_audit_csv(features: list[RespondentFeatures]) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    charted_slugs = [spec.slug for spec in config.QUESTION_SPECS]
    with open(RESPONDENT_AUDIT_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "bot_name",
            "matched_leaderboard_name",
            "rank",
            "sum_spot_peer",
            "is_winner",
            "is_top_10",
            "frontier",
        ]
        for slug in charted_slugs:
            header.append(f"{slug}__matched")
            header.append(f"{slug}__other")
        writer.writerow(header)
        for feature in features:
            respondent = feature.respondent
            row = [
                respondent.bot_name,
                respondent.matched_leaderboard_name or "",
                respondent.rank if respondent.rank is not None else "",
                f"{respondent.sum_spot_peer:.4f}" if respondent.sum_spot_peer is not None else "",
                respondent.is_winner,
                respondent.is_top_10,
                feature.frontier,
            ]
            for slug in charted_slugs:
                cell = feature.cells[slug]
                row.append(" | ".join(cell.matched))
                row.append(" | ".join(cell.other))
            writer.writerow(row)
    logger.info("Wrote respondent audit CSV to %s", RESPONDENT_AUDIT_CSV)


def write_review_report(features: list[RespondentFeatures]) -> None:
    markdown = generate_review_report(features)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.PARSING_REVIEW_MD, "w") as handle:
        handle.write(markdown)
    write_respondent_audit_csv(features)
    logger.info("Wrote parsing review to %s", config.PARSING_REVIEW_MD)
