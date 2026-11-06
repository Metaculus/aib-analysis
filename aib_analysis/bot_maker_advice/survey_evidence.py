"""Read the "Evidence summary" list from a season's bot survey analysis report (survey feature vs. score)."""

from __future__ import annotations

import re
from dataclasses import dataclass

EVIDENCE_HEADING = "## Evidence summary"
EVIDENCE_LINE = re.compile(
    r"^- \*\*(?P<label>.+?)\*\*: r = (?P<correlation>[+-]?\d*\.?\d+) · p = (?P<p>\S+) · q = (?P<q>\S+) · "
    r"significant: (?P<significant>yes|no)(?: · precommitted: (?P<precommitted>yes|no))?\s*$"
)


@dataclass(frozen=True)
class SurveyEvidence:
    label: str
    correlation: float
    p_text: str
    q_text: str
    significant: bool
    precommitted: bool


def parse_survey_evidence(report_markdown: str) -> list[SurveyEvidence]:
    if EVIDENCE_HEADING not in report_markdown:
        raise ValueError(f"The survey report has no {EVIDENCE_HEADING!r} section")
    section = report_markdown.split(EVIDENCE_HEADING, 1)[1].split("\n## ", 1)[0]
    evidence = []
    for line in section.splitlines():
        match = EVIDENCE_LINE.match(line.strip())
        if match:
            evidence.append(
                SurveyEvidence(
                    label=match["label"],
                    correlation=float(match["correlation"]),
                    p_text=match["p"],
                    q_text=match["q"],
                    significant=match["significant"] == "yes",
                    precommitted=match["precommitted"] == "yes",
                )
            )
        elif line.strip().startswith("- **"):
            raise ValueError(f"Unparsed evidence line in the survey report: {line.strip()!r}")
    if not evidence:
        raise ValueError("The survey report's evidence summary has no result lines")
    return evidence
