from __future__ import annotations

import pytest

from aib_analysis.bot_maker_advice.survey_evidence import parse_survey_evidence
from tests.bot_maker_advice.conftest import (
    NOISY_EVIDENCE,
    SIGNIFICANT_EVIDENCE,
    SURVEY_REPORT,
)


def test_parse_survey_evidence_reads_every_line_of_the_section() -> None:
    significant, noisy = parse_survey_evidence(SURVEY_REPORT)
    assert (significant.label, significant.correlation, significant.p_text, significant.q_text) == (SIGNIFICANT_EVIDENCE, -0.51, "<0.001", "0.012")
    assert significant.significant and not significant.precommitted
    assert noisy.label == NOISY_EVIDENCE and noisy.precommitted and not noisy.significant


def test_parse_survey_evidence_fails_loudly_on_an_unexpected_format() -> None:
    with pytest.raises(ValueError, match="no '## Evidence summary'"):
        parse_survey_evidence("# Report\n\n## Methodology\n")
    with pytest.raises(ValueError, match="Unparsed evidence line"):
        parse_survey_evidence("## Evidence summary\n\n- **Uses Exa**: rho = 0.2\n")
