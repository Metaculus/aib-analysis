from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest

from aib_analysis.bot_maker_advice import run
from aib_analysis.bot_maker_advice.llm import CachedOpenRouterLlm, ReplyType
from aib_analysis.bot_maker_advice.method import MAX_FORMAT_RETRIES
from aib_analysis.bot_maker_advice.prompts import (
    CandidateTheme,
    CandidateThemes,
    CitationCheckReply,
)
from aib_analysis.bot_maker_advice.results import AnalysisResult
from aib_analysis.bot_maker_advice.season_config import AdviceSeasonConfig
from aib_analysis.bot_maker_advice.template import CodeVersion, render_writeup
from aib_analysis.pro_vs_bot_writeup.phrasing import FILL_ME
from tests.bot_maker_advice.conftest import (
    ABANDONED_TEXT,
    ANSWER_PATTERN,
    FAKE_MODELS,
    ONE_OFF_SENTENCE,
    SIGNIFICANT_EVIDENCE,
    THEME_SENTENCES,
    THEMES_PER_ANSWER,
    ScriptedLlm,
)

EXPECTED_SECTIONS = [
    "# Advice from Bot Makers to Bot Makers (Synthetic 2026)",
    "## Main themes",
    "## Notable one-offs",
    "## Open-source bots",
    "## Responses in full",
    "## Appendix A: How we found the themes",
]
THEME_BULLET = re.compile(r'^\* (#\d+: ".+"|Also raised by .+\.|Counterpoint, #\d+: ".+"( \(also .+\))?|Argued against it: .+\.|Data check: .+)$')
COUNT_SENTENCE = re.compile(r"^\d+ of the \d+ bot makers gave this advice(, and \d+ argued against it)?\.$")
COMMITTED = CodeVersion(commit="abc1234", method_has_uncommitted_changes=False)


@pytest.fixture
def analysis(synthetic_config: AdviceSeasonConfig, scripted_llm: ScriptedLlm) -> AnalysisResult:
    return asyncio.run(run.analyze(scripted_llm, FAKE_MODELS, synthetic_config, smoke_test=False))


def _theme_blocks(writeup: str) -> list[str]:
    themes_section = writeup.split("## Main themes", 1)[1].split("## Notable one-offs", 1)[0]
    return [block for block in re.split(r"\n(?=### )", themes_section) if block.startswith("### ")]


def test_selection_follows_the_precommitted_rules(analysis: AnalysisResult) -> None:
    statuses = {stats.theme.label: stats.status for stats in analysis.theme_stats}
    assert statuses == {
        "Invest in research quality": "published",
        "Backtest changes on resolved questions": "published",
        "Keep the pipeline simple": "published",
        "Avoid overconfident forecasts": "held_low_agreement",
        "Track costs per question": "below_minimum",
    }
    simple = next(theme for theme in analysis.published_themes if theme.stats.theme.label == "Keep the pipeline simple")
    assert len(simple.stats.supporters) == 4 and len(simple.stats.opposers) == 1 and simple.stats.rejected_count == 1
    assert [check.evidence.label for check in simple.data_checks] == [SIGNIFICANT_EVIDENCE]
    assert len(analysis.data_checks) == 2 and sum(check.published for check in analysis.data_checks) == 1
    assert [review.excerpt for review in analysis.one_offs] == [ONE_OFF_SENTENCE.rstrip(".")]


def test_writeup_is_uniform_and_quotes_are_verbatim(analysis: AnalysisResult, synthetic_config: AdviceSeasonConfig) -> None:
    writeup = render_writeup(analysis, synthetic_config, COMMITTED)
    positions = [writeup.index(section) for section in EXPECTED_SECTIONS]
    assert positions == sorted(positions)
    blocks = _theme_blocks(writeup)
    assert [block.splitlines()[0] for block in blocks] == [
        "### 1. Invest in research quality",
        "### 2. Backtest changes on resolved questions",
        "### 3. Keep the pipeline simple",
    ]
    for block in blocks:
        _heading, count_sentence, bullets, *trailing = block.strip().split("\n\n")
        assert all(part.startswith(FILL_ME) for part in trailing)
        assert COUNT_SENTENCE.match(count_sentence), count_sentence
        assert all(THEME_BULLET.match(bullet) for bullet in bullets.splitlines()), bullets
    responses = writeup.split("## Responses in full", 1)[1]
    for quote in re.findall(r'#\d+: "([^"]+)"', writeup.split("## Responses in full", 1)[0]):
        assert quote in responses
    assert "BACKTEST" not in writeup
    assert "cuts against this advice" in writeup and "Uses explicit base rates" not in writeup
    assert writeup.count(f"*Tried and abandoned:* {ABANDONED_TEXT}") == 1
    assert "Avoid overconfident forecasts" in writeup.split("## Main themes", 1)[1].split("## Notable one-offs", 1)[0]
    assert "at commit abc1234" in writeup
    assert all(f"synthetic-bot-{row}" not in writeup for row in range(2, len(THEMES_PER_ANSWER)))


def test_fill_mes_cover_what_the_data_cannot(analysis: AnalysisResult, synthetic_config: AdviceSeasonConfig) -> None:
    writeup = render_writeup(analysis, synthetic_config, CodeVersion(commit="abc1234", method_has_uncommitted_changes=True))
    reminders = re.findall(re.escape(FILL_ME) + r" _\(Reminder: ([^)]*)", writeup)
    assert writeup.count(FILL_ME) == len(reminders) == 5
    assert any("Pros vs bots post" in reminder for reminder in reminders)
    assert any("Avoid overconfident forecasts" in reminder for reminder in reminders)
    assert any("repository link" in reminder for reminder in reminders)
    assert any("rank" in reminder for reminder in reminders)
    assert any("uncommitted changes" in reminder for reminder in reminders)


def test_outputs_and_qa_review(analysis: AnalysisResult, synthetic_config: AdviceSeasonConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "current_code_version", lambda: COMMITTED)
    run.write_outputs(analysis, synthetic_config, tmp_path, "scripted")
    review = (tmp_path / "qa_review.md").read_text()
    for heading in ("## Disagreements and tie-breaks", "## Citations the checker didn't confirm", "## Data check against the survey analysis", "## Consent answers that need a decision"):
        assert heading in review
    assert "yes, but anonymously" in review
    assert "Uses explicit base rates" in review
    codes = (tmp_path / "data" / "codes.csv").read_text().splitlines()
    assert len(codes) == 1 + len(THEMES_PER_ANSWER) * len(THEME_SENTENCES)
    assert (tmp_path / "respondent_key.csv").read_text().startswith("answer_id,bot_name")


class OverreachingGeneratorLlm(ScriptedLlm):
    async def ask(self, stage: str, model: str, prompt: str, reply_type: type[ReplyType]) -> ReplyType:
        reply = await super().ask(stage, model, prompt, reply_type)
        if isinstance(reply, CandidateThemes) and model == FAKE_MODELS.theme_generators[1]:
            lone_id = next(iter(ANSWER_PATTERN.finditer(prompt)))["answer_id"]
            reply.themes.append(CandidateTheme(label="Pray", definition="", include_when="", exclude_when="", supporting_ids=[int(lone_id)]))
        return reply


def test_candidates_still_invalid_after_retries_are_dropped_and_noted(synthetic_config: AdviceSeasonConfig) -> None:
    llm = OverreachingGeneratorLlm()
    result = asyncio.run(run.analyze(llm, FAKE_MODELS, synthetic_config, smoke_test=False))
    assert [theme.label for theme in result.candidate_themes[1]] == [label for label, _sentence in THEME_SENTENCES.values()]
    assert '"Pray" needs at least 3' in result.run_notes[0]
    assert "A feature that does not exist" in result.run_notes[1]
    assert llm.calls.count(("theme_generation", FAKE_MODELS.theme_generators[1])) == 1 + MAX_FORMAT_RETRIES


def test_cached_llm_reuses_replies_without_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm = CachedOpenRouterLlm(tmp_path)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(llm.ask("citation_check", "openrouter/fake/model", "prompt", CitationCheckReply))
    digest = hashlib.sha256(b"openrouter/fake/model\nCitationCheckReply\nprompt").hexdigest()[:24]
    (tmp_path / "citation_check").mkdir()
    reply = {"verdict": "partial", "excerpt_shows_claim": False, "reason": "cached"}
    (tmp_path / "citation_check" / f"{digest}.json").write_text(json.dumps({"reply": reply}))
    assert asyncio.run(llm.ask("citation_check", "openrouter/fake/model", "prompt", CitationCheckReply)).reason == "cached"
    assert (llm.new_calls, llm.cached_calls) == (0, 1)
