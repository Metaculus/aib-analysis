"""Assemble the advice post from the analysis results and precommitted sentences.

Theme and one-off labels come from the models; every other sentence is fixed here, and every quote is
the bot maker's own words. Anything the data can't provide is left as [===Fill me===] with a reminder.
"""

from __future__ import annotations

import re

from aib_analysis.bot_maker_advice import method
from aib_analysis.bot_maker_advice.answers import (
    ABANDONED_QUESTION,
    LESSONS_QUESTION,
    OpenSourceBot,
)
from aib_analysis.bot_maker_advice.results import (
    AnalysisResult,
    DataCheck,
    PublishedTheme,
)
from aib_analysis.bot_maker_advice.season_config import (
    ANALYSIS_CODE_URL,
    FUTUREEVAL_RESOURCES_OPEN_SOURCE_URL,
    AdviceSeasonConfig,
)
from aib_analysis.pro_vs_bot_writeup.phrasing import (
    count_word,
    fill_me,
    join_names,
    plural,
)

METHOD_REFERENCES = (
    "[Guest, MacQueen & Namey, *Applied Thematic Analysis* (2012)](https://doi.org/10.4135/9781483384436)",
    "[MacQueen et al. (1998)](https://doi.org/10.1177/1525822X980100020301)",
    "[Krippendorff (2011)](https://repository.upenn.edu/asc_papers/43)",
)


def _link(text: str, url: str | None, reminder: str) -> str:
    return f"[{text}]({url})" if url else f"{text} {fill_me(reminder)}"


def one_line(text: str) -> str:
    return " ".join(text.split())


def _ids(answer_ids: tuple[int, ...]) -> str:
    return join_names([f"#{answer_id}" for answer_id in answer_ids])


def _paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    return [re.sub(r"(?m)^#", r"\\#", paragraph) for paragraph in paragraphs]


def format_q(q_text: str) -> str:
    return f"q {q_text[0]} {q_text[1:]}" if q_text[:1] in "<>" else f"q = {q_text}"


def shows_data_checks(result: AnalysisResult, config: AdviceSeasonConfig) -> bool:
    """Data-check notes link to the season's survey analysis post, so they appear only once it is published."""
    return result.survey_evidence_available and config.survey_post_url is not None


def render_writeup(result: AnalysisResult, config: AdviceSeasonConfig) -> str:
    sections = [
        f"# Advice from Bot Makers to Bot Makers ({config.label})",
        _introduction(result, config),
        _themes_section(result, config),
        _one_offs_section(result),
        _open_source_section(result),
        _responses_section(result),
        _appendix(result, config),
    ]
    return "\n\n".join(sections) + "\n"


def _introduction(result: AnalysisResult, config: AdviceSeasonConfig) -> str:
    theme_count = len(result.published_themes)
    example_id = result.published_themes[0].exemplars[0].answer_id if theme_count and result.published_themes[0].exemplars else result.survey.answers[0].answer_id
    tournament = _link(f"{config.label} FutureEval tournament", config.tournament_url, "link the tournament page")
    return "\n\n".join(
        [
            f'As part of the {tournament}, we surveyed bot makers. Among other things, the survey asked, "{LESSONS_QUESTION}" '
            f"{result.answer_count} bot makers answered. This post groups their advice into {count_word(theme_count)} "
            f"{plural(theme_count, 'theme')}, notable one-off ideas, and the full list of quoted bot maker responses.",
            f"Each bot maker has a random ID (like #{example_id}) that stays the same throughout the post, and all quotes are "
            "verbatim. Many bot makers can't run extensive A/B tests, so some of this advice reflects patterns from anecdotes "
            "more than measured results. AI models ran and wrote the thematic analysis. The methodology is detailed in the "
            "appendix.",
        ]
    )


def data_check_sentence(check: DataCheck, survey_post_url: str) -> str:
    evidence = check.evidence
    return (
        f'Data check: in the [survey analysis]({survey_post_url}), "{evidence.label}" went with '
        f"{'higher' if evidence.correlation > 0 else 'lower'} average scores (r = {evidence.correlation:+.2f}, "
        f"{format_q(evidence.q_text)}). That {'supports' if check.relation == 'agrees' else 'cuts against'} this advice, "
        "though it is a correlation and may be confounded."
    )


def theme_block(theme: PublishedTheme, answer_count: int, survey_post_url: str | None) -> str:
    stats = theme.stats
    count_sentence = f"{len(stats.supporters)} of the {answer_count} bot makers gave this advice"
    if stats.opposers:
        count_sentence += f", and {len(stats.opposers)} argued against it"
    bullets = [f'* #{citation.answer_id}: "{one_line(citation.excerpt)}"' for citation in theme.exemplars]
    if theme.also_raised_by:
        bullets.append(f"* Also raised by {_ids(theme.also_raised_by)}.")
    if theme.counterpoint is not None:
        others = f" (also {_ids(theme.other_opposers)})" if theme.other_opposers else ""
        bullets.append(f'* Counterpoint, #{theme.counterpoint.answer_id}: "{one_line(theme.counterpoint.excerpt)}"{others}')
    elif theme.other_opposers:
        bullets.append(f"* Argued against it: {_ids(theme.other_opposers)}.")
    if survey_post_url is not None:
        bullets.extend(f"* {data_check_sentence(check, survey_post_url)}" for check in theme.data_checks)
    return "\n\n".join([f"### {theme.rank}. {stats.theme.label}", f"{count_sentence}.", "\n\n".join(bullets)])


def _themes_section(result: AnalysisResult, config: AdviceSeasonConfig) -> str:
    survey_post_url = config.survey_post_url if shows_data_checks(result, config) else None
    blocks = ["## Main themes"]
    blocks.extend(theme_block(theme, result.answer_count, survey_post_url) for theme in result.published_themes)
    for stats in result.stats_with_status("held_low_agreement"):
        blocks.append(
            fill_me(
                f'the theme "{stats.theme.label}" had {len(stats.supporters)} supporting bot makers, but the two coders agreed '
                f"too little to publish it automatically (Krippendorff's alpha {stats.alpha or 0:.2f}, below the precommitted "
                f"{method.MIN_CODER_ALPHA}). Publish it as another theme, sharpen its definition and rerun, or delete this "
                "note; qa_review.md lists its citations"
            )
        )
    if not result.published_themes:
        blocks.append(fill_me("no theme met the precommitted bar; see qa_review.md before deciding what to publish"))
    return "\n\n".join(blocks)


def _one_offs_section(result: AnalysisResult) -> str:
    if not result.one_offs:
        return "## Notable one-offs\n\n" + fill_me("no one-off idea passed the checks; pick one from qa_review.md or delete this section")
    lines = [f'* **{item.label}** (#{item.answer_id}): "{one_line(item.excerpt or "")}"' for item in result.one_offs]
    return "\n\n".join(["## Notable one-offs", "Ideas that one bot maker raised and the themes above don't cover:", "\n\n".join(lines)])


def leaderboard_note(bot: OpenSourceBot) -> str:
    """Spring 2026 style: participant rank with Metac Bots removed, or "MiniBench only" for a bot not on the season's board."""
    if bot.leaderboard_status == "ranked":
        return f", rank {bot.rank}"
    if bot.leaderboard_status == "not_found":
        return ", MiniBench only"
    if bot.leaderboard_status == "unranked":
        return ", unranked extra bot"
    return ""


def _open_source_section(result: AnalysisResult) -> str:
    linked = [bot for bot in result.survey.open_source_bots if bot.url]
    if not linked:
        return "## Open-source bots\n\n" + fill_me("no respondent gave a link to an open-source bot; delete this section")
    blocks = [
        "## Open-source bots",
        "The bot makers listed below told us their bot is open source, and its repository is public. The "
        f"[FutureEval resources page]({FUTUREEVAL_RESOURCES_OPEN_SOURCE_URL}) keeps a longer list of open-source bots, "
        "though these are the ones from this survey:",
        "\n".join(f"* [{bot.bot_name}]({bot.url}){leaderboard_note(bot)}" for bot in linked),
    ]
    if any(bot.leaderboard_status == "unknown" for bot in linked):
        blocks.append(fill_me('add each bot\'s rank (participant rank with Metac Bots removed) or "MiniBench only"'))
    return "\n\n".join(blocks)


def _responses_section(result: AnalysisResult) -> str:
    entries = []
    for answer in result.survey.answers:
        lessons = _paragraphs(answer.lessons)
        entry = [f"**Bot maker #{answer.answer_id}:** {lessons[0]}", *lessons[1:]]
        if answer.abandoned is not None:
            abandoned = _paragraphs(answer.abandoned)
            entry += [f"*Tried and abandoned:* {abandoned[0]}", *abandoned[1:]]
        entries.append("\n\n".join(entry))
    preface = (
        f'All {result.answer_count} answers to "{LESSONS_QUESTION}", verbatim and ordered by ID. Where a bot maker allowed us '
        f'to share their survey response publicly, their answer to "{ABANDONED_QUESTION}" follows under "Tried and abandoned."'
    )
    return "\n\n".join(["## Responses in full", preface, *entries])


def _theme_quality_table(result: AnalysisResult) -> str:
    lines = [
        "| Theme | Gave this advice | Argued against | Coder agreement (Krippendorff's alpha) | Citations the checker rejected |",
        "| --- | --- | --- | --- | --- |",
    ]
    for theme in result.published_themes:
        stats = theme.stats
        lines.append(
            f"| {theme.rank}. {stats.theme.label} | {len(stats.supporters)} of {result.answer_count} | {len(stats.opposers)} | "
            f"{stats.alpha or 0:.2f} | {stats.rejected_count} of {stats.checked_count} |"
        )
    return "\n".join(lines)


def _appendix(result: AnalysisResult, config: AdviceSeasonConfig) -> str:
    models = result.models
    label = method.model_label
    answer_count = result.answer_count
    codebook_size = len(result.codebook)
    proposed_by_both = sum(
        1 for theme in result.codebook if {code[0] for code in theme.source_candidates} == {"A", "B"}
    )
    disputed = sum(stats.disputed_count for stats in result.theme_stats)
    checked = sum(stats.checked_count for stats in result.theme_stats)
    rejected = sum(stats.rejected_count for stats in result.theme_stats)
    steps = [
        f"1. **Themes:** Two models from different companies ({label(models.theme_generators[0])} and "
        f"{label(models.theme_generators[1])}) each read all {answer_count} answers, in different random orders, and proposed "
        f"{method.MIN_CODEBOOK_THEMES} to {method.MAX_CODEBOOK_THEMES} themes, with definitions and rules for what does and "
        f"doesn't count. A third model ({label(models.codebook)}) merged them into a codebook of {codebook_size} themes; "
        f"{proposed_by_both} of them were proposed by both models.",
        f"2. **Coding:** Two coders ({label(models.coders[0])} and {label(models.coders[1])}) separately read every answer and "
        "marked, for each theme, whether it gives that advice, argues against it, or neither, with a supporting quote. Code "
        "confirmed that every quote appears word for word in the answer.",
        f"3. **Tie-breaks:** Where the coders disagreed ({disputed} of {answer_count * codebook_size} answer-theme pairs), a "
        f"fourth model ({label(models.tie_breaker)}) coded the pair without seeing their calls, and the majority decided.",
        f"4. **Citation checks:** A separate model ({label(models.verifier)}) skeptically checked each of the {checked} "
        f"remaining citations against the full answer and rejected {rejected}. Only citations it confirmed are counted or quoted.",
        f"5. **Selection:** A theme is published if at least {method.min_supporters(answer_count)} bot makers "
        f"({method.MIN_SUPPORTER_SHARE:.0%} of answers, and at least {method.MIN_SUPPORTERS_FLOOR}) give the advice and the "
        f"two coders' agreement (Krippendorff's alpha) is at least {method.MIN_CODER_ALPHA}. We publish at most "
        f"{method.MAX_PUBLISHED_THEMES} themes, ranked by how many bot makers gave the advice. Each theme quotes up to "
        f"{method.EXEMPLARS_PER_THEME} answers, the most concrete first, and no bot maker is quoted in two themes unless a "
        "theme runs out of other quotes.",
        f"6. **One-offs:** {label(models.one_offs)} proposed specific ideas that the themes don't cover, and "
        f"{label(models.verifier)} checked each one. We show up to {method.MAX_ONE_OFFS}, one per bot maker.",
    ]
    if shows_data_checks(result, config):
        steps.append(
            f"7. **Data check:** {label(models.data_check)} matched each published theme to results in the survey analysis "
            f"that directly measure the advice, and {label(models.data_check_confirmer)} confirmed each match. A theme gets a "
            "data-check note only for a confirmed match that the survey analysis found statistically significant; other "
            "matches can be noise, so we leave them out."
        )
    return "\n\n".join(
        [
            "## Appendix: How themes were found",
            "This analysis used codebook thematic analysis (" + "; ".join(METHOD_REFERENCES) + "). This involves building a "
            "codebook of themes with rules for what counts, applying it to every answer with independent coders, and "
            f"measuring how well they agree. [The code]({ANALYSIS_CODE_URL}) for the analysis is open source.",
            "\n".join(steps),
            _theme_quality_table(result),
            "Counts are floors: the survey didn't ask about each theme, so a bot maker who didn't mention one may still agree "
            "with it. Theme and one-off labels were written by the models; the quotes are the bot makers' own words.",
        ]
    )
