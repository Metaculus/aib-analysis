"""The internal QA review: everything a person needs to audit the analysis before publishing the post."""

from __future__ import annotations

from aib_analysis.bot_maker_advice import method
from aib_analysis.bot_maker_advice.results import AnalysisResult, CoderVote, Theme
from aib_analysis.bot_maker_advice.template import CodeVersion, format_q, one_line


def _cell(text: object) -> str:
    return one_line(str(text)).replace("|", "\\|")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _vote(vote: CoderVote | None) -> str:
    return "" if vote is None else vote.stance


def _theme_definition(theme: Theme, prefix: str = "") -> str:
    sources = f" (from {', '.join(theme.source_candidates)})" if theme.source_candidates else ""
    supporters = f" Proposer cited: {', '.join(f'#{answer_id}' for answer_id in theme.supporting_ids)}." if theme.supporting_ids else ""
    return (
        f"* **{prefix}{theme.number}. {theme.label}**{sources}: {theme.definition} Include when: {theme.include_when} "
        f"Exclude when: {theme.exclude_when}{supporters}"
    )


def render_qa_review(result: AnalysisResult, code_version: CodeVersion, usage_note: str) -> str:
    survey = result.survey
    labels = {theme.number: theme.label for theme in result.codebook}
    models = result.models
    pairs = result.pairs
    published_numbers = {theme.stats.theme.number for theme in result.published_themes}
    sections = [
        f"# QA review: Advice from Bot Makers to Bot Makers ({result.season_label})",
        "Internal. Check this before publishing. It shows answer IDs and quotes only; the ID-to-bot key is in respondent_key.csv.",
        "## Run summary",
        "\n".join(
            [
                f"* Respondents: {survey.respondent_count}. Answered the lessons question: {result.answer_count} "
                f"({survey.blank_lessons_count} blank and {survey.null_lessons_count} null-like answers left out).",
                f"* 'Tried and abandoned' answers shared: {sum(answer.abandoned is not None for answer in survey.answers)}; "
                f"withheld for lack of public consent: {survey.withheld_abandoned_count}.",
                f"* Code version: {code_version.commit or 'unknown'}"
                + (" (method files had uncommitted changes)" if code_version.method_has_uncommitted_changes else ""),
                f"* LLM calls: {usage_note}.",
                "* Models: "
                + "; ".join(
                    f"{role} {', '.join(map(method.model_label, value if isinstance(value, tuple) else (value,)))}"
                    for role, value in models.__dict__.items()
                ),
            ]
        ),
        "## Problems the run worked around",
        "\n".join(f"* {note}" for note in result.run_notes) or "_None._",
        "## Theme labels (model-written; edit freely)",
        "\n".join(f"{theme.rank}. {theme.stats.theme.label}" for theme in result.published_themes) or "_None published._",
        "## Codebook and selection",
        f"Selection rule: at least {method.min_supporters(result.answer_count)} verified supporters and Krippendorff's alpha "
        f">= {method.MIN_CODER_ALPHA} (alpha >= {method.FIRM_CODER_ALPHA} is firm agreement), at most "
        f"{method.MAX_PUBLISHED_THEMES} themes. Coder counts and agreement use each coder's own 'supports' calls before tie-breaks.",
        _table(
            ["#", "Theme", "Status", "Coder 1 / 2 supporters", "Alpha", "AC1", "Agreement", "Tie-breaks", "No majority", "Verified supporters", "Opposers", "Rejected / checked"],
            [
                [
                    stats.theme.number,
                    stats.theme.label,
                    stats.status,
                    f"{stats.coder_supporter_counts[0]} / {stats.coder_supporter_counts[1]}",
                    _number(stats.alpha),
                    _number(stats.ac1),
                    f"{stats.percent_agreement:.0%}",
                    stats.disputed_count,
                    stats.no_majority_count,
                    len(stats.supporters),
                    len(stats.opposers),
                    f"{stats.rejected_count} / {stats.checked_count}",
                ]
                for stats in result.theme_stats
            ],
        ),
        "\n".join(_theme_definition(theme) for theme in result.codebook),
        "## Every counted citation in a published theme",
        "Spot-check these against the full answers in the post. Quoted = shown as an example in the post.",
    ]
    for theme in result.published_themes:
        quoted_ids = {citation.answer_id for citation in theme.exemplars} | ({theme.counterpoint.answer_id} if theme.counterpoint else set())
        theme_pairs = [pair for pair in pairs if pair.theme_number == theme.stats.theme.number and pair.counted]
        sections.append(f"### {theme.rank}. {theme.stats.theme.label}")
        sections.append(
            _table(
                ["Answer", "Stance", "Quoted", "Excerpt", "Concreteness", "Checker's reason"],
                [
                    [f"#{pair.answer_id}", pair.final_stance, "yes" if pair.answer_id in quoted_ids else "", pair.excerpt or "(none)", pair.concreteness, pair.check.reason if pair.check else ""]
                    for pair in theme_pairs
                ],
            )
        )
    disputed_pairs = [pair for pair in pairs if pair.tie_break is not None]
    rejected_pairs = [pair for pair in pairs if pair.final_stance != "absent" and not pair.counted]
    unlocated_votes = [
        (pair, vote)
        for pair in pairs
        for vote in [*pair.votes, *([pair.tie_break] if pair.tie_break else [])]
        if vote.excerpt_problem is not None
    ]
    uncovered = [answer.answer_id for answer in survey.answers if not any(pair.counted and pair.theme_number in published_numbers for pair in pairs if pair.answer_id == answer.answer_id)]
    sections += [
        "## Disagreements and tie-breaks",
        _table(
            ["Answer", "Theme", f"Coder 1 ({method.model_label(models.coders[0])})", f"Coder 2 ({method.model_label(models.coders[1])})", f"Tie-break ({method.model_label(models.tie_breaker)})", "Final"],
            [
                [f"#{pair.answer_id}", f"{pair.theme_number}. {labels[pair.theme_number]}", _vote(pair.votes[0]), _vote(pair.votes[1]), _vote(pair.tie_break), pair.final_stance if pair.has_majority else "absent (no majority)"]
                for pair in disputed_pairs
            ],
        ),
        "## Citations the checker didn't confirm (not counted)",
        _table(
            ["Answer", "Theme", "Stance", "Verdict", "Reason", "Excerpt"],
            [
                [f"#{pair.answer_id}", f"{pair.theme_number}. {labels[pair.theme_number]}", pair.final_stance, pair.check.verdict if pair.check else "not checked", pair.check.reason if pair.check else "", pair.excerpt or "(none)"]
                for pair in rejected_pairs
            ],
        ),
        "## Quotes that failed the word-for-word check after retries",
        "The vote still counts toward the stance; only the quote is unusable.",
        _table(
            ["Answer", "Theme", "Model", "Problem"],
            [[f"#{pair.answer_id}", f"{pair.theme_number}. {labels[pair.theme_number]}", method.model_label(vote.model), vote.excerpt_problem] for pair, vote in unlocated_votes],
        ),
        "## Answers with no counted citation in a published theme",
        ", ".join(f"#{answer_id}" for answer_id in uncovered) or "_None._",
        "## One-off candidates",
        _table(
            ["Answer", "Label", "Verdict", "Covered by a theme", "Shown", "Reason", "Excerpt"],
            [
                [f"#{review.answer_id}", review.label, review.verdict or "not checked", "" if review.covered_by_theme is None else review.covered_by_theme, "yes" if review.selected else "", review.reason, review.excerpt or "(none)"]
                for review in result.one_off_reviews
            ],
        ),
        "## Data check against the survey analysis",
        "A note appears in the post only when the match is confirmed and the survey result is significant. The rest may be "
        "noise; they are listed so you can judge."
        if result.survey_evidence_available
        else "_Skipped: no survey analysis report was available._",
        _table(
            ["Theme", "Survey result", "r", "q", "Significant", "Proposed relation", "Confirmed", "In post", "Proposer's reason", "Confirmer's reason"],
            [
                [f"{check.theme_number}. {labels[check.theme_number]}", check.evidence.label, f"{check.evidence.correlation:+.2f}", format_q(check.evidence.q_text), "yes" if check.evidence.significant else "no", check.relation, "yes" if check.confirmed else "no", "yes" if check.published else "", check.reason, check.confirmer_reason]
                for check in result.data_checks
            ],
        ),
        "## Consent answers that need a decision",
        "These respondents answered 'tried and abandoned' with a consent answer that isn't a standard option, so that answer "
        "was withheld. Add it by hand if their wording allows public sharing.",
        _table(
            ["Answer", "Consent answer"],
            [[f"#{answer.answer_id}", answer.nonstandard_consent] for answer in survey.answers if answer.nonstandard_consent],
        ),
        "## Candidate themes from each proposer",
    ]
    for analyst, model, themes in zip("AB", models.theme_generators, result.candidate_themes):
        sections.append(f"### Analyst {analyst} ({method.model_label(model)})")
        sections.append("\n".join(_theme_definition(theme, prefix=analyst) for theme in themes))
    return "\n\n".join(sections) + "\n"
