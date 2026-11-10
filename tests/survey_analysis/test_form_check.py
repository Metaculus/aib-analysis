"""form_check.py: the config's option lists must match the survey form's text."""

from __future__ import annotations

from aib_analysis.survey_analysis import config, form_check


def _question(slug: str, options: list[str], marker: str = "Check all that apply.") -> str:
    lines = [f"  3.   {config.COLUMNS[slug]} *", "", f"       {marker}", ""]
    lines += [f"         {option}" for option in options]
    return "\n".join(lines + ["         Other:", ""])


def _form(**overrides: list[str]) -> str:
    """Form text whose option lists equal the config's, except for the overridden slugs."""
    sections = [
        _question(slug, overrides.get(slug, vocab))
        for slug, vocab in config.MULTISELECT_VOCAB.items()
    ]
    sections += [
        _question(slug, overrides.get(slug, vocab), marker="Mark only one oval.")
        for slug, vocab in config.SINGLE_SELECT_VOCAB.items()
    ]
    for slug in ("final_model", "support_model"):
        sections.append(_question(slug, overrides.get(slug, ["GPT-5.5", "Claude Opus 4.8"])))
    sections.append("12 of 17                    11/10/26, 2:31 PM\nSurvey   https://docs.google.com/forms/d/x")
    return "\n".join(sections)


def test_matching_form_has_no_mismatches():
    assert form_check.form_mismatches(_form()) == []


def test_option_added_to_the_form_is_reported():
    role = config.SINGLE_SELECT_VOCAB["role"] + ["Government Agency"]
    problems = form_check.form_mismatches(_form(role=role))
    assert problems == ["role: form text not covered by config options: 'Government Agency'"]


def test_option_dropped_from_the_form_is_reported():
    cost = config.SINGLE_SELECT_VOCAB["cost_per_q"][:-1]
    problems = form_check.form_mismatches(_form(cost_per_q=cost))
    assert problems == [f"cost_per_q: config options not on the form: {[config.SINGLE_SELECT_VOCAB['cost_per_q'][-1]]}"]


def test_wrapped_option_and_lost_ligature_still_match():
    development = [
        "LLM netuning" if option == "LLM finetuning" else option
        for option in config.MULTISELECT_VOCAB["development"]
    ]
    wrapped = [line for option in development for line in option.replace(" and resolving ", " and\nresolving ").split("\n")]
    assert form_check.form_mismatches(_form(development=wrapped)) == []


def test_model_without_its_own_registry_entry_is_reported():
    problems = form_check.form_mismatches(_form(final_model=["GPT-5.5", "GPT-5.5 Turbo Max"]))
    assert problems == ["final_model: form models without their own MODEL_REGISTRY entry: ['GPT-5.5 Turbo Max']"]
