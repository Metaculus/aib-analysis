"""Check the option lists in config.py against the season's survey form.

Google Forms cannot be exported as data, so the form is printed to PDF and read
with pdftotext. Every option list the analysis parses must match the form
exactly, and every model on the form must have its own registry entry, so the
config cannot silently fall behind a reworded or extended form.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import unicodedata

from aib_analysis.survey_analysis import config
from aib_analysis.survey_analysis.config import _norm_model

logger = logging.getLogger(__name__)

_QUESTION_LINE = re.compile(r"^\s*\d+\.\s+(?P<text>\S.*?)\s*\*?\s*$")
_OPTION_LIST_MARKERS = ("Check all that apply.", "Mark only one oval.")
_PAGE_FURNITURE = re.compile(r"docs\.google\.com/forms|^\s*\d+ of \d+\s")
_MODEL_SLUGS = ("final_model", "support_model")


def read_form_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"No survey form PDF at {pdf_path}. Print the season's Google Form (the blank form "
            "preview) to PDF and save it there; the run checks the option lists against it."
        )
    result = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True, check=True)
    return result.stdout


def form_option_lines(form_text: str) -> dict[str, list[str]]:
    """Option lines under each config.COLUMNS question that has a checkbox or radio list."""
    lines = [
        _collapse(unicodedata.normalize("NFKC", line))
        for line in form_text.splitlines()
        if not _PAGE_FURNITURE.search(line)
    ]
    options_by_slug: dict[str, list[str]] = {}
    current_slug: str | None = None
    collecting = False
    for line in lines:
        question = _QUESTION_LINE.match(line)
        if question:
            current_slug = _slug_for_question(question.group("text"))
            collecting = False
        elif current_slug is not None and line in _OPTION_LIST_MARKERS:
            collecting = True
            options_by_slug[current_slug] = []
        elif collecting and line.startswith("Other:"):
            collecting = False
        elif collecting and line:
            options_by_slug[current_slug].append(line)
    return options_by_slug


def form_mismatches(form_text: str) -> list[str]:
    options_by_slug = form_option_lines(form_text)
    problems: list[str] = []
    vocabularies = {**config.MULTISELECT_VOCAB, **config.SINGLE_SELECT_VOCAB}
    for slug, vocab in vocabularies.items():
        if slug not in options_by_slug:
            problems.append(f"{slug}: no option list found on the form")
            continue
        missing_from_form, leftover = _match_options(" ".join(options_by_slug[slug]), vocab)
        if missing_from_form:
            problems.append(f"{slug}: config options not on the form: {missing_from_form}")
        if leftover:
            problems.append(f"{slug}: form text not covered by config options: {leftover!r}")
    registry_keys = {model.normalized_key for model in config.MODEL_REGISTRY}
    for slug in _MODEL_SLUGS:
        unknown = [line for line in options_by_slug.get(slug, []) if _norm_model(line) not in registry_keys]
        if slug not in options_by_slug:
            problems.append(f"{slug}: no option list found on the form")
        if unknown:
            problems.append(f"{slug}: form models without their own MODEL_REGISTRY entry: {unknown}")
    return problems


def check_form_matches_config() -> None:
    problems = form_mismatches(read_form_text(config.SURVEY_FORM_PDF))
    if problems:
        raise ValueError(
            "The survey form and config.py disagree. Update the option lists (and midpoints) or the "
            "model registry to match the form:\n- " + "\n- ".join(problems)
        )
    logger.info("Survey form option lists match config.py")


def _slug_for_question(first_line: str) -> str | None:
    matches = [
        slug for slug, question in config.COLUMNS.items()
        if _collapse(question).startswith(first_line)
    ]
    return matches[0] if len(matches) == 1 else None


def _match_options(text: str, options: list[str]) -> tuple[list[str], str]:
    """Remove each option (longest first, whole words); return options not found and the leftover text."""
    missing: list[str] = []
    for option in sorted(options, key=len, reverse=True):
        pattern = rf"(?<!\S){_option_pattern(option)}(?!\S)"
        text, count = re.subn(pattern, " ", text, count=1)
        if count == 0:
            missing.append(option)
    return missing, _collapse(text)


def _option_pattern(option: str) -> str:
    """Regex for an option's text as pdftotext prints it.

    The form's option font has ligature glyphs (fi, fl, ...) that pdftotext cannot
    map, so it prints nothing or a space in their place ("Non-Pro t Entity").
    """
    pieces = re.split(r"(ffi|ffl|ff|fi|fl)", _collapse(option))
    return "".join(
        f"(?:{piece}| ?)" if index % 2 else re.escape(piece) for index, piece in enumerate(pieces)
    )


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
