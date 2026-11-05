from __future__ import annotations

import pytest
from sklearn.metrics import cohen_kappa_score

from aib_analysis.bot_maker_advice.quality import (
    gwet_ac1,
    krippendorff_alpha,
    locate_excerpt,
    majority_stance,
    percent_agreement,
)


def test_locate_excerpt_returns_the_source_text_despite_quote_style_case_and_whitespace() -> None:
    source = "First line.\n\nWe found that “more agents”   didn’t help at all. Then we stopped."
    assert locate_excerpt('"We found that "more agents" didn\'t HELP at all."', (source,)) == (
        "We found that “more agents”   didn’t help at all"
    )


def test_locate_excerpt_searches_every_source_and_strips_edge_ellipses() -> None:
    assert locate_excerpt("...dropped it because it cost too much…", ("Lessons.", "We dropped it because it cost too much.")) == (
        "dropped it because it cost too much"
    )


@pytest.mark.parametrize(
    "excerpt",
    ["We found that more agents helped", "We found ... help at all", "", '"..."'],
)
def test_locate_excerpt_rejects_paraphrases_elisions_and_empty_quotes(excerpt: str) -> None:
    assert locate_excerpt(excerpt, ("We found that more agents didn't help at all.",)) is None


def _labels(positives: int, total: int) -> list[bool]:
    return [index < positives for index in range(total)]


def test_alpha_matches_hand_computed_values() -> None:
    five_tagged = _labels(5, 45)
    one_disagreement = _labels(4, 45)
    assert krippendorff_alpha(five_tagged, one_disagreement) == pytest.approx(1 - 89 * 1 / (9 * 81))
    assert round(krippendorff_alpha(five_tagged, one_disagreement) or 0, 2) == 0.88
    disagree_on_one_each = [index in (0, 1, 2, 3, 5) for index in range(45)]
    assert round(krippendorff_alpha(five_tagged, disagree_on_one_each) or 0, 2) == 0.78


def test_alpha_is_close_to_kappa_and_undefined_without_variation() -> None:
    first = [index % 7 == 0 for index in range(40)]
    second = [index % 7 == 0 or index == 3 for index in range(40)]
    assert krippendorff_alpha(first, second) == pytest.approx(cohen_kappa_score(first, second), abs=0.02)
    assert krippendorff_alpha([False] * 10, [False] * 10) is None


def test_ac1_stays_high_for_a_rare_code_with_one_disagreement() -> None:
    first, second = _labels(2, 45), _labels(1, 45)
    assert percent_agreement(first, second) == pytest.approx(44 / 45)
    assert gwet_ac1(first, second) > 0.95
    assert (krippendorff_alpha(first, second) or 0) < 0.7


def test_majority_stance() -> None:
    assert majority_stance("supports", "supports", None) == ("supports", True)
    assert majority_stance("supports", "absent", "supports") == ("supports", True)
    assert majority_stance("supports", "absent", "absent") == ("absent", True)
    assert majority_stance("supports", "opposes", "absent") == ("absent", False)
    with pytest.raises(ValueError, match="tie-break"):
        majority_stance("supports", "absent", None)
