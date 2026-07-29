from datetime import datetime

from aib_analysis.data_structures.custom_types import QuestionType
from aib_analysis.data_structures.data_models import Question
from aib_analysis.data_structures.problem_questions import (
    _all_unscored_title_match_message,
)
from aib_analysis.main_logic.load_tournament import _parse_unscored_resolution_reason


def _unscored_question(post_id: int, reason: str | None) -> Question:
    return Question(
        post_id=post_id,
        question_id=post_id,
        type=QuestionType.BINARY,
        question_text="Same title",
        resolution=None,
        options=None,
        range_max=None,
        range_min=None,
        open_upper_bound=None,
        open_lower_bound=None,
        weight=1.0,
        spot_scoring_time=datetime(2030, 1, 1),
        created_at=datetime(2030, 1, 1),
        unscored_resolution_reason=reason,
    )


def test_parse_unscored_resolution_reason() -> None:
    assert _parse_unscored_resolution_reason({"resolution": "annulled"}) == "annulled"
    assert _parse_unscored_resolution_reason({"resolution": "ambiguous"}) == "ambiguous"
    assert _parse_unscored_resolution_reason({"resolution": float("nan")}) == "blank"
    assert _parse_unscored_resolution_reason({"resolution": "yes"}) is None


def test_all_unscored_title_match_message_differentiates() -> None:
    annulled = [
        _unscored_question(1, "annulled"),
        _unscored_question(2, "annulled"),
    ]
    assert "all are annulled" in _all_unscored_title_match_message(annulled)

    blank = [
        _unscored_question(1, "blank"),
        _unscored_question(2, "blank"),
    ]
    assert "blank resolutions" in _all_unscored_title_match_message(blank)
    assert "deleted or unresolved" in _all_unscored_title_match_message(blank)

    mixed = [
        _unscored_question(1, "annulled"),
        _unscored_question(2, "blank"),
    ]
    mixed_message = _all_unscored_title_match_message(mixed)
    assert "annulled or deleted" in mixed_message
    assert "1 annulled" in mixed_message
    assert "1 blank" in mixed_message

    unknown = [
        _unscored_question(1, None),
        _unscored_question(2, "annulled"),
    ]
    assert "annulled or deleted" in _all_unscored_title_match_message(unknown)
