import pytest

from aib_analysis.data_structures.custom_types import ScoreType, UserType
from aib_analysis.data_structures.data_models import Leaderboard, LeaderboardEntry, Score
from aib_analysis.data_structures.simulated_tournament import SimulatedTournament
from aib_analysis.main_logic.visualize_tournament import (
    accuracy_table,
    filter_leaderboard_for_display,
    infer_leaderboard_display_exclusions,
)
from tests.mock_data_maker import make_forecast, make_question_binary, make_user


def test_accuracy_table_reports_typical_probability_and_references() -> None:
    yes_question = make_question_binary("Will it rain?").model_copy(update={"question_id": 1, "post_id": 1})
    no_question = make_question_binary("Will it snow?").model_copy(
        update={"question_id": 2, "post_id": 2, "resolution": False}
    )
    pro_user = make_user(name="pro_user", user_type=UserType.PRO)
    bot_user = make_user(name="bot_user", user_type=UserType.BOT)
    tournament = SimulatedTournament(
        name="Pro With Bot Tourn",
        forecasts=[
            make_forecast(question=yes_question, user=pro_user, prediction=[0.8, 0.2]),
            make_forecast(question=no_question, user=pro_user, prediction=[0.2, 0.8]),
            make_forecast(question=yes_question, user=bot_user, prediction=[0.5, 0.5]),
            make_forecast(question=no_question, user=bot_user, prediction=[0.5, 0.5]),
        ],
    )
    table = accuracy_table(tournament, hidden_user_types=set())
    assert table is not None
    typical_probability = table.set_index("user")["typical_probability_pct"]
    assert typical_probability["pro_user"] == pytest.approx(80.0)
    assert typical_probability["bot_user"] == pytest.approx(50.0)
    assert typical_probability["Reference: always the Yes rate (50%)"] == pytest.approx(50.0)
    assert typical_probability["Reference: coin flip"] == 50.0
    hidden_bots_table = accuracy_table(tournament, hidden_user_types={UserType.BOT})
    assert hidden_bots_table is not None
    assert "bot_user" not in set(hidden_bots_table["user"])


def test_infer_leaderboard_display_exclusions() -> None:
    assert infer_leaderboard_display_exclusions("Pro Tournament") == {UserType.BOT}
    assert infer_leaderboard_display_exclusions("Bot Tournament") == {UserType.PRO}
    assert infer_leaderboard_display_exclusions("Bot Tournament | Binary") == {
        UserType.PRO
    }
    assert infer_leaderboard_display_exclusions(
        "Pro With Bot Tourn | No Teams"
    ) == set()
    assert infer_leaderboard_display_exclusions(
        "Pro Vs Bot Tournament | Teams Size 10"
    ) == set()


def test_filter_leaderboard_for_display_hides_user_types() -> None:
    question = make_question_binary()
    pro_user = make_user(name="pro_user", user_type=UserType.PRO)
    bot_user = make_user(name="bot_user", user_type=UserType.BOT)
    pro_forecast = make_forecast(user=pro_user, question=question, prediction=[0.7, 0.3])
    bot_forecast = make_forecast(user=bot_user, question=question, prediction=[0.6, 0.4])
    leaderboard = Leaderboard(
        type=ScoreType.SPOT_PEER,
        entries=[
            LeaderboardEntry(
                scores=[
                    Score(
                        score=10.0,
                        type=ScoreType.SPOT_PEER,
                        forecast=pro_forecast,
                    )
                ]
            ),
            LeaderboardEntry(
                scores=[
                    Score(
                        score=20.0,
                        type=ScoreType.SPOT_PEER,
                        forecast=bot_forecast,
                    )
                ]
            ),
        ],
    )

    visible, hidden = filter_leaderboard_for_display(leaderboard, {UserType.PRO})
    assert [entry.user.name for entry in visible.entries] == ["bot_user"]
    assert [entry.user.name for entry in hidden] == ["pro_user"]
