from aib_analysis.data_structures.custom_types import ScoreType, UserType
from aib_analysis.data_structures.data_models import Leaderboard, LeaderboardEntry, Score
from aib_analysis.main_logic.visualize_tournament import (
    filter_leaderboard_for_display,
    infer_leaderboard_display_exclusions,
)
from tests.mock_data_maker import make_forecast, make_question_binary, make_user


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
