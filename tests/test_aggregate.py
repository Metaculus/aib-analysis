from datetime import datetime, timedelta

import pytest

from aib_analysis.aggregate import (
    AggregateUser,
    aggregate_forecasts,
    create_aggregated_user_at_spot_time,
)
from aib_analysis.custom_types import UserType
from aib_analysis.data_models import User
from aib_analysis.simulated_tournament import SimulatedTournament
from tests.mock_data_maker import (
    make_forecast,
    make_question_binary,
    make_question_mc,
    make_tournament,
    make_user,
)


# TODO: Validate other edge cases
# - Test a specific numeric aggregate forecast
# - mixed question types
# - Signle and no forecasts provided
# - Duplicate forecasts
# - different prediction times


def test_binary_aggregate() -> None:
    forecasts = [[0.4], [0.7], [0.3]]
    question = make_question_binary()
    forecasts = [
        make_forecast(question, make_user(f"User {i}"), forecast)
        for i, forecast in enumerate(forecasts)
    ]
    aggregate_user = make_user("Aggregate User", UserType.AGGREGATE)
    aggregate_forecast = aggregate_forecasts(forecasts, aggregate_user, datetime.now())
    assert aggregate_forecast.aggregate.prediction == [0.4, 0.6]


def test_multiple_choice_aggregate() -> None:
    forecasts = [[0.4, 0.3, 0.3], [0.6, 0.3, 0.1], [0.3, 0.65, 0.05]]
    question = make_question_mc()
    forecasts = [
        make_forecast(question, make_user(f"User {i}"), forecast)
        for i, forecast in enumerate(forecasts)
    ]
    aggregate_user = make_user("Aggregate User", UserType.AGGREGATE)
    aggregate_forecast = aggregate_forecasts(forecasts, aggregate_user, datetime.now())
    prediction = aggregate_forecast.aggregate.prediction
    assert prediction
    assert len(prediction) == 3
    assert prediction[0] == pytest.approx(0.4333333333)
    assert sum(prediction) == 1.0


def test_prediction_time_after_aggregation_time() -> None:
    forecasts = [[0.4], [0.6], [0.3]]
    question = make_question_binary()
    forecast_time = datetime.now() + timedelta(days=1)
    forecasts = [
        make_forecast(question, make_user(f"User {i}"), forecast, forecast_time)
        for i, forecast in enumerate(forecasts)
    ]
    aggregate_user = make_user("Aggregate User", UserType.AGGREGATE)
    aggregation_time = datetime.now()
    with pytest.raises(ValueError):
        aggregate_forecasts(forecasts, aggregate_user, aggregation_time)


def test_create_aggregated_user_user_not_in_tournament() -> None:
    sample_tournament = make_tournament()
    invalid_user = User(name="Invalid User", type=UserType.PRO, aggregated_users=[])
    with pytest.raises(ValueError):
        create_aggregated_user_at_spot_time(
            [invalid_user], sample_tournament, "Aggregate User"
        )


def test_create_aggregated_user_empty_users_list() -> None:
    sample_tournament = make_tournament()
    with pytest.raises(ValueError):
        create_aggregated_user_at_spot_time([], sample_tournament, "Aggregate User")


def test_create_aggregated_user_generally_correct() -> None:
    tournament = make_tournament()
    all_users = tournament.users
    select_users = all_users[:3]
    aggregate_name = "Aggregate User"
    result = create_aggregated_user_at_spot_time(
        select_users, tournament, aggregate_name
    )
    _assert_aggregate_user_correct(result, select_users, tournament, aggregate_name)


def test_create_aggregated_user_pro_tournemaent(
    pro_tournament: SimulatedTournament,
) -> None:
    aggregate_name = "Aggregate User"
    pro_users = pro_tournament.users
    select_pro_users = pro_users[:3]
    result = create_aggregated_user_at_spot_time(
        select_pro_users, pro_tournament, aggregate_name
    )
    _assert_aggregate_user_correct(
        result, select_pro_users, pro_tournament, aggregate_name
    )


def test_create_aggregated_user_bot_tournament(
    bot_tournament: SimulatedTournament,
) -> None:
    aggregate_name = "Aggregate User"
    bot_users = bot_tournament.users
    select_bot_users = bot_users[:15]
    result = create_aggregated_user_at_spot_time(
        select_bot_users, bot_tournament, aggregate_name
    )
    _assert_aggregate_user_correct(
        result, select_bot_users, bot_tournament, aggregate_name
    )




def _assert_aggregate_user_correct(
    aggregate_user: AggregateUser,
    users_to_aggregate: list[User],
    tournament: SimulatedTournament,
    aggregate_name: str,
) -> None:
    assert len(aggregate_user.forecasts) == len(tournament.questions)
    assert aggregate_user.user.name == aggregate_name
    assert aggregate_user.user.type == UserType.AGGREGATE
    assert len(aggregate_user.user.aggregated_users) == len(users_to_aggregate)
    for user in users_to_aggregate:
        assert user in aggregate_user.user.aggregated_users
    for aggregate_forecast in aggregate_user.forecasts:
        for input_forecast in aggregate_forecast.forecasts_inputted:
            assert input_forecast.user in users_to_aggregate
