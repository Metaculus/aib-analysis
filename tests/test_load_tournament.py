from aib_analysis.simulated_tournament import SimulatedTournament


def test_load_pro_data(pro_tournament: SimulatedTournament):
    validate_load_tournament_data(pro_tournament, 2840)


def test_load_bot_data(bot_tournament: SimulatedTournament):
    validate_load_tournament_data(bot_tournament, 33121)


def validate_load_tournament_data(
    tournament: SimulatedTournament, expected_forecast_count: int
):
    forecasts = tournament.forecasts
    questions = tournament.questions
    scores = tournament.scores
    users = tournament.users
    spot_forecasts = tournament.spot_forecasts

    assert len(forecasts) == expected_forecast_count, f"Expected {expected_forecast_count} forecasts, got {len(forecasts)}"
    assert len(questions) > 0, "Expected at least one question"
    assert len(users) > 0, "Expected at least one user"
    assert len(scores) > 0, "Expected at least one score"
    assert len(spot_forecasts) < len(forecasts), f"Expected less than {len(forecasts)} spot forecasts, got {len(spot_forecasts)}"

    unique_questions = set(questions)
    unique_scores = set([score.id for score in scores])
    unique_forecasts = set([forecast.id for forecast in forecasts])
    unique_users = set([user.name for user in users])
    unique_spot_forecasts = set([forecast.id for forecast in spot_forecasts])

    assert len(unique_questions) == len(questions), f"Expected all questions to be unique, got {len(unique_questions)} unique questions and {len(questions)} questions"
    assert len(unique_scores) == len(scores), f"Expected all scores to be unique, got {len(unique_scores)} unique scores and {len(scores)} scores"
    assert len(unique_forecasts) == len(forecasts), f"Expected all forecasts to be unique, got {len(unique_forecasts)} unique forecasts and {len(forecasts)} forecasts"
    assert len(unique_users) == len(users), f"Expected all users to be unique, got {len(unique_users)} unique users and {len(users)} users"
    assert len(unique_spot_forecasts) == len(spot_forecasts), f"Expected all spot forecasts to be unique, got {len(unique_spot_forecasts)} unique spot forecasts and {len(spot_forecasts)} spot forecasts"

    assert len(forecasts) > len(unique_questions), f"Expected more forecasts than questions, got {len(forecasts)} forecasts and {len(unique_questions)} questions"
    assert len(forecasts) > len(spot_forecasts) > len(questions) > len(users) , f"This is a hueristic for number of objects in tournament. Got {len(scores)} scores, {len(forecasts)} forecasts, {len(spot_forecasts)} spot forecasts, {len(questions)} questions, and {len(users)} users"
    assert len(scores) > len(spot_forecasts), f"Expected more scores than spot forecasts, got {len(scores)} scores and {len(spot_forecasts)} spot forecasts"

    # TODO: Make other validations here
