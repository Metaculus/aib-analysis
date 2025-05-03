import numpy as np
import pytest

from refactored_notebook.data_models import ForecastType
from refactored_notebook.scoring import (
    calculate_spot_baseline_score,
    calculate_spot_peer_score,
)

# TODO:
# For each of Multiple Choice, Binary, and Numeric questions
# - Test spot peer score
#   - forecast this is further away than others gets worse scores (with 1-5 forecasts)
#   - forecast this is closer to the resolution gets better scores (with 1-5 forecasts)
#   - If everyone has the same forecast, the score is 0
#   - The sum (average?) of everyone's scores is 0
#   - The score for a weighted question is weighted by the question weight
# - Run a test of some forecasts from the site, and make sure the score generated matches the score the site gives

################################### HELPER FUNCTIONS ###################################


def generate_uniform_cdf(num_points: int) -> list[float]:
    return [(i + 1) / num_points for i in range(num_points)]


def generate_perfect_cdf(correct_index: int, inverse_cdf: bool = False) -> list[float]:
    assert correct_index >= 0 and correct_index <= 201
    length_of_cdf = 201
    perfect_forecast = 0.99999
    cdf = []
    for i in range(length_of_cdf):
        if i < correct_index:
            cdf.append(1 - perfect_forecast)
        else:
            cdf.append(perfect_forecast)

    if inverse_cdf:
        cdf = [1 - c for c in cdf]

    return cdf


################################### PEER SCORES ###################################


@pytest.mark.parametrize(
    "forecasts,resolution,options,range_min,range_max",
    [
        # Binary: forecast closer to resolution gets better score
        (
            [[0.9], [0.7], [0.5], [0.3], [0.1]],
            True,
            None,
            None,
            None,
        ),
        # Multiple Choice: forecast closer to resolution gets better score
        (
            [
                [0.9, 0.1, 0.0],
                [0.7, 0.2, 0.1],
                [0.5, 0.3, 0.2],
                [0.3, 0.4, 0.3],
                [0.1, 0.2, 0.7],
            ],
            "A",
            ["A", "B", "C"],
            None,
            None,
        ),
        # Numeric: forecast CDFs with more mass near resolution get better score
        (
            [
                [0.1] * 100 + [0.9] * 101,  # most mass above 0.5
                [0.2] * 100 + [0.8] * 101,
                [0.5] * 201,
                [0.8] * 100 + [0.2] * 101,
                [0.9] * 100 + [0.1] * 101,  # most mass below 0.5
            ],
            0.5,
            None,
            0.0,
            1.0,
        ),
    ],
)
def test_better_forecast_means_better_peer_score(
    forecasts: list[list[float]],
    resolution: bool | str | float,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
):
    scores = [
        calculate_spot_peer_score(
            forecast,
            [f for i, f in enumerate(forecasts) if i != idx],
            resolution,
            options,
            range_min,
            range_max,
            1.0,
        )
        for idx, forecast in enumerate(forecasts)
    ]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    assert sorted_indices == list(range(len(scores))), "Scores should be ordered as expected (descending)"


@pytest.mark.parametrize(
    "question_type,forecast,resolution,options,range_min,range_max",
    [
        ("binary", [0.5], True, None, None, None),
        ("mc", [1 / 3, 1 / 3, 1 / 3], "A", ["A", "B", "C"], None, None),
        ("numeric", [0.5] * 201, 0.5, None, 0.0, 1.0),
    ],
)
def test_peer_score_zero_when_all_same(
    question_type: str,
    forecast: list[float],
    resolution: bool | str | float,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
):
    forecasts = [forecast for _ in range(5)]
    scores = [
        calculate_spot_peer_score(
            f,
            [f2 for i2, f2 in enumerate(forecasts) if i2 != i],
            resolution,
            options,
            range_min,
            range_max,
            1.0,
        )
        for i, f in enumerate(forecasts)
    ]
    for score in scores:
        assert score == pytest.approx(0)


@pytest.mark.parametrize(
    "forecasts,resolution,options,range_min,range_max",
    [
        # Binary
        ([[0.7], [0.3], [0.5]], True, None, None, None),
        # Multiple Choice
        (
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]],
            "A",
            ["A", "B", "C"],
            None,
            None,
        ),
        # Numeric
        (
            [[0.1] * 100 + [0.9] * 101, [0.2] * 100 + [0.8] * 101, [0.5] * 201],
            0.5,
            None,
            0.0,
            1.0,
        ),
    ],
)
def test_peer_score_average_zero(
    forecasts: list[list[float]],
    resolution: bool | str | float,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
):
    scores = [
        calculate_spot_peer_score(
            forecast,
            [f for i, f in enumerate(forecasts) if i != idx],
            resolution,
            options,
            range_min,
            range_max,
            1.0,
        )
        for idx, forecast in enumerate(forecasts)
    ]
    assert np.mean(scores) == pytest.approx(0)


@pytest.mark.parametrize(
    "forecasts,resolution,options,range_min,range_max,weight",
    [
        # Binary
        ([[0.7], [0.3], [0.5]], True, None, None, None, 2.0),
        # Multiple Choice
        (
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]],
            "A",
            ["A", "B", "C"],
            None,
            None,
            0.5,
        ),
        # Numeric
        (
            [[0.1] * 100 + [0.9] * 101, [0.9] * 100 + [0.1] * 101, [0.5] * 201],
            0.5,
            None,
            0.0,
            1.0,
            3.0,
        ),
    ],
)
def test_peer_score_weighted(
    forecasts: list[ForecastType],
    resolution: bool | str | float,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
    weight: float,
):
    for idx, forecast in enumerate(forecasts):
        other_forecasts = [f for i, f in enumerate(forecasts) if i != idx]
        score_unweighted = calculate_spot_peer_score(
            forecast, other_forecasts, resolution, options, range_min, range_max, 1.0
        )
        score_weighted = calculate_spot_peer_score(
            forecast, other_forecasts, resolution, options, range_min, range_max, weight
        )
        assert score_weighted == pytest.approx(score_unweighted * weight)

# TODO: Test the below
# Best score for MC and binary is 996
# Worst score for MC and binary is -996
# Best score for numeric is 408
# Worst score for numeric is -408
# @Check: Can we even validate this (won't we need infinite other forecasters to get max score?)

################################### BASELINE SCORES ###################################


@pytest.mark.parametrize(
    "forecast,resolution,options,range_min,range_max,question_weight,expected",
    [
        # Binary: uniform forecast, should be 0
        ([0.5], True, None, None, None, 1.0, 0.0),
        ([0.5], False, None, None, None, 1.0, 0.0),
        ([0.5, 0.5], False, None, None, None, 1.0, 0.0),
        # Multiple Choice: uniform forecast, should be 0
        ([1 / 3, 1 / 3, 1 / 3], "A", ["A", "B", "C"], None, None, 1.0, 0.0),
        ([0.25, 0.25, 0.25, 0.25], "B", ["A", "B", "C", "D"], None, None, 1.0, 0.0),
        # Numeric: uniform CDF, should be 0
        (generate_uniform_cdf(201), 0.5, None, 0.0, 1.0, 1.0, 0.0),
    ],
)
def test_baseline_score_is_0_with_uniform_prediction(
    forecast: list[float],
    resolution: bool | str | None,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
    question_weight: float,
    expected: float,
):
    score = calculate_spot_baseline_score(
        forecast, resolution, options, range_min, range_max, question_weight
    )
    assert abs(score - expected) == pytest.approx(0)


def test_binary_baseline_score_when_perfect_forecast():
    score = calculate_spot_baseline_score(
        forecast=[0.99999999],
        resolution=True,
    )
    assert score == pytest.approx(100)


def test_binary_baseline_if_completly_incorrect_forecast():
    score = calculate_spot_baseline_score(
        forecast=[0.0000001],
        resolution=True,
    )
    assert score == pytest.approx(-897)


def test_numeric_baseline_when_perfect_forecast():
    correct_index = 30
    length_of_cdf = 201
    index_to_answer_ratio = 3
    correct_answer = correct_index * index_to_answer_ratio
    range_max = length_of_cdf * index_to_answer_ratio

    score = calculate_spot_baseline_score(
        forecast=generate_perfect_cdf(correct_index),
        resolution=correct_answer,
        range_min=0,
        range_max=range_max,
    )
    assert score == pytest.approx(183)


def test_numeric_baseline_if_completly_incorrect_forecast():
    correct_index = 30
    length_of_cdf = 201
    index_to_answer_ratio = 3
    correct_answer = correct_index * index_to_answer_ratio
    range_max = length_of_cdf * index_to_answer_ratio

    score = calculate_spot_baseline_score(
        forecast=generate_perfect_cdf(correct_index),
        resolution=correct_answer,
        range_min=0,
        range_max=range_max,
    )
    assert score == pytest.approx(-230)


def test_multiple_choice_perfect_forecast():
    forecast_for_answer_a = 0.999999999
    num_other_forecasts = 7
    other_forecasts = (1 - forecast_for_answer_a) / num_other_forecasts
    score = calculate_spot_baseline_score(
        forecast=[forecast_for_answer_a] + [other_forecasts] * num_other_forecasts,
        resolution="A",
        options=["A"] + [f"B{i}" for i in range(num_other_forecasts)],
    )
    assert score == pytest.approx(100)


def test_multiple_choice_if_completly_incorrect_forecast():
    forecast_for_answer_c = 0.999999999
    other_forecasts = (1 - forecast_for_answer_c) / 2
    score = calculate_spot_baseline_score(
        forecast=[other_forecasts, other_forecasts, forecast_for_answer_c],
        resolution="C",
        options=["A", "B", "C"],
    )
    assert score == pytest.approx(-232)


@pytest.mark.parametrize(
    "forecast_closer,forecast_further,resolution,options,range_min,range_max",
    [
        # Binary: closer to True
        ([0.8], [0.2], True, None, None, None),
        # Binary: closer to False
        ([0.2], [0.8], False, None, None, None),
        # Multiple Choice: closer to "A"
        ([0.7, 0.2, 0.1], [0.1, 0.2, 0.7], "A", ["A", "B", "C"], None, None),
        # Numeric: CDF with more mass near 0.5 vs near 0.0
        ([0.1] * 52 + [0.9] * 149, [0.9] * 52 + [0.1] * 149, 0.5, None, 0.0, 1.0),
    ],
)
def test_baseline_score_better_when_closer(
    forecast_closer: list[float],
    forecast_further: list[float],
    resolution: bool | str | None,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
):
    score_closer = calculate_spot_baseline_score(
        forecast_closer, resolution, options, range_min, range_max, 1.0
    )
    score_further = calculate_spot_baseline_score(
        forecast_further, resolution, options, range_min, range_max, 1.0
    )
    assert score_closer > score_further


@pytest.mark.parametrize(
    "forecast,resolution,options,range_min,range_max,question_weight",
    [
        # Binary
        ([0.8], True, None, None, None, 2.0),
        # Multiple Choice
        ([0.7, 0.2, 0.1], "A", ["A", "B", "C"], None, None, 0.5),
        # Numeric
        ([0.1] * 50 + [0.9] * 149, 0.5, None, 0.0, 1.0, 3.0),
    ],
)
def test_baseline_score_weighted(
    forecast: list[float],
    resolution: bool | str | None,
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
    question_weight: float,
):
    score_unweighted = calculate_spot_baseline_score(
        forecast, resolution, options, range_min, range_max, 1.0
    )
    score_weighted = calculate_spot_baseline_score(
        forecast, resolution, options, range_min, range_max, question_weight
    )
    assert abs(score_weighted - score_unweighted * question_weight) < 1e-8
