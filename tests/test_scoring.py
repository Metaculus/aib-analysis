from dataclasses import dataclass

import numpy as np
import pytest

from refactored_notebook.data_models import ForecastType
from refactored_notebook.scoring import calculate_baseline_score, calculate_peer_score

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


def generate_uniform_cdf() -> list[float]:
    num_points = 200 # cdf has 201 points, but first point is 0% if we assume closed bounds
    return [0] + [(i + 1) / num_points for i in range(num_points)]


def generate_cdf_with_forecast_at_index(index: int, forecast: float) -> list[float]:
    cdf = []
    for i in range(201):
        if i < index:
            cdf.append(0.0)
        else:
            cdf.append(forecast)
    return cdf


@dataclass
class Percentile:
    value: float
    probability_below: float


def generate_cdf(
    percentiles: list[Percentile],
    lower_bound: float,
    upper_bound: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
    zero_point: float | None = None,
) -> list[float]:
    # Copied from another notebook -> definitely could be cleaned up

    percentile_values: dict[float, float] = {
        percentile.probability_below * 100: percentile.value
        for percentile in percentiles
    }

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = lower_bound
    range_max = upper_bound
    range_size = abs(range_max - range_min)
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust any values that are exactly at the bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min + buffer:
            percentile_values[percentile] = range_min + buffer
        if not open_upper_bound and value >= range_max - buffer:
            percentile_values[percentile] = range_max - buffer

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
            percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
            percentile_values[int(0.5 * percentile_min)] = range_min
    else:
        percentile_values[0] = range_min

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value

    value_percentiles = {
        value: key for key, value in normalized_percentile_values.items()
    }

    # function for log scaled questions
    def generate_cdf_locations(
        range_min: float, range_max: float, zero_point: float | None
    ) -> list[float]:
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, 201)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    def linear_interpolation(
        x_values: list[float], xy_pairs: dict[float, float]
    ) -> list[float]:
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1
                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i - 1], known_x[i]
                    y0, y1 = known_y[i - 1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)

    percentiles = [
        Percentile(value=value, probability_below=percentile)
        for value, percentile in zip(cdf_xaxis, continuous_cdf)
    ]
    assert len(percentiles) == 201

    # Validate minimum spacing between consecutive values
    # for i in range(len(percentiles) - 1):
    #     assert (
    #         abs(percentiles[i + 1].probability_below - percentiles[i].probability_below)
    #         >= 5e-05
    #     ), (
    #         f"Percentiles at indices {i} and {i+1} are too close: "
    #         f"{percentiles[i].probability_below} and {percentiles[i+1].probability_below} "
    #         f"at values {percentiles[i].value} and {percentiles[i+1].value}. "
    #         "It is possible that your prediction is mostly or completely out of the upper/lower bound range "
    #         "Thus making this cdf mostly meaningless."
    #     )

    return [percentile.probability_below for percentile in percentiles]


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
        (generate_uniform_cdf(), 0.5, None, 0.0, 1.0, 1.0, 0.0),
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
    score = calculate_baseline_score(
        forecast, resolution, options, range_min, range_max, question_weight
    )
    assert abs(score - expected) == pytest.approx(0)


@pytest.mark.parametrize(
    "forecast,resolution,expected",
    [
        ([0.001], True, -896.57),  # Completely incorrect
        ([0.999], True, 99.86),  # Completely correct
        ([0.001], False, 99.86),  # Completely correct
        (
            [0.4],
            True,
            -32.19,
        ),  # Examples found here: https://www.metaculus.com/help/scores-faq/#:~:text=details%20for%20nerds-,Do%20all%20my%20predictions%20on%20a%20question%20count%20toward%20my%20score%3F,-Yes.%20Metaculus%20uses
        ([0.7], True, 48.542),
        ([0.4, 0.6], True, -32.19),
    ],
)
def test_binary_baseline_examples(
    forecast: list[float], resolution: bool, expected: float
):
    score = calculate_baseline_score(
        forecast=forecast,
        resolution=resolution,
    )
    assert score == pytest.approx(expected, abs=1e-1)


def test_numeric_baseline_when_perfect_forecast():
    correct_index = 31
    length_of_cdf = 201
    index_to_answer_ratio = 3
    correct_answer = correct_index * index_to_answer_ratio
    range_max = length_of_cdf * index_to_answer_ratio
    forecast = generate_cdf_with_forecast_at_index(correct_index, 0.59)
    # As of May 3, 2025, 0.59 is max difference between 2 points on a cdf

    score = calculate_baseline_score(
        forecast=forecast,
        resolution=correct_answer,
        range_min=0,
        range_max=range_max,
        open_upper_bound=False,
        open_lower_bound=False,
    )
    assert score == pytest.approx(183)


def test_numeric_baseline_if_completly_incorrect_forecast():
    correct_index = 31
    length_of_cdf = 201
    index_to_answer_ratio = 3
    correct_answer = correct_index * index_to_answer_ratio
    range_max = length_of_cdf * index_to_answer_ratio
    forecast = generate_cdf_with_forecast_at_index(correct_index, 0.01/200)

    score = calculate_baseline_score(
        forecast=forecast,
        resolution=correct_answer,
        range_min=0,
        range_max=range_max,
    )
    assert score == pytest.approx(-230.25, abs=1e-1)


@pytest.mark.parametrize(
    "forecast_for_answer_a,num_total_forecasts,expected",
    [
        (0.999, 8, 99.95),
        (0.001, 8, -232.19),
    ],
)
def test_multiple_choice_examples(
    forecast_for_answer_a: float, num_total_forecasts: int, expected: float
):
    num_other_forecasts = num_total_forecasts - 1
    other_forecasts = (1 - forecast_for_answer_a) / num_other_forecasts
    score = calculate_baseline_score(
        forecast=[forecast_for_answer_a] + [other_forecasts] * num_other_forecasts,
        resolution="A",
        options=["A"] + [f"B{i}" for i in range(num_other_forecasts)],
    )
    assert score == pytest.approx(expected, abs=1e-2)


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
        (
            generate_cdf(
                [
                    Percentile(value=40, probability_below=0.1),
                    Percentile(value=60, probability_below=0.9),
                ],
                lower_bound=-1,
                upper_bound=96,
                open_lower_bound=False,
                open_upper_bound=False,
            ),
            generate_cdf(
                [
                    Percentile(value=30, probability_below=0.1),
                    Percentile(value=49, probability_below=0.9),
                ],
                lower_bound=-1,
                upper_bound=96,
                open_lower_bound=False,
                open_upper_bound=False,
            ),
            50,
            None,
            -1,
            96,
        ),
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
    score_closer = calculate_baseline_score(
        forecast=forecast_closer,
        resolution=resolution,
        options=options,
        range_min=range_min,
        range_max=range_max,
    )
    score_further = calculate_baseline_score(
        forecast=forecast_further,
        resolution=resolution,
        options=options,
        range_min=range_min,
        range_max=range_max,
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
        (
            generate_cdf(
                [
                    Percentile(value=0.1, probability_below=0.1),
                    Percentile(value=0.9, probability_below=0.9),
                ],
                lower_bound=0.0,
                upper_bound=1.0,
                open_lower_bound=False,
                open_upper_bound=False,
            ),
            0.5,
            None,
            0.0,
            1.0,
            3.0,
        ),
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
    score_unweighted = calculate_baseline_score(
        forecast, resolution, options, range_min, range_max, 1.0
    )
    score_weighted = calculate_baseline_score(
        forecast, resolution, options, range_min, range_max, question_weight
    )
    assert abs(score_weighted - score_unweighted * question_weight) < 1e-8


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
                [0.9, 0.09, 0.01],
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
                generate_cdf(  # Best CDF
                    [
                        Percentile(value=40, probability_below=0.1),
                        Percentile(value=60, probability_below=0.9),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=False,
                    open_upper_bound=False,
                ),
                generate_cdf(
                    [
                        Percentile(value=20, probability_below=0.1),
                        Percentile(value=50, probability_below=0.9),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=False,
                    open_upper_bound=False,
                ),
                generate_cdf(  # worst CDF
                    [
                        Percentile(value=10, probability_below=0.1),
                        Percentile(value=20, probability_below=0.9),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=False,
                    open_upper_bound=False,
                ),
            ],
            49,
            None,
            -1,
            96,  # Not even range
        ),
        # Numeric: forecast CDFs with more mass near upper bound get better score
        (
            [
                generate_cdf(  # Best CDF
                    [
                        Percentile(value=110, probability_below=0.1),
                        Percentile(value=130, probability_below=0.9),
                    ],
                    lower_bound=0,
                    upper_bound=100,
                    open_lower_bound=False,
                    open_upper_bound=True,
                ),
                generate_cdf(
                    [
                        Percentile(value=90, probability_below=0.1),
                        Percentile(value=140, probability_below=0.9),
                    ],
                    lower_bound=0,
                    upper_bound=100,
                    open_lower_bound=False,
                    open_upper_bound=True,
                ),
                generate_cdf(  # worst CDF
                    [
                        Percentile(value=30, probability_below=0.1),
                        Percentile(value=110, probability_below=0.9),
                    ],
                    lower_bound=0,
                    upper_bound=100,
                    open_lower_bound=False,
                    open_upper_bound=True,  # No upper bound = no probability mass at upper bound
                ),
            ],
            120,
            None,
            0,
            100,
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
        calculate_peer_score(
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
    assert scores[1] > 0, "The first score should be positive"
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    assert len(scores) == len(set(scores)), "Scores should all be different"
    assert sorted_indices == list(
        range(len(scores))
    ), "Scores should be ordered as expected (descending)"


@pytest.mark.parametrize(
    "question_type,forecast,resolution,options,range_min,range_max",
    [
        ("binary", [0.5], True, None, None, None),
        ("mc", [0.25, 0.25, 0.25, 0.25], "A", ["A", "B", "C", "D"], None, None),
        ("numeric", generate_cdf_with_forecast_at_index(100, 0.999), 100, None, 0, 100),
        ("numeric", generate_uniform_cdf(), 50, None, 0, 100),
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
        calculate_peer_score(
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
            [
                generate_cdf(
                    [
                        Percentile(value=30, probability_below=0.1),
                        Percentile(value=60, probability_below=0.9),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=True,
                    open_upper_bound=False,
                ),
                generate_cdf(
                    [
                        Percentile(value=20, probability_below=0.4),
                        Percentile(value=80, probability_below=0.6),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=True,
                    open_upper_bound=True,
                ),
                generate_cdf(
                    [
                        Percentile(value=10, probability_below=0.1),
                        Percentile(value=70, probability_below=0.3),
                    ],
                    lower_bound=-1,
                    upper_bound=96,
                    open_lower_bound=False,
                    open_upper_bound=False,
                ),
            ],
            50,
            None,
            -1,
            96,
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
        calculate_peer_score(
            forecast,
            [f for i, f in enumerate(forecasts) if i != idx],
            resolution,
            options,
            range_min,
            range_max,
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
            [
                generate_uniform_cdf(),
                generate_cdf_with_forecast_at_index(100, 0.999),
                generate_cdf_with_forecast_at_index(101, 0.999),
            ],
            50,
            None,
            0,
            100,
            0.8,
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
        score_unweighted = calculate_peer_score(
            forecast, other_forecasts, resolution, options, range_min, range_max, 1.0
        )
        score_weighted = calculate_peer_score(
            forecast, other_forecasts, resolution, options, range_min, range_max, weight
        )
        assert score_weighted == pytest.approx(score_unweighted * weight)


# TODO: Test the below for peer scores
# Best score for MC and binary is 996
# Worst score for MC and binary is -996
# Best score for numeric is 408
# Worst score for numeric is -408
