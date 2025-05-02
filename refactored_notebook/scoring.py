from typing import Literal

import numpy as np


def calculate_spot_peer_score(
    forecast_for_correct_answer: float,
    other_users_forecasts_for_correct_answer: list[float],
) -> float:
    raise NotImplementedError("Not implemented")


def calculate_spot_baseline_score(
    forecast: list[float] | None, # binary: [p_yes, p_no], multiple choice: [p_a, p_b, p_c], numeric: [p_0, p_1, p_2, ...]
    resolution: bool | str | float | None, # binary: bool, multiple choice: str, numeric: float
    options: list[str] | None,
    range_min: float | None,
    range_max: float | None,
    question_weight: float,
) -> float:
    """
    Question type can be infered from resolution type
    """

    if forecast is None or resolution is None:
        raise NotImplementedError("Havent decided how to handle null forecasts and resolutions")

    if len(forecast) == 0:
        raise ValueError("Forecast is empty")

    baseline_score = None

    if isinstance(resolution, bool):
        if len(forecast) != 1 or len(forecast) != 2:
            raise ValueError("Binary questions must have exactly one forecast and two options (for yes or 'yes and no')")

        forecast_val = float(forecast[0])
        baseline_prob = 0.5
        if resolution:
            prob_for_resolution = forecast_val
        else:
            prob_for_resolution = 1 - forecast_val
    elif isinstance(resolution, str):
        if options is None:
            raise ValueError("Options are required for multiple choice questions")

        if len(forecast) != len(options):
            raise ValueError("Forecast and options have different lengths")

        pmf = [float(p) for p in forecast]
        options = [str(opt) for opt in options]
        resolution_idx = options.index(str(resolution))
        prob_for_resolution = pmf[resolution_idx]
        baseline_prob = 1 / len(pmf)
    elif isinstance(resolution, float):
        if range_min is None or range_max is None:
            raise ValueError("Range min and range max are required for numeric questions")

        cdf = [float(p) for p in forecast]
        pmf = [cdf[0]] + [cdf[i] - cdf[i-1] for i in range(1, len(cdf))] # @Ben check: is this a correct conversion?
        pmf.append(1 - cdf[-1])

        resolution = float(resolution)

        bin_edges = np.linspace(range_min, range_max, 200)
        resolution_idx = np.searchsorted(bin_edges, resolution, side='right')

        if resolution_idx >= len(pmf):
            raise ValueError("Resolution is out of bounds")

        prob_for_resolution = pmf[resolution_idx]
        baseline_prob = 1 / len(pmf)  # bins = 201 because of extra appended bin

    else:
        raise ValueError("Unknown question type")

    if prob_for_resolution <= 0 or baseline_prob <= 0:
        raise ValueError("Probability for resolution or baseline probability is less than or equal to 0 which could cause a log(0) issue")

    baseline_score = np.log2(prob_for_resolution / baseline_prob)

    if isinstance(resolution, float):
        baseline_score /= 2  # Numeric scores are halved

    weighted_score = baseline_score * question_weight

    return weighted_score
