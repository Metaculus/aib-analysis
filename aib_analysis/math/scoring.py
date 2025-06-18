from enum import Enum
from typing import Literal

import numpy as np
from scipy.stats.mstats import gmean

from aib_analysis.data_structures.custom_types import (
    ForecastType,
    QuestionType,
    ResolutionType,
)


def calculate_peer_score(
    forecast: ForecastType,
    forecast_for_other_users: list[ForecastType],
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    question_weight: float = 1.0,
    q_type: Literal["binary", "multiple_choice", "numeric"] | None = None,
) -> float:
    question_type = _determine_question_type(q_type, resolution)
    resolution = _normalize_resolution(question_type, resolution, range_min, range_max)
    forecast_for_resolution = _determine_probability_for_resolution(
        question_type, forecast, resolution, options, range_min, range_max
    )
    other_user_forecasts = [
        _determine_probability_for_resolution(
            question_type, forecast, resolution, options, range_min, range_max
        )
        for forecast in forecast_for_other_users
    ]

    geometric_mean = gmean(other_user_forecasts)
    peer_score = np.log(forecast_for_resolution / geometric_mean)
    if question_type == QuestionType.NUMERIC:
        peer_score /= 2
    return peer_score * question_weight * 100


def calculate_baseline_score(
    forecast: ForecastType,
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    question_weight: float = 1.0,
    open_upper_bound: bool | None = False,
    open_lower_bound: bool | None = False,
    q_type: Literal["binary", "multiple_choice", "numeric"] | None = None,
) -> float:
    """
    Question type can be infered from resolution type
    Scoring math: https://www.metaculus.com/help/scores-faq/#What:~:text=given%20score%20type.-,What%20is%20the%20Baseline%20score%3F,-The%20Baseline%20score
    """
    question_type = _determine_question_type(q_type, resolution)
    resolution = _normalize_resolution(question_type, resolution, range_min, range_max)
    prob_for_resolution = _determine_probability_for_resolution(
        question_type, forecast, resolution, options, range_min, range_max
    )
    baseline_prob = _determine_baseline(
        question_type,
        resolution,
        options,
        range_min,
        range_max,
        open_upper_bound,
        open_lower_bound,
    )
    divisor = _determine_divisor_for_baseline_score(question_type, options)
    if prob_for_resolution <= 0 or baseline_prob <= 0:
        raise ValueError(
            "Probability for resolution or baseline probability is less than or equal to 0 which could cause a log(0) issue"
        )

    baseline_score = np.log(prob_for_resolution / baseline_prob) / divisor * 100

    weighted_score = baseline_score * question_weight

    return weighted_score


def _determine_baseline(
    question_type: QuestionType,
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    open_upper_bound: bool | None = None,
    open_lower_bound: bool | None = None,
) -> float:
    resolution = _normalize_resolution(question_type, resolution, range_min, range_max)
    if question_type == QuestionType.BINARY:
        baseline_prob = 0.5
    elif question_type == QuestionType.MULTIPLE_CHOICE:
        if options is None:
            raise ValueError("Options are required for multiple choice questions")
        baseline_prob = 1 / len(options)
    elif question_type == QuestionType.NUMERIC:
        if open_upper_bound is None or open_lower_bound is None:
            raise ValueError(
                "Open upper bound and lower bound are required for numeric questions"
            )
        if range_min is None or range_max is None:
            raise ValueError(
                "Range min and range max are required for numeric questions"
            )
        if not isinstance(resolution, float):
            raise ValueError("Resolution must be a float for numeric questions")


        resolved_outside_bounds = False
        assert (
            range_min is not None and range_max is not None and resolution is not None
        ), f"These need to be not None: Range min: {range_min}, range max: {range_max}, resolution: {resolution}"
        if resolution > range_max or resolution < range_min:
            resolved_outside_bounds = True
        if resolved_outside_bounds:
            baseline_prob = 0.05
        else:
            open_bound_count = bool(open_upper_bound) + bool(open_lower_bound)
            baseline_prob = (
                1 - 0.05 * open_bound_count
            ) / 200  # PMF has 202 bins, 2 of which represent the bounds. So 200 is the internal bins
    else:
        raise ValueError("Unknown question type")
    assert (
        0 <= baseline_prob <= 1
    ), f"Baseline probability is {baseline_prob} which is not between 0 and 1"
    return baseline_prob


def _determine_probability_for_resolution(
    q_type: QuestionType,
    forecast: ForecastType,
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
) -> float:
    """
    Returns a 0 to 1 probability for the resolution
    Also returns the baseline probability used in baseline scoring
    """
    resolution = _normalize_resolution(q_type, resolution, range_min, range_max)

    if resolution is None:
        raise ValueError(
            "Cannot score a forecast with an annulled or ambiguous resolution"
        )

    if forecast is None:
        raise NotImplementedError(
            "Havent decided how to handle null forecasts. I think we can probably just avoid scoring these?"
        )

    try:
        if len(forecast) == 0:
            raise ValueError("Forecast is empty")
    except Exception as e:
        raise ValueError(
            f"Error encountered for question of type {q_type} with resolution {resolution} and forecast {forecast}: {e}"
        )

    if not q_type == QuestionType.NUMERIC and any(p <= 0 or p >= 1 for p in forecast):
        raise ValueError("Forecast contains probabilities outside of 0 to 1 range")

    if q_type == QuestionType.BINARY:
        assert isinstance(resolution, bool)
        prob_for_resolution = _binary_resolution_prob(forecast, resolution)
    elif q_type == QuestionType.MULTIPLE_CHOICE:
        assert isinstance(resolution, str)
        if options is None:
            raise ValueError("Options are required for multiple choice questions")
        prob_for_resolution = _multiple_choice_resolution_prob(
            forecast, resolution, options
        )
    elif q_type == QuestionType.NUMERIC:
        if range_min is None or range_max is None:
            raise ValueError(
                "Range min and range max are required for numeric questions"
            )
        assert isinstance(
            resolution, float
        ), f"Resolution is {resolution} which is not a float"
        prob_for_resolution = _numeric_resolution_prob(
            forecast, resolution, range_min, range_max
        )
    else:
        raise ValueError(f"Unknown question type: {q_type}")

    if not 0 < prob_for_resolution < 1:
        raise ValueError(
            f"Probability for resolution is {prob_for_resolution} which is not between 0 and 1"
        )
    return prob_for_resolution


def _binary_resolution_prob(forecast: list[float], resolution: bool) -> float:
    if len(forecast) != 1 and len(forecast) != 2:
        raise ValueError(
            "Binary questions must have exactly one or two forecasts (for yes or 'yes and no')"
        )

    forecast_val = float(forecast[0])
    if resolution:
        prob_for_resolution = forecast_val
    else:
        prob_for_resolution = 1 - forecast_val
    return prob_for_resolution


def _multiple_choice_resolution_prob(
    forecast: list[float], resolution: str, options: list[str]
) -> float:
    if len(forecast) != len(options):
        raise ValueError("Forecast and options have different lengths")

    pmf = [float(p) for p in forecast]
    options = [
        str(opt) for opt in options
    ]
    resolution_idx = options.index(str(resolution))
    prob_for_resolution = pmf[resolution_idx]
    return prob_for_resolution


def _numeric_resolution_prob(
    forecast: list[float], resolution: float, range_min: float, range_max: float
) -> float:
    if len(forecast) != 201:
        raise ValueError("CDF should have 201 bins")

    previous_prob = 0
    for current_prob in forecast:
        if current_prob < previous_prob:
            raise ValueError("CDF should be in increasing order")
        previous_prob = current_prob

    cdf = [float(p) for p in forecast]
    pmf = cdf_to_pmf(cdf)

    resolution_bin_idx = _resolution_value_to_pmf_index(
        pmf, resolution, range_min, range_max
    )

    prob_for_resolution = pmf[resolution_bin_idx]
    if not 0 < prob_for_resolution < 1:
        raise ValueError(
            f"Numeric forecast probability for resolution is {prob_for_resolution} which is not between 0 and 1"
        )

    return prob_for_resolution


def _determine_divisor_for_baseline_score(
    question_type: QuestionType, options: list[str] | None = None
) -> float:
    if question_type == QuestionType.BINARY:
        return np.log(2)
    elif question_type == QuestionType.MULTIPLE_CHOICE:
        if options is None:
            raise ValueError("Options are required for multiple choice questions")
        return np.log(len(options))
    elif question_type == QuestionType.NUMERIC:
        return 2
    else:
        raise ValueError("Unknown question type")


def _resolution_value_to_pmf_index(
    pmf: list[float], resolution: float, range_min: float, range_max: float
) -> int:
    """
    PMF explanation:
    - 200 bins for the internal range
    - 1 bin for the 'above upper bound'
    - 1 bin for the 'below lower bound'
    - 202 total bins
    """
    if len(pmf) != 202:
        raise ValueError(f"PMF should have 202 bins, but has {len(pmf)}")
    position_in_range = _resolution_value_to_position_in_numeric_range(
        resolution, range_min, range_max
    )
    resolution_bin_idx = _position_in_range_to_bucket_index(position_in_range)
    if resolution_bin_idx >= len(pmf) or resolution_bin_idx < 0:
        raise ValueError(
            f"Invalid resolution bin index: {resolution_bin_idx}. Resolution: {resolution}, Range min: {range_min}, Range max: {range_max}"
        )
    _test_resolution_bin_idx_edge_cases(
        pmf, position_in_range, resolution_bin_idx, resolution, range_min, range_max
    )
    return resolution_bin_idx

def _position_in_range_to_bucket_index(
    position_in_range: float
) -> int:
    outcome_count = 200
    if position_in_range < 0:
        return 0
    if position_in_range > 1:
        return outcome_count + 1
    if position_in_range == 1:
        return outcome_count
    return max(
        int(position_in_range * outcome_count + 1 - 1e-10), 1
    )

def _test_resolution_bin_idx_edge_cases(
    pmf: list[float],
    position_in_range: float | None,
    resolution_bin_idx: int,
    resolution: float,
    range_min: float,
    range_max: float,
) -> None:
    """
    Test the edge cases for the resolution bin index
    An index of 0 means the resolution is BELOW the lower bound
    An index of 1 means the resolution is AT (or very near) the lower bound
    etc
    """
    assert (
        0 <= resolution_bin_idx < 202
    ), f"Resolution bin index is {resolution_bin_idx} which is not between 0 and 201 (i.e. 202 options)"
    if resolution > range_max:
        assert (
            resolution_bin_idx == len(pmf) - 1
        ), f"Resolution bin index is {resolution_bin_idx} which is not the last index. The position in range is {position_in_range}"
    elif resolution < range_min:
        assert (
            resolution_bin_idx == 0
        ), f"Resolution bin index is {resolution_bin_idx} which is not the first index. The position in range is {position_in_range}"
    elif resolution == range_max:
        assert (
            resolution_bin_idx == len(pmf) - 2
        ), f"Resolution bin index is {resolution_bin_idx} which is not the second to last index. The position in range is {position_in_range}"
    elif resolution == range_min:
        assert (
            resolution_bin_idx == 1
        ), f"Resolution bin index is {resolution_bin_idx} which is not the second index. The position in range is {position_in_range}"


def _resolution_value_to_position_in_numeric_range(
    resolution: float,
    range_min: float,
    range_max: float,
    zero_point: float | None = None,
) -> float:
    """
    Takes a location in nominal format (e.g. resolution 176 for a question with bounds 0-500) and scales it to
    metaculus's "internal representation" range [0, 1] incorporating question scaling
    0.8 would incidate the nomial locatoin is at cdf index 201 * 0.8
    Values higher/lower than 0 and 1 are resolutions that are above/below the upper/lower bound
    """
    assert isinstance(
        zero_point, float | None
    ), f"Zero point is {zero_point} which is not a float or None"

    # TODO: Make sure to use datetime.fromisoformat(nominal_location).timestamp() if you start using date questions
    scaled_location = float(resolution)

    # Unscale the value to put it into the range [0,1]
    if zero_point is not None:
        # logarithmically scaled question
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        unscaled_location = (
            np.log(
                (scaled_location - range_min) * (deriv_ratio - 1)
                + (range_max - range_min)
            )
            - np.log(range_max - range_min)
        ) / np.log(deriv_ratio)
    else:
        # linearly scaled question
        unscaled_location = (scaled_location - range_min) / (range_max - range_min)
    return unscaled_location


def _normalize_resolution(
    question_type: QuestionType,
    resolution: ResolutionType,
    range_min: float | None,
    range_max: float | None,
) -> ResolutionType:
    if resolution == "annulled" or resolution == "ambiguous":
        return None

    if question_type == QuestionType.NUMERIC:
        if range_min is None or range_max is None:
            raise ValueError(
                "Range min and range max are required for numeric questions"
            )
        if resolution == "above_upper_bound":
            updated_resolution = range_max + 1.1
            # Adding arbitrary buffer to put the resolution beyond the bounds
        elif resolution == "below_lower_bound":
            updated_resolution = range_min - 1.1
        elif not isinstance(resolution, float):
            try:
                updated_resolution = float(resolution)  # type: ignore
            except Exception:
                raise ValueError(
                    f"Resolution {resolution} could not be cast to float for numeric question."
                )
        else:
            updated_resolution = resolution
        return updated_resolution
    else:
        return resolution


def _determine_question_type(
    question_type: Literal["binary", "multiple_choice", "numeric"] | None,
    resolution: ResolutionType,
) -> QuestionType:
    if question_type is None:
        if isinstance(resolution, bool):
            return QuestionType.BINARY
        elif (
            isinstance(resolution, float)
            or isinstance(resolution, int)
            or resolution == "above_upper_bound"
            or resolution == "below_lower_bound"
        ):
            return QuestionType.NUMERIC
        elif isinstance(resolution, str):
            return QuestionType.MULTIPLE_CHOICE
        else:
            raise ValueError(
                f"Cannot infer question type from resolution. Please provide a question type. Resolution: {resolution}"
            )
    else:
        return QuestionType(question_type)


def cdf_to_pmf(cdf: list[float]) -> list[float]:
    assert len(cdf) == 201, f"There should be 201 bins, but there are {len(cdf)}"
    lower_bound_prob = cdf[0]
    upper_bound_prob = 1 - cdf[-1]
    pmf = (
        [lower_bound_prob]
        + [cdf[i] - cdf[i - 1] for i in range(1, len(cdf))]
        + [upper_bound_prob]
    )
    assert len(pmf) == 202, f"There should be 202 bins, but there are {len(pmf)}"
    return pmf


def pmf_to_cdf(pmf: list[float]) -> list[float]:
    assert len(pmf) == 202, f"There should be 202 bins, but there are {len(pmf)}"
    cdf = []
    total = 0.0
    for p in pmf:
        total += p
        cdf.append(total)
    assert len(cdf) == 201, f"There should be 201 bins, but there are {len(cdf)}"
    return cdf


# HOW TO CALCULATE PEER SCORE W/ GEOMETRIC MEAN AVERAGES
# def calculate_peer_score(
#     forecast: ForecastType,
#     forecast_for_other_users: list[ForecastType],
#     resolution: ResolutionType,
#     options: list[str] | None = None,
#     range_min: float | None = None,
#     range_max: float | None = None,
#     question_weight: float = 1.0,
#     q_type: Literal["binary", "multiple_choice", "numeric"] | None = None,
# ) -> float:
#     num_other_forecasters = len(forecast_for_other_users)
#     if num_other_forecasters <= 0:
#         raise ValueError(f"Number of other forecasts has to be greater than 0")
#     num_forecasters = num_other_forecasters + 1
#     all_forecasts = [forecast] + forecast_for_other_users

#     question_type = _determine_question_type(q_type, resolution)
#     normalized_resolution = _normalize_resolution(
#         question_type, resolution, range_min, range_max
#     )

#     forecast_for_resolution = _determine_probability_for_resolution(
#         question_type, forecast, normalized_resolution, options, range_min, range_max
#     )

#     geometric_mean_prediction = _get_geometric_mean_prediction(
#         question_type=question_type,
#         forecasts=all_forecasts,
#         resolution=normalized_resolution,
#         options=options,
#         range_min=range_min,
#         range_max=range_max,
#     )
#     peer_score = (
#         100
#         * (num_forecasters / (num_forecasters - 1))
#         * np.log(forecast_for_resolution / geometric_mean_prediction)
#     )

#     if question_type == QuestionType.NUMERIC:
#         peer_score /= 2

#     return peer_score * question_weight

# def _get_geometric_mean_prediction(
#     question_type: QuestionType,
#     forecasts: list[ForecastType],
#     resolution: ResolutionType,
#     options: list[str] | None = None,
#     range_min: float | None = None,
#     range_max: float | None = None,
# ) -> float:
#     pmfs = []
#     for forecast in forecasts:
#         assert forecast is not None
#         if question_type == QuestionType.NUMERIC:
#             pmf = cdf_to_pmf(forecast)
#         else:
#             pmf = forecast
#         pmfs.append(pmf)

#     geometric_mean_pmf: list[float] = gmean(forecasts, axis=0)
#     if question_type == QuestionType.NUMERIC:
#         geometric_mean_forecast = pmf_to_cdf(geometric_mean_pmf)
#     else:
#         geometric_mean_forecast = geometric_mean_pmf

#     probability_for_forecast = _determine_probability_for_resolution(
#         question_type,
#         geometric_mean_forecast,
#         resolution,
#         options,
#         range_min,
#         range_max,
#     )

#     return probability_for_forecast
