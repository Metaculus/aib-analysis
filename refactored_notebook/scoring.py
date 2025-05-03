from datetime import datetime
import numpy as np
from scipy.stats.mstats import gmean

from refactored_notebook.data_models import ForecastType, ResolutionType


def calculate_spot_peer_score(
    forecast: ForecastType,
    forecast_for_other_users: list[ForecastType],
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    question_weight: float = 1.0,
) -> float:
    forecast_for_resolution = _determine_probability_for_resolution(
        forecast, resolution, options, range_min, range_max
    )
    other_user_forecasts = [
        _determine_probability_for_resolution(
            forecast, resolution, options, range_min, range_max
        )
        for forecast in forecast_for_other_users
    ]

    geometric_mean = gmean(other_user_forecasts)
    peer_score = np.log(forecast_for_resolution / geometric_mean)
    if isinstance(resolution, float):  # @Check: shouldn't other q types get a divsor?
        peer_score /= 2
    return peer_score * question_weight


def nominal_location_to_cdf_location(
    nominal_location: float,
    range_min: float,
    range_max: float,
    zero_point: float | None = None,
) -> float:
    """
    Takes a location in nominal format (e.g. 123, "123", or datetime in iso format) and scales it to
    metaculus's "internal representation" range [0, 1] incorporating question scaling
    0.8 would incidate the nomial locatoin is at cdf index 201 * 0.8
    Values higher/lower than 0 and 1 are resolutions that are above/below the upper/lower bound
    """
    assert isinstance(zero_point, float | None)

    # TODO: Make sure to use datetime.fromisoformat(nominal_location).timestamp() if you start using date questions
    scaled_location = float(nominal_location)

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


def calculate_spot_baseline_score(
    forecast: ForecastType,
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
    question_weight: float = 1.0,
) -> float:
    """
    Question type can be infered from resolution type
    Scoring math: https://www.metaculus.com/help/scores-faq/#What:~:text=given%20score%20type.-,What%20is%20the%20Baseline%20score%3F,-The%20Baseline%20score
    """

    prob_for_resolution = _determine_probability_for_resolution(
        forecast, resolution, options, range_min, range_max
    )
    baseline_prob = _determine_baseline(resolution, options)
    divisor = _determine_divisor_for_baseline_score(resolution, options)
    if prob_for_resolution <= 0 or baseline_prob <= 0:
        raise ValueError(
            "Probability for resolution or baseline probability is less than or equal to 0 which could cause a log(0) issue"
        )

    # if resolution_bucket in [0, len(pmf) - 1]:
    #     baseline = 0.05
    # else:
    #     open_bound_count = bool(question.open_upper_bound) + bool(
    #     question.open_lower_bound
    # )
    #     baseline = (1 - 0.05 * open_bounds_count) / (len(pmf) - 2)
    # forecast_score = 100 * np.log(pmf[resolution_bucket] / baseline) / 2

    baseline_score = np.log(prob_for_resolution / baseline_prob) / divisor * 100

    weighted_score = baseline_score * question_weight

    return weighted_score


def _determine_baseline(
    resolution: ResolutionType, options: list[str] | None = None
) -> float:
    is_binary = isinstance(resolution, bool)
    is_multiple_choice = isinstance(resolution, str)
    is_numeric = isinstance(resolution, float) or isinstance(resolution, int)

    if is_binary:
        baseline_prob = 0.5
    elif is_multiple_choice:
        if options is None:
            raise ValueError("Options are required for multiple choice questions")
        baseline_prob = 1 / len(options)
    elif is_numeric:
        baseline_prob = (
            1 / 202
        )  # len(pmf) # ??? -> bins = 201 because of extra appended bin # @Check: This comment seems off since its the cdf that has 201 bins
        # @Check: Should this be either 1, 0.9, or 0.95 based on whether open or closed bounds
    else:
        raise ValueError("Unknown question type")
    assert (
        0 <= baseline_prob <= 1
    ), f"Baseline probability is {baseline_prob} which is not between 0 and 1"
    return baseline_prob


def _determine_probability_for_resolution(
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

    if resolution == "above_upper_bound" or resolution == "below_lower_bound":
        raise ValueError(
            "'above_upper_bound' or 'below_lower_bound' format not supported"
        )

    is_numeric = isinstance(resolution, float) or isinstance(resolution, int)
    is_binary = isinstance(resolution, bool)
    is_multiple_choice = isinstance(resolution, str)

    if forecast is None or resolution is None:
        raise NotImplementedError(
            "Havent decided how to handle null forecasts or anulled resolutions"
        )

    if len(forecast) == 0:
        raise ValueError("Forecast is empty")

    if not is_numeric and any(p <= 0 or p >= 1 for p in forecast):
        raise ValueError("Forecast contains probabilities outside of 0 to 1 range")

    if is_binary:
        prob_for_resolution = _binary_resolution_prob(forecast, resolution)
    elif is_multiple_choice:
        if options is None:
            raise ValueError("Options are required for multiple choice questions")
        prob_for_resolution = _multiple_choice_resolution_prob(
            forecast, resolution, options
        )
    elif is_numeric:
        if range_min is None or range_max is None:
            raise ValueError(
                "Range min and range max are required for numeric questions"
            )
        prob_for_resolution = _numeric_resolution_prob(
            forecast, resolution, range_min, range_max
        )
    else:
        raise ValueError("Unknown question type")

    assert (
        0 <= prob_for_resolution <= 1
    ), f"Probability for resolution is {prob_for_resolution} which is not between 0 and 1"
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
    options = [str(opt) for opt in options]
    resolution_idx = options.index(str(resolution))
    prob_for_resolution = pmf[resolution_idx]
    return prob_for_resolution


def _numeric_resolution_prob(
    forecast: list[float], resolution: float | str, range_min: float, range_max: float
) -> float:
    if len(forecast) != 201:
        raise ValueError("CDF should have 201 bins")

    previous_prob = 0
    for current_prob in forecast:
        if current_prob < previous_prob:
            raise ValueError("CDF should be in increasing order")
        previous_prob = current_prob

    cdf = [float(p) for p in forecast]
    assert len(cdf) == 201, f"There should be 201 bins, but there are {len(cdf)}"
    lower_bound_prob = cdf[0]
    upper_bound_prob = 1 - cdf[-1]
    pmf = (
        [lower_bound_prob]
        + [cdf[i] - cdf[i - 1] for i in range(1, len(cdf))]
        + [upper_bound_prob]
    )  # @Check: is this a correct conversion?
    # pmf = np.diff(np.concatenate([[0], cdf]))
    assert len(pmf) == 202, f"There should be 202 bins, but there are {len(pmf)}"

    resolution = float(resolution)
    # bin_edges = np.linspace(range_min, range_max, 200)
    # resolution_bin_idx = np.searchsorted(bin_edges, resolution, side="right")
    cdf_location = nominal_location_to_cdf_location(resolution, range_min, range_max)
    resolution_bin_idx = min(int(cdf_location * (len(pmf) - 1)), len(pmf) - 1)

    if resolution_bin_idx >= len(pmf):
        raise ValueError("Resolution is out of bounds")

    prob_for_resolution = pmf[resolution_bin_idx]

    return prob_for_resolution


def _determine_divisor_for_baseline_score(
    resolution: ResolutionType, options: list[str] | None = None
) -> float:
    is_binary = isinstance(resolution, bool)
    is_multiple_choice = isinstance(resolution, str)
    is_numeric = isinstance(resolution, float) or isinstance(resolution, int)

    if is_binary:
        return np.log(2)
    elif is_multiple_choice:
        if options is None:
            raise ValueError("Options are required for multiple choice questions")
        return np.log(len(options))
    elif is_numeric:
        return 2
    else:
        raise ValueError("Unknown question type")
