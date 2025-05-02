import numpy as np

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
    raise NotImplementedError("Not implemented")


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

    prob_for_resolution, baseline_prob = (
        _determine_probability_for_resolution_and_baseline(
            forecast, resolution, options, range_min, range_max
        )
    )

    if prob_for_resolution <= 0 or baseline_prob <= 0:
        raise ValueError(
            "Probability for resolution or baseline probability is less than or equal to 0 which could cause a log(0) issue"
        )

    baseline_score = (
        np.log2(prob_for_resolution / baseline_prob) * 100
    )  # @Check: check correctness (also shouldn't this be natural log?)

    if isinstance(resolution, float):
        baseline_score /= 2  # Numeric scores are halved

    weighted_score = baseline_score * question_weight

    return weighted_score


def _determine_probability_for_resolution_and_baseline(
    forecast: ForecastType,
    resolution: ResolutionType,
    options: list[str] | None = None,
    range_min: float | None = None,
    range_max: float | None = None,
) -> tuple[float, float]:

    is_binary = isinstance(resolution, bool)
    is_multiple_choice = isinstance(resolution, str)
    is_numeric = isinstance(resolution, float) or isinstance(resolution, int)

    if forecast is None or resolution is None:
        raise NotImplementedError(
            "Havent decided how to handle null forecasts or anulled resolutions"
        )

    if len(forecast) == 0:
        raise ValueError("Forecast is empty")

    if not is_numeric and any(p <= 0 or p >= 1 for p in forecast):
        # @Check: Is it valid to have a numeric forecast with 0 probability for a number?
        raise ValueError("Forecast contains probabilities outside of 0 to 1 range")

    if is_binary:
        if len(forecast) != 1 and len(forecast) != 2:
            raise ValueError(
                "Binary questions must have exactly one or two forecasts (for yes or 'yes and no')"
            )

        forecast_val = float(forecast[0])
        baseline_prob = 0.5
        if resolution:
            prob_for_resolution = forecast_val
        else:
            prob_for_resolution = 1 - forecast_val
    elif is_multiple_choice:
        if options is None:
            raise ValueError("Options are required for multiple choice questions")

        if len(forecast) != len(options):
            raise ValueError("Forecast and options have different lengths")

        pmf = [float(p) for p in forecast]
        options = [str(opt) for opt in options]
        resolution_idx = options.index(str(resolution))
        prob_for_resolution = pmf[resolution_idx]
        baseline_prob = 1 / len(pmf)
    elif is_numeric:
        if range_min is None or range_max is None:
            raise ValueError(
                "Range min and range max are required for numeric questions"
            )
        if len(forecast) != 201:
            raise ValueError("CDF should have 201 bins")
        previous_prob = 0
        for current_prob in forecast:
            if current_prob < previous_prob:
                raise ValueError("CDF should be in increasing order")
            previous_prob = current_prob

        cdf = [float(p) for p in forecast]
        pmf = [cdf[0]] + [
            cdf[i] - cdf[i - 1] for i in range(1, len(cdf))
        ]  # @Check: is this a correct conversion?
        pmf.append(1 - cdf[-1])

        resolution = float(resolution)

        bin_edges = np.linspace(range_min, range_max, 200)
        resolution_idx = np.searchsorted(bin_edges, resolution, side="right")

        if resolution_idx >= len(pmf):
            raise ValueError("Resolution is out of bounds")

        prob_for_resolution = pmf[resolution_idx]
        baseline_prob = 1 / len(
            pmf
        )  # bins = 201 because of extra appended bin # @Check: This comment seems off since its the cdf that has 201 bins

    else:
        raise ValueError("Unknown question type")

    return prob_for_resolution, baseline_prob
