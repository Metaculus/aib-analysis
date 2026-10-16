"""Typical probability on the outcome: an absolute-accuracy companion to peer scores.

For binary questions this is the weighted geometric mean of the probability a forecaster gave
to the outcome that happened, i.e. the exponentiated log score. It is used instead of "% of
questions on the right side of the base rate" because that measure is improper (it rewards
pushing forecasts past the base rate) and ignores confidence. Because peer scores are log
scores too, a binary peer score of X means about e^(X/100) times the field's probability.
"""

from __future__ import annotations

import math
import statistics

from scipy import stats

CONTINUOUS_PEER_SCORE_DIVISOR = 2


def binary_probability_on_outcome(probability_yes: float, resolved_yes: bool) -> float:
    return probability_yes if resolved_yes else 1 - probability_yes


def geometric_mean_probability(probabilities: list[float], weights: list[float] | None = None) -> float:
    if not probabilities:
        raise ValueError("Need at least one probability")
    weights = weights if weights is not None else [1.0] * len(probabilities)
    if len(weights) != len(probabilities):
        raise ValueError("Probabilities and weights have different lengths")
    if any(probability <= 0 or probability > 1 for probability in probabilities):
        raise ValueError("Probabilities on the outcome must be in (0, 1]")
    total_weight = sum(weights)
    return math.exp(sum(weight * math.log(probability) for probability, weight in zip(probabilities, weights)) / total_weight)


def _exp_t_interval(log_values: list[float], confidence: float) -> tuple[float, float] | None:
    if len(log_values) < 2:
        return None
    half_width = stats.t.ppf(0.5 + confidence / 2, len(log_values) - 1) * statistics.stdev(log_values) / math.sqrt(len(log_values))
    center = statistics.fmean(log_values)
    return math.exp(center - half_width), math.exp(center + half_width)


def geometric_mean_probability_ci(probabilities: list[float], confidence: float = 0.95) -> tuple[float, float] | None:
    """Unweighted t-interval on log probabilities, mapped back with exp."""
    return _exp_t_interval([math.log(probability) for probability in probabilities], confidence)


def paired_probability_ratio(
    numerator_probabilities: list[float], denominator_probabilities: list[float], confidence: float = 0.95
) -> tuple[float, tuple[float, float] | None]:
    """Geometric mean of per-question probability ratios, with an unweighted t-interval on the log ratios."""
    if len(numerator_probabilities) != len(denominator_probabilities) or not numerator_probabilities:
        raise ValueError("Need the same, non-zero number of probabilities for both forecasters")
    log_ratios = [
        math.log(numerator) - math.log(denominator)
        for numerator, denominator in zip(numerator_probabilities, denominator_probabilities)
    ]
    return math.exp(statistics.fmean(log_ratios)), _exp_t_interval(log_ratios, confidence)


def yes_rate(resolutions: list[bool]) -> float:
    if not resolutions:
        raise ValueError("Need at least one resolution")
    return sum(resolutions) / len(resolutions)


def base_rate_geometric_mean_probability(resolutions: list[bool], weights: list[float] | None = None) -> float:
    """Typical probability on the outcome for a forecaster who always predicts the observed Yes rate."""
    observed_yes_rate = yes_rate(resolutions)
    return geometric_mean_probability(
        [binary_probability_on_outcome(observed_yes_rate, resolved_yes) for resolved_yes in resolutions], weights
    )


def field_probability_multiplier(peer_scores: list[float], weights: list[float], is_continuous: list[bool]) -> float:
    """How many times the field's probability a forecaster put on the outcome, from its spot peer scores.

    Spot peer scores are 100 * weight * ln(p / field p), halved for numeric and discrete questions,
    so this undoes the halving and averages the log ratios with the question weights.
    """
    if not peer_scores or not (len(peer_scores) == len(weights) == len(is_continuous)):
        raise ValueError("Need matching, non-empty peer scores, weights, and question kinds")
    log_ratio_total = sum(
        peer_score * (CONTINUOUS_PEER_SCORE_DIVISOR if continuous else 1) / 100
        for peer_score, continuous in zip(peer_scores, is_continuous)
    )
    return math.exp(log_ratio_total / sum(weights))
