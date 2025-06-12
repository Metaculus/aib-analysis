from pydantic import BaseModel
import numpy as np
from scipy.stats import t, shapiro

class ConfidenceInterval(BaseModel):
    mean: float
    margin_of_error: float
    standard_deviation: float

    @property
    def lower_bound(self) -> float:
        return self.mean - self.margin_of_error

    @property
    def upper_bound(self) -> float:
        return self.mean + self.margin_of_error


class ConfidenceIntervalCalculator:

    @classmethod
    def confidence_interval_from_observations(
        cls, observations: list[float], confidence: float = 0.9
    ) -> ConfidenceInterval:
        """
        This solves the following stats problem:
        'estimating population mean with unknown population standard deviation'

        Requirements
        - Simple random sample
        - Either the sample is from a normally distributed population or n >30
        - Observations are independent
        """
        assert 0 <= confidence <= 1, "Confidence must be between 0 and 1"
        assert len(observations) > 0, "Observations must be non-empty"
        assert (
            len(observations) > 3
        ), "Must have at least 3 observations to check for normality"

        sample_size = len(observations)
        if sample_size < 2:
            raise ValueError("Not enough data for T-based confidence interval")

        if sample_size < 30:
            _, normality_pvalue = shapiro(observations)
            if normality_pvalue < 0.05:
                raise ValueError(
                    "Data fails normality assumption for T-based confidence interval"
                )

        sample_mean = np.mean(observations)
        sample_std = np.std(observations, ddof=1)

        return cls.confidence_interval_from_mean_and_std(
            float(sample_mean), float(sample_std), sample_size, confidence
        )

    @classmethod
    def confidence_interval_from_mean_and_std(
        cls,
        sample_mean: float,
        sample_std: float,
        sample_size: int,
        confidence: float,
    ) -> ConfidenceInterval:
        standard_error = sample_std / np.sqrt(sample_size)
        alpha = 1 - confidence
        critical_value = t.ppf(1 - alpha / 2, sample_size - 1)
        margin_of_error = critical_value * standard_error

        return ConfidenceInterval(
            mean=float(sample_mean),
            margin_of_error=margin_of_error,
            standard_deviation=float(sample_std),
        )
