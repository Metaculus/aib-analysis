"""Bot-maker survey analysis pipeline.

Usage:
    poetry run python aib_analysis/run_survey_analysis.py --season spring-2026

To analyse a new season, add a `SeasonConfig` to `seasons.py`. The rest of the
pipeline is season-agnostic.
"""

from aib_analysis.survey_analysis.analysis import AnalysisResults, run_analysis
from aib_analysis.survey_analysis.config import SeasonConfig
from aib_analysis.survey_analysis.loading import Dataset, load_dataset

__all__ = [
    "AnalysisResults",
    "Dataset",
    "SeasonConfig",
    "load_dataset",
    "run_analysis",
]
