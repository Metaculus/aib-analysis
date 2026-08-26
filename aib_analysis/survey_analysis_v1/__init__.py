"""Bot-maker survey analysis pipeline.

Usage:
    poetry run python aib_analysis/survey_analysis_v1/run_survey_analysis.py --season spring-2026

To analyse a new season, add a `SeasonConfig` to `seasons.py`. The rest of the
pipeline is season-agnostic.
"""

from aib_analysis.survey_analysis_v1.analysis import AnalysisResults, run_analysis
from aib_analysis.survey_analysis_v1.config import SeasonConfig
from aib_analysis.survey_analysis_v1.loading import Dataset, load_dataset

__all__ = [
    "AnalysisResults",
    "Dataset",
    "SeasonConfig",
    "load_dataset",
    "run_analysis",
]
