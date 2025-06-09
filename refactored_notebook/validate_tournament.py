from collections import defaultdict

from refactored_notebook.data_models import Forecast, ScoreType
from refactored_notebook.simulated_tournament import SimulatedTournament


def validate_simulated_tournament(tournament: SimulatedTournament):
    _validate_forecast_uniqueness(tournament.forecasts)


def _validate_forecast_uniqueness(forecasts: list[Forecast]) -> None:
    id_to_forecasts = defaultdict(list)
    for forecast in forecasts:
        id_to_forecasts[forecast.id].append(forecast)
    duplicate_ids = [
        fid for fid, items in id_to_forecasts.items() if len(items) > 1
    ]
    if len(duplicate_ids) > 0:
        raise ValueError(
            f"Forecasts should be unique, but found duplicate ids: {duplicate_ids}"
        )

# def _test_caches(self) -> None:
#     assert len(self._question_to_forecast_cache.values()) == len(self.questions)
#     for forecast_list in self._question_to_forecast_cache.values():
#         assert (
#             len(forecast_list) >= len(self.users) / 2
#         ), "Heuristic: something is up if less than half of people forecast on a question"

#         unique_forecasts = set([forecast.id for forecast in forecast_list])
#         assert len(unique_forecasts) == len(
#             forecast_list
#         ), "Forecasts should be unique"

#     assert len(self._question_to_spot_forecasts_cache.values()) == len(
#         self.questions
#     )
#     for forecast_list in self._question_to_spot_forecasts_cache.values():
#         for forecast in forecast_list:
#             assert (
#                 forecast in self._spot_forecasts_cache
#             ), "Forecast should be in spot forecasts cache"

#         unique_forecasts = set([forecast.id for forecast in forecast_list])
#         assert len(unique_forecasts) == len(
#             forecast_list
#         ), "Forecasts should be unique"
