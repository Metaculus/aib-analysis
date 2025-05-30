import pytest
from archived.functions import *

def test_get_median_forecast_multiple_choice():
  test_row = {'options': ['0-4', '5-9', '>9'], 'resolution': '0-4'}
  test_forecasts = [
    [0.1, 0.2, 0.7],
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3]
  ]

  result = get_median_forecast_multiple_choice(test_row, test_forecasts)
  assert result == 0.2, "Expected median forecast for '0-4' to be 0.2"