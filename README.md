## Installing Dependencies

Make sure you have Python and Poetry installed (Poetry is a Python package manager).

If you don't have Poetry installed:
```
sudo apt update -y
sudo apt install -y pipx
pipx install poetry

# Optional
poetry config virtualenvs.in-project true
```

From the repo root:
```
poetry install
```

to install all required dependencies.

## Input data

Forecast CSVs live under `local/private_input_data/` (gitignored). For Spring 2026 the simulation script expects:

- `local/private_input_data/pro_forecasts_2026_spring.csv`
- `local/private_input_data/bot_forecasts_2026_spring.csv`

You can:

- If you have access, download prior analysis outputs (and sometimes inputs) from [this Google Drive folder](https://drive.google.com/drive/folders/1m7e8AQd4M-Y4oPuj--dDAEYwxO-J2kth?usp=drive_link), or
- Request access to analysis data from the instructions shared [here](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#what-data-do-i-have-access-to-via-api-how-can-i-get-access-to-more) which will show you how to use the [Metaculus Data Needs Form](https://docs.google.com/forms/d/e/1FAIpQLSeJhtZzHl5qMvBjbXbatyaqoS4IU7RE0GGw_vlhs6I9syqn1g/viewform?usp=pp_url&entry.192763438=https://www.metaculus.com/api/)

## Running the analysis

For the current (Spring 2026) tournament:

```
poetry run python aib_analysis/run_spring_simulations.py
```

That writes JSON artifacts to `local/spring_2026_simulations_teams_comparison/`.

Check `logs/latest_info.log` for warnings (annulled questions, weight mismatches, unmatched questions, missing control bots, etc.).

For older seasons (Q1–Q4), use `aib_analysis/run_q1_q2_simulations.py` and edit the `__main__` paths/folders as needed.

## Viewing results

```
poetry run streamlit run aib_analysis/front_end.py
```

In the UI, set the tournaments folder to the simulation output path (e.g. `local/spring_2026_simulations_teams_comparison`). Restart Streamlit or use **Clear cache** in the ⋮ menu after changing underlying JSON or loading/scoring code.

## Structure

The project is built around `SimulatedTournament` (see `data_models.py` / `simulated_tournament.py`). It is initialized from `Forecast` objects and derives users, scores, etc. Most analyses are filtered, aggregated, or combined tournaments built from those forecasts.
