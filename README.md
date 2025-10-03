The main entry point to the project is `aib_analysis/main.py`

## Installing Dependencies
To set up the project please run:
Installing dependencies
Make sure you have python and poetry installed (poetry is a python package manager).

If you don't have poetry installed run the below:
```
sudo apt update -y
sudo apt install -y pipx
pipx install poetry

# Optional
poetry config virtualenvs.in-project true
```

Inside the terminal, go to the directory you cloned the repository into and run the following command:

```
poetry install
```

to install all required dependencies.

You may also need to follow instructions [here](https://git-lfs.com/) for setting up git management for large files.

## Running the Analysis

To generate the analysis data run `aib_analysis/run_q1_q1_simulations.py`.

Check the logs for warnings to see edge cases that have come up.

To display the analysis please execute:
```
poetry run streamlit run aib_analysis/front_end.py
```

This will bring up the visuals for the analysis. Make sure you have chosen the right input data in this script. Front end uses caching, so restart streamlit if you update the underlying data.


Make sure to restart the site or click "clear cache" in the triple dot menu on the site if you change the underlying data or simulated_tournament loading/intialization code.


## Structure
The project is focused around the SimulatedTournament object. This is initialized with a number of Forecast objects (see `data_models.py`) and used to create other parts of a tournament (Users, Scores, etc). Every data analysis item we care about is just a tournament of some sort. Often this is a filter, aggregation, intersection, etc of forecasts from another tournament, but even a comparison of 2 people is a tournament.

Refactor advantages
- Pydantic model validation (type safety, and consistency checks at initialization time)
- Easier to unit test
- More reusable (e.g. We can use this structure for regular peer score comparisons in the future)
- Git diffs are easier to read
- We can eventually publish our results as an interactive app if we want to (and if not, we can more easily check and interact with the data ourselves)

