from refactored_notebook.data_models import User, Question, Forecast, Score


# Generate test csvs to input into the notebook, and assert the below tests pass

# Things that could go wrong:
# - bad math in scoring
# - didn't load in data correctly
# - bad filtering/manipulation of scoring data (did we take out the right people)
#    - make sure to determine the bot team only by the bot-only questions
#    - make sure best bot team is decided by baseline score comparison to each other
#    - make sure best bots for bot team are decided by lower bound of t test
#    - make sure that worse bots come out on bottom
# - Confidence interval code is wrong
#   - make sure that there are large intervals if only a few forecasts, and small intervals if many forecasts
#   - make sure bootstrap and t tests indicate the same things generally
# ... continue through and consider other final outputs (e.g. calibration curve)


