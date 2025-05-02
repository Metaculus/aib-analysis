
# TODO:
# For each of Multiple Choice, Binary, and Numeric questions
# - Test spot peer score
#   - forecast this is further away than others gets worse scores (with 1-5 forecasts)
#   - forecast this is closer to the resolution gets better scores (with 1-5 forecasts)
#   - If everyone has the same forecast, the score is 0
#   - The sum (average?) of everyone's scores is 0
#   - The score for a weighted question is weighted by the question weight
# - Test spot baseline score
#   - 0 with 50% forecast, ? for a uniform distribution, and 0 for uniform multiple choice questions
#   - better score when closer to resolution, and worse when further away (for forecasts on both sides of 50% forecast)
#   - The score for a weighted question is weighted by the question weight
# - Run a test of some forecasts from the site, and make sure the score generated matches the score the site gives
