"""
There are 3 questions of:
- How many arms sales globally will the US State Department approve in March 2025?

Bot Dataclip:
420 unique questions
424 rows
Duplicates:
- What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?
    - There is one of weight 1 and one of weight 0.5. They have different ids.
- How many arms sales globally will the US State Department approve in March 2025?
    - Different ids
    - https://www.metaculus.com/questions/34260/ vs https://www.metaculus.com/questions/34706/
- Which party will win the most seats in Curaçao in the March 2025 general election?
    - different resolutions (one annulled?)
    - https://www.metaculus.com/questions/35994/ vs https://www.metaculus.com/questions/35892/
- Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace?
Unique to dataclip:
- How many Oscars will A Complete Unknown win in 2025?
- (see candidates)

Sheet
420 unique questoins
422 rows
- What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?
- How many arms sales globally will the US State Department approve in March 2025?
Unique to sheet:
- This counts awards either won by the movie itself (such as Best Picture) or awards won by someone who worked on the movie (such as Timothée Chalame for Actor in a Leading Role).
- (see candidates)

Q1:
???

Problem question candidates:
    1st: Dataclip, 2nd: Sheet
    Will the reported rate of incidents of unruly passengers per 10,000 flights reported by the FAA exceed the long-term average for any week before April 1, 2025?
    Will tee reported rate of incidents of unruly passengers per 10,000 flights reported by the FAA exceeds the long-term average for any week before April 1, 2025?

    1st: Dataclip, 2nd: Sheet
    Before March 15, 2025, will Reform UK be the highest polling party in the UK by at least 2 points, according to Politico?
    Before March 1, 2025, kill Reform UK be the highest polling party in the UK by at least 2 points, according to Politico?
"""


poor_questions = [
    "How many Grammy awards will Taylor Swift win in 2025?",  # Pro/Bot question have different options (but the one that resolved was the same)
    "Which party will win the 2nd highest number of seats in the 2025 German federal election?",  # Same as above
    "What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?",  # The spot scoring time is different for bot/pro question (but only off by 2 days).
]

problem_questions = [
    "How many arms sales globally will the US State Department approve in March 2025?",  # The options for the pro vs bot questions are different, and different options resolved. Also the spot scoring time is off by 1.2 weeks.
    # https://www.metaculus.com/questions/34706/ vs https://www.metaculus.com/questions/34382/
]
