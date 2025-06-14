from __future__ import annotations

from pydantic import BaseModel

from aib_analysis.data_structures.data_models import Question

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


"""
# TODO: Always keep the one that matches pro/quarterly cup more
# TODO: Always keep the non annulled question
# TODO: Always keep the later version (spot score? creation time?)
# TODO: Remove one if they are exactly the same (no differences)
# Do the above if 'smart-deduplicate' is true
# Don't log a warning if the ProblemQuestion is in manually listed and has the correct ID

# TODO: 5-9 version matches pro question, 0-5 matches site question
    # "How many arms sales globally will the US State Department approve in March 2025?",  # The options for the pro vs bot questions are different, and different options resolved. Also the spot scoring time is off by 1.2 weeks.
    # https://www.metaculus.com/questions/34706/ vs https://www.metaculus.com/questions/34382/
# Duplicates for question text: How many arms sales globally will the US State Department approve in March 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/34260/ | https://www.metaculus.com/questions/34706/ |
| Question Id | 33757 | 34220 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many arms sales globally will the US State Department approve in March 2025? | How many arms sales globally will the US State Department approve in March 2025? |
| Resolution | 5-9 | 0-5 |
| Options | ['0-4', '5-9', '>9'] | ['0-5', '6-10', '>10'] |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 34260 | 34706 |
| Created At | 2025-01-25 06:31:51.259600+00:00 | 2025-02-01 05:24:04.045627+00:00 |
| Spot Scoring Time | 2025-01-29 07:00:00+00:00 | 2025-02-09 00:44:00+00:00 |

# TODO: Use second one (b/c matches quarterly cup)
# Duplicates for question text: What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/34281/ | https://www.metaculus.com/questions/34667/ |
| Question Id | 33778 | 34181 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? |
| Resolution | 3rd | 3rd |
| Options | ['1st', '2nd', '3rd', '4th', '≥5th'] | ['1st', '2nd', '3rd', '4th', '≥5th'] |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 0.5 |
| Post Id | 34281 | 34667 |
| Created At | 2025-01-25 06:31:52.795962+00:00 | 2025-02-01 05:24:00.456127+00:00 |
| Spot Scoring Time | 2025-01-31 02:00:00+00:00 | 2025-02-02 17:00:00+00:00 |

# TODO: Take second (b/c later and not annulled)
# Duplicates for question text: Which party will win the most seats in Curaçao in the March 2025 general election?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/35892/ | https://www.metaculus.com/questions/35994/ |
| Question Id | 35326 | 35426 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | Which party will win the most seats in Curaçao in the March 2025 general election? | Which party will win the most seats in Curaçao in the March 2025 general election? |
| Resolution | None | Movement for the Future of Curaçao |
| Options | ['Movement for the Future of Curaçao', 'Real Alternative Party', 'Another outcome'] | ['Movement for the Future of Curaçao', 'Real Alternative Party', 'Another outcome'] |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 35892 | 35994 |
| Created At | 2025-03-08 04:57:09.780762+00:00 | 2025-03-11 14:35:21.855687+00:00 |
| Spot Scoring Time | 2025-03-10 12:00:00+00:00 | 2025-03-12 12:00:00+00:00 |

# TODO: Keep the second one (b/c later and not annulled)
# Duplicates for question text: Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/36161/ | https://www.metaculus.com/questions/36264/ |
| Question Id | 35598 | 35705 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace? | Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace? |
| Resolution | None | Candace |
| Options | ['The New York Times Daily', 'The Tucker Carlson Show'] | ['Call Her Daddy', 'Candace'] |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 36161 | 36264 |
| Created At | 2025-03-15 15:49:27.084578+00:00 | 2025-03-20 19:35:15.771896+00:00 |
| Spot Scoring Time | 2025-03-18 20:00:00+00:00 | 2025-03-20 20:00:00+00:00 |
"""


class ProblemQuestion(BaseModel):
    question_text: str
    urls: list[str]
    notes: str
    proposed_action: str | None = None

    def question_matches(self, question: Question) -> bool:
        post_id = str(question.post_id)
        input_url = f"https://www.metaculus.com/questions/{post_id}/"

        input_question_text = question.question_text
        problem_question_text = self.question_text
        text_matches = input_question_text.strip() == problem_question_text.strip()
        url_matches = input_url in self.urls

        if text_matches and url_matches:
            return True
        elif not text_matches and not url_matches:
            return False
        else:
            raise ValueError(
                f"Input Question {input_url} matches some parts of problem but not all | "
                f"Input Question_text: {question.question_text} | "
                f"Problem Question_text: {self.question_text} | "
                f"Problem: {self.model_dump_json()}"
            )


class ProblemManager:

    _prequalified_questions_when_matching_tournaments: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text="How many Grammy awards will Taylor Swift win in 2025?",
            urls=[
                "https://www.metaculus.com/questions/31797/",
                "https://www.metaculus.com/questions/31865/",
            ],
            notes="Pro/Bot question have different options (but the one that resolved was the same)",
        ),
        ProblemQuestion(
            question_text="Which party will win the 2nd highest number of seats in the 2025 German federal election?",
            urls=[
                "https://www.metaculus.com/questions/35002/",
                "https://www.metaculus.com/questions/34940/",
            ],
            notes="Pro/Bot question have different options (but the one that resolved was the same)",
        ),
        ProblemQuestion(
            question_text="What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?",
            urls=[
                "https://www.metaculus.com/questions/34389/",
                "https://www.metaculus.com/questions/34281/",
                "https://www.metaculus.com/questions/34667/"
            ],
            notes=(
                "The spot scoring time is different for one of the bot/pro questions "
                "(but only off by 2 days). There are 2 bots and 2 pros"
            )
        ),
    ]

    _q1_bot_tournament_duplicates: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text="How many arms sales globally will the US State Department approve in March 2025?",
            urls=[
                "https://www.metaculus.com/questions/34260/",
                "https://www.metaculus.com/questions/34706/",
            ],
            notes="Different options and resolutions: first has ('0-4', '5-9', '>9') resolved to 5-9, second has ('0-5', '6-10', '>10') resolved to 0-5. They were launched a week apart (so tests updating)",
            proposed_action="Keep this, since it tests bot's ability to update",
        ),
        ProblemQuestion(
            question_text="What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?",
            urls=[
                "https://www.metaculus.com/questions/34281/",
                "https://www.metaculus.com/questions/34667/",
            ],
            notes="Same options and resolution (3rd), but different weights (1.0 vs 0.5) and spot scoring times (off by ~2 days). Accidental rerelease",
            proposed_action="Remove this",
        ),
        ProblemQuestion(
            question_text="Which party will win the most seats in Curaçao in the March 2025 general election?",
            urls=[
                "https://www.metaculus.com/questions/35892/",
                "https://www.metaculus.com/questions/35994/",
            ],
            notes="Same options but different resolutions: first unresolved, second resolved to 'Movement for the Future of Curaçao'. Spot scoring time 2 days off. First was annulled",
            proposed_action="Leave this. The first one was annulled",
        ),
        ProblemQuestion(
            question_text="Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace?",
            urls=[
                "https://www.metaculus.com/questions/36161/",
                "https://www.metaculus.com/questions/36264/",
            ],
            notes="Completely different options: first has ('The New York Times Daily', 'The Tucker Carlson Show') resolved to None, second has ('Call Her Daddy', 'Candace') and resolved to 'Candace'. Spot scoring time 2 days off",
            proposed_action="Leave this. The first one was annulled",
        ),
    ]


    @classmethod
    def is_prequalified_in_tournament_matching(cls, question_1: Question, question_2: Question) -> bool:
        question_1_match = False
        question_2_match = False
        for q in ProblemManager._prequalified_questions_when_matching_tournaments:
            if q.question_matches(question_1):
                question_1_match = True
            if q.question_matches(question_2):
                question_2_match = True
        if question_1_match != question_2_match:
            raise ValueError(f"If question 1 is a problem question, then question 2 should be one as well. Question 1: {question_1.url}, Question 2: {question_2.url}")
        return question_1_match or question_2_match


    @classmethod
    def is_prequalified_duplicate_within_tournament(cls, questions: list[Question]) -> bool:
        for q in cls._q1_bot_tournament_duplicates:
            matches = [q.question_matches(question) for question in questions]
            if all(matches):
                return True
            elif any(matches):
                raise ValueError(f"One of the questions matches the problem question, but not all of them. Problem question: {q.question_text}, Questions: {questions}")
            else:
                continue
        return False

"""
##################### Q1 Duplicate Question - Bot Tournament #####################
# Duplicates for question text: How many arms sales globally will the US State Department approve in March 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/34260/ | https://www.metaculus.com/questions/34706/ |
| Question Id | 33757 | 34220 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many arms sales globally will the US State Department approve in March 2025? | How many arms sales globally will the US State Department approve in March 2025? |
| Resolution | 5-9 | 0-5 |
| Options | ('0-4', '5-9', '>9') | ('0-5', '6-10', '>10') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 34260 | 34706 |
| Created At | 2025-01-25 06:31:51.259600+00:00 | 2025-02-01 05:24:04.045627+00:00 |
| Spot Scoring Time | 2025-01-29 07:00:00+00:00 | 2025-02-09 00:44:00+00:00 |
| Notes | None | None |


# Duplicates for question text: What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/34281/ | https://www.metaculus.com/questions/34667/ |
| Question Id | 33778 | 34181 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? |
| Resolution | 3rd | 3rd |
| Options | ('1st', '2nd', '3rd', '4th', '≥5th') | ('1st', '2nd', '3rd', '4th', '≥5th') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 0.5 |
| Post Id | 34281 | 34667 |
| Created At | 2025-01-25 06:31:52.795962+00:00 | 2025-02-01 05:24:00.456127+00:00 |
| Spot Scoring Time | 2025-01-31 02:00:00+00:00 | 2025-02-02 17:00:00+00:00 |
| Notes | None | None |


# Duplicates for question text: Which party will win the most seats in Curaçao in the March 2025 general election?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/35892/ | https://www.metaculus.com/questions/35994/ |
| Question Id | 35326 | 35426 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | Which party will win the most seats in Curaçao in the March 2025 general election? | Which party will win the most seats in Curaçao in the March 2025 general election? |
| Resolution | None | Movement for the Future of Curaçao |
| Options | ('Movement for the Future of Curaçao', 'Real Alternative Party', 'Another outcome') | ('Movement for the Future of Curaçao', 'Real Alternative Party', 'Another outcome') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 35892 | 35994 |
| Created At | 2025-03-08 04:57:09.780762+00:00 | 2025-03-11 14:35:21.855687+00:00 |
| Spot Scoring Time | 2025-03-10 12:00:00+00:00 | 2025-03-12 12:00:00+00:00 |
| Notes | None | None |


# Duplicates for question text: Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/36161/ | https://www.metaculus.com/questions/36264/ |
| Question Id | 35598 | 35705 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace? | Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace? |
| Resolution | None | Candace |
| Options | ('The New York Times Daily', 'The Tucker Carlson Show') | ('Call Her Daddy', 'Candace') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 36161 | 36264 |
| Created At | 2025-03-15 15:49:27.084578+00:00 | 2025-03-20 19:35:15.771896+00:00 |
| Spot Scoring Time | 2025-03-18 20:00:00+00:00 | 2025-03-20 20:00:00+00:00 |
| Notes | None | None |

"""