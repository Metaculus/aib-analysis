from __future__ import annotations
from pydantic import BaseModel
from aib_analysis.data_structures.data_models import Question
from enum import Enum
import json

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

class ProblemType(Enum):
    BETWEEN_TOURNAMENT = "between_tournament"
    WITHIN_TOURNAMENT = "within_tournament"

class SolutionType(Enum):
    FORCE_MATCH = "force_match" # For "Between tournament" problem
    REMOVE = "remove" # For "Between tournament" problem
    TAKE_LATEST = "take_latest" # For "Within tournament" problem
    TAKE_INDEX = "take_index" # For "Within tournament" problem

class ProblemGroup(BaseModel):
    notes: str
    questions: list[Question]
    problem_type: ProblemType
    solution_type: SolutionType | None = None
    index: int | None = None

    def solve_problem_group(self, questions_to_choose_from: list[Question]) -> Question | None:
        if self.solution_type is None:
            raise ValueError("Solution type is not set")
        elif self.solution_type == SolutionType.FORCE_MATCH:
            return self.questions[0]
        elif self.solution_type == SolutionType.REMOVE:
            return None
        elif self.solution_type == SolutionType.TAKE_LATEST:
            sorted_questions = sorted(questions_to_choose_from, key=lambda x: x.created_at)
            return sorted_questions[-1]
        elif self.solution_type == SolutionType.TAKE_INDEX:
            assert self.index is not None, "Index is not set"
            return questions_to_choose_from[self.index]
        else:
            raise ValueError(f"Invalid solution type: {self.solution_type}")

class ProblemManager:
    PROBLEM_GROUP_FILE_PATH = "problem_questions.json"
    _problem_groups: list[ProblemGroup] | None = None

    @classmethod
    def _load_problem_groups(cls) -> list[ProblemGroup]:
        with open(cls.PROBLEM_GROUP_FILE_PATH, "r") as f:
            json_data = json.load(f)
        return [ProblemGroup(**problem_group) for problem_group in json_data]

    @classmethod
    def find_matching_problem_group(cls, input_questions: list[Question]) -> ProblemGroup | None:
        if cls._problem_groups is None:
            cls._problem_groups = cls._load_problem_groups()
        matches = []
        for problem_group in cls._problem_groups:
            problem_group_question_titles = set([question.question_text for question in problem_group.questions])
            input_question_titles = set([question.question_text for question in input_questions])
            if problem_group_question_titles == input_question_titles:
                assert set(problem_group.questions) == set(input_questions), "Question titles match, but questions do not"
                matches.append(problem_group)
        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches[0]
        else:
            raise ValueError(f"Multiple problem groups found for {input_questions}")

    @classmethod
    def save_problem_group(cls, problem_group: ProblemGroup):
        assert cls._problem_groups is not None, "Problem groups must be loaded first"
        cls._problem_groups.append(problem_group)
        with open(cls.PROBLEM_GROUP_FILE_PATH, "w") as f:
            json.dump([problem_group.model_dump() for problem_group in cls._problem_groups], f)

poor_questions = [
    "How many Grammy awards will Taylor Swift win in 2025?",  # Pro/Bot question have different options (but the one that resolved was the same)
    "Which party will win the 2nd highest number of seats in the 2025 German federal election?",  # Same as above
    "What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?",  # The spot scoring time is different for bot/pro question (but only off by 2 days).
]

problem_questions = [
    # "How many arms sales globally will the US State Department approve in March 2025?",  # The options for the pro vs bot questions are different, and different options resolved. Also the spot scoring time is off by 1.2 weeks.
    # https://www.metaculus.com/questions/34706/ vs https://www.metaculus.com/questions/34382/
]
