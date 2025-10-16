import logging
from enum import Enum

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from aib_analysis.data_structures.data_models import Question

logger = logging.getLogger(__name__)


class Tournament(Enum):
    Q3_2024 = "q3_2024"
    Q4_2024 = "q4_2024"
    Q1_2025 = "q1_2025"
    Q1_2025_VS_CUP = "q1_2025_vs_cup"
    Q2_2025 = "q2_2025"
    Fall_2025 = "fall_2025"


class ProblemQuestion2(BaseModel):
    question_text: str
    question_1_url: str
    question_2_url: str
    notes: str
    tournament: Tournament
    proposed_action: str

    @model_validator(mode="after")
    def check_urls(self) -> Self:
        urls = [self.question_1_url, self.question_2_url]
        if len(urls) != len(set(urls)):
            raise ValueError(f"URLs must be unique, got: {urls}")
        for url in urls:
            if not url.startswith(
                "https://www.metaculus.com/questions/"
            ) and not url.endswith("/"):
                raise ValueError(
                    f"URL must start with 'https://www.metaculus.com/questions/', or end with '/', got: {url}"
                )
        return self


force_match_questions: list[ProblemQuestion2] = [
    # Q1 2025
    ProblemQuestion2(
        question_text="How many Grammy awards will Taylor Swift win in 2025?",
        question_1_url="https://www.metaculus.com/questions/31797/",
        question_2_url="https://www.metaculus.com/questions/31865/",
        notes="Different options: first has '3 or more', second has 'Greater than 2'",
        proposed_action="Keep since resolution is same",
        tournament=Tournament.Q1_2025,
    ),
    ProblemQuestion2(
        question_text="Which party will win the 2nd highest number of seats in the 2025 German federal election?",
        question_1_url="https://www.metaculus.com/questions/35002/",
        question_2_url="https://www.metaculus.com/questions/34940/",
        notes="Different options: first has 'Greens', second has 'Social Democratic Party' as an option twice. Same resolution (Alternative for Germany) and spot scoring time, created 16 minutes apart",
        proposed_action="Keep this since the resolution is the same",
        tournament=Tournament.Q1_2025,
    ),
    # Q1 2025 vs Cup
    ProblemQuestion2(
        question_text="What will the total number of Tesla vehicle deliveries be for Q1 2025?",
        question_1_url="https://www.metaculus.com/questions/35589/",
        question_2_url="https://www.metaculus.com/questions/35888/",
        notes="Different resolutions ('below lower bound' vs 336681.0), though they are both below lower bound. Created 8 days apart.",
        proposed_action="Force match",
        tournament=Tournament.Q1_2025_VS_CUP,
    ),
    ProblemQuestion2(
        question_text="How many earthquakes of magnitude ≥ 4 will happen near Santorini, Greece in the first week of March, 2025?",
        question_1_url="https://www.metaculus.com/questions/34862/",
        question_2_url="https://www.metaculus.com/questions/34968/",
        notes="Different open bounds (True vs False for upper bound). Created a day apart.",
        proposed_action="",
        tournament=Tournament.Q1_2025_VS_CUP,
    ),
    # ProblemQuestion(
    #     question_text="[TITLE MISMATCH] Premier League position",
    #     urls=[
    #         "https://www.metaculus.com/questions/34667/",
    #         "https://www.metaculus.com/questions/31672/",
    #     ],
    #     notes="Titles mismatch. One resolves Mar 10 while ther other March 8"
    # ),
    ProblemQuestion2(
        question_text="What will be the IMDb rating of Severance's second season finale?",
        question_1_url="https://www.metaculus.com/questions/35318/",
        question_2_url="https://www.metaculus.com/questions/35470/",
        notes="Different open bounds (False vs True for upper bound). Created 2 days apart.",
        proposed_action="",
        tournament=Tournament.Q1_2025_VS_CUP,
    ),
]

questions_to_remove_from_comparison: list[ProblemQuestion2] = []


class ProblemManager2:

    @classmethod
    def should_be_forced_matched(
        cls, question_1: Question, question_2: Question
    ) -> bool:
        return cls._pair_matches_problem_question_in_list(
            question_1, question_2, force_match_questions
        )

    @classmethod
    def remove_question_pairing_from_comparison(
        cls, question_1: Question, question_2: Question
    ) -> bool:
        return cls._pair_matches_problem_question_in_list(
            question_1, question_2, questions_to_remove_from_comparison
        )

    @classmethod
    def _pair_matches_problem_question_in_list(
        cls,
        question_1: Question,
        question_2: Question,
        problem_question_list: list[ProblemQuestion2],
    ) -> bool:
        for problem_question in problem_question_list:
            url_1 = problem_question.question_1_url
            url_2 = problem_question.question_2_url

            match_direction_1 = (
                str(question_1.post_id) in url_1 and str(question_2.post_id) in url_2
            )
            match_direction_2 = (
                str(question_2.post_id) in url_1 and str(question_1.post_id) in url_2
            )
            if match_direction_1 or match_direction_2:
                return True
        return False
