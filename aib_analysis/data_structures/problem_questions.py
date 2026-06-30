import logging
from enum import Enum

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from aib_analysis.data_structures.data_models import Question

logger = logging.getLogger(__name__)

"""
Second version of problem question management
"""


class Tournament(Enum):
    Q3_2024 = "q3_2024"
    Q4_2024 = "q4_2024"
    Q1_2025 = "q1_2025"
    Q1_2025_VS_CUP = "q1_2025_vs_cup"
    Q2_2025 = "q2_2025"
    Fall_2025 = "fall_2025"
    Spring_2026 = "spring_2026"


class ProblemQuestion(BaseModel):

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


force_match_questions: list[ProblemQuestion] = [
    # Q1 2025
    ProblemQuestion(
        question_text="How many Grammy awards will Taylor Swift win in 2025?",
        question_1_url="https://www.metaculus.com/questions/31797/",
        question_2_url="https://www.metaculus.com/questions/31865/",
        notes="Different options: first has '3 or more', second has 'Greater than 2'",
        proposed_action="Keep since resolution is same",
        tournament=Tournament.Q1_2025,
    ),
    ProblemQuestion(
        question_text="Which party will win the 2nd highest number of seats in the 2025 German federal election?",
        question_1_url="https://www.metaculus.com/questions/35002/",
        question_2_url="https://www.metaculus.com/questions/34940/",
        notes="Different options: first has 'Greens', second has 'Social Democratic Party' as an option twice. Same resolution (Alternative for Germany) and spot scoring time, created 16 minutes apart",
        proposed_action="Keep this since the resolution is the same",
        tournament=Tournament.Q1_2025,
    ),
    # Q1 2025 vs Cup
    ProblemQuestion(
        question_text="What will the total number of Tesla vehicle deliveries be for Q1 2025?",
        question_1_url="https://www.metaculus.com/questions/35589/",
        question_2_url="https://www.metaculus.com/questions/35888/",
        notes="Different resolutions ('below lower bound' vs 336681.0), though they are both below lower bound. Created 8 days apart.",
        proposed_action="Force match",
        tournament=Tournament.Q1_2025_VS_CUP,
    ),
    ProblemQuestion(
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
    ProblemQuestion(
        question_text="What will be the IMDb rating of Severance's second season finale?",
        question_1_url="https://www.metaculus.com/questions/35318/",
        question_2_url="https://www.metaculus.com/questions/35470/",
        notes="Different open bounds (False vs True for upper bound). Created 2 days apart.",
        proposed_action="",
        tournament=Tournament.Q1_2025_VS_CUP,
    ),
    # Spring 2026
    ProblemQuestion(
        question_text="How many commercial aircraft deliveries will Airbus report for March 2026 ?",
        question_1_url="https://www.metaculus.com/questions/42109/",
        question_2_url="https://www.metaculus.com/questions/42272/",
        notes="Duplicate questions in the same tournament. One is discrete, one is numeric. Both resolved to 60.",
        proposed_action="Remove from comparison",
        tournament=Tournament.Spring_2026,
    ),
]

class ProblemManager:

    @classmethod
    def should_be_forced_matched(
        cls, question_1: Question, question_2: Question
    ) -> bool:
        return cls._pair_matches_problem_question_in_list(
            question_1, question_2, force_match_questions
        )

    @classmethod
    def _pair_matches_problem_question_in_list(
        cls,
        question_1: Question,
        question_2: Question,
        problem_question_list: list[ProblemQuestion],
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


def title_matched_questions_are_problematic(
    title_matched_questions: list[Question], log_results: bool
) -> bool:
    assert len(title_matched_questions) > 1
    assert (
        len(set([q.question_text.lower().strip() for q in title_matched_questions]))
        == 1
    ), "When matching question titles, question titles must be the same"

    warning_message: str = ""
    info_message: str = ""
    non_annulled_questions = [
        q for q in title_matched_questions if not q.is_annulled_or_ambiguous
    ]

    log_comparison_table = False
    if len(non_annulled_questions) == 0:
        info_message = f"Matching question titles found, but all are annulled: {[q.url for q in title_matched_questions]}"
    elif len(non_annulled_questions) == 1:
        unique_projects = list(set([q.project for q in title_matched_questions]))
        if len(unique_projects) > 1:
            warning_message = f"[No action needed] Same-titles questions appear from both projects, but are not matched. Not much we can do to match them if leaderboards are finalized.: {[q.url for q in title_matched_questions]}"
            log_comparison_table = False
    elif len(non_annulled_questions) == 2:
        project_name_1 = non_annulled_questions[0].project
        project_name_2 = non_annulled_questions[1].project
        hash_1 = non_annulled_questions[0].get_hash_for_tournament_matching()
        hash_2 = non_annulled_questions[1].get_hash_for_tournament_matching()

        from_same_tournament = project_name_1 == project_name_2
        hashes_match = hash_1 == hash_2

        if from_same_tournament and hashes_match:
            logger.debug(
                f"Identical questions found in same tournament: {[q.url for q in title_matched_questions]}"
            )  # Probably triggered by combining two tournaments with different teams.
        elif from_same_tournament and not hashes_match:
            warning_message = f"[Probably fine] Same-titled non-annulled questions found in same tournament. Could be mistake or testing scope sensitivity: {[q.url for q in title_matched_questions]}"
        elif not from_same_tournament and hashes_match:
            pass  # The questions are intended to be matched.
    elif len(non_annulled_questions) > 2:
        unique_hashes = list(
            set([q.get_hash_for_tournament_matching() for q in non_annulled_questions])
        )
        only_2_questions_match = len(unique_hashes) == len(non_annulled_questions) - 1

        if only_2_questions_match:
            warning_message = f"[Probably fine] Matching question titles found, but more than 2 are non-annulled. Only 2 of the questions have matching hashes which shouldn't cause problems.: {[q.url for q in title_matched_questions]}"
        else:
            warning_message = f"Matching question titles found, but more than 2 are non-annulled: {[q.url for q in title_matched_questions]}"
            log_comparison_table = True

    if log_comparison_table:
        question_comparison_table = Question.question_comparison_table(
            title_matched_questions
        )
        warning_message += f"\nQuestion comparison table:\n{question_comparison_table}"

    if log_results:
        if warning_message != "":
            logger.warning(warning_message)
        if info_message != "":
            logger.info(info_message)

    problem_found = warning_message != "" or log_comparison_table
    return problem_found


"""
TODO: Record these in some way:

Q4 Remove from comparison
| URL | https://www.metaculus.com/questions/30994/ | https://www.metaculus.com/questions/31033/ |
One was incorrectly marked as "No" When it should be "ambiguous"

Q2 Remove from comparison
    ProblemQuestion(
        question_text="How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025?",
        urls=[
            "https://www.metaculus.com/questions/37295/",
            "https://www.metaculus.com/questions/37651/",
            "https://www.metaculus.com/questions/37248/",
        ],
        notes="Same question text but different option formats: ('0', '1', '2-3', '4 or more') vs ('Zero', 'One', 'Two or Three', 'Four or more'). First resolved to '0', second to 'Zero', third unresolved.",
        proposed_action="Remove from comparison becasuse one question resolved as 'Zero' while the other '0'",
    ),

Q1 Remove from comparison
    ProblemQuestion(
        question_text="For Q1 2025, how many banks will be listed on the FDIC's Failed Bank List?",
        urls=[
            "https://www.metaculus.com/questions/31736/",
            "https://www.metaculus.com/questions/31730/",
        ],
        notes="Different resolutions (1 vs 0)",
        proposed_action="Remove from comparison",
    ),


_q1_bot_v_cup_to_remove_from_comparison: list[ProblemQuestion] = [
    ProblemQuestion(
        question_text="[TITLE MISMATCH] Cherry blossom peak bloom",
        urls=[
            "https://www.metaculus.com/questions/35670/",
            "https://www.metaculus.com/questions/35588/",
        ],
        notes=(
            "Titles mismatch, but they are asking the same idea. "
            "The options are different enough to make this not viable"
            "This question will probably already be excluded due to title mismatch"
        ),
        proposed_action="Remove from comparison",
    ),
    ProblemQuestion(
        question_text="How many hostages will Hamas release after January 26 and before April 5, 2025?",
        urls=[
            "https://www.metaculus.com/questions/31849/",
            "https://www.metaculus.com/questions/34274/",
        ],
        notes="Different resolutions (20-29 vs 30-39) and spot scoring times (17:00 vs 02:00). Created 2 days apart.",
        proposed_action="Remove from comparison due to different resolutions",
    ),
]

"""
