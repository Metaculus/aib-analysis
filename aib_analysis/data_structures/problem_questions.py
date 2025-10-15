from __future__ import annotations

import logging

from pydantic import BaseModel, model_validator
from typing_extensions import Self
from enum import Enum

from aib_analysis.data_structures.data_models import Question

logger = logging.getLogger(__name__)



class ProblemQuestion(BaseModel):
    question_text: str
    urls: list[str]
    notes: str
    proposed_action: str | None = None

    @model_validator(mode="after")
    def check_urls(self) -> Self:
        if len(self.urls) != len(set(self.urls)):
            raise ValueError(f"URLs must be unique, got: {self.urls}")
        for url in self.urls:
            if not url.startswith("https://www.metaculus.com/questions/") and not url.endswith("/"):
                raise ValueError(f"URL must start with 'https://www.metaculus.com/questions/', or end with '/', got: {url}")
        return self

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
            logger.debug(
                f"Input Question {input_url} matches some parts of problem but not all | "
                f"Input Question_text: {question.question_text} | "
                f"Problem Question_text: {self.question_text} | "
                f"Problem: {self.model_dump_json()}"
            )
            return False


class ProblemManager:
    """
    When adding to this manager:
    1. Run the simulation
    2. Find the log statements indicating either
        a) A duplicate was found within a tournament
        b) A mismatch was found between tournaments
    3. Copy those logs into this file
    4. Make a list of problem questions matching these questions
    5. Sort the lists into the right action
        a) For duplicates categorize into the below options:
            - Leave as is - Both questions should be included in analysis - make sure its in dont_log_in_duplicate_detection_within_tournament
            - One is annulled - One will already be excluded - make sure its in dont_log_in_duplicate_detection_within_tournament
            - Exclude the pair (or choose one) - Currently this hasn't come up and is not supported
        b) For mismatches between tournaments (more than one possible match). Categorize them into the following buckets:
            - Leave as is - For groups (not just pairs) that will have at least one correct match (e.g. one is annulled and wouldn't be included) - include in dont_log_in_tournament_matching
            - Exclude the pair - For pairs that can't be resolved - include in dont_log_in_tournament_matching
            - Force match - make sure they are in the is_prequalified_for_tournament_matching function and dont_log_in_tournament_matching

    For reference you can tell if a question needs a force match by checking the question.get_hash_for_tournament_matching() method which is a unique identifier for matching questions.
    """

    @classmethod
    def dont_log_in_duplicate_detection_within_tournament(
        cls, questions: list[Question]
    ) -> bool:
        return cls._question_list_fully_matches_a_problem_question(
            questions,
            cls._q1_bot__in_tournament_title_duplicates
            + cls._q2_bot__in_tournament_title_duplicates,
        )

    @classmethod
    def dont_log_in_tournament_matching(cls, questions: list[Question]) -> bool:
        return cls._question_list_fully_matches_a_problem_question(
            questions,
            cls._q1_bot_v_cup_inconsistencies_to_force_match
            + cls._q1_bot_v_cup_to_remove_from_comparison
            + cls._q1_bot_v_pro_inconsistencies_to_force_match
            + cls._q1_bot_v_pro_to_remove_from_comparison
            + cls._q1_bot_v_pro_inconsistencies_that_have_at_least_one_good_match
            + cls._q2_bot_v_pro_inconsistencies_to_force_match
            + cls._q2_bot_v_pro_to_remove_from_comparison
            + cls._q2_bot_v_pro_inconsistencies_that_have_at_least_one_good_match,
        )

    @classmethod
    def find_prequalified_matches_for_tournament_matching(
        cls, questions: list[Question]
    ) -> list[list[Question]]:
        question_title_map: dict[str, list[Question]] = {}
        for question in questions:
            question_title_map.setdefault(question.question_text, []).append(question)

        matches = []
        for question_group in question_title_map.values():
            if len(question_group) < 2:
                continue
            if cls.is_prequalified_for_tournament_matching(question_group):
                matches.append(question_group)
        for match in matches:
            if len(match) != 2:
                raise ValueError(
                    f"Found a group of {len(match)} questions. All matches should produce 2 questions"
                )
        return matches

    @classmethod
    def is_prequalified_for_tournament_matching(cls, questions: list[Question]) -> bool:
        if cls._question_list_fully_matches_a_problem_question(
            questions,
            cls._q1_bot_v_pro_inconsistencies_to_force_match
            + cls._q1_bot_v_cup_inconsistencies_to_force_match
            + cls._q2_bot_v_pro_inconsistencies_to_force_match,
        ):
            return True
        return False

    @classmethod
    def _question_list_fully_matches_a_problem_question(
        cls, questions: list[Question], problem_question_list: list[ProblemQuestion]
    ) -> bool:
        for pq in problem_question_list:
            matches = [pq.question_matches(question) for question in questions]
            if all(matches):
                return True
            elif any(matches):
                logger.debug(
                    f"One of the input questions matches the problem question, but not all of them. Input Questions: {[question.url for question in questions]}, Problem question: {pq.model_dump_json()}"
                )
        return False



    _q2_bot__in_tournament_title_duplicates: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text='How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025?',
            urls=[
                "https://www.metaculus.com/questions/38539/",
                "https://www.metaculus.com/questions/38052/",
            ],
            notes="Different options: first has ('Zero', 'One', 'Two', 'Three or more'), second has ('Zero', 'One', 'Two', 'Greater than two'). Different resolutions: first was annulled, second 'Two'",
            proposed_action="Leave as is, the first one was annulled",
        ),
        ProblemQuestion(
            question_text="How many people will be in space on June 27, 2025, according to whoisinspace.com?",
            urls=[
                "https://www.metaculus.com/questions/38532/",
                "https://www.metaculus.com/questions/38083/",
                "https://www.metaculus.com/questions/37480/",
            ],
            notes="Three duplicates with slightly different options: ('Less than Ten', 'Exactly Ten', 'Greater than Ten') vs ('Less than Ten', 'Ten', 'Greater than Ten') vs ('Less than 10', '10', 'Greater than 10'). First resolved to 'Greater than Ten', others annulled",
            proposed_action="Leave as is. 2 of them were annulled",
        ),
        ProblemQuestion(
            question_text='At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)?',
            urls=[
                "https://www.metaculus.com/questions/38331/",
                "https://www.metaculus.com/questions/38092/",
            ],
            notes="Binary questions with different resolutions: first resolved to True, second unresolved. One got annulled because 'annulling this question, since Pakistan got moved from Current to Former before the launch of this question'",
            proposed_action="Leave as is. The first one was annulled",
        ),
        ProblemQuestion(
            question_text="How many unique posters will Bluesky have on June 28, 2025?",
            urls=[
                "https://www.metaculus.com/questions/38326/",
                "https://www.metaculus.com/questions/38076/",
            ],
            notes="Numeric questions with different resolutions: first resolved to 640344.0, second annulled because of bad background",
            proposed_action="Leave as is. The second one was annulled",
        ),
        ProblemQuestion(
            question_text="For Q2 2025, how many banks will be listed on the FDIC's Failed Bank List?",
            urls=[
                "https://www.metaculus.com/questions/38096/",
                "https://www.metaculus.com/questions/37247/",
            ],
            notes="Different options: first has ('Exactly 0', 'Exactly 1', '2-3', '4-6', '7-20', '>20'), second has ('0', '1', '2-3', '4-6', '7-20', '>20'). Different resolutions: first 'Exactly 1', second anulled",
            proposed_action="Leave as is. The second one was annulled",
        ),
        ProblemQuestion(
            question_text="How many Chinese universities will be in the top 20 of the QS World University Rankings 2026?",
            urls=[
                "https://www.metaculus.com/questions/38072/",
                "https://www.metaculus.com/questions/37046/",
            ],
            notes="Different options: first has ('Zero or One', 'Two', 'Three or more'), second has ('0 or 1', '2', '3 or more'). Different resolutions: first 'Two', second annulled",
            proposed_action="Leave as is. The second one was annulled",
        ),
        ProblemQuestion(
            question_text="How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025?",
            urls=[
                "https://www.metaculus.com/questions/37651/",
                "https://www.metaculus.com/questions/37248/",
            ],
            notes="Different options: first has ('Zero', 'One', 'Two or Three', 'Four or more'), second has ('0', '1', '2-3', '4 or more'). Different resolutions: first 'Zero', second annulled",
            proposed_action="Leave as is. The second one was annulled",
        ),
        ProblemQuestion(
            question_text="How many foreign visitors to the United States will the International Trade Administration report for April 2025?",
            urls=[
                "https://www.metaculus.com/questions/37216/",
                "https://www.metaculus.com/questions/37010/",
            ],
            notes="Numeric questions with different resolutions: first resolved to 5040051.0, second annulled due to a fine print typo",
            proposed_action="Leave as is. The second one was annulled",
        ),
    ]

    _q2_bot_v_pro_inconsistencies_to_force_match: list[ProblemQuestion] = [
    ]
    _q2_bot_v_pro_to_remove_from_comparison: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text="Will a Gemini model be ranked #1 overall on the Chatbot Arena Leaderboard at the end of the 2nd Quarter of 2025?",
            urls=[
                "https://www.metaculus.com/questions/38565/",
                "https://www.metaculus.com/questions/38538/",
            ],
            notes="Identical question text but different weights (1.0 vs 0.8) and different tournaments. First resolved to True, second annulled due to API bug.",
            proposed_action="Remove from comparison due to different resolutions",
        ),
        ProblemQuestion(
            question_text="What will the US national debt be on June 27, 2025?",
            urls=[
                "https://www.metaculus.com/questions/38564/",
                "https://www.metaculus.com/questions/38537/",
            ],
            notes="Identical question text and parameters but different tournaments. First resolved to 36.21512431338216, second anulled due to a bug in the posts API.",
            proposed_action="Remove from comparison due to different resolutions",
        ),
        ProblemQuestion(
            question_text="Will the word 'tariff(s)' disappear from the front print pages of The New York Times and Wall Street Journal by June 1, 2025?",
            urls=[
                "https://www.metaculus.com/questions/37510/",
                "https://www.metaculus.com/questions/37477/",
            ],
            notes="Identical question text but different tournaments. First anulled, second resolved to False.",
            proposed_action="Remove from comparison due to different resolutions",
        ),
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
    ]
    _q2_bot_v_pro_inconsistencies_that_have_at_least_one_good_match: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text="How many people will be in space on June 27, 2025, according to whoisinspace.com?",
            urls=[
                "https://www.metaculus.com/questions/38560/",
                "https://www.metaculus.com/questions/38532/",
                "https://www.metaculus.com/questions/38083/",
                "https://www.metaculus.com/questions/37480/",
            ],
            notes="Same question text but different option formats: ('Less than Ten', 'Exactly Ten', 'Greater than Ten') vs ('Less than Ten', 'Ten', 'Greater than Ten') vs ('Less than 10', '10', 'Greater than 10'). First two resolved to 'Greater than Ten', others annulled.",
            proposed_action="Leave as is. The first two will be correctly matched",
        ),
        ProblemQuestion(
            question_text="At the end of June 2025, will Wikipedia still list all these countries as 'currently' blocking access to X (formerly Twitter)?",
            urls=[
                "https://www.metaculus.com/questions/38360/",
                "https://www.metaculus.com/questions/38331/",
                "https://www.metaculus.com/questions/38092/",
            ],
            notes="Identical question text but different tournaments and spot scoring times. First two resolved to True, third unresolved.",
            proposed_action="Leave as is. The two will be correctly matched",
        ),
        ProblemQuestion(
            question_text="How many 'Level 4 – Do Not Travel' travel advisories will the US State Department issue in June 2025?",
            urls=[
                "https://www.metaculus.com/questions/38124/",
                "https://www.metaculus.com/questions/38539/",
                "https://www.metaculus.com/questions/38052/",
            ],
            notes="Same question text but different option formats: ('Zero', 'One', 'Two', 'Greater than two') vs ('Zero', 'One', 'Two', 'Three or more'). First and third resolved to 'Two', second unresolved.",
            proposed_action="Leave as is. The two will be correctly matched",
        ),
    ]


    # These are questions with duplicate titles in the q1 tournament
    _q1_bot__in_tournament_title_duplicates: list[ProblemQuestion] = [
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
            notes="Different weights (1.0 vs 0.5) and spot scoring times (off by ~2 days). Accidental rerelease",
            proposed_action="Remove this",  # TODO: Create mechanism to remove one of the versions of this question from scoring
        ),
        ProblemQuestion(
            question_text="Which party will win the most seats in Curaçao in the March 2025 general election?",
            urls=[
                "https://www.metaculus.com/questions/35892/",
                "https://www.metaculus.com/questions/35994/",
            ],
            notes="Different resolutions: first unresolved, second resolved to 'Movement for the Future of Curaçao'. Spot scoring time 2 days off. First was annulled",
            proposed_action="Leave this. The first one was annulled",
        ),
        ProblemQuestion(
            question_text="Which podcast will be ranked higher on Spotify on March 31, 2025: Call Her Daddy or Candace?",
            urls=[
                "https://www.metaculus.com/questions/36161/",
                "https://www.metaculus.com/questions/36264/",
            ],
            notes="Completely different options: first has ('The New York Times Daily', 'The Tucker Carlson Show') resolved to None, second has ('Call Her Daddy', 'Candace') and resolved to 'Candace'. Spot scoring time 2 days off. First was annulled",
            proposed_action="Leave this. The first one was annulled",
        ),
    ]


    # These are questions that are close enough to each other to match, but do have differences (like different options)
    _q1_bot_v_pro_inconsistencies_to_force_match: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text="How many Grammy awards will Taylor Swift win in 2025?",
            urls=[
                "https://www.metaculus.com/questions/31797/",
                "https://www.metaculus.com/questions/31865/",
            ],
            notes="Different options: first has '3 or more', second has 'Greater than 2'",
            proposed_action="Keep since resolution is same",
        ),
        ProblemQuestion(
            question_text="Which party will win the 2nd highest number of seats in the 2025 German federal election?",
            urls=[
                "https://www.metaculus.com/questions/35002/",
                "https://www.metaculus.com/questions/34940/",
            ],
            notes="Different options: first has 'Greens', second has 'Social Democratic Party' as an option twice. Same resolution (Alternative for Germany) and spot scoring time, created 16 minutes apart",
            proposed_action="Keep this since the resolution is the same",
        ),
    ]

    _q1_bot_v_pro_to_remove_from_comparison: list[ProblemQuestion] = [
        ProblemQuestion(
            question_text="For Q1 2025, how many banks will be listed on the FDIC's Failed Bank List?",
            urls=[
                "https://www.metaculus.com/questions/31736/",
                "https://www.metaculus.com/questions/31730/",
            ],
            notes="Different resolutions (1 vs 0)",
            proposed_action="Remove from comparison",
        ),
    ]

    _q1_bot_v_pro_inconsistencies_that_have_at_least_one_good_match: list[
        ProblemQuestion
    ] = [
        ProblemQuestion(
            question_text="How many arms sales globally will the US State Department approve in March 2025?",
            urls=[
                "https://www.metaculus.com/questions/34382/",
                "https://www.metaculus.com/questions/34260/",
                "https://www.metaculus.com/questions/34706/",
            ],
            notes="Three versions with different options and resolutions. First two have same options ('0-4', '5-9', '>9') and resolution (5-9), third has different options ('0-5', '6-10', '>10') and resolution (0-5).  Third has spot scoring time off by 9 days. First is pro tournament, second 2 bot tournament",
            proposed_action="Leave as is. The first two will be correctly matched for pro v bot tournament and the 3rd automatically excluded",
            # TODO: Match this with CP comparison, and the in-tournament duplicates above
        ),
        ProblemQuestion(
            question_text="What Premier League position will Nottingham Forest F.C. be in on March 8, 2025?",
            urls=[
                "https://www.metaculus.com/questions/34389/",
                "https://www.metaculus.com/questions/34281/",
                "https://www.metaculus.com/questions/34667/",
            ],
            notes="Three versions. different weights (1.0 for first 2 vs 0.5 for third) and spot scoring times (third one is 2 days after first 2 which are same). Tournaments are Pro, Bot, Bot",
            proposed_action="Leave as is. The first two will be correctly matched for pro v bot tournament and the 3rd automatically excluded",
            # TODO: Match this with the in-tournament duplicates above
        ),
    ]

    _q1_bot_v_cup_inconsistencies_to_force_match: list[ProblemQuestion] = [
        # ProblemQuestion(
        #     question_text="[TITLE MISMATCH] Premier League position",
        #     urls=[
        #         "https://www.metaculus.com/questions/34667/",
        #         "https://www.metaculus.com/questions/31672/",
        #     ],
        #     notes="Titles mismatch. One resolves Mar 10 while ther other March 8"
        # ),
        ProblemQuestion(
            question_text="What will the total number of Tesla vehicle deliveries be for Q1 2025?",
            urls=[
                "https://www.metaculus.com/questions/35589/",
                "https://www.metaculus.com/questions/35888/",
            ],
            notes="Different resolutions ('below lower bound' vs 336681.0), though they are both below lower bound. Created 8 days apart.",
            proposed_action="Force match",
        ),
        ProblemQuestion(
            question_text="How many earthquakes of magnitude ≥ 4 will happen near Santorini, Greece in the first week of March, 2025?",
            urls=[
                "https://www.metaculus.com/questions/34862/",
                "https://www.metaculus.com/questions/34968/",
            ],
            notes="Different open bounds (True vs False for upper bound). Created a day apart.",
            proposed_action="",
        ),
        ProblemQuestion(
            question_text="What will be the IMDb rating of Severance's second season finale?",
            urls=[
                "https://www.metaculus.com/questions/35318/",
                "https://www.metaculus.com/questions/35470/",
            ],
            notes="Different open bounds (False vs True for upper bound). Created 2 days apart.",
            proposed_action="",
        ),
    ]

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
##################### Q2 Duplicate Question - Bot Tournament #####################

# Duplicates for question text: How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38539/ | https://www.metaculus.com/questions/38052/ |
| Question Id | 37770 | 37358 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025? | How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025? |
| Resolution | None | Two |
| Options | ('Zero', 'One', 'Two', 'Three or more') | ('Zero', 'One', 'Two', 'Greater than two') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 38539 | 38052 |
| Created At | 2025-06-06 23:33:33.485716+00:00 | 2025-05-24 05:50:05.303416+00:00 |
| Spot Scoring Time | 2025-06-14 14:00:00+00:00 | 2025-05-26 04:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |


# Duplicates for question text: How many people will be in space on June 27, 2025, according to whoisinspace.com?
| Parameter | Question 1 | Question 2 | Question 3 |
|-----------|---|---|---|
| URL | https://www.metaculus.com/questions/38532/ | https://www.metaculus.com/questions/38083/ | https://www.metaculus.com/questions/37480/ |
| Question Id | 37763 | 37389 | 36840 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many people will be in space on June 27, 2025, according to whoisinspace.com? | How many people will be in space on June 27, 2025, according to whoisinspace.com? | How many people will be in space on June 27, 2025, according to whoisinspace.com? |
| Resolution | Greater than Ten | None | None |
| Options | ('Less than Ten', 'Exactly Ten', 'Greater than Ten') | ('Less than Ten', 'Ten', 'Greater than Ten') | ('Less than 10', '10', 'Greater than 10') |
| Range Max | None | None | None |
| Range Min | None | None | None |
| Open Upper Bound | None | None | None |
| Open Lower Bound | None | None | None |
| Zero Point | None | None | None |
| Weight | 1.0 | 1.0 | 1.0 |
| Post Id | 38532 | 38083 | 37480 |
| Created At | 2025-06-06 23:33:32.267839+00:00 | 2025-05-24 05:50:06.929652+00:00 | 2025-05-03 02:27:38.243986+00:00 |
| Spot Scoring Time | 2025-06-11 12:00:00+00:00 | 2025-05-29 00:00:00+00:00 | 2025-05-09 18:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None | None |


# Duplicates for question text: At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38331/ | https://www.metaculus.com/questions/38092/ |
| Question Id | 37607 | 37398 |
| Type | QuestionType.BINARY | QuestionType.BINARY |
| Question Text | At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)? | At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)? |
| Resolution | True | None |
| Options | None | None |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 38331 | 38092 |
| Created At | 2025-05-31 04:11:49.586557+00:00 | 2025-05-24 05:50:07.272330+00:00 |
| Spot Scoring Time | 2025-06-05 16:00:00+00:00 | 2025-05-29 18:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |


# Duplicates for question text: How many unique posters will Bluesky have on June 28, 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38326/ | https://www.metaculus.com/questions/38076/ |
| Question Id | 37602 | 37382 |
| Type | QuestionType.NUMERIC | QuestionType.NUMERIC |
| Question Text | How many unique posters will Bluesky have on June 28, 2025? | How many unique posters will Bluesky have on June 28, 2025? |
| Resolution | 640344.0 | None |
| Options | None | None |
| Range Max | 2000000.0 | 2000000.0 |
| Range Min | 200000.0 | 200000.0 |
| Open Upper Bound | True | True |
| Open Lower Bound | True | True |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 38326 | 38076 |
| Created At | 2025-05-31 04:11:49.263493+00:00 | 2025-05-24 05:50:06.475972+00:00 |
| Spot Scoring Time | 2025-06-05 04:00:00+00:00 | 2025-05-28 08:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |


# Duplicates for question text: For Q2 2025, how many banks will be listed on the FDIC's Failed Bank List?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38096/ | https://www.metaculus.com/questions/37247/ |
| Question Id | 37402 | 36625 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | For Q2 2025, how many banks will be listed on the FDIC's Failed Bank List? | For Q2 2025, how many banks will be listed on the FDIC's Failed Bank List? |
| Resolution | Exactly 1 | None |
| Options | ('Exactly 0', 'Exactly 1', '2-3', '4-6', '7-20', '>20') | ('0', '1', '2-3', '4-6', '7-20', '>20') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 38096 | 37247 |
| Created At | 2025-05-24 05:50:07.291011+00:00 | 2025-04-26 05:58:11.711088+00:00 |
| Spot Scoring Time | 2025-05-30 02:00:00+00:00 | 2025-05-01 02:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |


# Duplicates for question text: How many Chinese universities will be in the top 20 of the QS World University Rankings 2026?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38072/ | https://www.metaculus.com/questions/37046/ |
| Question Id | 37378 | 36441 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many Chinese universities will be in the top 20 of the QS World University Rankings 2026? | How many Chinese universities will be in the top 20 of the QS World University Rankings 2026? |
| Resolution | Two | None |
| Options | ('Zero or One', 'Two', 'Three or more') | ('0 or 1', '2', '3 or more') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 38072 | 37046 |
| Created At | 2025-05-24 05:50:06.456786+00:00 | 2025-04-18 04:06:42.027745+00:00 |
| Spot Scoring Time | 2025-05-28 00:00:00+00:00 | 2025-04-24 18:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |


# Duplicates for question text: How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/37651/ | https://www.metaculus.com/questions/37248/ |
| Question Id | 37004 | 36626 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025? | How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025? |
| Resolution | Zero | None |
| Options | ('Zero', 'One', 'Two or Three', 'Four or more') | ('0', '1', '2-3', '4 or more') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 37651 | 37248 |
| Created At | 2025-05-09 21:48:35.077445+00:00 | 2025-04-26 05:58:11.716639+00:00 |
| Spot Scoring Time | 2025-05-15 16:00:00+00:00 | 2025-05-01 06:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |


# Duplicates for question text: How many foreign visitors to the United States will the International Trade Administration report for April 2025?
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/37216/ | https://www.metaculus.com/questions/37010/ |
| Question Id | 36594 | 36405 |
| Type | QuestionType.NUMERIC | QuestionType.NUMERIC |
| Question Text | How many foreign visitors to the United States will the International Trade Administration report for April 2025? | How many foreign visitors to the United States will the International Trade Administration report for April 2025? |
| Resolution | 5040051.0 | None |
| Options | None | None |
| Range Max | 6000000.0 | 6000000.0 |
| Range Min | 4000000.0 | 4000000.0 |
| Open Upper Bound | True | True |
| Open Lower Bound | True | True |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 37216 | 37010 |
| Created At | 2025-04-26 05:58:08.840506+00:00 | 2025-04-18 04:06:40.480103+00:00 |
| Spot Scoring Time | 2025-04-28 04:00:00+00:00 | 2025-04-21 18:00:00+00:00 |
| Project | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |
"""


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


"""
###################### Q2 Bot v Pro Matching Inconsistencies ######################

2025-09-18 12:57:06,481 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38565/ | https://www.metaculus.com/questions/38538/ |
| Question Id | 37796 | 37769 |
| Type | QuestionType.BINARY | QuestionType.BINARY |
| Question Text | Will a Gemini model be ranked #1 overall on the Chatbot Arena Leaderboard at the end of the 2nd Quarter of 2025? | Will a Gemini model be ranked #1 overall on the Chatbot Arena Leaderboard at the end of the 2nd Quarter of 2025? |
| Resolution | True | None |
| Options | None | None |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 0.8 |
| Post Id | 38565 | 38538 |
| Created At | 2025-06-06 23:36:39.387331+00:00 | 2025-06-06 23:33:33.480377+00:00 |
| Spot Scoring Time | 2025-06-14 04:00:00+00:00 | 2025-06-14 04:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-09-18 12:57:06,493 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/38564/ | https://www.metaculus.com/questions/38537/ |
| Question Id | 37795 | 37768 |
| Type | QuestionType.NUMERIC | QuestionType.NUMERIC |
| Question Text | What will the US national debt be on June 27, 2025? | What will the US national debt be on June 27, 2025? |
| Resolution | 36.21512431338216 | None |
| Options | None | None |
| Range Max | 37.0 | 37.0 |
| Range Min | 36.2 | 36.2 |
| Open Upper Bound | True | True |
| Open Lower Bound | True | True |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 38564 | 38537 |
| Created At | 2025-06-06 23:36:39.380631+00:00 | 2025-06-06 23:33:33.474442+00:00 |
| Spot Scoring Time | 2025-06-13 20:00:00+00:00 | 2025-06-13 20:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-09-18 12:57:06,511 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 | Question 3 | Question 4 |
|-----------|---|---|---|---|
| URL | https://www.metaculus.com/questions/38560/ | https://www.metaculus.com/questions/38532/ | https://www.metaculus.com/questions/38083/ | https://www.metaculus.com/questions/37480/ |
| Question Id | 37791 | 37763 | 37389 | 36840 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many people will be in space on June 27, 2025, according to whoisinspace.com? | How many people will be in space on June 27, 2025, according to whoisinspace.com? | How many people will be in space on June 27, 2025, according to whoisinspace.com? | How many people will be in space on June 27, 2025, according to whoisinspace.com? |
| Resolution | Greater than Ten | Greater than Ten | None | None |
| Options | ('Less than Ten', 'Exactly Ten', 'Greater than Ten') | ('Less than Ten', 'Exactly Ten', 'Greater than Ten') | ('Less than Ten', 'Ten', 'Greater than Ten') | ('Less than 10', '10', 'Greater than 10') |
| Range Max | None | None | None | None |
| Range Min | None | None | None | None |
| Open Upper Bound | None | None | None | None |
| Open Lower Bound | None | None | None | None |
| Zero Point | None | None | None | None |
| Weight | 1.0 | 1.0 | 1.0 | 1.0 |
| Post Id | 38560 | 38532 | 38083 | 37480 |
| Created At | 2025-06-06 23:36:38.494108+00:00 | 2025-06-06 23:33:32.267839+00:00 | 2025-05-24 05:50:06.929652+00:00 | 2025-05-03 02:27:38.243986+00:00 |
| Spot Scoring Time | 2025-06-11 12:00:00+00:00 | 2025-06-11 12:00:00+00:00 | 2025-05-29 00:00:00+00:00 | 2025-05-09 18:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None | None | None |
| Tournament 1 | True | False | False | False |
| Tournament 2 | False | True | True | True |

2025-09-18 12:57:06,531 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 | Question 3 |
|-----------|---|---|---|
| URL | https://www.metaculus.com/questions/38360/ | https://www.metaculus.com/questions/38331/ | https://www.metaculus.com/questions/38092/ |
| Question Id | 37636 | 37607 | 37398 |
| Type | QuestionType.BINARY | QuestionType.BINARY | QuestionType.BINARY |
| Question Text | At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)? | At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)? | At the end of June 2025, will Wikipedia still list all these countries as "currently" blocking access to X (formerly Twitter)? |
| Resolution | True | True | None |
| Options | None | None | None |
| Range Max | None | None | None |
| Range Min | None | None | None |
| Open Upper Bound | None | None | None |
| Open Lower Bound | None | None | None |
| Zero Point | None | None | None |
| Weight | 1.0 | 1.0 | 1.0 |
| Post Id | 38360 | 38331 | 38092 |
| Created At | 2025-05-31 04:19:35.799176+00:00 | 2025-05-31 04:11:49.586557+00:00 | 2025-05-24 05:50:07.272330+00:00 |
| Spot Scoring Time | 2025-06-05 16:00:00+00:00 | 2025-06-05 16:00:00+00:00 | 2025-05-29 18:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None | None |
| Tournament 1 | True | False | False |
| Tournament 2 | False | True | True |

2025-09-18 12:57:06,555 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 | Question 3 |
|-----------|---|---|---|
| URL | https://www.metaculus.com/questions/38124/ | https://www.metaculus.com/questions/38539/ | https://www.metaculus.com/questions/38052/ |
| Question Id | 37430 | 37770 | 37358 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025? | How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025? | How many "Level 4 – Do Not Travel" travel advisories will the US State Department issue in June 2025? |
| Resolution | Two | None | Two |
| Options | ('Zero', 'One', 'Two', 'Greater than two') | ('Zero', 'One', 'Two', 'Three or more') | ('Zero', 'One', 'Two', 'Greater than two') |
| Range Max | None | None | None |
| Range Min | None | None | None |
| Open Upper Bound | None | None | None |
| Open Lower Bound | None | None | None |
| Zero Point | None | None | None |
| Weight | 1.0 | 1.0 | 1.0 |
| Post Id | 38124 | 38539 | 38052 |
| Created At | 2025-05-24 06:00:32.410658+00:00 | 2025-06-06 23:33:33.485716+00:00 | 2025-05-24 05:50:05.303416+00:00 |
| Spot Scoring Time | 2025-05-26 04:00:00+00:00 | 2025-06-14 14:00:00+00:00 | 2025-05-26 04:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None | None |
| Tournament 1 | True | False | False |
| Tournament 2 | False | True | True |

2025-09-18 12:57:06,582 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/37510/ | https://www.metaculus.com/questions/37477/ |
| Question Id | 36870 | 36837 |
| Type | QuestionType.BINARY | QuestionType.BINARY |
| Question Text | Will the word "tariff(s)" disappear from the front print pages of The New York Times and Wall Street Journal by June 1, 2025? | Will the word "tariff(s)" disappear from the front print pages of The New York Times and Wall Street Journal by June 1, 2025? |
| Resolution | None | False |
| Options | None | None |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 37510 | 37477 |
| Created At | 2025-05-03 02:30:25.725508+00:00 | 2025-05-03 02:27:38.070353+00:00 |
| Spot Scoring Time | 2025-05-09 11:00:00+00:00 | 2025-05-09 11:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-09-18 12:57:06,625 - WARNING - aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 | Question 3 |
|-----------|---|---|---|
| URL | https://www.metaculus.com/questions/37295/ | https://www.metaculus.com/questions/37651/ | https://www.metaculus.com/questions/37248/ |
| Question Id | 36670 | 37004 | 36626 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025? | How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025? | How many mentions of Ghana will Pharma Manufacturing magazine make before July 1, 2025? |
| Resolution | 0 | Zero | None |
| Options | ('0', '1', '2-3', '4 or more') | ('Zero', 'One', 'Two or Three', 'Four or more') | ('0', '1', '2-3', '4 or more') |
| Range Max | None | None | None |
| Range Min | None | None | None |
| Open Upper Bound | None | None | None |
| Open Lower Bound | None | None | None |
| Zero Point | None | None | None |
| Weight | 1.0 | 1.0 | 1.0 |
| Post Id | 37295 | 37651 | 37248 |
| Created At | 2025-04-26 06:03:07.838553+00:00 | 2025-05-09 21:48:35.077445+00:00 | 2025-04-26 05:58:11.716639+00:00 |
| Spot Scoring Time | 2025-05-01 06:00:00+00:00 | 2025-05-15 16:00:00+00:00 | 2025-05-01 06:00:00+00:00 |
| Project | Pro Forecasters - AI Forecasting Benchmark Q2 2025 | Q2 AI Forecasting Benchmark Tournament | Q2 AI Forecasting Benchmark Tournament |
| Notes | None | None | None |
| Tournament 1 | True | False | False |
| Tournament 2 | False | True | True |
"""

"""
###################### Q1 Bot v Pro Matching Inconsistencies ######################

# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/31736/ | https://www.metaculus.com/questions/31730/ |
| Question Id | 31268 | 31262 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | For Q1 2025, how many banks will be listed on the FDIC's Failed Bank List? | For Q1 2025, how many banks will be listed on the FDIC's Failed Bank List? |
| Resolution | 1 | 0 |
| Options | ('0', '1', '2-3', '4-6', '>6') | ('0', '1', '2-3', '4-6', '>6') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 31736 | 31730 |
| Created At | 2025-01-17 19:06:22.013528+00:00 | 2025-01-17 19:02:43.857529+00:00 |
| Spot Scoring Time | 2025-01-20 03:27:00+00:00 | 2025-01-20 03:27:00+00:00 |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-06-14 13:00:13,710 - WARNING - from aib_analysis.main_logic.process_tournament - _log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/31797/ | https://www.metaculus.com/questions/31865/ |
| Question Id | 31321 | 31370 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many Grammy awards will Taylor Swift win in 2025? | How many Grammy awards will Taylor Swift win in 2025? |
| Resolution | 0 | 0 |
| Options | ('0', '1', '2', '3 or more') | ('0', '1', '2', 'Greater than 2') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 31797 | 31865 |
| Created At | 2025-01-21 13:57:50.512496+00:00 | 2025-01-23 18:06:36.599465+00:00 |
| Spot Scoring Time | 2025-01-23 23:23:00+00:00 | 2025-01-23 23:23:00+00:00 |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-06-14 13:00:13,714 - WARNING - from aib_analysis.main_logic.process_tournament - _log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 | Question 3 |
|-----------|---|---|---|
| URL | https://www.metaculus.com/questions/34382/ | https://www.metaculus.com/questions/34260/ | https://www.metaculus.com/questions/34706/ |
| Question Id | 33879 | 33757 | 34220 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many arms sales globally will the US State Department approve in March 2025? | How many arms sales globally will the US State Department approve in March 2025? | How many arms sales globally will the US State Department approve in March 2025? |
| Resolution | 5-9 | 5-9 | 0-5 |
| Options | ('0-4', '5-9', '>9') | ('0-4', '5-9', '>9') | ('0-5', '6-10', '>10') |
| Range Max | None | None | None |
| Range Min | None | None | None |
| Open Upper Bound | None | None | None |
| Open Lower Bound | None | None | None |
| Weight | 1.0 | 1.0 | 1.0 |
| Post Id | 34382 | 34260 | 34706 |
| Created At | 2025-01-25 07:08:58.779381+00:00 | 2025-01-25 06:31:51.259600+00:00 | 2025-02-01 05:24:04.045627+00:00 |
| Spot Scoring Time | 2025-01-29 07:00:00+00:00 | 2025-01-29 07:00:00+00:00 | 2025-02-09 00:44:00+00:00 |
| Notes | None | None | None |
| Tournament 1 | True | False | False |
| Tournament 2 | False | True | True |

2025-06-14 13:00:13,717 - WARNING - from aib_analysis.main_logic.process_tournament - _log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 | Question 3 |
|-----------|---|---|---|
| URL | https://www.metaculus.com/questions/34389/ | https://www.metaculus.com/questions/34281/ | https://www.metaculus.com/questions/34667/ |
| Question Id | 33886 | 33778 | 34181 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? | What Premier League position will Nottingham Forest F.C. be in on March 8, 2025? |
| Resolution | 3rd | 3rd | 3rd |
| Options | ('1st', '2nd', '3rd', '4th', '≥5th') | ('1st', '2nd', '3rd', '4th', '≥5th') | ('1st', '2nd', '3rd', '4th', '≥5th') |
| Range Max | None | None | None |
| Range Min | None | None | None |
| Open Upper Bound | None | None | None |
| Open Lower Bound | None | None | None |
| Weight | 1.0 | 1.0 | 0.5 |
| Post Id | 34389 | 34281 | 34667 |
| Created At | 2025-01-25 07:08:59.118741+00:00 | 2025-01-25 06:31:52.795962+00:00 | 2025-02-01 05:24:00.456127+00:00 |
| Spot Scoring Time | 2025-01-31 02:00:00+00:00 | 2025-01-31 02:00:00+00:00 | 2025-02-02 17:00:00+00:00 |
| Notes | None | None | None |
| Tournament 1 | True | False | False |
| Tournament 2 | False | True | True |

2025-06-14 13:00:13,721 - WARNING - from aib_analysis.main_logic.process_tournament - _log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/35002/ | https://www.metaculus.com/questions/34940/ |
| Question Id | 34488 | 34426 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | Which party will win the 2nd highest number of seats in the 2025 German federal election? | Which party will win the 2nd highest number of seats in the 2025 German federal election? |
| Resolution | Alternative for Germany | Alternative for Germany |
| Options | ('CDU/CSU', 'Alternative for Germany', 'Social Democratic Party', 'Greens', 'Another party') | ('CDU/CSU', 'Alternative for Germany', 'Social Democratic Party', 'Social Democratic Party', 'Another party') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 35002 | 34940 |
| Created At | 2025-02-08 04:20:42.357783+00:00 | 2025-02-08 04:04:07.666456+00:00 |
| Spot Scoring Time | 2025-02-10 06:00:00+00:00 | 2025-02-10 06:00:00+00:00 |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |
"""


"""
###################### Q1 Bot v Cup Matching Inconsistencies (excluding mismatched titles) ######################
2025-06-16 18:57:57,787 - WARNING - from aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/34862/ | https://www.metaculus.com/questions/34968/ |
| Question Id | 34356 | 34454 |
| Type | QuestionType.NUMERIC | QuestionType.NUMERIC |
| Question Text | How many earthquakes of magnitude ≥ 4 will happen near Santorini, Greece in the first week of March, 2025? | How many earthquakes of magnitude ≥ 4 will happen near Santorini, Greece in the first week of March, 2025? |
| Resolution | 0.0 | 0.0 |
| Options | None | None |
| Range Max | 150.0 | 150.0 |
| Range Min | 0.0 | 0.0 |
| Open Upper Bound | True | False |
| Open Lower Bound | False | False |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 34862 | 34968 |
| Created At | 2025-02-07 00:18:51.368391+00:00 | 2025-02-08 04:04:10.387471+00:00 |
| Spot Scoring Time | 2025-02-13 17:00:00+00:00 | 2025-02-13 17:00:00+00:00 |
| Project | 🏆 Quarterly Cup 🏆 | Q1 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-06-16 18:57:57,791 - WARNING - from aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/35318/ | https://www.metaculus.com/questions/35470/ |
| Question Id | 34788 | 34937 |
| Type | QuestionType.NUMERIC | QuestionType.NUMERIC |
| Question Text | What will be the IMDb rating of Severance's second season finale? | What will be the IMDb rating of Severance's second season finale? |
| Resolution | 9.6 | 9.6 |
| Options | None | None |
| Range Max | 10.0 | 10.0 |
| Range Min | 5.0 | 5.0 |
| Open Upper Bound | False | True |
| Open Lower Bound | True | True |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 35318 | 35470 |
| Created At | 2025-02-20 19:02:43.938942+00:00 | 2025-02-22 03:56:11.035398+00:00 |
| Spot Scoring Time | 2025-02-27 17:00:00+00:00 | 2025-02-27 17:00:00+00:00 |
| Project | 🏆 Quarterly Cup 🏆 | Q1 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-06-16 18:57:57,795 - WARNING - from aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/31849/ | https://www.metaculus.com/questions/34274/ |
| Question Id | 31360 | 33771 |
| Type | QuestionType.MULTIPLE_CHOICE | QuestionType.MULTIPLE_CHOICE |
| Question Text | How many hostages will Hamas release after January 26 and before April 5, 2025? | How many hostages will Hamas release after January 26 and before April 5, 2025? |
| Resolution | 20-29 | 30-39 |
| Options | ('≤9', '10-19', '20-29', '30-39', '≥40') | ('≤9', '10-19', '20-29', '30-39', '≥40') |
| Range Max | None | None |
| Range Min | None | None |
| Open Upper Bound | None | None |
| Open Lower Bound | None | None |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 31849 | 34274 |
| Created At | 2025-01-23 15:52:21.322919+00:00 | 2025-01-25 06:31:52.370373+00:00 |
| Spot Scoring Time | 2025-01-30 17:00:00+00:00 | 2025-01-30 02:00:00+00:00 |
| Project | 🏆 Quarterly Cup 🏆 | Q1 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |

2025-06-16 18:57:57,805 - WARNING - from aib_analysis.main_logic.process_tournament - log_title_mapping_inconsistencies  -
# Text-matched questions have different tournament-matching hashes (NOTE: If more than 2 questions are in this list then a question pair that matches will still be combined):
| Parameter | Question 1 | Question 2 |
|-----------|---|---|
| URL | https://www.metaculus.com/questions/35589/ | https://www.metaculus.com/questions/35888/ |
| Question Id | 35032 | 35322 |
| Type | QuestionType.NUMERIC | QuestionType.NUMERIC |
| Question Text | What will the total number of Tesla vehicle deliveries be for Q1 2025? | What will the total number of Tesla vehicle deliveries be for Q1 2025? |
| Resolution | -1e+32 | 336681.0 |
| Options | None | None |
| Range Max | 500000.0 | 500000.0 |
| Range Min | 350000.0 | 350000.0 |
| Open Upper Bound | True | True |
| Open Lower Bound | True | True |
| Zero Point | None | None |
| Weight | 1.0 | 1.0 |
| Post Id | 35589 | 35888 |
| Created At | 2025-02-28 14:37:26.012903+00:00 | 2025-03-08 04:57:09.648719+00:00 |
| Spot Scoring Time | 2025-03-08 17:00:00+00:00 | 2025-03-08 17:00:00+00:00 |
| Project | 🏆 Quarterly Cup 🏆 | Q1 AI Forecasting Benchmark Tournament |
| Notes | None | None |
| Tournament 1 | True | False |
| Tournament 2 | False | True |
"""
