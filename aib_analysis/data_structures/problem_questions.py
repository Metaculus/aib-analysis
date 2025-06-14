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
    @classmethod
    def dont_log_in_tournament_matching(cls, questions: list[Question]) -> bool:
        return cls._is_in_problem_question_list(
            questions, cls._q1_bot_v_pro_matching_inconsistencies
        )

    @classmethod
    def dont_log_in_duplicate_detection_within_tournament(
        cls, questions: list[Question]
    ) -> bool:
        return cls._is_in_problem_question_list(
            questions, cls._q1_bot__in_tournament_title_duplicates
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
            if cls._is_in_problem_question_list(
                question_group, cls._q1_bot_v_pro_inconsistencies_to_force_match
            ):
                matches.append(question_group)
        return matches

    @classmethod
    def should_force_match_between_tournaments(cls, questions: list[Question]) -> bool:
        return cls._is_in_problem_question_list(
            questions, cls._q1_bot_v_pro_inconsistencies_to_force_match
        )

    @classmethod
    def _is_in_problem_question_list(
        cls, questions: list[Question], problem_question_list: list[ProblemQuestion]
    ) -> bool:
        for q in problem_question_list:
            matches = [q.question_matches(question) for question in questions]
            if all(matches):
                return True
            elif any(matches):
                raise ValueError(
                    f"One of the questions matches the problem question, but not all of them. Problem question: {q.question_text}, Questions: {questions}"
                )
            else:
                continue
        return False

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
            proposed_action="Remove this",  # TODO: Remove @Check
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

    _q1_bot_v_pro_negligable_inconsistencies: list[ProblemQuestion] = [
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

    # This is the full list of questions who have inconsistencies (e.g. options, resolutions, etc) between q1 pro tournament and q1 bot tournament
    _q1_bot_v_pro_matching_inconsistencies: list[ProblemQuestion] = [
        *_q1_bot_v_pro_inconsistencies_to_force_match,
        *_q1_bot_v_pro_to_remove_from_comparison,
        *_q1_bot_v_pro_negligable_inconsistencies,
    ]


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

2025-06-14 13:00:13,710 - WARNING - aib_analysis.process_tournament - _log_title_mapping_inconsistencies  -
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

2025-06-14 13:00:13,714 - WARNING - aib_analysis.process_tournament - _log_title_mapping_inconsistencies  -
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

2025-06-14 13:00:13,717 - WARNING - aib_analysis.process_tournament - _log_title_mapping_inconsistencies  -
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

2025-06-14 13:00:13,721 - WARNING - aib_analysis.process_tournament - _log_title_mapping_inconsistencies  -
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
