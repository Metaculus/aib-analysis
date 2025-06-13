
class ProblemType(Enum):
    BETWEEN_TOURNAMENT = "between_tournament"
    WITHIN_TOURNAMENT = "within_tournament"

class SolutionType(Enum):
    FORCE_MATCH = "force_match" # For "Between tournament" problem
    SKIP_QUESTION = "skip_question" # For "Between tournament" problem
    TAKE_LATEST = "take_latest" # For "Within tournament" problem
    TAKE_INDEX = "take_index" # For "Within tournament" problem

class ProblemGroup(BaseModel):
    notes: str
    questions: list[Question]
    problem_type: ProblemType
    solution_type: SolutionType | None = None
    index: int | None = None

    def solve_problem_group(self, questions_to_choose_from: list[Question]) -> Question | Literal["skip_question"]:
        if self.solution_type is None:
            raise ValueError("Solution type is not set")
        elif self.solution_type == SolutionType.FORCE_MATCH:
            return self.questions[0]
        elif self.solution_type == SolutionType.SKIP_QUESTION:
            return "skip_question"
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
        if len(input_questions) < 2:
            raise ValueError("Need at least 2 questions to find a problem group")

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
