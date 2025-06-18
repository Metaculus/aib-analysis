from __future__ import annotations

import logging

from pydantic import BaseModel, PrivateAttr, model_validator
from typing_extensions import Self

from aib_analysis.data_structures.custom_types import (
    AnnulledAmbiguousResolutionType,
    QuestionType,
)
from aib_analysis.data_structures.data_models import (
    Forecast,
    Question,
    Score,
    ScoreType,
    User,
)
from aib_analysis.data_structures.problem_questions import ProblemManager

logger = logging.getLogger(__name__)


class SimulatedTournament(BaseModel):
    name: str | None = None
    forecasts: list[Forecast]

    @property
    def users(self) -> list[User]:
        assert self._user_cache is not None, "User cache is not initialized"
        return list(self._user_cache.values())

    @property
    def questions(self) -> list[Question]:
        assert self._question_cache is not None, "Question cache is not initialized"
        return list(self._question_cache.values())

    @property
    def scores(self) -> list[Score]:
        assert self._scores_cache is not None, "Scores cache is not initialized"
        return list(self._scores_cache.values())

    @property
    def spot_forecasts(self) -> list[Forecast]:
        """
        The latest forecast for each user for each question that is before the spot scoring time
        """
        assert (
            self._spot_forecasts_cache is not None
        ), "Spot forecasts cache is not initialized"
        return list(self._spot_forecasts_cache.values())

    _user_cache: dict[str, User] = PrivateAttr(default_factory=dict)
    _question_cache: dict[int, Question] = PrivateAttr(default_factory=dict)
    _scores_cache: dict[str, Score] = PrivateAttr(default_factory=dict)

    _spot_forecasts_cache: dict[str, Forecast] = PrivateAttr(default_factory=dict)
    _question_to_spot_forecasts_cache: dict[int, list[Forecast]] = PrivateAttr(
        default_factory=dict
    )

    def question_to_spot_forecasts(self, question_id: int) -> list[Forecast]:
        if len(self._question_to_spot_forecasts_cache) == 0:
            for forecast in self.spot_forecasts:
                question_id_to_cache = forecast.question.question_id
                if question_id_to_cache not in self._question_to_spot_forecasts_cache:
                    self._question_to_spot_forecasts_cache[question_id_to_cache] = []
                self._question_to_spot_forecasts_cache[question_id_to_cache].append(
                    forecast
                )
        spot_forecasts = self._question_to_spot_forecasts_cache[question_id]
        return (
            spot_forecasts.copy()
        )  # Shallow copy (so you don't modify order of original list)

    def question_to_forecasts(self, question_id: int) -> list[Forecast]:
        return [
            forecast
            for forecast in self.forecasts
            if forecast.question.question_id == question_id
        ]

    def question_and_user_to_forecasts(
        self, question_id: int, user_name: str
    ) -> list[Forecast]:
        return [
            forecast
            for forecast in self.forecasts
            if forecast.question.question_id == question_id
            and forecast.user.name == user_name
        ]

    def user_to_scores(
        self, user_name: str, score_type: ScoreType | None = None
    ) -> list[Score]:
        scores = [
            score for score in self.scores if score.forecast.user.name == user_name
        ]
        if score_type is not None:
            scores = [score for score in scores if score.type == score_type]
        return scores.copy()

    def user_to_spot_forecasts(self, user_name: str) -> list[Forecast]:
        return [
            forecast
            for forecast in self.spot_forecasts
            if forecast.user.name == user_name
        ]

    def get_spot_score_for_question_and_user(
        self, question_id: int, user_name: str, score_type: ScoreType
    ) -> Score:
        assert (
            score_type.is_spot_score()
        ), "Only spot scores are supported for this method"
        scores = [
            score
            for score in self.scores
            if score.forecast.user.name == user_name
            and score.forecast.question.question_id == question_id
            and score.type == score_type
        ]
        assert len(scores) == 1, "Expected exactly for question for user if spot score"
        return scores[0]

    @model_validator(mode="after")
    def initialize_tournament(self) -> Self:
        logger.info(f"Initializing tournament {self.name}")
        self._remove_log_scale_questions()
        self._initialize_spot_forecast_cache()
        self._initialize_user_and_question_caches()
        self._initialize_scores_cache()

        logger.info(
            f"Finished initializing scoring caches for {len(self.scores)} scores"
        )

        self._validate_no_duplicate_questions()
        self._validate_one_user_question_per_spot_score()
        self._validate_num_scores_match_num_spot_forecasts()

        self._log_if_less_than_half_users_forecasted()
        self._log_if_weights_are_too_low()
        return self

    def _remove_log_scale_questions(self) -> None:
        non_log_scale_forecasts = [
            forecast
            for forecast in self.forecasts
            if not forecast.question.is_log_scale
        ]
        if not (len(non_log_scale_forecasts) == len(self.forecasts)):
            logger.warning(f"Removed {len(self.forecasts) - len(non_log_scale_forecasts)} log scale questions")
        self.forecasts = non_log_scale_forecasts

    def _initialize_spot_forecast_cache(self) -> None:
        spot_forecasts: dict[tuple[str, int], Forecast] = {}
        for forecast in self.forecasts:
            question = forecast.question
            user_name = forecast.user.name
            spot_time = question.spot_scoring_time
            if forecast.prediction_time >= spot_time:
                continue
            key = (user_name, question.question_id)
            current = spot_forecasts.get(key)
            if current is None or forecast.prediction_time > current.prediction_time:
                spot_forecasts[key] = forecast
        self._spot_forecasts_cache = {
            forecast.id: forecast for forecast in spot_forecasts.values()
        }

    def _initialize_user_and_question_caches(self) -> None:
        self._user_cache = {
            forecast.user.name: forecast.user for forecast in self.forecasts
        }
        self._question_cache = {
            forecast.question.question_id: forecast.question
            for forecast in self.forecasts
        }

    def _initialize_scores_cache(self) -> None:
        log_every_n = 1000
        all_scores: list[Score] = []
        for i, forecast in enumerate(self.spot_forecasts):
            should_log_scoring = i % log_every_n == 0
            if should_log_scoring:
                logger.info(
                    f"Caching scores for forecast {i} of {len(self.spot_forecasts)}"
                )
            if isinstance(
                forecast.question.resolution, AnnulledAmbiguousResolutionType
            ):
                continue
            new_scores = self._calculate_spot_scores_for_forecast(forecast)
            all_scores.extend(new_scores)
        self._scores_cache = {score.id: score for score in all_scores}

    def _calculate_spot_scores_for_forecast(
        self, forecast_to_score: Forecast
    ) -> list[Score]:
        if forecast_to_score.id not in self._spot_forecasts_cache:
            raise ValueError("Forecast to score must be in spot forecasts cache")

        resolution = forecast_to_score.question.resolution
        if isinstance(resolution, AnnulledAmbiguousResolutionType):
            return []

        spot_forecasts_from_others: list[Forecast] = self.question_to_spot_forecasts(
            forecast_to_score.question.question_id
        )
        spot_forecasts_from_others.remove(forecast_to_score)

        spot_peer_score = forecast_to_score.get_spot_peer_score(
            resolution, spot_forecasts_from_others
        )
        spot_baseline_score = forecast_to_score.get_spot_baseline_score(resolution)
        scores = [spot_peer_score, spot_baseline_score]
        return scores

    def _validate_one_user_question_per_spot_score(self) -> None:
        # Group scores by question_id, user_name, and score_type
        score_groups: dict[tuple[int, str, ScoreType], list[Score]] = {}
        for score in self.scores:
            if not score.type.is_spot_score():
                continue
            key = (
                score.forecast.question.question_id,
                score.forecast.user.name,
                score.type,
            )
            score_groups.setdefault(key, []).append(score)

        # Check each group has exactly one score
        for (question_id, user_name, score_type), scores in score_groups.items():
            if len(scores) != 1:
                raise ValueError(
                    f"Expected exactly one {score_type} score for question {question_id} and user {user_name}, "
                    f"but found {len(scores)} scores"
                )

    def _validate_num_scores_match_num_spot_forecasts(self) -> None:
        spot_forecasts = self.spot_forecasts
        non_annulled_spot_forecasts = len(
            [
                forecast
                for forecast in spot_forecasts
                if not forecast.question.is_annulled_or_ambiguous
            ]
        )
        num_scores = len(self.scores)
        num_score_types = len(
            [score_type for score_type in ScoreType if score_type.is_spot_score()]
        )
        expected_scores = non_annulled_spot_forecasts * num_score_types
        assert (
            num_scores == expected_scores
        ), f"Number of non-annulled spot forecasts ({non_annulled_spot_forecasts}) and scores ({num_scores}) do not match (expected {expected_scores} scores)"
        assert len(self.forecasts) > 0, "No forecasts found"
        assert len(spot_forecasts) > 0, "No spot forecasts found"


    def _validate_spot_forecasters_equal_forecasters(self) -> None:
        for question in self.questions:
            forecasts = self.question_to_forecasts(question.question_id)
            spot_forecasts = self.question_to_spot_forecasts(question.question_id)
            num_of_forecasts = len(forecasts)
            num_of_forecasters = len(set([f.user.name for f in forecasts]))
            num_of_spot_forecasts = len(spot_forecasts)
            num_of_spot_forecasters = len(set([f.user.name for f in spot_forecasts]))
            assert (
                num_of_forecasters == num_of_spot_forecasters
            ), f"Number of forecasts ({num_of_forecasts}) and number of spot forecasts ({num_of_spot_forecasts}) do not match for question {question.question_id} ({question.url})"
            assert (
                num_of_forecasts >= num_of_spot_forecasts
            ), f"Number of forecasts ({num_of_forecasts}) and number of spot forecasts ({num_of_spot_forecasts}) do not match for question {question.question_id} ({question.url})"

    def _validate_no_duplicate_questions(self) -> None:
        question_text_map: dict[str, list[Question]] = {}
        for question in self.questions:
            question_text_map.setdefault(question.question_text, []).append(question)

        duplicate_error_messages = []
        for question_text, questions in question_text_map.items():
            assert len(questions) > 0
            if len(questions) == 1:
                continue
            if ProblemManager.dont_log_in_duplicate_detection_within_tournament(
                questions
            ):
                logger.info(
                    f"Duplicate question is prequalified for q1 bot tournament: {[q.url for q in questions]}"
                )
                continue
            error_message = "# Duplicates for question text: " + question_text + "\n"
            error_message += Question.question_comparison_table(questions)
            duplicate_error_messages.append(error_message)

        if len(duplicate_error_messages) > 0:
            combined_error_message = "\n\n".join(duplicate_error_messages)
            logger.warning(
                f"Duplicate question texts found in questions: \n{combined_error_message}"
            )

        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Duplicate question IDs found in questions")

        if len(self.questions) != len(set(self.questions)):
            raise ValueError("Duplicate questions found in questions")

    def _log_if_less_than_half_users_forecasted(self) -> None:
        total_users = len(self.users)
        min_expected_forecasts = total_users / 4

        questions_with_too_few_forecasts: list[Question] = []
        messages = []
        for question in self.questions:
            forecasts_for_question = self.question_to_spot_forecasts(
                question.question_id
            )

            if len(forecasts_for_question) < min_expected_forecasts:
                messages.append(
                    f"Question {question.question_id} ({question.url}) has only "
                    f"{len(forecasts_for_question)} forecasts out of {total_users} participants"
                )
                questions_with_too_few_forecasts.append(question)

        if len(questions_with_too_few_forecasts) > 0:
            logger.warning(
                f"Found {len(questions_with_too_few_forecasts)} questions with too few forecasts. First 5 instances: {messages[:5]}"
            )

    def _log_if_weights_are_too_low(self) -> None:
        min_weight = 0.3
        for question in self.questions:
            if question.weight < min_weight:
                logger.warning(
                    f"Question {question.question_id} ({question.url}) has a weight of {question.weight}, "
                    f"which is less than the minimum expected weight of {min_weight}"
                )
