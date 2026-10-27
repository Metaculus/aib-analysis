"""Compute every number in the Pros vs bots write-up from a season's simulation JSONs.

The JSONs are read with the json module rather than SimulatedTournament because pydantic
validation of the 400MB+ bot tournament is too slow for a report run.
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from scipy import stats as scipy_stats

from aib_analysis.math.accuracy import (
    base_rate_geometric_mean_probability,
    binary_probability_on_outcome,
    field_probability_multiplier,
    geometric_mean_probability,
    geometric_mean_probability_ci,
    paired_probability_ratio,
    yes_rate,
)
from aib_analysis.math.stats import (
    ConfidenceIntervalCalculator,
    MeanHypothesisCalculator,
)
from aib_analysis.pro_vs_bot_writeup.season_config import (
    PUBLISHED_BASELINE_BOT_RANKS,
    PUBLISHED_PRO_VS_BOT_RESULTS,
    SEASONS_WITHOUT_PRO_COMPARISON,
    SeasonConfig,
    seasons_before,
)

logger = logging.getLogger(__name__)

CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05
PRO_TEAM_NAME = "Pro Team"
BOT_TEAM_NAME = "Bot Team"
UNIFORM_REFERENCE_BOT = "metac-uniform-probability-bot"
BASELINE_METAC_BOT_PATTERN = re.compile(r"^metac-(?P<model>[^+\[\]]+)\+(?P<search>[^\[\]]+)$")
NO_RESEARCH_ONE_SHOT_SUFFIX = "-no-research-one-shot"
VARIANCE_TEST_PATTERN = re.compile(r"\((?P<label>variance-test[^)]*)\)$")
TOP_N = 10
ELIGIBLE_SHARE_OF_QUESTIONS = 0.5
PROMISING_MIN_SHARE_OF_QUESTIONS = 0.1
FIELD_CHART_MIN_QUESTIONS = 10
LUCK_BAND_Z = scipy_stats.norm.ppf(0.975)
QUESTION_TYPE_ORDER = ("binary", "numeric", "discrete", "multiple_choice")
TEAM_SIZE_FILE_PATTERN = re.compile(r"^\d+_pro_vs_bot_tournament__teams_size_(?P<size>\d+|all)\.json$")
COMBINED_QUESTION_NOTES_PATTERN = re.compile(r"^Combined \S+ \(QID:(?P<pro_id>\d+)\) and \S+ \(QID:(?P<bot_id>\d+)\)")

Leader = Literal["pros", "bots"]
Evidence = Literal["strong", "weak", "none"]


@dataclass(frozen=True)
class QuestionScore:
    question_id: int
    question_type: str
    score: float
    weight: float


@dataclass
class Participant:
    name: str
    user_type: str
    members: tuple[tuple[str, str], ...]
    scores: list[QuestionScore]

    @property
    def sum_of_scores(self) -> float:
        return sum(question_score.score for question_score in self.scores)

    @property
    def average_score(self) -> float:
        return self.sum_of_scores / len(self.scores)

    @property
    def question_count(self) -> int:
        return len(self.scores)

    @property
    def score_values(self) -> list[float]:
        return [question_score.score for question_score in self.scores]


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    name: str
    user_type: str
    sum_of_scores: float
    average_score: float
    ci: tuple[float, float] | None
    question_count: int


@dataclass(frozen=True)
class CloseTimeDifference:
    """A shared question whose bot copy closed at a different time than its Pro copy."""

    bot_question_id: int
    bot_close: datetime
    minutes_bot_closed_after_pro: float


@dataclass(frozen=True)
class TournamentOverview:
    bot_count: int
    metac_bot_count: int
    bot_question_count: int
    bot_scored_count: int
    bot_type_counts: dict[str, int]
    binary_yes_count: int
    binary_scored_count: int
    weighted_question_count: int
    shared_question_weights_below_1: tuple[float, ...]
    bot_only_scored_count: int
    pro_count: int
    non_pro_accounts_in_pro_tournament: int
    pro_question_count: int
    pro_scored_count: int
    pro_weighted_question_count: int
    first_question_close: date
    last_question_close: date
    close_time_differences: tuple[CloseTimeDifference, ...]
    non_primary_bots: tuple[str, ...]

    @property
    def bot_annulled_count(self) -> int:
        return self.bot_question_count - self.bot_scored_count

    @property
    def pro_annulled_count(self) -> int:
        return self.pro_question_count - self.pro_scored_count

    @property
    def non_metac_bot_count(self) -> int:
        return self.bot_count - self.metac_bot_count


@dataclass(frozen=True)
class HeadToHeadResult:
    question_count: int
    bot_team_mean: float
    ci: tuple[float, float]
    p_pros_better: float
    p_bots_better: float
    pro_wins: int
    bot_wins: int
    ties: int
    average_pro_win_margin: float | None
    average_bot_win_margin: float | None
    bot_team_members: tuple[str, ...]
    pro_member_count: int
    non_pro_member_count: int
    question_type_counts: dict[str, int]

    @property
    def leader(self) -> Leader:
        return "bots" if self.bot_team_mean > 0 else "pros"

    @property
    def leader_p_value(self) -> float:
        return self.p_bots_better if self.leader == "bots" else self.p_pros_better

    @property
    def evidence(self) -> Evidence:
        lower, upper = self.ci
        if upper < 0 or lower > 0:
            return "strong"
        if self.leader_p_value < SIGNIFICANCE_LEVEL:
            return "weak"
        return "none"


@dataclass(frozen=True)
class TeamSizeResult:
    """difference_from_main_ci compares the team with the main bot team on questions both forecast (None for the main team)."""

    team_size: int
    is_all_bots: bool
    mean: float
    ci: tuple[float, float] | None
    question_count: int
    difference_from_main_ci: tuple[float, float] | None = None


@dataclass(frozen=True)
class SeasonTrendPoint:
    season_label: str
    mean: float
    ci: tuple[float, float]
    question_count: int | None
    is_current: bool


@dataclass(frozen=True)
class SeasonTrend:
    points: list[SeasonTrendPoint]
    change_p_value: float | None
    skipped_seasons: tuple[str, ...]

    @property
    def current(self) -> SeasonTrendPoint:
        return self.points[-1]

    @property
    def previous(self) -> SeasonTrendPoint | None:
        return self.points[-2] if len(self.points) >= 2 else None


@dataclass(frozen=True)
class IndividualResults:
    rows: list[LeaderboardRow]
    pro_aliases: dict[str, str]
    excluded_account_count: int
    question_count: int

    @property
    def pro_count(self) -> int:
        return sum(1 for row in self.rows if row.user_type == "pro")

    @property
    def bot_count(self) -> int:
        return sum(1 for row in self.rows if row.user_type == "bot")

    @property
    def pros_in_top_10(self) -> int:
        return sum(1 for row in self.rows[:TOP_N] if row.user_type == "pro")

    @property
    def best_bot(self) -> LeaderboardRow:
        return next(row for row in self.rows if row.user_type == "bot")

    @property
    def pros_above_best_bot(self) -> int:
        return self.best_bot.rank - 1

    @property
    def lowest_pro_rank(self) -> int:
        return max(row.rank for row in self.rows if row.user_type == "pro")

    @property
    def bots_above_lowest_pro(self) -> int:
        return sum(1 for row in self.rows[: self.lowest_pro_rank] if row.user_type == "bot")


@dataclass(frozen=True)
class LuckBandCounts:
    shown: int
    above: int
    below: int
    expected_per_side: float
    more_spread_than_luck: bool


@dataclass(frozen=True)
class FieldSummary:
    rows: list[LeaderboardRow]
    scored_question_count: int
    eligible_min_questions: int
    eligible_count: int
    best_eligible: LeaderboardRow
    median_eligible_average: float
    worst_eligible: LeaderboardRow
    tenth_place: LeaderboardRow
    promising: list[LeaderboardRow]
    promising_min_questions: int
    per_question_score_sd: float
    uniform_reference: LeaderboardRow | None
    top_10_average_ci_width: float | None

    @property
    def top_10(self) -> list[LeaderboardRow]:
        return self.rows[:TOP_N]

    @property
    def top_bot_clearly_ahead(self) -> bool:
        top_ci = self.rows[0].ci
        tenth_ci = self.tenth_place.ci
        return top_ci is not None and tenth_ci is not None and top_ci[0] > tenth_ci[1]

    def luck_band_half_width(self, question_count: float) -> float:
        """Where a bot with a true average of 0 lands about 95% of the time, treating its scores like the field's."""
        return LUCK_BAND_Z * self.per_question_score_sd / math.sqrt(question_count)

    def luck_band_counts(self, min_questions: int) -> LuckBandCounts:
        shown = [row for row in self.rows if row.question_count >= min_questions]
        above = sum(1 for row in shown if row.average_score > self.luck_band_half_width(row.question_count))
        below = sum(1 for row in shown if row.average_score < -self.luck_band_half_width(row.question_count))
        share_outside_each_side = 1 - scipy_stats.norm.cdf(LUCK_BAND_Z)
        p_values = [
            scipy_stats.binomtest(count, len(shown), share_outside_each_side, alternative="greater").pvalue
            for count in (above, below)
        ]
        return LuckBandCounts(
            shown=len(shown),
            above=above,
            below=below,
            expected_per_side=share_outside_each_side * len(shown),
            more_spread_than_luck=min(p_values) < SIGNIFICANCE_LEVEL,
        )


@dataclass(frozen=True)
class BotTypicalProbability:
    name: str
    typical_probability: float
    binary_question_count: int


@dataclass(frozen=True)
class BotFieldAccuracy:
    """Typical probability on the outcome across the bot tournament's binary questions."""

    binary_question_count: int
    yes_rate: float
    base_rate_probability: float
    field_probability: float
    eligible_bots: list[BotTypicalProbability]
    top_bot: BotTypicalProbability
    top_bot_average_peer_score: float
    top_bot_field_multiplier: float
    worst_bot: BotTypicalProbability

    @property
    def median_probability(self) -> float:
        return statistics.median(bot.typical_probability for bot in self.eligible_bots)

    @property
    def share_below_base_rate(self) -> float:
        below = sum(1 for bot in self.eligible_bots if bot.typical_probability < self.base_rate_probability)
        return below / len(self.eligible_bots)


@dataclass(frozen=True)
class TeamAccuracy:
    """Typical probability on the outcome for the Pro team and bot team on the shared binary questions."""

    binary_question_count: int
    yes_rate: float
    base_rate_probability: float
    pro_team_probability: float
    pro_team_ci: tuple[float, float] | None
    bot_team_probability: float
    bot_team_ci: tuple[float, float] | None
    bot_to_pro_ratio: float
    bot_to_pro_ratio_ci: tuple[float, float] | None


@dataclass(frozen=True)
class BaselineRankPoint:
    season_label: str
    rank: int
    bot_count: int
    bot_name: str | None


@dataclass(frozen=True)
class RunnerUpComparison:
    """The top same-prompt Metac Bot's score minus the runner-up's, on the questions both forecast."""

    runner_up: LeaderboardRow
    shared_question_count: int
    mean_difference: float
    difference_ci: tuple[float, float] | None


@dataclass(frozen=True)
class MetacResults:
    baseline_rows: list[LeaderboardRow]
    other_rows: list[LeaderboardRow]
    metac_bot_count: int
    scored_question_count: int
    average_questions_per_baseline_bot: float
    baseline_history: list[BaselineRankPoint]
    first_question_close: date
    first_scored_close_by_name: dict[str, date]
    runner_up_comparison: RunnerUpComparison | None

    @property
    def top_baseline(self) -> LeaderboardRow:
        return self.baseline_rows[0]

    def higher_average_on_fewer_questions(self) -> list[LeaderboardRow]:
        """Same-prompt bots that beat the top bot's average but rank lower by sum because they forecast less."""
        top = self.top_baseline
        return sorted(
            (
                row
                for row in self.baseline_rows[1:]
                if row.average_score > top.average_score and row.question_count < top.question_count
            ),
            key=lambda row: row.average_score,
            reverse=True,
        )

    def days_after_season_start(self, name: str) -> int:
        return (self.first_scored_close_by_name[name] - self.first_question_close).days

    @property
    def current_standing(self) -> BaselineRankPoint:
        return self.baseline_history[-1]



@dataclass(frozen=True)
class MetacExperimentGroups:
    search_variants: dict[str, list[LeaderboardRow]]
    reasoning_pairs: list[tuple[LeaderboardRow, LeaderboardRow]]
    variance_tests: dict[str, list[LeaderboardRow]]
    no_research_pairs: list[tuple[LeaderboardRow, list[LeaderboardRow]]]


@dataclass(frozen=True)
class ExampleQuestion:
    question_type: str
    title: str
    options: tuple[str, ...] | None
    resolution_text: str
    url: str
    was_pinned: bool


@dataclass(frozen=True)
class SeasonResults:
    overview: TournamentOverview
    head_to_head: HeadToHeadResult
    season_trend: SeasonTrend
    individuals: IndividualResults
    team_sizes: list[TeamSizeResult]
    field: FieldSummary
    metac: MetacResults
    experiment_groups: MetacExperimentGroups
    example_questions: list[ExampleQuestion]
    bot_field_accuracy: BotFieldAccuracy
    team_accuracy: TeamAccuracy


def load_json(path: Path) -> dict[str, Any]:
    logger.info(f"Loading {path.name}")
    with open(path) as file_handle:
        return json.load(file_handle)


def find_simulation_file(folder: Path, suffix: str) -> Path:
    pattern = re.compile(r"^\d+_" + re.escape(suffix) + "$")
    matches = sorted(path for path in folder.iterdir() if pattern.match(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '<number>_{suffix}' in {folder}, found {[path.name for path in matches]}. "
            "This template needs the output folder of a season simulation that ran a Pro tournament "
            "(e.g. aib_analysis/run_spring_simulations.py)."
        )
    return matches[0]


def find_team_size_files(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        match = TEAM_SIZE_FILE_PATTERN.match(path.name)
        if match and match.group("size") not in files:
            files[match.group("size")] = path
    if len(files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 '<number>_pro_vs_bot_tournament__teams_size_<N>.json' files in {folder}, "
            f"found {sorted(files)}"
        )
    return files


def spot_peer_participants(raw: dict[str, Any]) -> dict[str, Participant]:
    participants: dict[str, Participant] = {}
    for score in raw["scores_cache"].values():
        if score["type"] != "spot_peer":
            continue
        forecast = score["forecast"]
        user = forecast["user"]
        participant = participants.get(user["name"])
        if participant is None:
            participant = Participant(
                name=user["name"],
                user_type=user["type"],
                members=tuple((member["name"], member["type"]) for member in user["aggregated_users"]),
                scores=[],
            )
            participants[user["name"]] = participant
        question = forecast["question"]
        participant.scores.append(
            QuestionScore(question["question_id"], question["type"], score["score"], question["weight"])
        )
    return participants


def binary_outcome_probabilities(raw: dict[str, Any]) -> dict[str, dict[int, tuple[float, float]]]:
    """user name -> question id -> (probability on the outcome, question weight), for scored binary spot forecasts."""
    probabilities: dict[str, dict[int, tuple[float, float]]] = {}
    for score in raw["scores_cache"].values():
        forecast = score["forecast"]
        question = forecast["question"]
        if score["type"] != "spot_peer" or question["type"] != "binary" or not is_scored(question):
            continue
        probability = binary_probability_on_outcome(forecast["prediction"][0], question["resolution"])
        probabilities.setdefault(forecast["user"]["name"], {})[question["question_id"]] = (probability, question["weight"])
    return probabilities


def questions_by_id(raw: dict[str, Any]) -> dict[int, dict[str, Any]]:
    questions: dict[int, dict[str, Any]] = {}
    for forecast in raw["forecasts"]:
        questions.setdefault(forecast["question"]["question_id"], forecast["question"])
    return questions


def is_scored(question: dict[str, Any]) -> bool:
    return question["resolution"] is not None and question["unscored_resolution_reason"] is None


def close_date(question: dict[str, Any]) -> date:
    return datetime.fromisoformat(question["spot_scoring_time"]).date()


def bot_label(name: str, non_primary_bots: tuple[str, ...]) -> str:
    return f"{name} (non-primary)" if name in non_primary_bots else name


def find_non_primary_bots(raw: dict[str, Any]) -> tuple[str, ...]:
    """Bots outside Metaculus that their makers didn't mark as primary; the official leaderboard doesn't rank them."""
    names = {
        score["forecast"]["user"]["name"]
        for score in raw["scores_cache"].values()
        if score["forecast"]["user"]["type"] == "bot" and score["forecast"]["user"]["is_primary_bot"] is False
    }
    return tuple(sorted(name for name in names if not is_metac_bot(name)))


def shared_question_pairs(team_raw: dict[str, Any]) -> list[tuple[int, int]]:
    """(Pro question id, bot question id) for each question in a Pro vs bot team comparison."""
    pairs = []
    for question in questions_by_id(team_raw).values():
        match = COMBINED_QUESTION_NOTES_PATTERN.match(question["notes"] or "")
        if match is None:
            raise ValueError(f"Can't find the Pro and bot question ids in the notes of question {question['question_id']}")
        pairs.append((int(match["pro_id"]), int(match["bot_id"])))
    return pairs


def check_comparison_uses_pro_weights(comparison_weights: dict[int, float], pro_questions: dict[int, dict[str, Any]]) -> None:
    """The write-up says the Pro vs bot comparison uses the Pro questions' weights; stop if the simulation didn't."""
    mismatched = [question_id for question_id, weight in comparison_weights.items() if weight != pro_questions[question_id]["weight"]]
    if mismatched:
        raise ValueError(f"{len(mismatched)} Pro vs bot questions don't use the Pro question's weight, e.g. question {mismatched[0]}")


def find_close_time_differences(
    pairs: list[tuple[int, int]], pro_questions: dict[int, dict[str, Any]], bot_questions: dict[int, dict[str, Any]]
) -> tuple[CloseTimeDifference, ...]:
    differences = []
    for pro_id, bot_id in pairs:
        pro_close = datetime.fromisoformat(pro_questions[pro_id]["spot_scoring_time"])
        bot_close = datetime.fromisoformat(bot_questions[bot_id]["spot_scoring_time"])
        if bot_close != pro_close:
            differences.append(CloseTimeDifference(bot_id, bot_close, (bot_close - pro_close).total_seconds() / 60))
    return tuple(sorted(differences, key=lambda difference: difference.bot_close))


def first_scored_close_by_name(raw: dict[str, Any]) -> dict[str, date]:
    first_closes: dict[str, date] = {}
    for score in raw["scores_cache"].values():
        if score["type"] != "spot_peer":
            continue
        name = score["forecast"]["user"]["name"]
        question_close = close_date(score["forecast"]["question"])
        first_closes[name] = min(first_closes.get(name, question_close), question_close)
    return first_closes


def is_metac_bot(name: str) -> bool:
    lowered = name.lower()
    return "metac-" in lowered or "mf-bot-" in lowered


def is_baseline_metac_bot(name: str) -> bool:
    return BASELINE_METAC_BOT_PATTERN.match(name) is not None


def confidence_interval(values: list[float]) -> tuple[float, float] | None:
    try:
        interval = ConfidenceIntervalCalculator.confidence_interval_from_observations(
            values, CONFIDENCE_LEVEL
        )
    except (ValueError, AssertionError) as error:
        logger.debug(f"No confidence interval for {len(values)} observations: {error}")
        return None
    return interval.lower_bound, interval.upper_bound


def build_leaderboard(participants: list[Participant]) -> list[LeaderboardRow]:
    ranked = sorted(participants, key=lambda participant: participant.sum_of_scores, reverse=True)
    rows = [
        LeaderboardRow(
            rank=rank,
            name=participant.name,
            user_type=participant.user_type,
            sum_of_scores=participant.sum_of_scores,
            average_score=participant.average_score,
            ci=confidence_interval(participant.score_values),
            question_count=participant.question_count,
        )
        for rank, participant in enumerate(ranked, start=1)
    ]
    rows_without_ci = sum(1 for row in rows if row.ci is None)
    if rows_without_ci:
        logger.info(
            f"{rows_without_ci} of {len(rows)} leaderboard rows have no 95% CI "
            "(3 or fewer questions, or under 30 questions and failing the normality check)"
        )
    return rows


def rerank(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    return [replace(row, rank=rank) for rank, row in enumerate(rows, start=1)]


def compute_head_to_head(raw: dict[str, Any], config: SeasonConfig) -> HeadToHeadResult:
    participants = spot_peer_participants(raw)
    for team_name in (BOT_TEAM_NAME, PRO_TEAM_NAME):
        if team_name not in participants:
            raise ValueError(f"No '{team_name}' scores in the team comparison file. Found: {sorted(participants)}")
    bot_team = participants[BOT_TEAM_NAME]
    pro_team = participants[PRO_TEAM_NAME]
    bot_scores_by_question = {question_score.question_id: question_score.score for question_score in bot_team.scores}
    pro_scores_by_question = {question_score.question_id: question_score.score for question_score in pro_team.scores}
    if set(bot_scores_by_question) != set(pro_scores_by_question):
        raise ValueError("The Pro Team and Bot Team were scored on different questions")
    asymmetric_questions = [
        question_id
        for question_id, bot_score in bot_scores_by_question.items()
        if abs(bot_score + pro_scores_by_question[question_id]) > 1e-6
    ]
    if asymmetric_questions:
        raise ValueError(
            f"Head-to-head scores should be exact opposites, but {len(asymmetric_questions)} questions are not "
            f"(e.g. question {asymmetric_questions[0]}). Is a third participant in the team comparison file?"
        )
    if len(bot_team.members) != config.main_bot_team_size:
        raise ValueError(
            f"Expected a {config.main_bot_team_size}-bot team, found {len(bot_team.members)} members"
        )
    non_bot_members = sum(1 for _name, member_type in bot_team.members if member_type != "bot")
    if non_bot_members:
        raise ValueError(f"The main bot team has {non_bot_members} non-bot members")
    non_pro_members = sum(1 for _name, member_type in pro_team.members if member_type != "pro")
    if non_pro_members != config.expected_non_pro_members_in_pro_team:
        raise ValueError(
            f"The Pro Team aggregate has {non_pro_members} non-Pro members, but the season config "
            f"expects {config.expected_non_pro_members_in_pro_team}. Check who forecast in the Pro "
            "tournament and update expected_non_pro_members_in_pro_team only after confirming it is intended."
        )

    question_ids = sorted(bot_scores_by_question)
    bot_scores = [bot_scores_by_question[question_id] for question_id in question_ids]
    pro_scores = [pro_scores_by_question[question_id] for question_id in question_ids]
    bot_team_mean = statistics.fmean(bot_scores)
    if bot_team_mean == 0:
        raise ValueError("The bot team and Pro team tied exactly; the precommitted phrasing has no tie branch")
    interval = confidence_interval(bot_scores)
    if interval is None:
        raise ValueError(
            f"Could not compute a t-based 95% CI for the head-to-head scores ({len(bot_scores)} questions)"
        )
    bot_win_scores = [score for score in bot_scores if score > 0]
    pro_win_scores = [-score for score in bot_scores if score < 0]
    return HeadToHeadResult(
        question_count=len(bot_scores),
        bot_team_mean=bot_team_mean,
        ci=interval,
        p_pros_better=MeanHypothesisCalculator.test_if_mean_is_greater_than_hypothesis_mean(
            pro_scores, 0, CONFIDENCE_LEVEL
        ).p_value,
        p_bots_better=MeanHypothesisCalculator.test_if_mean_is_greater_than_hypothesis_mean(
            bot_scores, 0, CONFIDENCE_LEVEL
        ).p_value,
        pro_wins=len(pro_win_scores),
        bot_wins=len(bot_win_scores),
        ties=len(bot_scores) - len(pro_win_scores) - len(bot_win_scores),
        average_pro_win_margin=statistics.fmean(pro_win_scores) if pro_win_scores else None,
        average_bot_win_margin=statistics.fmean(bot_win_scores) if bot_win_scores else None,
        bot_team_members=tuple(name for name, _member_type in bot_team.members),
        pro_member_count=len(pro_team.members) - non_pro_members,
        non_pro_member_count=non_pro_members,
        question_type_counts=dict(Counter(question_score.question_type for question_score in bot_team.scores)),
    )


def standard_error_from_ci(point: SeasonTrendPoint) -> float:
    half_width = (point.ci[1] - point.ci[0]) / 2
    if point.question_count is None:
        critical_value = scipy_stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2)
    else:
        critical_value = scipy_stats.t.ppf(0.5 + CONFIDENCE_LEVEL / 2, point.question_count - 1)
    return half_width / critical_value


def season_change_p_value(previous: SeasonTrendPoint, current: SeasonTrendPoint) -> float:
    """Two-sided z-test for a difference between two seasons' means, using each season's CI."""
    combined_standard_error = math.hypot(standard_error_from_ci(previous), standard_error_from_ci(current))
    z_statistic = (current.mean - previous.mean) / combined_standard_error
    return float(2 * scipy_stats.norm.sf(abs(z_statistic)))


def compute_season_trend(config: SeasonConfig, head_to_head: HeadToHeadResult) -> SeasonTrend:
    earlier_seasons = set(seasons_before(config.label))
    points = [
        SeasonTrendPoint(result.season_label, result.mean, (result.ci_lower, result.ci_upper), result.question_count, False)
        for result in PUBLISHED_PRO_VS_BOT_RESULTS
        if result.season_label in earlier_seasons
    ]
    for published in PUBLISHED_PRO_VS_BOT_RESULTS:
        if published.season_label == config.label and abs(published.mean - head_to_head.bot_team_mean) > 0.005:
            logger.warning(
                f"{config.label} was published with a bot team score of {published.mean}, but these "
                f"simulations give {head_to_head.bot_team_mean:.2f}"
            )
    points.append(
        SeasonTrendPoint(config.label, head_to_head.bot_team_mean, head_to_head.ci, head_to_head.question_count, True)
    )
    change_p_value = season_change_p_value(points[-2], points[-1]) if len(points) >= 2 else None
    skipped_seasons = tuple(season for season in SEASONS_WITHOUT_PRO_COMPARISON if season in earlier_seasons)
    return SeasonTrend(points=points, change_p_value=change_p_value, skipped_seasons=skipped_seasons)


def compute_team_sizes(
    files: dict[str, Path], pro_questions: dict[int, dict[str, Any]], main_size_label: str, humans_allowed: bool
) -> list[TeamSizeResult]:
    teams: dict[str, Participant] = {}
    for size_label, path in files.items():
        raw = load_json(path)
        check_comparison_uses_pro_weights(
            {question_id: question["weight"] for question_id, question in questions_by_id(raw).items()}, pro_questions
        )
        bot_team = spot_peer_participants(raw).get(BOT_TEAM_NAME)
        if bot_team is None:
            raise ValueError(f"No '{BOT_TEAM_NAME}' scores in {path.name}")
        if size_label != "all" and int(size_label) != len(bot_team.members):
            raise ValueError(f"{path.name} says size {size_label} but its bot team has {len(bot_team.members)} members")
        human_count = sum(1 for _name, member_type in bot_team.members if member_type != "bot")
        if human_count and not humans_allowed:
            raise ValueError(
                f"{path.name}: the bot team has {human_count} human accounts. Rerun the simulations, which remove "
                "humans after the bot leaderboard, or set human_accounts_in_bot_teams_allowed for this season."
            )
        teams[size_label] = bot_team
    main_scores = {question_score.question_id: question_score.score for question_score in teams[main_size_label].scores}
    results: list[TeamSizeResult] = []
    for size_label, bot_team in teams.items():
        scores = {question_score.question_id: question_score.score for question_score in bot_team.scores}
        differences = [scores[question_id] - main_scores[question_id] for question_id in scores.keys() & main_scores.keys()]
        results.append(
            TeamSizeResult(
                team_size=len(bot_team.members),
                is_all_bots=size_label == "all",
                mean=bot_team.average_score,
                ci=confidence_interval(bot_team.score_values),
                question_count=bot_team.question_count,
                difference_from_main_ci=None if size_label == main_size_label else confidence_interval(differences),
            )
        )
    return sorted(results, key=lambda result: result.team_size)


def anonymize_pros(rows: list[LeaderboardRow]) -> tuple[list[LeaderboardRow], dict[str, str]]:
    pro_names = [row.name for row in rows if row.user_type == "pro"]
    aliases = {
        name: f"Pro {chr(ord('A') + index)}" if index < 26 else f"Pro {index + 1}"
        for index, name in enumerate(pro_names)
    }
    return [replace(row, name=aliases.get(row.name, row.name)) for row in rows], aliases


def compute_individuals(
    raw: dict[str, Any], pro_names: set[str], bot_names: set[str], question_count: int
) -> IndividualResults:
    participants = spot_peer_participants(raw)
    included = [
        participant
        for participant in participants.values()
        if (participant.user_type == "pro" and participant.name in pro_names)
        or (participant.user_type == "bot" and participant.name in bot_names)
    ]
    excluded_count = len(participants) - len(included)
    if excluded_count:
        logger.info(
            f"Leaving {excluded_count} accounts out of the individual leaderboard (humans from the bot "
            "tournament and bots from the Pro tournament); they stay in everyone's peer baseline"
        )
    missing_pros = pro_names - {participant.name for participant in included}
    if missing_pros:
        raise ValueError(f"{len(missing_pros)} Pros have no scores in the combined individual tournament")
    rows, aliases = anonymize_pros(build_leaderboard(included))
    return IndividualResults(
        rows=rows, pro_aliases=aliases, excluded_account_count=excluded_count, question_count=question_count
    )


def compute_field(bot_rows: list[LeaderboardRow], all_scores: list[float], scored_question_count: int) -> FieldSummary:
    eligible_min_questions = math.ceil(scored_question_count * ELIGIBLE_SHARE_OF_QUESTIONS)
    promising_min_questions = math.ceil(scored_question_count * PROMISING_MIN_SHARE_OF_QUESTIONS)
    eligible = [
        row for row in bot_rows if row.question_count >= eligible_min_questions and row.name != UNIFORM_REFERENCE_BOT
    ]
    if not eligible:
        raise ValueError(f"No bot forecast on at least {eligible_min_questions} questions")
    eligible_by_average = sorted(eligible, key=lambda row: row.average_score)
    tenth_place = bot_rows[min(TOP_N, len(bot_rows)) - 1]
    promising = sorted(
        (
            row
            for row in bot_rows
            if promising_min_questions <= row.question_count < eligible_min_questions
            and row.average_score > tenth_place.average_score
        ),
        key=lambda row: row.average_score,
        reverse=True,
    )
    top_10_widths = [row.ci[1] - row.ci[0] for row in bot_rows[:TOP_N] if row.ci is not None]
    uniform_rows = [row for row in bot_rows if row.name == UNIFORM_REFERENCE_BOT]
    return FieldSummary(
        rows=bot_rows,
        scored_question_count=scored_question_count,
        eligible_min_questions=eligible_min_questions,
        eligible_count=len(eligible),
        best_eligible=eligible_by_average[-1],
        median_eligible_average=statistics.median(row.average_score for row in eligible),
        worst_eligible=eligible_by_average[0],
        tenth_place=tenth_place,
        promising=promising,
        promising_min_questions=promising_min_questions,
        per_question_score_sd=statistics.pstdev(all_scores),
        uniform_reference=uniform_rows[0] if uniform_rows else None,
        top_10_average_ci_width=statistics.fmean(top_10_widths) if top_10_widths else None,
    )


def compare_with_runner_up(
    baseline_rows: list[LeaderboardRow], participants: dict[str, Participant]
) -> RunnerUpComparison | None:
    if len(baseline_rows) < 2:
        return None
    top_scores = {score.question_id: score.score for score in participants[baseline_rows[0].name].scores}
    runner_up_scores = {score.question_id: score.score for score in participants[baseline_rows[1].name].scores}
    shared = sorted(top_scores.keys() & runner_up_scores.keys())
    if not shared:
        raise ValueError(f"{baseline_rows[0].name} and {baseline_rows[1].name} share no scored questions")
    differences = [top_scores[question_id] - runner_up_scores[question_id] for question_id in shared]
    return RunnerUpComparison(
        runner_up=baseline_rows[1],
        shared_question_count=len(shared),
        mean_difference=statistics.fmean(differences),
        difference_ci=confidence_interval(differences),
    )


def compute_metac(metac_raw: dict[str, Any], bot_rows: list[LeaderboardRow], season_label: str) -> MetacResults:
    metac_participants = spot_peer_participants(metac_raw)
    scored_question_count = sum(1 for question in questions_by_id(metac_raw).values() if is_scored(question))
    all_rows = build_leaderboard([participant for participant in metac_participants.values() if participant.user_type == "bot"])
    baseline_rows = rerank([row for row in all_rows if is_baseline_metac_bot(row.name)])
    if not baseline_rows:
        raise ValueError("No Metac Bot names match the same-prompt baseline pattern metac-<model>+<search>")
    other_rows = [row for row in all_rows if not is_baseline_metac_bot(row.name)]
    best_baseline_in_tournament = next((row for row in bot_rows if is_baseline_metac_bot(row.name)), None)
    if best_baseline_in_tournament is None:
        raise ValueError("No same-prompt baseline Metac Bot appears in the bot tournament leaderboard")
    earlier_seasons = set(seasons_before(season_label))
    history = [
        BaselineRankPoint(published.season_label, published.rank, published.bot_count, None)
        for published in PUBLISHED_BASELINE_BOT_RANKS
        if published.season_label in earlier_seasons
    ]
    history.append(
        BaselineRankPoint(season_label, best_baseline_in_tournament.rank, len(bot_rows), best_baseline_in_tournament.name)
    )
    for published in PUBLISHED_BASELINE_BOT_RANKS:
        if published.season_label == season_label and (published.rank, published.bot_count) != (
            best_baseline_in_tournament.rank,
            len(bot_rows),
        ):
            logger.warning(
                f"{season_label} was published with the best baseline bot at {published.rank} of "
                f"{published.bot_count}, but these simulations give {best_baseline_in_tournament.rank} of {len(bot_rows)}"
            )
    return MetacResults(
        baseline_rows=baseline_rows,
        other_rows=other_rows,
        metac_bot_count=len(all_rows),
        scored_question_count=scored_question_count,
        average_questions_per_baseline_bot=statistics.fmean(row.question_count for row in baseline_rows),
        baseline_history=history,
        first_question_close=min(close_date(question) for question in questions_by_id(metac_raw).values()),
        first_scored_close_by_name=first_scored_close_by_name(metac_raw),
        runner_up_comparison=compare_with_runner_up(baseline_rows, metac_participants),
    )


def _typical_probability(name: str, probabilities_by_question: dict[int, tuple[float, float]]) -> BotTypicalProbability:
    probabilities = [probability for probability, _weight in probabilities_by_question.values()]
    weights = [weight for _probability, weight in probabilities_by_question.values()]
    return BotTypicalProbability(name, geometric_mean_probability(probabilities, weights), len(probabilities))


def compute_bot_field_accuracy(
    bot_raw: dict[str, Any], field: FieldSummary, bot_participants: dict[str, Participant]
) -> BotFieldAccuracy:
    outcome_probabilities = binary_outcome_probabilities(bot_raw)
    bot_names = {row.name for row in field.rows}
    questions = {
        question_id: question
        for question_id, question in questions_by_id(bot_raw).items()
        if question["type"] == "binary" and is_scored(question)
    }
    question_ids = sorted(questions)
    resolutions = [questions[question_id]["resolution"] for question_id in question_ids]
    weights = [questions[question_id]["weight"] for question_id in question_ids]
    field_probabilities = []
    for question_id in question_ids:
        bot_probabilities = [
            outcome_probabilities[name][question_id][0]
            for name in bot_names
            if question_id in outcome_probabilities.get(name, {})
        ]
        field_probabilities.append(geometric_mean_probability(bot_probabilities))
    eligible_names = [
        row.name
        for row in field.rows
        if row.question_count >= field.eligible_min_questions and row.name != UNIFORM_REFERENCE_BOT
    ]
    eligible_without_binary = [name for name in eligible_names if not outcome_probabilities.get(name)]
    if eligible_without_binary:
        logger.warning(f"{len(eligible_without_binary)} eligible bots made no binary forecasts and are left out of the accuracy numbers")
    eligible_bots = [
        _typical_probability(name, outcome_probabilities[name]) for name in eligible_names if outcome_probabilities.get(name)
    ]
    top_row = field.rows[0]
    if not outcome_probabilities.get(top_row.name):
        raise ValueError(f"The top bot {top_row.name} made no binary forecasts")
    top_scores = bot_participants[top_row.name].scores
    return BotFieldAccuracy(
        binary_question_count=len(question_ids),
        yes_rate=yes_rate(resolutions),
        base_rate_probability=base_rate_geometric_mean_probability(resolutions, weights),
        field_probability=geometric_mean_probability(field_probabilities, weights),
        eligible_bots=eligible_bots,
        top_bot=_typical_probability(top_row.name, outcome_probabilities[top_row.name]),
        top_bot_average_peer_score=top_row.average_score,
        top_bot_field_multiplier=field_probability_multiplier(
            [question_score.score for question_score in top_scores],
            [question_score.weight for question_score in top_scores],
            [question_score.question_type in ("numeric", "discrete") for question_score in top_scores],
        ),
        worst_bot=next(bot for bot in eligible_bots if bot.name == field.worst_eligible.name),
    )


def compute_team_accuracy(team_raw: dict[str, Any]) -> TeamAccuracy:
    outcome_probabilities = binary_outcome_probabilities(team_raw)
    pro_team = outcome_probabilities.get(PRO_TEAM_NAME, {})
    bot_team = outcome_probabilities.get(BOT_TEAM_NAME, {})
    if set(pro_team) != set(bot_team) or not pro_team:
        raise ValueError("The Pro Team and Bot Team need forecasts on the same, non-empty set of binary questions")
    question_ids = sorted(pro_team)
    weights = [pro_team[question_id][1] for question_id in question_ids]
    if any(weight != 1 for weight in weights):
        logger.warning("Some shared binary questions have weights below 1; the team accuracy CIs ignore weights")
    questions = questions_by_id(team_raw)
    resolutions = [questions[question_id]["resolution"] for question_id in question_ids]
    pro_probabilities = [pro_team[question_id][0] for question_id in question_ids]
    bot_probabilities = [bot_team[question_id][0] for question_id in question_ids]
    ratio, ratio_ci = paired_probability_ratio(bot_probabilities, pro_probabilities)
    return TeamAccuracy(
        binary_question_count=len(question_ids),
        yes_rate=yes_rate(resolutions),
        base_rate_probability=base_rate_geometric_mean_probability(resolutions, weights),
        pro_team_probability=geometric_mean_probability(pro_probabilities, weights),
        pro_team_ci=geometric_mean_probability_ci(pro_probabilities),
        bot_team_probability=geometric_mean_probability(bot_probabilities, weights),
        bot_team_ci=geometric_mean_probability_ci(bot_probabilities),
        bot_to_pro_ratio=ratio,
        bot_to_pro_ratio_ci=ratio_ci,
    )


def split_baseline_name(name: str) -> tuple[str, str, str | None]:
    """(model, search, variance-test label) for a same-prompt baseline Metac Bot name."""
    match = BASELINE_METAC_BOT_PATTERN.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not a same-prompt baseline Metac Bot name")
    search = match.group("search")
    variance_match = VARIANCE_TEST_PATTERN.search(search)
    if variance_match:
        return match.group("model"), search[: variance_match.start()], variance_match.group("label")
    return match.group("model"), search, None


def find_metac_experiment_groups(baseline_rows: list[LeaderboardRow], other_rows: list[LeaderboardRow]) -> MetacExperimentGroups:
    parsed = [(row, *split_baseline_name(row.name)) for row in baseline_rows]
    searches_by_model: dict[str, list[LeaderboardRow]] = {}
    standard_by_model_and_search: dict[tuple[str, str], LeaderboardRow] = {}
    runs_by_setup: dict[tuple[str, str], list[LeaderboardRow]] = {}
    for row, model, search, variance_label in parsed:
        runs_by_setup.setdefault((model, search), []).append(row)
        if variance_label is None:
            searches_by_model.setdefault(model, []).append(row)
            standard_by_model_and_search[(model, search)] = row
    reasoning_pairs = [
        (standard_by_model_and_search[(model.removesuffix("-high"), search)], row)
        for (model, search), row in standard_by_model_and_search.items()
        if model.endswith("-high") and (model.removesuffix("-high"), search) in standard_by_model_and_search
    ]
    variance_tests = {
        f"{model}+{search}": rows
        for (model, search), rows in runs_by_setup.items()
        if len(rows) >= 2
    }
    no_research_pairs = []
    for row in other_rows:
        if not row.name.endswith(NO_RESEARCH_ONE_SHOT_SUFFIX):
            continue
        model = row.name.removeprefix("metac-").removesuffix(NO_RESEARCH_ONE_SHOT_SUFFIX)
        standard = searches_by_model.get(model, []) + searches_by_model.get(f"{model}-high", [])
        if standard:
            no_research_pairs.append((row, standard))
    return MetacExperimentGroups(
        search_variants={model: rows for model, rows in searches_by_model.items() if len(rows) >= 3},
        reasoning_pairs=reasoning_pairs,
        variance_tests=variance_tests,
        no_research_pairs=no_research_pairs,
    )


def format_resolution(question: dict[str, Any]) -> str:
    resolution = question["resolution"]
    if question["type"] == "binary":
        return "Yes" if resolution else "No"
    if question["type"] == "multiple_choice":
        return str(resolution)
    if resolution == "above_upper_bound":
        return "above the upper bound"
    if resolution == "below_lower_bound":
        return "below the lower bound"
    value = float(resolution)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def pick_example_questions(
    questions: dict[int, dict[str, Any]], pinned_question_ids: dict[str, int]
) -> list[ExampleQuestion]:
    """One scored question per type: the pinned one if configured, else the shortest title."""
    scored_by_type: dict[str, list[dict[str, Any]]] = {}
    for question in questions.values():
        if is_scored(question):
            scored_by_type.setdefault(question["type"], []).append(question)
    examples: list[ExampleQuestion] = []
    for question_type in QUESTION_TYPE_ORDER:
        if question_type not in scored_by_type:
            continue
        pinned_id = pinned_question_ids.get(question_type)
        if pinned_id is not None:
            if pinned_id not in questions or not is_scored(questions[pinned_id]):
                raise ValueError(f"Pinned example question {pinned_id} is not a scored question in the bot tournament")
            question = questions[pinned_id]
        else:
            question = min(
                scored_by_type[question_type],
                key=lambda candidate: (len(candidate["question_text"]), candidate["question_id"]),
            )
        examples.append(
            ExampleQuestion(
                question_type=question_type,
                title=question["question_text"],
                options=tuple(question["options"]) if question["options"] else None,
                resolution_text=format_resolution(question),
                url=f"https://www.metaculus.com/questions/{question['post_id']}/",
                was_pinned=pinned_id is not None,
            )
        )
    return examples


def compute_season_results(simulations_folder: Path, config: SeasonConfig) -> SeasonResults:
    logger.info(f"Computing {config.label} results from {simulations_folder}")

    logger.info("Step 1/6: Pro team vs bot team head-to-head")
    team_size_files = find_team_size_files(simulations_folder)
    main_size_key = str(config.main_bot_team_size)
    if main_size_key not in team_size_files:
        raise FileNotFoundError(f"No team comparison file for the main bot team size {main_size_key}")
    main_team_raw = load_json(team_size_files[main_size_key])
    head_to_head = compute_head_to_head(main_team_raw, config)
    team_accuracy = compute_team_accuracy(main_team_raw)
    question_pairs = shared_question_pairs(main_team_raw)
    del main_team_raw
    season_trend = compute_season_trend(config, head_to_head)

    logger.info("Step 2/6: Pro tournament")
    pro_raw = load_json(find_simulation_file(simulations_folder, "pro_tournament.json"))
    pro_participants = spot_peer_participants(pro_raw)
    pro_names = {name for name, participant in pro_participants.items() if participant.user_type == "pro"}
    pro_questions = questions_by_id(pro_raw)
    del pro_raw

    logger.info("Step 3/6: bot tournament leaderboard and question mix")
    bot_raw = load_json(find_simulation_file(simulations_folder, "bot_tournament.json"))
    bot_participants = spot_peer_participants(bot_raw)
    bot_questions = questions_by_id(bot_raw)
    bots = [participant for participant in bot_participants.values() if participant.user_type == "bot"]
    bot_rows = build_leaderboard(bots)
    scored_bot_questions = [question for question in bot_questions.values() if is_scored(question)]
    scored_binary = [question for question in scored_bot_questions if question["type"] == "binary"]
    field = compute_field(
        bot_rows,
        [score for participant in bots for score in participant.score_values],
        len(scored_bot_questions),
    )
    bot_field_accuracy = compute_bot_field_accuracy(bot_raw, field, bot_participants)
    non_primary_bots = find_non_primary_bots(bot_raw)
    del bot_raw

    logger.info("Step 4/6: bot-only questions and Metac Bots")
    bot_only_raw = load_json(find_simulation_file(simulations_folder, "bot_tournament_without_pro_questions.json"))
    bot_only_scored_count = sum(1 for question in questions_by_id(bot_only_raw).values() if is_scored(question))
    del bot_only_raw
    metac = compute_metac(
        load_json(find_simulation_file(simulations_folder, "metac_bot_tournament.json")), bot_rows, config.label
    )

    logger.info("Step 5/6: Pros and bots as individuals")
    individuals = compute_individuals(
        load_json(find_simulation_file(simulations_folder, "pro_with_bot_tourn__no_teams.json")),
        pro_names,
        {participant.name for participant in bots},
        head_to_head.question_count,
    )

    logger.info("Step 6/6: bot team sizes")
    team_sizes = compute_team_sizes(team_size_files, pro_questions, main_size_key, config.human_accounts_in_bot_teams_allowed)

    overview = TournamentOverview(
        bot_count=len(bots),
        metac_bot_count=sum(1 for participant in bots if is_metac_bot(participant.name)),
        bot_question_count=len(bot_questions),
        bot_scored_count=len(scored_bot_questions),
        bot_type_counts=dict(Counter(question["type"] for question in scored_bot_questions)),
        binary_yes_count=sum(1 for question in scored_binary if question["resolution"] is True),
        binary_scored_count=len(scored_binary),
        weighted_question_count=sum(1 for question in scored_bot_questions if question["weight"] < 1),
        shared_question_weights_below_1=tuple(
            sorted(bot_questions[bot_id]["weight"] for _pro_id, bot_id in question_pairs if bot_questions[bot_id]["weight"] < 1)
        ),
        bot_only_scored_count=bot_only_scored_count,
        pro_count=len(pro_names),
        non_pro_accounts_in_pro_tournament=len(pro_participants) - len(pro_names),
        pro_question_count=len(pro_questions),
        pro_scored_count=sum(1 for question in pro_questions.values() if is_scored(question)),
        pro_weighted_question_count=sum(
            1 for question in pro_questions.values() if is_scored(question) and question["weight"] < 1
        ),
        first_question_close=min(close_date(question) for question in bot_questions.values()),
        last_question_close=max(close_date(question) for question in bot_questions.values()),
        close_time_differences=find_close_time_differences(question_pairs, pro_questions, bot_questions),
        non_primary_bots=non_primary_bots,
    )
    logger.info(
        f"{overview.bot_count} bots, {overview.bot_scored_count} scored bot questions, {overview.pro_count} Pros, "
        f"{head_to_head.question_count} shared questions; bot team score {head_to_head.bot_team_mean:.2f}"
    )
    return SeasonResults(
        overview=overview,
        head_to_head=head_to_head,
        season_trend=season_trend,
        individuals=individuals,
        team_sizes=team_sizes,
        field=field,
        metac=metac,
        experiment_groups=find_metac_experiment_groups(metac.baseline_rows, metac.other_rows),
        example_questions=pick_example_questions(bot_questions, config.example_question_ids),
        bot_field_accuracy=bot_field_accuracy,
        team_accuracy=team_accuracy,
    )
