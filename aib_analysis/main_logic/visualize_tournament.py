import logging
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aib_analysis.data_structures.custom_types import QuestionType
from aib_analysis.data_structures.data_models import (
    Leaderboard,
    Question,
    Score,
    ScoreType,
)
from aib_analysis.data_structures.simulated_tournament import (
    SimulatedTournament,
)
from aib_analysis.main_logic.process_tournament import (
    calculate_calibration_curve,
    constrain_question_types,
    find_question_titles_unique_to_first_tournament,
    get_leaderboard,
)
from aib_analysis.data_structures.data_models import (
    LeaderboardEntry,
)
from aib_analysis.math.stats import (
    MeanHypothesisCalculator,
    HypothesisTest,
    ConfidenceIntervalCalculator,
)

logger = logging.getLogger(__name__)


def display_tournament_and_variations(
    tournament: SimulatedTournament, name: str, divide_into_types: bool = False
):
    display_individual_tournament(tournament, name)
    if divide_into_types:
        binary_combined_tournament = constrain_question_types(
            tournament, [QuestionType.BINARY]
        )
        if binary_combined_tournament is not None:
            display_individual_tournament(
                binary_combined_tournament, f"{name} (Binary)"
            )
        multiple_choice_combined_tournament = constrain_question_types(
            tournament, [QuestionType.MULTIPLE_CHOICE]
        )
        if multiple_choice_combined_tournament is not None:
            display_individual_tournament(
                multiple_choice_combined_tournament, f"{name} (Multiple Choice)"
            )
        numeric_combined_tournament = constrain_question_types(
            tournament, [QuestionType.NUMERIC]
        )
        if numeric_combined_tournament is not None:
            display_individual_tournament(
                numeric_combined_tournament, f"{name} (Numeric)"
            )


def display_individual_tournament(tournament: SimulatedTournament, name: str):
    st.subheader(f"{name}")

    # Display tournament statistics
    with st.expander(f"{name} Spot Peer Leaderboard"):
        leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
        display_leaderboard(leaderboard)
    with st.expander(f"{name} Spot Baseline Leaderboard"):
        leaderboard = get_leaderboard(tournament, ScoreType.SPOT_BASELINE)
        display_leaderboard(leaderboard)
    with st.expander(f"{name} Stats"):
        display_tournament_stats(tournament)
    with st.expander(f"{name} Forecasts"):
        display_forecasts(tournament)
    with st.expander(f"{name} Questions"):
        display_questions(tournament.questions, tournament)
    with st.expander(f"{name} Scores"):
        display_scores(tournament.scores)
    # with st.expander(f"{name} Calibration Curve"):
    #     display_calibration_curve(tournament)


def display_bot_v_pro_hypothesis_test(
    team_tournament: SimulatedTournament, expander_name: str
) -> None:
    hypothesis_mean = 0
    confidence_level = 0.95
    leaderboard = get_leaderboard(team_tournament, ScoreType.SPOT_PEER)
    st.subheader(expander_name)
    with st.expander(expander_name):
        st.write(f"## The Test")
        st.write(
            "The below runs 2 tests: 1) tests if the each team's average spot peer score is not equal to zero and 2) if it is greater than zero. If its not equal to zero, then we can conclude that there is a statistically significant difference between the two teams (not necessarily that one group is doing better than another). If its greater than zero, then we can conclude that one group is doing better than another."
        )
        entries = leaderboard.entries_via_sum_of_scores()
        if len(entries) != 2:
            raise ValueError(f"Expected 2 entries, got {len(entries)}")
        for entry in entries:
            try:
                assert all([s.type == ScoreType.SPOT_PEER for s in entry.scores])
                observations = [s.score for s in entry.scores]
                equal_to_hypothesis_test = MeanHypothesisCalculator.test_if_mean_is_equal_to_than_hypothesis_mean(
                    observations, hypothesis_mean, confidence_level
                )
                greater_than_hypothesis_test = MeanHypothesisCalculator.test_if_mean_is_greater_than_hypothesis_mean(
                    observations, hypothesis_mean, confidence_level
                )
                confidence_interval = (
                    ConfidenceIntervalCalculator.confidence_interval_from_observations(
                        observations, confidence_level
                    )
                )
                bootstrap_ci = confidence_interval.bootstrap_confidence_interval
                t_based_ci = confidence_interval.t_based_confidence_interval

                st.write(f"## {entry.user.name}")
                st.write(f"### Equal to {hypothesis_mean}")
                _display_hypothesis_sub_section(equal_to_hypothesis_test, observations)
                st.write(f"### Greater than {hypothesis_mean}")
                _display_hypothesis_sub_section(
                    greater_than_hypothesis_test, observations
                )
                st.write(f"### Confidence Intervals")
                st.write(f"Observed Mean: {confidence_interval.mean:.5f}")
                st.write(f"Hypothesis Mean: {hypothesis_mean:.5f}")
                percentage_confidence_level = confidence_level * 100
                if t_based_ci is not None:
                    st.write(
                        f"T-based {percentage_confidence_level}% Confidence Interval: "
                        f"{t_based_ci.lower_bound:.5f} to "
                        f"{t_based_ci.upper_bound:.5f}"
                    )
                if bootstrap_ci is not None:
                    st.write(
                        f"Bootstrap {percentage_confidence_level}% Confidence Interval: "
                        f"{bootstrap_ci.lower_bound:.5f} to "
                        f"{bootstrap_ci.upper_bound:.5f}"
                    )
                st.write(f"--------------------------------")

            except Exception as e:
                st.write(f"Error: {e}")


def _display_hypothesis_sub_section(
    hypothesis_test: HypothesisTest, observations: list[float]
):
    st.write(f"**P-value**: {hypothesis_test.p_value:.5f}")
    st.write(f"**Shapiro test passes**: {hypothesis_test.shapiro_test_passes}")
    st.write(f"**N > 30**: {len(observations) > 30} (N = {len(observations)})")
    st.write(f"**Interval Type**: {hypothesis_test.interval_type}")
    st.write(f"**Conclusion**: {hypothesis_test.written_conclusion}")


def display_tournament_stats(tournament: SimulatedTournament) -> None:
    forecasts = tournament.forecasts
    if not forecasts:
        st.write("No forecasts available.")
        return

    # Calculate basic statistics
    num_forecasts = len(forecasts)
    num_users = len(tournament.users)
    num_questions = len(tournament.questions)
    num_scores_calculated = len(tournament.scores)
    num_peer_scores_calculated = len(
        [s for s in tournament.scores if s.type == ScoreType.SPOT_PEER]
    )
    num_baseline_scores_calculated = len(
        [s for s in tournament.scores if s.type == ScoreType.SPOT_BASELINE]
    )
    annulled_questions = [q for q in tournament.questions if q.is_annulled_or_ambiguous]
    num_annulled_questions = len(annulled_questions)
    num_annulled_forecasts = len(
        [f for f in forecasts if f.question.is_annulled_or_ambiguous]
    )

    # Calculate averages
    forecasts_per_user = num_forecasts / num_users if num_users > 0 else 0
    forecasts_per_question = num_forecasts / num_questions if num_questions > 0 else 0
    forecasts_per_user_per_question = (
        forecasts_per_user / num_questions if num_questions > 0 else 0
    )

    # Display statistics
    st.write(
        "*Note that if tournaments are loaded only from jsons with only scores, then all stats will not include questions/forecasts from annulled/ambiguous questions. Stats will be off since you cannot score annulled questions.*"
    )
    st.write("### Basic Statistics")
    st.write(f"Number of forecasts: {num_forecasts}")
    st.write(f"Number of users: {num_users}")
    st.write(f"Number of questions: {num_questions}")
    st.write(
        f"Number of annulled or ambiguous questions: {num_annulled_questions} (affected forecasts {num_annulled_forecasts})"
    )
    st.write(f"Number of scores calculated: {num_scores_calculated}")
    st.write(f"Number of peer scores calculated: {num_peer_scores_calculated}")
    st.write(f"Number of baseline scores calculated: {num_baseline_scores_calculated}")

    st.write("### Average Statistics")
    st.write(f"Average forecasts per user: {forecasts_per_user:.2f}")
    st.write(f"Average forecasts per question: {forecasts_per_question:.2f}")
    st.write(
        f"Average forecasts per user per question: {forecasts_per_user_per_question:.2f}"
    )

    # Calculate and display user type distribution
    user_types = {}
    for forecast in forecasts:
        user_type = forecast.user.type.value
        user_types[user_type] = user_types.get(user_type, 0) + 1

    st.write("### User Type Distribution")
    for user_type, count in user_types.items():
        st.write(f"{user_type}: {count} forecasts")

    # Calculate and display question type distribution
    question_type_forecasts = {}
    for forecast in forecasts:
        question_type = forecast.question.type.value
        question_type_forecasts[question_type] = (
            question_type_forecasts.get(question_type, 0) + 1
        )

    question_type_questions = {}
    for question in tournament.questions:
        question_type = question.type.value
        question_type_questions[question_type] = (
            question_type_questions.get(question_type, 0) + 1
        )

    percent_resolved_yes = (
        len([q for q in tournament.questions if q.resolution == True])
        / num_questions
        * 100
    )

    st.write("### Question Type Distribution")
    st.write(f"**Forecasts**: {num_forecasts}")
    for question_type, count in question_type_forecasts.items():
        st.write(f"- {question_type}: {count} forecasts")
    st.write(f"**Questions**: {num_questions}")
    for question_type, count in question_type_questions.items():
        st.write(f"- {question_type}: {count} questions")
    st.write(f"**Percent Binary that resolved yes**: {percent_resolved_yes:.2f}%")

    st.write("### Annulled or Ambiguous Questions")
    for question in annulled_questions:
        st.write(f"- {question.question_text} [link]({question.url})")


def display_forecasts(tournament: SimulatedTournament):
    forecasts = tournament.forecasts
    if not forecasts:
        st.write("No forecasts available.")
        return
    # Convert forecasts to DataFrame for display
    data = [
        {
            "user": f.user.name,
            "user_type": f.user.type.value,
            "question_url": f.question.url,
            "question": f.question.question_text,
            "question_id": f.question.question_id,
            "question_type": f.question.type.value,
            "prediction_time": f.prediction_time,
            "prediction": f.prediction,
            "resolution": f.question.resolution,
            "weight": f.question.weight,
            "options": f.question.options,
            "range_max": f.question.range_max,
            "range_min": f.question.range_min,
            "open_lower_bound": f.question.open_lower_bound,
            "open_upper_bound": f.question.open_upper_bound,
        }
        for f in forecasts
    ]
    st.write(f"**Number of forecasts**: {len(forecasts)}")
    df = pd.DataFrame(data)
    # Truncate to first 100 rows for performance, allow filtering
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def display_questions(
    questions: list[Question], tournament: SimulatedTournament | None = None
):
    data = []
    for question in questions:
        datum = {"url": question.url, **question.model_dump()}
        datum["type"] = question.type.value
        if tournament is not None:
            forecasts = tournament.question_to_forecasts(question.question_id)
            spot_forecasts = tournament.question_to_spot_forecasts(question.question_id)
            num_of_forecasts = len(forecasts)
            num_of_forecasters = len(set([f.user.name for f in forecasts]))
            num_of_spot_forecasts = len(spot_forecasts)
            num_of_spot_forecasters = len(set([f.user.name for f in spot_forecasts]))
            datum["num_forecasts"] = num_of_forecasts
            datum["num_forecasters"] = num_of_forecasters
            datum["num_spot_forecasts"] = num_of_spot_forecasts
            datum["num_spot_forecasters"] = num_of_spot_forecasters
        data.append(datum)
    st.write(f"**Number of questions**: {len(questions)}")
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def display_scores(scores: list[Score]):
    st.write(f"**Number of scores**: {len(scores)}")
    data = []
    for score in scores:
        data.append(
            {
                "score": score.score,
                "type": score.type.value,
                "user": score.forecast.user.name,
                "url": score.forecast.question.url,
                "question_text": score.forecast.question.question_text,
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def display_leaderboard(leaderboard: Leaderboard):
    confidence_level = 0.95
    _display_average_scores_plot(leaderboard, confidence_level)
    _display_leaderboard_table(leaderboard, confidence_level)
    _display_score_histogram_by_user(
        leaderboard.all_scores,
        title="All Users Scores Histogram (overlay, not stacked)",
    )


def _display_leaderboard_table(leaderboard: Leaderboard, confidence_level: float):
    data = []
    entries = leaderboard.entries_via_sum_of_scores()
    include_bootstrap = len(entries) <= 10
    for i, entry in enumerate(entries):
        num_to_display = min(5, entry.question_count)
        random_sample_of_scores = entry.randomly_sample_scores(num_to_display)
        top_n_scores = entry.top_n_scores(num_to_display)
        bottom_n_scores = entry.bottom_n_scores(num_to_display)
        try:
            confidence_interval = entry.get_confidence_interval(
                confidence_level,
                num_bootstraps=(
                    ConfidenceIntervalCalculator.DEFAULT_NUM_BOOTSTRAPS
                    if include_bootstrap
                    else None
                ),
            )
            t_based_confidence_interval = (
                confidence_interval.t_based_confidence_interval
            )
            if t_based_confidence_interval is not None:
                t_based_upper_bound = t_based_confidence_interval.upper_bound
                t_based_lower_bound = t_based_confidence_interval.lower_bound
            else:
                t_based_upper_bound = None
                t_based_lower_bound = None
            bootstrap_confidence_interval = (
                confidence_interval.bootstrap_confidence_interval
            )
            if bootstrap_confidence_interval is not None:
                bootstrap_upper_bound = bootstrap_confidence_interval.upper_bound
                bootstrap_lower_bound = bootstrap_confidence_interval.lower_bound
            else:
                bootstrap_upper_bound = None
                bootstrap_lower_bound = None
        except Exception as e:
            logger.debug(
                f"Failed to get confidence interval for entry {entry.user.name}: {e}"
            )
            t_based_upper_bound = None
            t_based_lower_bound = None
            bootstrap_upper_bound = None
            bootstrap_lower_bound = None
        data.append(
            {
                "rank": i + 1,
                "user": entry.user.name,
                "sum_of_scores": entry.sum_of_scores,
                "average_score": entry.average_score,
                "avg_lower_bound_t_based": t_based_lower_bound,
                "avg_upper_bound_t_based": t_based_upper_bound,
                "avg_lower_bound_bootstrap": bootstrap_lower_bound,
                "avg_upper_bound_bootstrap": bootstrap_upper_bound,
                "num_questions_with_scores": entry.question_count,
                "user_type": entry.user.type.value,
                "aggregated_users": [user.name for user in entry.user.aggregated_users],
                "random_sample_of_scores": [
                    score.display_score_and_question()
                    for score in random_sample_of_scores
                ],
                "top_n_scores": [
                    score.display_score_and_question() for score in top_n_scores
                ],
                "bottom_n_scores": [
                    score.display_score_and_question() for score in bottom_n_scores
                ],
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(
        df.sort_values(by="sum_of_scores", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _display_average_scores_plot(
    leaderboard: Leaderboard, confidence_level: float
) -> None:
    """Display a plotly graph of average scores with error bars."""
    entries = []
    score_type = leaderboard.type

    for entry in leaderboard.entries_via_sum_of_scores():
        try:
            confidence_interval = entry.get_confidence_interval(confidence_level)
            upper_bound = confidence_interval.upper_bound
            lower_bound = confidence_interval.lower_bound
        except Exception as e:
            logger.debug(
                f"Failed to get confidence interval for entry {entry.user.name}: {e}"
            )
            upper_bound = 0
            lower_bound = 0

        entries.append(
            {
                "user": entry.user.name,
                "average_score": entry.average_score,
                "sum_of_scores": entry.sum_of_scores,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "num_questions": entry.question_count,
            }
        )

    if not entries:
        st.warning("No valid entries with confidence intervals available for plotting.")
        return

    df = pd.DataFrame(entries)
    df = df.sort_values("sum_of_scores", ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["user"],
            y=df["average_score"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=df["upper_bound"] - df["average_score"],
                arrayminus=df["average_score"] - df["lower_bound"],
                visible=True,
            ),
            marker=dict(
                color=df["average_score"],
                colorscale="Viridis",
            ),
            hovertemplate="User: %{x}<br>Score: %{y:.3f}<br>Questions: %{customdata}<extra></extra>",
            customdata=df["num_questions"],
        )
    )

    friendly_score_type = score_type.value.replace('_', ' ').title()
    fig.update_layout(
        title=f"Average Score ({friendly_score_type}) with {confidence_level*100}% Confidence Intervals",
        xaxis_title="User",
        yaxis_title=f"Average Score ({friendly_score_type})",
        showlegend=False,
        height=600,
        xaxis=dict(tickangle=45),
    )
    random_number = random.randint(0, 1000000)
    plot_key = f"{random_number}"
    st.plotly_chart(fig, use_container_width=True, key=plot_key)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color (e.g. #1f77b4) to rgba string with given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    raise ValueError(f"Invalid hex color: {hex_color}")


def display_calibration_curve(tournament: SimulatedTournament) -> None:
    # Get all binary forecasts with resolutions
    binary_forecasts = [
        f
        for f in tournament.forecasts
        if f.question.type == QuestionType.BINARY and f.question.resolution is not None
    ]

    if not binary_forecasts:
        st.warning(
            "No binary forecasts with resolutions available for calibration curve."
        )
        return

    fig = go.Figure()

    # Add perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect Calibration",
        )
    )

    color_sequence = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    user_colors = {}
    for idx, user in enumerate(tournament.users):
        user_colors[user.name] = color_sequence[idx % len(color_sequence)]

    for user in tournament.users:
        user_forecasts = [f for f in binary_forecasts if f.user == user]
        if not user_forecasts:
            continue

        calibration_curve = calculate_calibration_curve(user_forecasts)
        bin_centers = [b.bin_center for b in calibration_curve.curve]
        avg_resolutions = [b.average_resolution for b in calibration_curve.curve]
        lower_ci = [b.lower_confidence_interval for b in calibration_curve.curve]
        upper_ci = [b.upper_confidence_interval for b in calibration_curve.curve]
        bin_counts = [b.forecast_count for b in calibration_curve.curve]
        color = user_colors[user.name]
        fill_color = _hex_to_rgba(color, 0.15)

        fig.add_trace(
            go.Scatter(
                x=bin_centers + bin_centers[::-1],
                y=upper_ci + lower_ci[::-1],
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=f"{user.name} CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=avg_resolutions,
                mode="lines+markers",
                name=f"{user.name} ({len(user_forecasts)} forecasts)",
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate="Probability: %{x:.2f}<br>Resolution Rate: %{y:.2f}<br>Forecasts in Bin: %{customdata}<extra></extra>",
                customdata=bin_counts,
            )
        )

    fig.update_layout(
        title="Calibration Curves by User",
        xaxis_title="Assigned Probability",
        yaxis_title='Fraction that Resolved "Yes"',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        height=600,
    )

    random_number = random.randint(0, 1000000)
    plot_key = f"{random_number}"
    st.plotly_chart(fig, use_container_width=True, key=plot_key)


def _display_score_histogram_by_user(
    scores: list[Score], title: str | None = None
) -> None:
    score_types = set([s.type for s in scores])
    if len(score_types) > 1:
        raise ValueError(
            "Cannot display score histogram by user for multiple score types"
        )
    fig = go.Figure()

    # Group scores by user
    user_scores: dict[str, list[float]] = {}
    for score in scores:
        user_name = score.forecast.user.name
        if user_name not in user_scores:
            user_scores[user_name] = []
        user_scores[user_name].append(score.score)

    # Add a trace for each user
    for user_name, user_score_list in user_scores.items():
        fig.add_trace(go.Histogram(x=user_score_list, name=user_name, opacity=0.7))

    if title is not None:
        fig.update_layout(title=title)
    fig.update_layout(xaxis_title="Score", yaxis_title="Count", barmode="overlay")
    random_number = random.randint(0, 1000000)
    plot_key = f"{random_number}"
    st.plotly_chart(fig, use_container_width=True, key=plot_key)


def display_unique_questions(
    tournament_1: SimulatedTournament, tournament_2: SimulatedTournament
) -> None:
    t1_name = tournament_1.name if tournament_1.name else "First Tournament"
    t2_name = tournament_2.name if tournament_2.name else "Second Tournament"

    st.subheader(f'Questions titles in "{t1_name}" but not in "{t2_name}"')
    with st.expander(f'Questions titles in "{t1_name}" but not in "{t2_name}"'):
        unique_questions = find_question_titles_unique_to_first_tournament(
            tournament_1, tournament_2
        )
        display_questions(unique_questions, tournament_1)


def display_aggregate_comparison(team_comparison_tourns: list[SimulatedTournament]):
    entries_to_graph: list[LeaderboardEntry] = []
    for tournament in team_comparison_tourns:
        bot_team_users = [user for user in tournament.users if user.name == "Bot Team"]
        assert len(bot_team_users) == 1, f"Expected 1 bot team user, got {len(bot_team_users)}"
        bot_team_user = bot_team_users[0]
        leaderboard = get_leaderboard(tournament, ScoreType.SPOT_PEER)
        bot_entry = [entry for entry in leaderboard.entries if entry.user == bot_team_user][0]
        entries_to_graph.append(bot_entry)
    
    st.subheader("Aggregate comparison")
    with st.expander("Aggregate comparison"):
        entries_to_graph = sorted(entries_to_graph, key=lambda x: len(x.user.aggregated_users))
        
        scores = [entry.average_score for entry in entries_to_graph]
        confidence_intervals = [entry.get_confidence_interval(confidence_level=0.95) for entry in entries_to_graph]
        lower_bounds = [ci.lower_bound for ci in confidence_intervals]
        upper_bounds = [ci.upper_bound for ci in confidence_intervals]
        
        error_y_minus = [score - lower for score, lower in zip(scores, lower_bounds)]
        error_y_plus = [upper - score for score, upper in zip(scores, upper_bounds)]
        
        team_sizes = [len(entry.user.aggregated_users) for entry in entries_to_graph]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=team_sizes,
            y=scores,
            mode='markers+lines',
            name='Bot Team Score',
            marker={'size': 10},
            error_y={
                'type': 'data',
                'symmetric': False,
                'array': error_y_plus,
                'arrayminus': error_y_minus,
                'visible': True
            }
        ))
        
        fig.update_layout(
            title='Bot Team Performance vs Team Size',
            xaxis_title='Team Size',
            yaxis_title='Average Score',
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

        for entry in entries_to_graph:
            st.write(f"- Bot Team Size: {len(entry.user.aggregated_users)} | Score: {entry.average_score:.3f}")

        
