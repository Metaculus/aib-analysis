import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize_scalar
from scipy.stats import binom
import re
from datetime import datetime

def convert_baseline_to_forecasts(df):
    """
    Converts baseline scores to forecasts based on resolution.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'bot_question_id', 'resolution', and 'score'.

    Returns:
        pandas.DataFrame: DataFrame with an additional 'forecast' column.
    """
    result_df = df.copy()

    def score_to_forecast(score, resolution):
        if resolution == 'yes':
            return 2 ** (score/100 - 1)
        else:
            return 1 - 2 ** (score/100 - 1)

    score_columns = ['score']
    #score_columns = [col for col in score_columns if col not in ['bot_question_id', 'resolution', 'pro_question_id']]

    for col in score_columns:
        result_df['forecast'] = result_df.apply(lambda row: score_to_forecast(row[col], row['resolution']), axis=1)

    return result_df

def add_is_median(df):
    """
    Marks exactly one row per question_id as the median.
    Guarantees one median per question by taking the forecaster with 
    the actual median value for that question.

    Args:
        df (pandas.DataFrame): DataFrame with 'question_id' and 'forecast' columns.

    Returns:
        pandas.DataFrame: DataFrame with an additional 'is_median' column.
    """
    # Initialize median column
    df['is_median'] = False
    
    # For each question_id
    for qid in df['question_id'].unique():
        # Get just the rows for this question
        question_mask = df['question_id'] == qid
        question_df = df[question_mask]
        
        # Get the median value index (middle position after sorting)
        median_idx = question_df['forecast'].sort_values().index[len(question_df)//2]
        
        # Mark that row
        df.loc[median_idx, 'is_median'] = True
        
    return df

def add_median_rows(df, prefix):
    """
    For each row where is_median=True, creates a duplicate row with forecaster='median'.

    Args:
        df (pandas.DataFrame): DataFrame with 'is_median' and 'forecaster' columns.
        prefix (str): Prefix to add to the 'forecaster' column for median rows.

    Returns:
        pandas.DataFrame: Original DataFrame plus duplicate rows for medians.
    """
    # Get the median rows
    median_rows = df[df['is_median']].copy()
    
    # Change forecaster to 'median'
    median_rows['forecaster'] = f'{prefix}_median'
    
    # Combine original and new median rows
    whole = pd.concat([df, median_rows], ignore_index=True).sort_values('question_id').drop_duplicates(['question_id', 'forecaster'])

    return whole

def calculate_weighted_stats(df):
    """
    Calculates weighted statistics (mean, sum, standard error, confidence intervals) for each forecaster.

    Args:
        df (pandas.DataFrame): DataFrame with 'forecaster', 'score', and 'question_weight' columns.

    Returns:
        pandas.DataFrame: DataFrame with weighted statistics for each forecaster.
    """
    results = []

    # For each forecaster
    for forecaster in df['forecaster'].unique():
        forecaster_data = df[df['forecaster'] == forecaster]

        # Get scores and weights
        scores = forecaster_data['score']
        weights = forecaster_data['question_weight']

        # Calculate weighted mean
        weighted_mean = np.average(scores, weights=weights)
        weighted_sum = np.sum(scores * weights)

        # Calculate weighted standard error
        # Using weighted variance formula
        weighted_var = np.average((scores - weighted_mean)**2, weights=weights)
        n = len(scores)
        weighted_se = np.sqrt(weighted_var / n)

        # Calculate t-statistic for 95% confidence interval
        t_value = stats.t.ppf(0.975, n-1)
        ci_lower = weighted_mean - (t_value * weighted_se)

        results.append({
            'forecaster': forecaster,
            'weighted_mean': weighted_mean,
            'weighted_sum': weighted_sum,
            'n_questions': n,
            'ci_lower': ci_lower,
            'weighted_se': weighted_se
        })

    # Convert to dataframe and sort by lower bound
    results_df = pd.DataFrame(results)
    return results_df.sort_values('weighted_sum', ascending=False)


def make_wide(df_bot_peer, df_pro_bot_resolved_questions):
    """
    Converts a long-format DataFrame to a wide-format DataFrame and merges question weights.

    Args:
        df_bot_peer (pandas.DataFrame): DataFrame with 'bot_question_id', 'forecaster', and 'score' columns.
        df_pro_bot_resolved_questions (pandas.DataFrame): DataFrame with 'bot_question_id' and 'question_weight' columns.

    Returns:
        pandas.DataFrame: Wide-format DataFrame with question weights merged.
    """
    df_pivoted = df_bot_peer.pivot(index='bot_question_id', columns='forecaster', values='score')
    df_pivoted = df_pivoted.reset_index()
    df_pivoted = df_pivoted.reindex(sorted(df_pivoted.columns), axis=1)

    # Step 4: Move 'question_id' to be the first column
    cols = df_pivoted.columns.tolist()
    cols = ['bot_question_id'] + [col for col in cols if col != 'bot_question_id']
    df_pivoted = df_pivoted[cols]

    all_columns = df_pivoted.columns.tolist()
    ## Remove 'question_id' and 'bot_median' from the list if they exist
    all_columns = [col for col in all_columns if col not in ['bot_question_id']]
    new_column_order = ['bot_question_id'] + all_columns
    df_pivoted = df_pivoted[new_column_order]
    df_bot_peer_wide = df_pivoted
    df_bot_peer_wide['bot_question_id'] = pd.to_numeric(df_bot_peer_wide['bot_question_id'], errors='coerce')

    # Join with df_pro_bot_resolved_questions to get question weights
    df_bot_peer_wide = pd.merge(
        df_bot_peer_wide,
        df_pro_bot_resolved_questions[['bot_question_id', 'question_weight']],
        on='bot_question_id',
        how='left'
    )

    return df_bot_peer_wide

"""
Options from https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance
I didn't think (B) beared trying, but could be wrong. - MGH
It makes very little difference here but (C) does seem to be the correct formula - corrects for 
the bias in the sample variance.
"""

def calc_weighted_std_dev(df3, bot, weighted_score, weighted_count, weight_col):
    """
    Calculates the weighted standard deviation using Molly's method - (A) from stack exchange post.

    Args:
        df3 (pandas.DataFrame): DataFrame containing the data.
        bot (str): Column name for the bot's scores.
        weighted_score (float): Weighted score.
        weighted_count (int): Weighted count.
        weight_col (str): Column name for the weights.

    Returns:
        float: Weighted standard deviation.
    """
    weighted_average = weighted_score / weighted_count
    return np.sqrt(((df3[bot] - weighted_average) ** 2 * df3[weight_col]).sum() / (weighted_count - 1))
  
def calc_weighted_std_dev2(df3, bot, weighted_score, weighted_count, weight_col):
    """
    Calculates the weighted standard deviation using Claude (via Nikos) method - (C) from stack exchange post.

    Args:
        df3 (pandas.DataFrame): DataFrame containing the data.
        bot (str): Column name for the bot's scores.
        weighted_score (float): Weighted score.
        weighted_count (int): Weighted count.
        weight_col (str): Column name for the weights.

    Returns:
        float: Weighted standard deviation.
    """
    weighted_average = weighted_score / weighted_count
    return np.sqrt(
        (df3[weight_col] * (df3[bot] - weighted_average) ** 2).sum() / 
        (df3[weight_col].sum() * (1 - (df3[weight_col] ** 2).sum() / (df3[weight_col].sum() ** 2)))
    )

def weighted_bootstrap_analysis(df_bot_peer_wide, bots, NUM, ITER):
    """
    Performs weighted bootstrap analysis to calculate confidence intervals and medians.

    Args:
        df_bot_peer_wide (pandas.DataFrame): DataFrame with bot scores and question weights.
        bots (list): List of bot column names.
        NUM (int): Number of samples to draw in each bootstrap iteration.
        ITER (int): Number of bootstrap iterations.

    Returns:
        pandas.DataFrame: DataFrame with confidence intervals and medians for each bot.
    """
    # Function to perform a single bootstrap iteration
    def single_bootstrap(df):
        # Weighted sampling of questions
        sampled_df = df.sample(n=NUM, weights='question_weight', replace=True)
        # Calculate total weighted score for each bot
        return sampled_df[bots].sum()

    # Perform bootstrap ITER times
    bootstrap_results = [single_bootstrap(df_bot_peer_wide) for _ in range(ITER)]

    # Convert results to DataFrame
    results_df = pd.DataFrame(bootstrap_results)

    # Calculate confidence intervals and median
    ci_low = results_df.quantile(0.025)
    ci_10 = results_df.quantile(0.1)
    ci_high = results_df.quantile(0.975)
    ci_90 = results_df.quantile(0.9)
    median = results_df.median()

    # Create output DataFrame
    output_df = pd.DataFrame({
        '2.5% CI': ci_low,
        '10% CI': ci_10,
        'Median': median,
        '90% CI': ci_90,
        '97.5% CI': ci_high
    })

    # Sort by median descending
    output_df = output_df.sort_values('Median', ascending=False)

    return output_df

def calculate_median_forecast(df, bots):
    """
    Calculates the median forecast for a given set of bots.

    Args:
        df (pandas.DataFrame): DataFrame with bot forecast columns.
        bots (list): List of bot column names.

    Returns:
        pandas.Series: Median forecast for each row.
    """
    return df[bots].median(axis=1)

def calculate_weighted_scores(df_bot_team_forecasts, teams):
    """
    Calculates weighted scores for each team based on their forecasts and question weights.

    Args:
        df_bot_team_forecasts (pandas.DataFrame): DataFrame with team forecasts and question weights.
        teams (list): List of team column names.

    Returns:
        pandas.Series: Weighted scores for each team.
    """
    # Initialize a dictionary to store the weighted scores for each team
    team_scores = {team: 0 for team in teams}

    # Iterate through each row (question) in the dataframe
    for _, row in df_bot_team_forecasts.iterrows():
        resolution = row['resolution']
        question_weight = row['question_weight']

        # Calculate the baseline score for each team
        for team in teams:
            forecast = row[team]

            # Calculate the baseline score based on the resolution
            if resolution == 'yes':
                baseline_score = np.log2(forecast / 0.5)
            elif resolution == 'no':
                baseline_score = np.log2((1 - forecast) / 0.5)
            else:
                # Skip if resolution is neither 0 nor 1
                continue

            # Calculate the weighted score and add it to the team's total
            weighted_score = baseline_score * question_weight
            team_scores[team] += weighted_score

    # Convert the dictionary to a pandas Series for easier handling
    return pd.Series(team_scores)

def calculate_t_test(df_input, bot_list, weight_col='question_weight'):
    """
    Calculates weighted statistics, including t-test and p-values, for multiple bots.

    Args:
        df_input (pandas.DataFrame): 
            DataFrame with peer scores, such as `df_bot_vs_pro_peer`, comparing each bot to the pro median.
        bot_list (list): 
            List of column names corresponding to bot scores.
        weight_col (str, optional): 
            Name of the column containing weights. Defaults to 'question_weight'.

    Returns:
        pandas.DataFrame: 
            Leaderboard DataFrame with calculated statistics for each bot, including:
            - W_score: Weighted score.
            - W_count: Weighted count.
            - W_ave: Weighted average.
            - W_stdev: Weighted standard deviation.
            - std_err: Standard error.
            - t_stat: T-statistic.
            - t_crit: Critical t-value.
            - upper_bound: Upper confidence bound.
            - lower_bound: Lower confidence bound.
            - cdf: Cumulative distribution function value.
            - p_value: Two-tailed p-value.
    """
    # Initialize results dataframe
    df_W_leaderboard = pd.DataFrame(index=bot_list)
    
    for bot in bot_list:
        # Create working copy with just needed columns
        df3 = df_input[[bot, weight_col]].copy()
        df3 = df3.dropna()
        df3 = df3.reset_index(drop=True)
        
        # Calculate weighted statistics
        weighted_score = (df3[bot] * df3[weight_col]).sum()
        weighted_count = df3[weight_col].sum()
        
        if weighted_count > 2:  # Only calculate if we have enough data
            weighted_average = weighted_score / weighted_count
            weighted_std_dev = calc_weighted_std_dev2(df3, bot, weighted_score, weighted_count, weight_col)
            std_error = weighted_std_dev / np.sqrt(weighted_count)
            t_statistic = (weighted_average - 0) / std_error
            
            # Get t-critical value and confidence bounds
            effective_n = (df3[weight_col].sum() ** 2) / (df3[weight_col] ** 2).sum()
            t_crit = t.ppf(0.975, df=effective_n - 1)  # 95% confidence level
            upper_bound = weighted_average + t_crit * std_error
            lower_bound = weighted_average - t_crit * std_error
            
            # Calculate CDF and p-value
            cdf = t.cdf(t_statistic, df=weighted_count-1)
            p_value = 2 * min(cdf, 1 - cdf)  # Two-tailed p-value
            
        else:  # Not enough data
            weighted_average = weighted_score / weighted_count if weighted_count > 0 else np.nan
            weighted_std_dev = np.nan
            std_error = np.nan
            t_statistic = np.nan
            t_crit = np.nan
            upper_bound = np.nan
            lower_bound = np.nan
            cdf = np.nan
            p_value = np.nan
        
        # Store results
        df_W_leaderboard.loc[bot, 'W_score'] = weighted_score
        df_W_leaderboard.loc[bot, 'W_count'] = weighted_count
        df_W_leaderboard.loc[bot, 'W_ave'] = weighted_average
        df_W_leaderboard.loc[bot, 'W_stdev'] = weighted_std_dev
        df_W_leaderboard.loc[bot, 'std_err'] = std_error
        df_W_leaderboard.loc[bot, 't_stat'] = t_statistic
        df_W_leaderboard.loc[bot, 't_crit'] = t_crit
        df_W_leaderboard.loc[bot, 'upper_bound'] = upper_bound
        df_W_leaderboard.loc[bot, 'lower_bound'] = lower_bound
        df_W_leaderboard.loc[bot, 'cdf'] = cdf
        df_W_leaderboard.loc[bot, 'p_value'] = p_value
    
    # Format and round the results
    df_W_leaderboard['W_score'] = df_W_leaderboard['W_score'].round(1)

    # Store numerical p-values temporarily for sorting
    df_W_leaderboard['_p_value_sort'] = df_W_leaderboard['p_value']
    
    # Format p-values as percentages
    df_W_leaderboard['p_value'] = df_W_leaderboard['p_value'].apply(
        lambda x: f"{x:.6f}" if pd.notnull(x) else "NA"
    )
    
    # Round other columns
    df_W_leaderboard[['W_ave', 'W_count', 'lower_bound', 'upper_bound']] = \
        df_W_leaderboard[['W_ave', 'W_count', 'lower_bound', 'upper_bound']].round(1)
    
    # Sort by the numerical p-values
    df_W_leaderboard = df_W_leaderboard.sort_values(
        by='W_score',
        ascending=False,
        na_position='last'
    )
    
    # Drop the temporary sorting column
    df_W_leaderboard = df_W_leaderboard.drop('_p_value_sort', axis=1)
     
    return df_W_leaderboard

def calculate_head_to_head(row, a, b):
    """
    Calculates the head-to-head score for two forecasters. If positive, a did better than b; if negative, a did worse than b.

    Args:
        row (pandas.Series): Row of the DataFrame containing 'resolution' and forecaster columns.
        a (str): Column name for the first forecaster.
        b (str): Column name for the second forecaster.

    Returns:
        float: Head-to-head score.
    """
    if (row['resolution'] == 'yes') | (row['resolution'] == 1):
        return 100* np.log(row[a] / row[b])
    elif (row['resolution'] == 'no') | (row['resolution'] == 0):
        return 100* np.log((1 - row[a]) / (1 - row[b]))
    else:
        return np.nan

def plot_head_to_head_distribution(df_forecasts, col='head_to_head', vs=('Bot Team', 'Pros')):
    """
    Plots the distribution of head-to-head scores and fits a Gaussian curve.

    Args:
        df_forecasts (pandas.DataFrame): DataFrame with head-to-head scores.
        col (str): Column name for head-to-head scores.
        vs (tuple): Tuple of labels for the comparison (e.g., ('Bot Team', 'Pros')), for plot title.

    Returns:
        None
    """
    # Extract the 'head_to_head' data
    data = df_forecasts[col]

    # Calculate the mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Create the histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')

    # Generate points for the fitted Gaussian curve
    x = np.linspace(min(data), max(data), 100)
    y = norm.pdf(x, mean, std)

    # Plot the fitted Gaussian curve
    plt.plot(x, y, 'r-', linewidth=2, label='Fitted Gaussian')

    # Customize the plot
    plt.title(f'{vs[0]} Head-to-Head Scores vs {vs[1]}')
    plt.xlabel('Head-to-Head Score')
    plt.ylabel('Density')
    plt.legend()

    # Add text annotation for the mean
    #plt.text(0.95, 0.95, f'Mean: {mean:.2f}', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

    # Display the plot
    plt.show()

    # Print the average
    print(f"The average of 'head_to_head' is: {mean:.2f}")

def calculate_calibration_curve(forecasts, resolutions, weights):
    """
    Calculates a calibration curve for forecasts.

    Args:
        forecasts (list[float]): List of forecast probabilities.
        resolutions (list[int]): List of actual outcomes (0 or 1).
        weights (list[float]): List of weights for each forecast.

    Returns:
        dict: Calibration curve data.
    """
    calibration_curve = []
    # Same number of forecasts in each bin
    quintiles = np.quantile(forecasts, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = []
    for i in range(len(quintiles) - 1):
        bins.append((quintiles[i], quintiles[i + 1]))
    for p_min, p_max in bins:
        resolutions_for_bucket = []
        weights_for_bucket = []
        bin_center = (p_min + p_max) / 2
        for value, weight, resolution in zip(forecasts, weights, resolutions):
            # For the last bin, include the upper bound
            if i == len(bins) - 1:
                if p_min <= value <= p_max:
                    resolutions_for_bucket.append(resolution)
                    weights_for_bucket.append(weight)
            else:
                if p_min <= value < p_max:
                    resolutions_for_bucket.append(resolution)
                    weights_for_bucket.append(weight)
        count = max(len(resolutions_for_bucket), 1)
        average_resolution = (
            np.average(resolutions_for_bucket, weights=weights_for_bucket)
            if sum(weights_for_bucket) > 0
            else None
        )
        lower_confidence_interval = binom.ppf(0.05, count, p_min) / count
        perfect_calibration = binom.ppf(0.50, count, bin_center) / count
        upper_confidence_interval = binom.ppf(0.95, count, p_max) / count

        calibration_curve.append(
            {
                "bin_lower": p_min,
                "bin_upper": p_max,
                "lower_confidence_interval": lower_confidence_interval,
                "average_resolution": average_resolution,
                "upper_confidence_interval": upper_confidence_interval,
                "perfect_calibration": perfect_calibration,
            }
        )

    return {
        "calibration_curve": calibration_curve,
    }

def plot_calibration_curve(df, column_name, label, color):
    """
    Plots a calibration curve with confidence intervals.

    Args:
        df (pandas.DataFrame): DataFrame with forecast and resolution data.
        column_name (str): Column name for forecast probabilities.
        label (str): Label for the plot.
        color (str): Color for the plot.

    Returns:
        None
    """
    y_true = df['resolution']
    y_pred = df[column_name]
    weights = [1.0 for _ in y_true]
    calibration_curve = calculate_calibration_curve(y_pred, y_true, weights)['calibration_curve']
    prob_true = [item['average_resolution'] for item in calibration_curve]
    bin_center = [(item['bin_lower'] + item['bin_upper']) / 2 for item in calibration_curve]
    ci_lower = [item['lower_confidence_interval'] for item in calibration_curve]
    ci_upper = [item['upper_confidence_interval'] for item in calibration_curve]

    plt.plot(bin_center, prob_true, marker='o', linewidth=2, label=label, color=color)
    plt.fill_between(bin_center, ci_lower, ci_upper, alpha=0.2, color=color)
    for x, y in zip(bin_center, prob_true):
        if x is None or y is None:
            continue
        plt.annotate(f'({x:.2f}, {y:.2f})',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    color=color,
                    fontsize=8)

def calculate_confidence(predictions, outcomes):
    """
    Calculates over- or under-confidence for a set of predictions.

    Args:
        predictions (pandas.Series): Series of predicted probabilities.
        outcomes (pandas.Series): Series of actual outcomes (0 or 1).

    Returns:
        float: Confidence score (positive for overconfidence, negative for underconfidence).
    """
    # Bin predictions into 10 equally spaced bins
    bins = pd.cut(predictions, bins=10)

    # Calculate mean prediction and actual outcome for each bin
    grouped = pd.DataFrame({'prediction': predictions, 'outcome': outcomes}).groupby(bins)
    mean_prediction = grouped['prediction'].mean()
    mean_outcome = grouped['outcome'].mean()

    # Calculate the difference between mean prediction and mean outcome
    confidence_diff = mean_prediction - mean_outcome

    # Return the average difference (excluding NaN values)
    return np.nanmean(confidence_diff)

def interpret_confidence(score):
    """
    Interprets the confidence score.

    Args:
        score (float): Confidence score.

    Returns:
        str: Interpretation of the confidence score.
    """
    if score > 0:
        return f"Overconfident by {score:.4f}"
    elif score < 0:
        return f"Underconfident by {abs(score):.4f}"
    else:
        return "Perfectly calibrated"

def create_discrimination_histogram(df, bot_col, pro_col, resolution_col):
    """
    Creates histograms to compare discrimination between bot and pro teams.

    Args:
        df (pandas.DataFrame): DataFrame with forecast and resolution data.
        bot_col (str): Column name for bot forecasts.
        pro_col (str): Column name for pro forecasts.
        resolution_col (str): Column name for resolutions.

    Returns:
        None
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Define bin edges
    bins = np.linspace(0, 1, 6)
    
    # Top plot: Questions that resolved 1
    ax1.hist([df[df[resolution_col] == 1][bot_col],
              df[df[resolution_col] == 1][pro_col]],
             bins=bins, label=['Bot Team', 'Pro Team'], alpha=0.7)
    ax1.set_title('Questions that Resolved \'Yes\'')
    ax1.set_xlabel('Assigned Probability')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Set integer y-ticks for top plot
    ymax1 = int(np.ceil(ax1.get_ylim()[1]))
    ax1.set_yticks(range(0, ymax1 + 1, 2))
    
    # Bottom plot: Questions that resolved 0
    ax2.hist([df[df[resolution_col] == 0][bot_col],
              df[df[resolution_col] == 0][pro_col]],
             bins=bins, label=['Bot Team', 'Pro Team'], alpha=0.7)
    ax2.set_title('Questions that Resolved \'No\'')
    ax2.set_xlabel('Assigned Probability')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Set integer y-ticks for bottom plot
    ymax2 = int(np.ceil(ax2.get_ylim()[1]))
    ax2.set_yticks(range(0, ymax2 + 1, 10))
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def get_weighted_score(df_forecasts):
    """
    Calculates the weighted total score for forecasts.

    Args:
        df_forecasts (pandas.DataFrame): DataFrame with 'head_to_head' and 'question_weight' columns.

    Returns:
        float: Weighted total score.
    """
    # Calculate the weighted score for each row
    df_forecasts['weighted_score'] = df_forecasts['head_to_head'] * df_forecasts['question_weight']

    # Calculate the total weighted score
    total_weighted_score = df_forecasts['weighted_score'].sum()

    # Calculate the sum of weights
    total_weight = df_forecasts['question_weight'].sum()

    # Calculate the weighted total score
    weighted_total_score = total_weighted_score / total_weight

    print(f"Weighted Total Score: {weighted_total_score:.4f}")

    return weighted_total_score

# ====== CODE FROM LUKE, REFACTORED BY CHATGPT =======

def string_location_to_scaled_location(string_location, question_row):
    """
    Converts a string location to a scaled location based on question type.

    Args:
        string_location (str): String representation of the location.
        question_row (pandas.Series): Row of the DataFrame containing question metadata.

    Returns:
        float: Scaled location.
    """
    if string_location in ["ambiguous", "annulled"]:
        raise ValueError("Cannot convert ambiguous or annulled to any real locations")

    question_type = question_row["type"]

    if question_type == "binary":
        return 1.0 if string_location == "yes" else 0.0

    if question_type == "multiple_choice":
        return float(question_row["options"].index(string_location))

    # continuous
    if string_location == "below_lower_bound":
        return question_row["range_min"] - 1.0
    if string_location == "above_upper_bound":
        return question_row["range_max"] + 1.0

    if question_type == "date":
        return datetime.fromisoformat(string_location).timestamp()

    # question.type == "numeric"
    return float(string_location)

def scaled_location_to_unscaled_location(scaled_location, question_row):
    """
    Converts a scaled location to an unscaled location based on question type.

    Args:
        scaled_location (float): Scaled location.
        question_row (pandas.Series): Row of the DataFrame containing question metadata.

    Returns:
        float: Unscaled location.
    """
    question_type = question_row["type"]

    if question_type in ["binary", "multiple_choice"]:
        return scaled_location

    zero_point = question_row.get("zero_point")
    range_max = question_row["range_max"]
    range_min = question_row["range_min"]

    if zero_point is not None:
        deriv_ratio = (range_max - zero_point) / max((range_min - zero_point), 1e-7)
        return (
            np.log((scaled_location - range_min) * (deriv_ratio - 1) + (range_max - range_min))
            - np.log(range_max - range_min)
        ) / np.log(deriv_ratio)

    return (scaled_location - range_min) / (range_max - range_min)

def nominal_location_to_cdf_location(nominal_location, question_data):
    """
    Takes a location in nominal format (e.g. 123, "123", or datetime in iso format) and scales it to
    metaculus's "internal representation" range [0, 1] incorporating question scaling

    Args:
        nominal_location (str or float): Nominal location.
        question_data (dict): Dictionary containing question metadata.

    Returns:
        float: CDF location.
    """
    if question_data["type"] == "date":
        scaled_location = datetime.fromisoformat(nominal_location).timestamp()
    else:
        scaled_location = float(nominal_location)
    # Unscale the value to put it into the range [0,1]
    range_min = question_data["range_min"]
    range_max = question_data["range_max"]
    zero_point = question_data["zero_point"]
    if ~np.isnan(zero_point) and (zero_point is not None):
        # logarithmically scaled question
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        unscaled_location = (
            np.log(
                (scaled_location - range_min) * (deriv_ratio - 1)
                + (range_max - range_min)
            )
            - np.log(range_max - range_min)
        ) / np.log(deriv_ratio)
    else:
        # linearly scaled question
        unscaled_location = (scaled_location - range_min) / (range_max - range_min)
    return unscaled_location

def get_cdf_at(cdf, unscaled_location):
    """
    Retrieves the CDF value at a given unscaled location.

    Args:
        cdf (list[float]): List of CDF values.
        unscaled_location (float): Unscaled location, with 0 meaning lower bound and 1 meaning upper bound.

    Returns:
        float: CDF value.
    """
    if unscaled_location <= 0:
        return cdf[0]
    if unscaled_location >= 1:
        return cdf[-1]
    index_scaled_location = unscaled_location * (len(cdf) - 1)
    if index_scaled_location.is_integer():
        return cdf[int(index_scaled_location)]
    # linear interpolation step
    left_index = int(index_scaled_location) # This is the floor, which is what we want
    right_index = left_index + 1
    left_value = cdf[left_index]
    right_value = cdf[right_index]
    return left_value + (right_value - left_value) * (
        index_scaled_location - left_index
    )

# ======== END OF LUKE'S CODE ==========

def cdf_between(row, cdf, lower_bound, upper_bound):
    """
    Calculates the probability between two bounds using the CDF.

    Args:
        row (pandas.Series): Row of the DataFrame containing question metadata.
        cdf (list[float]): List of CDF values.
        lower_bound (float): Lower bound.
        upper_bound (float): Upper bound.

    Returns:
        float: Probability between the bounds.
    """
    a = get_cdf_at(cdf, nominal_location_to_cdf_location(lower_bound, row))
    b = get_cdf_at(cdf, nominal_location_to_cdf_location(upper_bound, row))
    return (b - a)

def extract_year(title):
    """
    Extracts the year from a title string.

    Args:
        title (str): Title string.

    Returns:
        int or None: Extracted year or None if not found.
    """
    match = re.search(r'\b(19|20)\d{2}\b', title)
    return int(match.group(0)) if match else None

def extract_month(title):
    """
    Extracts the month from a title string.

    Args:
        title (str): Title string.

    Returns:
        str or None: Extracted month or None if not found.
    """
    match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', title)
    return match.group(0) if match else None

def compute_cp_baseline_score(value):
    """
    Gracefully computes the cp_baseline_score.

    Args:
        value (float): Input value.

    Returns:
        float: Computed cp_baseline_score or NaN if invalid.
    """
    try:
        # Ensure the value is numeric and not NaN
        if pd.isna(value) or not isinstance(value, (int, float)):
            return np.nan
        # Perform the calculation
        return 100 * np.log(value - np.log(0.5)) / np.log(2)
    except Exception:
        # Handle any unexpected errors
        return np.nan

def process_forecast_values(df):
    """
    Adds a 'bucket_forecast_value' column to the DataFrame (for interpreting CP distribution as a
    binary forecast, e.g. what probability mass did CP assign to "less than 7")
    Handles 'binary_version_tuple' and applies logic for 'less', 'greater', and 'complicated'.

    Args:
        df (pandas.DataFrame): DataFrame with 'binary_version_tuple', 'forecast_values', and other question-specific columns.

    Returns:
        pandas.DataFrame: Updated DataFrame with 'bucket_forecast_value' column added.
    """
    def compute_bucket_forecast_value(row):
        # Handle binary_version_tuple gracefully
        if pd.isna(row['binary_version_tuple']) or not isinstance(row['binary_version_tuple'], (list, tuple)):
            return None
        
        # Extract the first and second elements of the tuple
        comparison_type = row['binary_version_tuple'][0]
        string_location = row['binary_version_tuple'][1]
        
        # Skip if comparison_type is 'complicated'
        if comparison_type == 'complicated':
            return None
        
        # Compute forecast_value using the extracted string_location
        forecast_value = get_cdf_at(row['cdf'], nominal_location_to_cdf_location(string_location, row))
        
        # Apply logic based on comparison_type
        if comparison_type == 'less':
            return forecast_value
        elif comparison_type == 'greater':
            return 1 - forecast_value
        
        return None

    # Apply the function to each row and overwrite forecast_value (currently contains cdf, which we no longer need)
    df['forecast_values'] = df.apply(compute_bucket_forecast_value, axis=1)
    return df
