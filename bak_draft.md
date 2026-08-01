# Main Takeaways

* **Relative forecast performance on near-term questions appears to be a modest predictor for far-term questions:** This means that someone who ranks highly based on forecasts in the near term is also likely to rank highly based on forecasts in the far term.
* **The signal appears to weaken between questions as the time window between forecast and resolution increases:** This suggests that there may be some specialization effect in play.
* **Both the old and new Minibench designs show a statistically significant signal for predicting performance on far-term questions:** a direct, head-to-head test still does not find a statistical difference between the two Minibenches' performance on long term.
* **This analysis was conducted on all question types in aggregate:** There may be some question type specialization underlying the aggregate data.

# Introduction

One of the basic rules of forecasting is that the accuracy of a well-calibrated forecast will decrease as the window of time between a forecast and its respective event increases. As the time delta increases, so does the opportunity for noise, generally resulting in a regression towards the base case. For a trivial example, given a competition between two similarly postured teams that is yet to start, you may predict a 50:50 chance of winning for respective teams A:B, but if the game is near completion and one team has a significant lead over the other, the predictions will eventually converge to either 100:0 or 0:100 in favor of whatever team has a lead. Once again, assuming a well-calibrated forecaster, the accuracy of the later predictions will be higher (i.e., the prediction is closer to the resolution).

This analysis is not seeking to prove or disprove the fundamental principle that accuracy improves or decreases according to time. Rather, the intent is to discern whether forecasters (specifically, Metaculus' forecasting bots) may have specific competitive advantages for near-term or far-term questions, measured by baseline score. Simply: we want to know how well being good at short-term forecasts predicts being good at long-term prediction.

# Data Selection

This analysis uses data for all question types (binary, multiple-choice, and numeric) from Metaculus bot forecasters only. An earlier version of this analysis used only binary questions, but the smaller sample size produced noisier statistics, particularly in the Minibench-specific comparisons where the number of eligible forecasters and questions was already limited. This was addressed by scoring the same bot forecasters' multiple-choice and numeric questions as well, taking the dataset from roughly 124,000 to roughly 212,000 scored forecasts (about a 70% increase).

Not every question could be scored. Excluded due to data or methodological limitations: `discrete`-type questions (a variable-length outcome format the scoring method used here doesn't support), numeric questions on a log-scaled range (the scoring method assumes a linear scale), a small number of numeric questions with a malformed probability distribution, and questions with an "ambiguous" resolution (no real outcome to score against). These exclusions are small in aggregate, but not evenly distributed across tracks: the discrete-type exclusion in particular removes a disproportionate share of new Minibench's data relative to old Minibench's. The mix of binary vs. multiple-choice vs. numeric questions also differs meaningfully by track, new Minibench skews noticeably more binary-heavy than old Minibench or non-Minibench. Since baseline score is not on an identical scale across question types, this composition difference is a limitation worth keeping in mind when comparing tracks below, rather than a fully controlled experiment.

# Analysis

First, it is useful to understand how performance varies across time windows. For this, all forecasts were split into time windows of (0, 7], (7, 14], (14, 30], (30, 120], and (120, 365] days, and each forecast's [baseline score](https://www.metaculus.com/help/scores-faq/#baseline-score) was calculated. From there, a 95% CI bootstrap was constructed for the mean baseline score of each (forecaster, time window) pairing. The following chart was generated using the top 13 forecasters by volume in order to get quality bootstrapped intervals.

**Note:** This chart, and every correlation test in the sections below, requires a forecaster to have a minimum number of questions forecasted on in each time window being compared in order to get a reliable bootstrap; this restricts the analysis to established, higher-volume bots, and the results below may not generalize to newer or lower-volume forecasters.

![box plots for forecast terms per forecaster](https://cdn.metaculus.com/user_uploaded/Dropped_Image_12_dodBJTk.png)

Immediately, there appears to be a general pattern, although clearly not without exceptions, that bots tend to perform better in terms of scoring on far-term questions. Looking at this chart, we can already hypothesize that the correlation scores for performance between near-term and far-term forecasts will not be near 1. For example, looking at only the red (0, 7] day means and the purple (120, 365] day means for metac-o1+asknews (3rd from top) and metac-gemini-4o-pro+asknews (1st from top), despite having a similar general trend of an increase in forecast horizon resulting in an increase in score, 4o performs better on the near term than o1, but worse on the far term. Picking any random set of 2 forecasters and 2 time periods, it appears that the correlation is fractional at best.

## Spearman's Rank Correlation

To run this analysis, we are leveraging [Spearman's rank correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient), which calculates a correlation value $\rho = 1 - \frac{6\sum^{N}_{n=1}(R_0[F_n] - R_1[F_n])^2}{N(N^2 - 1)}$ where $N$ is the number of forecasters, $R_x[F_n]$ is the rank of forecaster $F_n$  in a leader-board $x$. A correlation value $\rho$ of 1 indicated perfect correlation (someone who places nth place in leader-board 1 also places nth place in leader-board 2), a value of -1 indicated perfect negative correlation (someone who places nth place in leader-board 1 places (N-n)th place in leader-board 2), and a value of 0 indicates that placement between the two leader-board is completely unrelated.

Since we are mainly concerned about competitive advantages, and since this metric doesn't depend on any statistical relationship between the forecasters, this works as a metric to address the research question.

One challenge is that "short term" and "long term" are not well defined, so this will be addressed by looking at multiple time windows, and, for the Minibench questions, simply comparing the Minibench performance (2-week window) with performance outside of Minibench on forecasts with a resolution window exceeding 2 weeks. The time chunks will be based on the same periods as the prior graph.

**Note:** For all statistical tests, the standard p-value threshold of 0.05 is being used for statistical significance. Since multiple p-values are being calculated on the same dataset, the  [Benjamini–Hochberg correction](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure) is additionally being used, although the correction appears to have no effect on the outcomes in this circumstance.

### Base Case

First off is the base case. For the following correlation matrix, questions were not pulled from Minibench. Each cell displays the rank correlation as r (short for rho), and a p-value for $H_1: \rho > 0, H_0: \rho <= 0$. The matrix is mirrored across the diagonal.

![correlation matrix for non-minibench](https://cdn.metaculus.com/user_uploaded/Dropped_Image_7_eDrhvx4.png)

<details>
  Pairwise Spearman correlations — no minibench:
  Multiple-comparison check (10 cells tested): 10 significant at raw α\=0.05, 10 remain significant after Benjamini-Hochberg correction (q\<0.05)  \['\*' below \= BH-significant]
  Pair                                          r      p(ρ≤0)      n
  \--------------------------------------------------------------------------------
  (0,7]d vs (7,14]d                         0.777      \<.001\*    123
  (0,7]d vs (14,30]d                        0.659      \<.001\*    123
  (0,7]d vs (30,120]d                       0.508      \<.001\*     97
  (0,7]d vs (120,365]d                      0.377      0.009\*     74
  (7,14]d vs (14,30]d                       0.610      \<.001\*    123
  (7,14]d vs (30,120]d                      0.441      \<.001\*     97
  (7,14]d vs (120,365]d                     0.524      \<.001\*     74
  (14,30]d vs (30,120]d                     0.685      \<.001\*     97
  (14,30]d vs (120,365]d                    0.602      \<.001\*     74
  (30,120]d vs (120,365]d                   0.682      \<.001\*     74
</details>

For this scenario, all (100%) of the comparisons show statistical significance that the correlation is above 0, implying that performances in any time window are an indicator of performance in any other time window. Looking in detail, the stronger correlation generally lies closer to the diagonal, which makes intuitive sense.

### Minibench Comparison

Now moving on to the Minibench matrices. These matrices have a few major stipulations:

* Since Minibench questions only exist within a (0, 14] day time period, they have to be compared to non-Minibench questions to determine correlation with long-term forecasts.
* Since the questions are formulated differently, a lower correlation value across the boundary of 14 days can be anticipated.
* Since the far-term questions are from the non-Minibench dataset, their mutual correlations (bottom right 3x3) are exactly the values from the non-Minibench matrix, and are not re-tested here.

First, the old Minibench questions, which were automatically generated based off of data sources which could be programmatically checked. These are acknowledged to be a lower quality of question than both classic Metaculus questions and the newer LLM-generated Minibench questions.

![correlation matrix with old minibench](https://cdn.metaculus.com/user_uploaded/Dropped_Image_8_wea5bHG.png)

<details>
  Pairwise Spearman correlations — old minibench + non minibench:
  Multiple-comparison check (7 cells tested): 7 significant at raw α\=0.05, 7 remain significant after Benjamini-Hochberg correction (q\<0.05)  \['\*' below \= BH-significant]
  Pair                                          r      p(ρ≤0)      n
  \--------------------------------------------------------------------------------
  (0,7]d vs (7,14]d                         0.583      \<.001\*     91
  (0,7]d vs (14,30]d                        0.760      \<.001\*     91
  (0,7]d vs (30,120]d                       0.571      \<.001\*     91
  (0,7]d vs (120,365]d                      0.472      0.009\*     69
  (7,14]d vs (14,30]d                       0.396      0.002\*     91
  (7,14]d vs (30,120]d                      0.208      0.031\*     91
  (7,14]d vs (120,365]d                     0.460      0.001\*     69
  (14,30]d vs (30,120]d                     0.685  (same as base)
  (14,30]d vs (120,365]d                    0.602  (same as base)
  (30,120]d vs (120,365]d                   0.682  (same as base)
</details>

Correlations remain demonstrably positive, with generally slight variations that can be attributed to noise (see cross-comparison section). Even the worst-scoring correlation score (0.031) is within the bounds of statistical significance.
Now, the matrix for the new, LLM-generated Minibench questions. The same stipulation applies: a different procedure will result in weaker results.

![correlation matrix with new minibench](https://cdn.metaculus.com/user_uploaded/Dropped_Image_9_6mpbB5R.png)

<details>
  Pairwise Spearman correlations — new minibench + non minibench:
  Multiple-comparison check (7 cells tested): 7 significant at raw α\=0.05, 7 remain significant after Benjamini-Hochberg correction (q\<0.05)  \['\*' below \= BH-significant]
  Pair                                          r      p(ρ≤0)      n
  \--------------------------------------------------------------------------------
  (0,7]d vs (7,14]d                         0.412      0.042\*     55
  (0,7]d vs (14,30]d                        0.529      0.007\*     55
  (0,7]d vs (30,120]d                       0.634      \<.001\*     29
  (0,7]d vs (120,365]d                      0.635      0.006\*     28
  (7,14]d vs (14,30]d                       0.778      \<.001\*     55
  (7,14]d vs (30,120]d                      0.735      0.003\*     29
  (7,14]d vs (120,365]d                     0.718      0.009\*     28
  (14,30]d vs (30,120]d                     0.685  (same as base)
  (14,30]d vs (120,365]d                    0.602  (same as base)
  (30,120]d vs (120,365]d                   0.682  (same as base)

  Pooled new vs. old minibench comparison (single joint test across all 7 bin-pairs):
  Common population (present in every tested bin-pair, both minibench designs): n\=28
  Pooled effect (mean Δρ \= ρ\_new − ρ\_old across bin-pairs): -0.033
  95% bootstrap CI: \[-0.331, 0.085]
  p (H0: pooled Δρ ≤ 0): 0.890
  Descriptive only (not an independent-replication count): 2/7 individual bin-pairs numerically favor new\_minibench.
</details>

Once again, correlations remain demonstrably positive.

## Cross-Comparison

Part of this project is determining if there is a significant difference between the correlation using the newer LLM generated Minibench questions vs the older automated Minibench questions. This analysis goes into detail on this point of comparison.

### Final, Near vs. Far Term Correlations

For the final correlation matrix, the following correlation matrix reduced the buckets into (0, 14] days and (14, 365] days, with the following bins:

* Non-Minibench forecasts in the (0, 14] day time window.
* Old Minibench forecasts.
* New Minibench forecasts.
* Non-Minibench forecasts in the (14, 365] day time window.

Minibench questions are all in the (0, 14] day time window.

![correlation matrix for all datasets](https://cdn.metaculus.com/user_uploaded/Dropped_Image_10_L8SXt4o.png)

<details>
  Pairwise Spearman correlations — focused (0,14]d by track vs (14,365]d non-minibench:
  Multiple-comparison check (6 cells tested): 6 significant at raw α\=0.05, 6 remain significant after Benjamini-Hochberg correction (q\<0.05)  \['\*' below \= BH-significant]
  Pair                                                    r      p(ρ≤0)      n
  \------------------------------------------------------------------------------------------
  (0,14]d old minibench vs (0,14]d new minibench      0.650      0.003\*     29
  (0,14]d old minibench vs (0,14]d non minibench      0.496      \<.001\*     91
  (0,14]d old minibench vs (14,365]d non minibench    0.433      0.001\*     91
  (0,14]d new minibench vs (0,14]d non minibench      0.768      \<.001\*     50
  (0,14]d new minibench vs (14,365]d non minibench    0.624      0.001\*     55
  (0,14]d non minibench vs (14,365]d non minibench    0.366      \<.001\*    120
</details>

Once again, all comparisons show a correlation that is statistically significantly greater than 0. Interestingly, the weakest entry appears to be the comparing the near-term and far-term resolutions within the non-Minibench questions.

The other five comparisons in this matrix are not fully clean: three (old Minibench vs. non-Minibench, new Minibench vs. non-Minibench, and old vs. new Minibench) hold the time window fixed at (0, 14] but still mix in a shift in question format, and two (old Minibench vs. far-term non-Minibench, new Minibench vs. far-term non-Minibench) mix both a shift in time horizon and a shift in question format at once. This design can't cleanly separate which of these is driving any individual correlation, which is worth keeping in mind when weighing the Minibench-specific numbers against the one clean comparison above.

### Pairwise Comparison

The following table tests, for each near-term forecasting bucket, whether one's near-to-far-term correlation exceeds the other's (one-sided, H0: ρ\_row ≤ ρ\_col). Entries mirroring across the diagonal will sum to 1.

![pairwise comparison results](https://cdn.metaculus.com/user_uploaded/Dropped_Image_11_6BFYhxN.png)

<details>
  Pairwise comparisons: H0: rho(row) \<\= rho(col), one-sided cluster bootstrap
  (each period has 2 tests -- one vs. each other period; small p \= row's rho is significantly higher)
  BH check across the 3 unique pairs: 0/3 significant after correction (raw: 0/3)
  '\*' \= significant after BH correction (q\<0.05, shared by both directional readouts of a pair)
  vs old minibench      vs new minibench      vs no minibench
  old minibench         —                     p\=0.166 (n\=29)      p\=0.464 (n\=91)
  new minibench         p\=0.836 (n\=29)      —                     p\=0.133 (n\=50)
  no minibench          p\=0.536 (n\=91)      p\=0.868 (n\=50)                   —
  Point estimates:
  old minibench    r\=0.433  95% CI\=\[0.180, 0.567]  n\=91
  new minibench    r\=0.624  95% CI\=\[0.139, 0.622]  n\=55
  no minibench     r\=0.366  95% CI\=\[0.122, 0.443]  n\=120
</details>

None of the three pairwise comparisons reach significance, so the table does not support a claim that any track is a better predictor of long-term performance than another.  There are no truly significant results here, with the closest (p\=0.133) coming from the comparison between new Minibench and non-Minibench: whether new Minibench's near-to-far-term correlation exceeds non-Minibench's own near-to-far-term correlation.

# Conclusions

Across every time window tested, inside Minibench, outside of it, and across the Minibench boundary, near-term forecasting performance is a statistically significant, if modest, predictor of far-term performance. The signal weakens as the two time windows get farther apart. This pattern held up, and got stronger, once the dataset was enlarged to include all question types rather than binary questions alone.

One of the questions this analysis originally set out to answer, whether the redesigned, LLM-generated Minibench carries a stronger long-term signal than the original automated Minibench, is not resolved by this data. Both designs individually show a statistically significant correlation with far-term performance, but a direct head-to-head test finds no significant difference between them. Which design nominally leads also depends on how much data is used: an earlier, smaller version of this analysis favored new Minibench, while old Minibench's near-zero correlation in one time window turned out to be a small-sample artifact that resolved once the dataset was enlarged. Neither result should be read as evidence favoring one design over the other.

This analysis also can't fully separate whether the redesign helps from whether the type of question being asked changes the answer. Baseline score is not on a directly comparable scale across binary, multiple-choice, and numeric questions, and the type composition genuinely differs by track. New Minibench's short-window population is disproportionately binary, relative to old Minibench and non-Minibench. A cleaner test would either compare each track within a single question type at a time, or normalize baseline score within type before pooling across tracks. Absent that, whether the new Minibench design is actually better remains an open question this dataset cannot settle yet.
