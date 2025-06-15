-- Snapshot of SQL for dataclips as of 2025-06-14

-- Forecasts for Pros Q1

WITH forecasts AS (
    SELECT
        qf.question_id,
        qf.author_id,
        qf.probability_yes,
        qf.probability_yes_per_category,
        qf.continuous_cdf,
        qf.start_time
    FROM
        questions_forecast qf
)
SELECT
    q.id AS question_id,
    q.title AS question_title,
    q.created_at, -- ?? same as publish time?
    f.author_id,
    f.probability_yes,
    f.probability_yes_per_category,
    f.continuous_cdf,
    f.start_time AS forecast_timestamp,
    u.username AS forecaster,
    -- q.mod_status,
    q.resolution,
    q.scheduled_close_time,
    q.actual_close_time,
    q.question_weight,
    q.type,
    q.options,
    q.range_min,
    q.range_max,
    q.open_lower_bound,
    q.open_upper_bound,
    qp.post_id
FROM
    questions_question q
INNER JOIN
    forecasts f ON q.id = f.question_id
INNER JOIN
    users_user u ON f.author_id = u.id -- for usernames
INNER JOIN
    questions_question_post qp ON qp.question_id = q.id OR qp.id = q.group_id -- for post ids
INNER JOIN
    posts_post pp ON pp.id = qp.post_id
WHERE
    pp.default_project_id = 32631
    -- sc.question_id = 30285
    -- pp.default_project_id = 3345 -- Q3 FOR TESTING
    -- AND sc.score_type = 'spot_baseline'
ORDER BY
    q.id, u.id;












-- Forecasts for bots Q1
WITH forecasts AS (
    SELECT
        qf.question_id,
        qf.author_id,
        qf.probability_yes,
        qf.probability_yes_per_category,
        qf.continuous_cdf,
        qf.start_time
    FROM
        questions_forecast qf
)
SELECT
    q.id AS question_id,
    q.title AS question_title,
    q.created_at, -- ?? same as publish time?
    f.author_id,
    f.probability_yes,
    f.probability_yes_per_category,
    f.continuous_cdf,
    f.start_time AS forecast_timestamp,
    u.username AS forecaster,
    u.is_bot,
    -- q.mod_status,
    q.resolution,
    q.scheduled_close_time,
    q.actual_close_time,
    q.question_weight,
    q.cp_reveal_time,
    q.type,
    q.options,
    q.range_min,
    q.range_max,
    q.open_lower_bound,
    q.open_upper_bound,
    qp.post_id
FROM
    questions_question q
INNER JOIN
    forecasts f ON q.id = f.question_id
INNER JOIN
    users_user u ON f.author_id = u.id -- for usernames
INNER JOIN
    questions_question_post qp ON qp.question_id = q.id OR qp.id = q.group_id -- for post ids
INNER JOIN
    posts_post pp ON pp.id = qp.post_id
WHERE
    pp.default_project_id = 32627 AND
    -- pp.default_project_id = 3349 AND -- Q3 FOR TESTING
    u.is_bot = TRUE AND
    q.title NOT LIKE '%PRACTICE%'
    -- AND q.close_time <= CURRENT_DATE
    -- AND (q.resolution >= 0 OR q.resolution IS NULL)
ORDER BY
    q.id, u.id;






-- Forecasts for Quarterly Cup Q1
WITH forecasts AS (
    SELECT
        qf.question_id,
        qf.author_id,
        qf.probability_yes,
        qf.probability_yes_per_category,
        qf.continuous_cdf,
        qf.start_time
    FROM
        questions_forecast qf
)
SELECT
    q.id AS question_id,
    q.title AS question_title,
    q.created_at, -- ?? same as publish time?
    f.author_id,
    f.probability_yes,
    f.probability_yes_per_category,
    f.continuous_cdf,
    f.start_time AS forecast_timestamp,
    u.username AS forecaster,
    u.is_bot,
    -- q.mod_status,
    q.resolution,
    q.scheduled_close_time,
    q.actual_close_time,
    q.question_weight,
    q.cp_reveal_time,
    q.type,
    q.options,
    q.range_min,
    q.range_max,
    q.open_lower_bound,
    q.open_upper_bound,
    qp.post_id
FROM
    questions_question q
INNER JOIN
    forecasts f ON q.id = f.question_id
INNER JOIN
    users_user u ON f.author_id = u.id -- for usernames
INNER JOIN
    questions_question_post qp ON qp.question_id = q.id OR qp.id = q.group_id -- for post ids
INNER JOIN
    posts_post pp ON pp.id = qp.post_id
WHERE
    pp.default_project_id = 32630 AND
    u.is_bot = FALSE AND
    f.start_time < q.cp_reveal_time
ORDER BY
    q.id, u.id;







-- Scores for Pros Q1

WITH scores AS (
  SELECT
    cp.question_id,
    cp.user_id,
    cp.score,
    cp.score_type
  FROM
    scoring_score cp
)
SELECT
    q.id AS question_id,
    q.title AS question_title,
    q.created_at, -- ?? same as publish time?
    sc.user_id,
    sc.score,
    sc.score_type,
    u.username AS forecaster,
    -- q.mod_status,
    q.resolution,
    q.scheduled_close_time,
    q.actual_close_time,
    q.question_weight,
    qp.post_id
FROM
    questions_question q
INNER JOIN
    scores sc ON q.id = sc.question_id
INNER JOIN
    users_user u ON sc.user_id = u.id -- for usernames
INNER JOIN
    questions_question_post qp ON qp.question_id = q.id OR qp.id = q.group_id -- for post ids
INNER JOIN
    posts_post pp ON pp.id = qp.post_id
WHERE
    pp.default_project_id = 32631
    -- sc.question_id = 30285
    -- pp.default_project_id = 3345 -- Q3 FOR TESTING
    -- AND sc.score_type = 'spot_baseline'
ORDER BY
    q.id, u.id;







-- Scores for Bots Q1

WITH scores AS (
  SELECT
    cp.question_id,
    cp.user_id,
    cp.score,
    cp.score_type
  FROM
    scoring_score cp
)
SELECT
    q.id AS question_id,
    q.title AS question_title,
    q.created_at, -- ?? same as publish time?
    sc.user_id,
    sc.score,
    sc.score_type,
    u.username AS forecaster,
    u.is_bot,
    -- q.mod_status,
    q.resolution,
    q.scheduled_close_time,
    q.actual_close_time,
    q.question_weight,
    q.cp_reveal_time,
    qp.post_id
FROM
    questions_question q
INNER JOIN
    scores sc ON q.id = sc.question_id
INNER JOIN
    users_user u ON sc.user_id = u.id -- for usernames
INNER JOIN
    questions_question_post qp ON qp.question_id = q.id OR qp.id = q.group_id -- for post ids
INNER JOIN
    posts_post pp ON pp.id = qp.post_id
WHERE
    pp.default_project_id = 32627 AND
    -- pp.default_project_id = 3349 AND -- Q3 FOR TESTING
    u.is_bot = TRUE AND
    q.title NOT LIKE '%PRACTICE%'
    -- AND sc.score_type = 'spot_baseline'
    --  AND q.close_time <= CURRENT_DATE
    --  AND (q.resolution >= 0 OR q.resolution IS NULL)
ORDER BY
    q.id, u.id;




-- Q4 CP Forecasts
WITH forecasts AS (
   SELECT forecast_values, means, centers, method, start_time, end_time, question_id
   FROM questions_aggregateforecast cp
   WHERE method = 'recency_weighted'
)
SELECT
   qp.post_id,
   cp.question_id,
   cp.forecast_values,
   cp.means,
   cp.centers,
   cp.method,
   cp.start_time ,
   cp.end_time,
   q.title,
   q.options,
   q.range_min,
   q.range_max,
   q.zero_point,
   q.type
FROM forecasts cp
INNER JOIN questions_question_post qp
   ON cp.question_id = qp.id
INNER JOIN questions_question q
   ON cp.question_id = q.id
WHERE qp.post_id IN (1621, 4779, 5587, 24806, 28972, 28974, 28925, 28925, 28925, 28925, 28925, 13924, 20694, 26616, 25749, 26304, 28854, 28841, 28657, 28650, 28658, 29086, 29140, 29027, 29090, 28742, 17373, 29222, 28395, 29242, 29077, 29077, 29077, 29028, 29028, 29028, 28783, 28546, 8001, 29145, 29221, 28704, 5201, 3458, 6806, 11122, 17104, 29507, 29507, 29141, 22017, 22017, 28706, 17100, 2788, 19824, 7792, 29609, 27881, 27881, 27881, 27881, 28834, 29524, 29524, 29524, 29524, 29524, 29524, 29608, 29608, 29608, 29608, 29608, 19724, 29807, 29807, 29807, 29807, 11589, 1624, 25801, 20694, 20557, 27141, 27881, 27881, 27881, 27881, 15537, 20172, 26884, 21167, 29900, 29841, 29816, 29903, 29902, 29901, 29847, 29847, 29847, 29847, 29847, 29847, 23018, 17102, 22046, 29709, 5313, 1568, 12217, 30213, 30297, 30251, 30252, 30295, 27868, 29956, 30472, 30472, 20776, 30516, 30476, 30475, 30475, 30475, 30434, 30434, 30434, 8533, 30517, 30477, 30477, 30477, 30709, 30709, 30709, 30708, 30651, 16014, 4862, 30809, 30922, 30870, 30870, 30870, 28860, 28860, 28860, 28860, 30925, 30925, 30925)
--WHERE qp.post_id = 29507