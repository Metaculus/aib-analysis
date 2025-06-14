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