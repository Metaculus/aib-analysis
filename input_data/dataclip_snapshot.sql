-- Dataclip query for the data used in Q1 analysis
SELECT
      f.id AS forecast_id,
      q.id AS question_id,
      qp.post_id,
      q.title AS question_title,
      q.created_at, -- ?? same as publish time?
      f.author_id,
      f.probability_yes,
      f.probability_yes_per_category,
      f.continuous_cdf,
      f.start_time AS forecast_timestamp,
      u.username AS forecaster,
      u.is_bot,
      q.resolution,
      q.scheduled_close_time,
      q.actual_close_time,
      q.scheduled_resolve_time,
      q.actual_resolve_time,
      q.spot_scoring_time,
      p.published_at,
      q.question_weight,
      q.cp_reveal_time,
      -- q.resolution_criteria,
      -- q.description,
      -- q.fine_print,
      q.type,
      q.options,
      q.range_min,
      q.range_max,
      q.open_lower_bound,
      q.open_upper_bound,
      q.zero_point,
      pr.name as project_title
FROM questions_forecast f
LEFT JOIN users_user u ON f.author_id = u.id
LEFT JOIN questions_question q ON f.question_id = q.id
LEFT JOIN questions_question_post qp ON q.id = qp.question_id
LEFT JOIN posts_post p ON p.id = qp.post_id
LEFT JOIN projects_project pr ON p.default_project_id = pr.id
WHERE q.title NOT ILIKE '%[practice]%'
AND p.default_project_id = 32627 -- Q1 AIB BOT Tournament
-- AND p.default_project_id = 32631 -- Q1 AIB PRO Tournament
-- AND p.default_project_id = 32630 -- Q1 Quarterly Cup
AND u.is_bot = true
-- AND f.start_time < q.cp_reveal_time -- If for quarterly cup
ORDER BY f.start_time DESC;