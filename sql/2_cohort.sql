-- cohort: who is in the study + label
--
-- snapshot ("index date") on 1 jan of every year 2019-2025
--   features: 2 years before the index date (3_features.sql)
--   label = 1 if first kidney disease diagnosis in the 12 months after it
-- adults only, alive on the index date, no kidney disease yet (we want new patients)
-- same patient can be in several snapshots


-- indexes, otherwise the joins on observations (~9M rows) are very slow
CREATE INDEX IF NOT EXISTS idx_encounters   ON encounters   (patient_id, start_time);
CREATE INDEX IF NOT EXISTS idx_conditions   ON conditions   (patient_id, start_date);
CREATE INDEX IF NOT EXISTS idx_observations ON observations (patient_id, code, obs_time);
CREATE INDEX IF NOT EXISTS idx_medications  ON medications  (patient_id, start_time);
ANALYZE;

DROP TABLE IF EXISTS first_kidney_dx, cohort CASCADE;


-- first kidney disease diagnosis per patient
-- also counting diabetic kidney disease and transplant records,
-- so nobody with existing kidney problems ends up in the cohort
CREATE TABLE first_kidney_dx AS
SELECT patient_id,
       MIN(start_date) AS first_dx_date
FROM conditions
WHERE description LIKE 'Chronic kidney disease stage%'
   OR description LIKE 'End-stage renal disease%'
   OR description LIKE 'Disorder of kidney due to diabetes mellitus%'
   OR description LIKE 'Awaiting transplantation of kidney%'
   OR description LIKE 'History of renal transplant%'
GROUP BY patient_id;


CREATE TABLE cohort AS
WITH index_dates AS (
    -- 2019-01-01, 2020-01-01, ..., 2025-01-01
    SELECT make_date(year, 1, 1) AS index_date
    FROM generate_series(2019, 2025) AS year
),
active_patients AS (
    -- at least 1 visit in the 2 years before the index date
    SELECT DISTINCT e.patient_id, d.index_date
    FROM encounters e
    JOIN index_dates d
      ON e.start_time >= d.index_date - INTERVAL '2 years'
     AND e.start_time <  d.index_date
)
SELECT a.patient_id,
       a.index_date,
       -- 1 if diagnosed within 1 year after the index date
       -- (never diagnosed -> first_dx_date is NULL -> 0)
       CASE WHEN k.first_dx_date < a.index_date + INTERVAL '1 year' THEN 1 ELSE 0 END AS label
FROM active_patients a
JOIN patients p ON p.patient_id = a.patient_id
LEFT JOIN first_kidney_dx k ON k.patient_id = a.patient_id
-- note: first tried AGE(...) >= INTERVAL '18 years', but postgres counts a month as 30 days
-- when comparing intervals, so some 17-year-olds got in. comparing dates is exact.
WHERE p.birth_date <= a.index_date - INTERVAL '18 years'             -- 18+
  AND (p.death_date IS NULL OR p.death_date > a.index_date)         -- alive
  AND (k.first_dx_date IS NULL OR k.first_dx_date >= a.index_date); -- not diagnosed yet
