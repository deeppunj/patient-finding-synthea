-- diagnostic delay for adults newly diagnosed with kidney disease in 2019-2025
-- = time between the first abnormal kidney test and the diagnosis
-- abnormal: egfr < 60 or creatinine > 1.3 mg/dL
-- uses first_kidney_dx from 2_cohort.sql

DROP TABLE IF EXISTS diagnostic_delay;

CREATE TABLE diagnostic_delay AS
WITH new_patients AS (
    SELECT k.patient_id, k.first_dx_date
    FROM first_kidney_dx k
    JOIN patients p ON p.patient_id = k.patient_id
    WHERE k.first_dx_date >= '2019-01-01'
      AND p.birth_date <= k.first_dx_date - INTERVAL '18 years'   -- adult at diagnosis
),
first_warning AS (
    -- earliest abnormal test before the diagnosis
    SELECT n.patient_id,
           MIN(o.obs_time)::date AS first_abnormal_test_date
    FROM new_patients n
    JOIN observations o
      ON o.patient_id = n.patient_id
     AND o.obs_time < n.first_dx_date
    WHERE (o.code = '33914-3' AND o.value::double precision < 60)               -- egfr
       OR (o.code IN ('38483-4', '2160-0') AND o.value::double precision > 1.3) -- creatinine
    GROUP BY n.patient_id
),
hypertension AS (
    -- high blood pressure before the diagnosis
    SELECT n.patient_id, 1 AS had_hypertension
    FROM new_patients n
    JOIN conditions c
      ON c.patient_id = n.patient_id
     AND c.start_date < n.first_dx_date
    WHERE c.description LIKE 'Essential hypertension%'
    GROUP BY n.patient_id
)
SELECT n.patient_id,
       n.first_dx_date,
       w.first_abnormal_test_date,
       n.first_dx_date - w.first_abnormal_test_date AS delay_days,   -- date - date = days. NULL if no warning
       COALESCE(h.had_hypertension, 0)               AS had_hypertension
FROM new_patients n
LEFT JOIN first_warning w ON w.patient_id = n.patient_id
LEFT JOIN hypertension h  ON h.patient_id = n.patient_id;
