-- features for every row in the cohort
-- only data from before the index date:
--   visits, labs, medicines -> 2 years before
--   diagnoses -> any time before (long-term conditions)
-- each CTE below = 1 group of features, joined together at the end

DROP TABLE IF EXISTS features;

CREATE TABLE features AS
WITH

-- visits in the 2-year window
visits AS (
    SELECT c.patient_id, c.index_date,
           COUNT(*)                                               AS n_visits,
           COUNT(*) FILTER (WHERE e.encounter_class = 'emergency') AS n_emergency_visits,
           MAX(e.start_time)::date                                AS last_visit_date  -- for the leakage test
    FROM cohort c
    JOIN encounters e
      ON e.patient_id = c.patient_id
     AND e.start_time >= c.index_date - INTERVAL '2 years'
     AND e.start_time <  c.index_date
    GROUP BY c.patient_id, c.index_date
),

-- earlier diagnoses, 1/0 flags (MAX over all diagnoses = 1 if they ever had it)
diagnoses AS (
    SELECT c.patient_id, c.index_date,
           MAX(CASE WHEN d.description LIKE 'Essential hypertension%'   THEN 1 ELSE 0 END) AS has_hypertension,
           MAX(CASE WHEN d.description LIKE 'Diabetes mellitus type 2%' THEN 1 ELSE 0 END) AS has_diabetes,
           MAX(CASE WHEN d.description LIKE 'Prediabetes%'              THEN 1 ELSE 0 END) AS has_prediabetes,
           MAX(CASE WHEN d.description LIKE 'Body mass index 30+%'      THEN 1 ELSE 0 END) AS has_obesity,
           MAX(CASE WHEN d.description LIKE 'Ischemic heart disease%'   THEN 1 ELSE 0 END) AS has_heart_disease
    FROM cohort c
    JOIN conditions d
      ON d.patient_id = c.patient_id
     AND d.start_date < c.index_date
    GROUP BY c.patient_id, c.index_date
),

-- lab results / vital signs in the 2-year window
-- LOINC code -> short name. creatinine, urea and glucose have 2 codes each (blood / serum), same name
labs AS (
    SELECT c.patient_id, c.index_date, o.obs_time,
           CASE o.code
               WHEN '33914-3' THEN 'egfr'                       -- kidney filtering, < 60 = CKD
               WHEN '38483-4' THEN 'creatinine'
               WHEN '2160-0'  THEN 'creatinine'
               WHEN '6299-2'  THEN 'urea'
               WHEN '3094-0'  THEN 'urea'
               WHEN '14959-1' THEN 'albumin_creatinine_ratio'   -- protein in urine, early kidney damage
               WHEN '2339-0'  THEN 'glucose'
               WHEN '2345-7'  THEN 'glucose'
               WHEN '4548-4'  THEN 'hba1c'
               WHEN '8480-6'  THEN 'systolic_bp'
               WHEN '39156-5' THEN 'bmi'
           END AS lab,
           o.value::double precision AS value                   -- value is stored as text
    FROM cohort c
    JOIN observations o
      ON o.patient_id = c.patient_id
     AND o.obs_time >= c.index_date - INTERVAL '2 years'
     AND o.obs_time <  c.index_date
    WHERE o.type = 'numeric'
      AND o.code IN ('33914-3', '38483-4', '2160-0', '6299-2', '3094-0', '14959-1',
                     '2339-0', '2345-7', '4548-4', '8480-6', '39156-5')
),

-- most recent value per lab (DISTINCT ON keeps the first row per group, sorted newest first)
latest_labs AS (
    SELECT DISTINCT ON (patient_id, index_date, lab)
           patient_id, index_date, lab, value
    FROM labs
    ORDER BY patient_id, index_date, lab, obs_time DESC
),

-- long -> wide: 1 column per lab. lab not done = NULL
lab_columns AS (
    SELECT patient_id, index_date,
           MAX(value) FILTER (WHERE lab = 'egfr')                     AS last_egfr,
           MAX(value) FILTER (WHERE lab = 'creatinine')               AS last_creatinine,
           MAX(value) FILTER (WHERE lab = 'urea')                     AS last_urea,
           MAX(value) FILTER (WHERE lab = 'albumin_creatinine_ratio') AS last_albumin_creatinine_ratio,
           MAX(value) FILTER (WHERE lab = 'glucose')                  AS last_glucose,
           MAX(value) FILTER (WHERE lab = 'hba1c')                    AS last_hba1c,
           MAX(value) FILTER (WHERE lab = 'systolic_bp')              AS last_systolic_bp,
           MAX(value) FILTER (WHERE lab = 'bmi')                      AS last_bmi
    FROM latest_labs
    GROUP BY patient_id, index_date
),

lab_extra AS (
    SELECT patient_id, index_date,
           MIN(value) FILTER (WHERE lab = 'egfr') AS min_egfr,       -- lowest egfr in the window
           MAX(obs_time)::date                    AS last_lab_date   -- for the leakage test
    FROM labs
    GROUP BY patient_id, index_date
),

-- medicines active at some point in the window
-- (started before the index date, not stopped before the window started)
medicines AS (
    SELECT c.patient_id, c.index_date,
           COUNT(DISTINCT m.description) AS n_medications,
           -- ace inhibitor / arb (blood pressure, also protect the kidneys)
           MAX(CASE WHEN m.description ILIKE '%lisinopril%' OR m.description ILIKE '%losartan%'
                    THEN 1 ELSE 0 END) AS on_blood_pressure_medicine,
           MAX(CASE WHEN m.description ILIKE '%metformin%' OR m.description ILIKE '%insulin%'
                    THEN 1 ELSE 0 END) AS on_diabetes_medicine,
           -- nsaids can damage the kidneys with long use
           MAX(CASE WHEN m.description ILIKE '%ibuprofen%' OR m.description ILIKE '%naproxen%'
                    THEN 1 ELSE 0 END) AS on_nsaid_painkiller
    FROM cohort c
    JOIN medications m
      ON m.patient_id = c.patient_id
     AND m.start_time < c.index_date
     AND (m.stop_time IS NULL OR m.stop_time >= c.index_date - INTERVAL '2 years')
    GROUP BY c.patient_id, c.index_date
)

-- everything together, LEFT JOIN so patients without labs/medicines are kept
-- counts and flags: NULL -> 0. lab values stay NULL (= not tested), the model handles that
SELECT c.patient_id,
       c.index_date,
       c.label,
       DATE_PART('year', AGE(c.index_date, p.birth_date))::int AS age,
       CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END             AS male,
       COALESCE(v.n_visits, 0)           AS n_visits,
       COALESCE(v.n_emergency_visits, 0) AS n_emergency_visits,
       COALESCE(d.has_hypertension, 0)   AS has_hypertension,
       COALESCE(d.has_diabetes, 0)       AS has_diabetes,
       COALESCE(d.has_prediabetes, 0)    AS has_prediabetes,
       COALESCE(d.has_obesity, 0)        AS has_obesity,
       COALESCE(d.has_heart_disease, 0)  AS has_heart_disease,
       l.last_egfr,
       x.min_egfr,
       l.last_creatinine,
       l.last_urea,
       l.last_albumin_creatinine_ratio,
       l.last_glucose,
       l.last_hba1c,
       l.last_systolic_bp,
       l.last_bmi,
       COALESCE(m.n_medications, 0)              AS n_medications,
       COALESCE(m.on_blood_pressure_medicine, 0) AS on_blood_pressure_medicine,
       COALESCE(m.on_diabetes_medicine, 0)       AS on_diabetes_medicine,
       COALESCE(m.on_nsaid_painkiller, 0)        AS on_nsaid_painkiller,
       -- not features, only for the leakage test
       v.last_visit_date,
       x.last_lab_date
FROM cohort c
JOIN patients p         ON p.patient_id = c.patient_id
LEFT JOIN visits v      ON v.patient_id = c.patient_id AND v.index_date = c.index_date
LEFT JOIN diagnoses d   ON d.patient_id = c.patient_id AND d.index_date = c.index_date
LEFT JOIN lab_columns l ON l.patient_id = c.patient_id AND l.index_date = c.index_date
LEFT JOIN lab_extra x   ON x.patient_id = c.patient_id AND x.index_date = c.index_date
LEFT JOIN medicines m   ON m.patient_id = c.patient_id AND m.index_date = c.index_date;
