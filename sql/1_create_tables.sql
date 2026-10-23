-- raw synthea tables, 1 per csv file
-- columns must be in the same order as in the csv (COPY loads by position)
-- renamed some columns to make the later queries easier to read (START -> start_date, PATIENT -> patient_id, ...)

DROP TABLE IF EXISTS patients, encounters, conditions, observations, medications CASCADE;

-- 1 row per person (most columns not used, but they're in the csv so they have to be here)
CREATE TABLE patients (
    patient_id          TEXT PRIMARY KEY,
    birth_date          DATE,
    death_date          DATE,          -- empty if alive
    ssn                 TEXT,
    drivers             TEXT,
    passport            TEXT,
    prefix              TEXT,
    first_name          TEXT,
    middle_name         TEXT,
    last_name           TEXT,
    suffix              TEXT,
    maiden              TEXT,
    marital             TEXT,
    race                TEXT,
    ethnicity           TEXT,
    gender              TEXT,          -- M / F
    birthplace          TEXT,
    address             TEXT,
    city                TEXT,
    state               TEXT,
    county              TEXT,
    fips                TEXT,
    zip                 TEXT,
    lat                 DOUBLE PRECISION,
    lon                 DOUBLE PRECISION,
    healthcare_expenses DOUBLE PRECISION,
    healthcare_coverage DOUBLE PRECISION,
    income              DOUBLE PRECISION
);

-- 1 row per visit
CREATE TABLE encounters (
    encounter_id        TEXT PRIMARY KEY,
    start_time          TIMESTAMPTZ,
    stop_time           TIMESTAMPTZ,
    patient_id          TEXT,
    organization_id     TEXT,
    provider_id         TEXT,
    payer_id            TEXT,
    encounter_class     TEXT,          -- wellness, ambulatory, emergency, inpatient, ...
    code                TEXT,
    description         TEXT,
    base_encounter_cost DOUBLE PRECISION,
    total_claim_cost    DOUBLE PRECISION,
    payer_coverage      DOUBLE PRECISION,
    reason_code         TEXT,
    reason_description  TEXT
);

-- 1 row per diagnosis
CREATE TABLE conditions (
    start_date   DATE,
    stop_date    DATE,                 -- empty for long-term conditions
    patient_id   TEXT,
    encounter_id TEXT,
    system       TEXT,                 -- SNOMED-CT
    code         TEXT,
    description  TEXT                  -- I filter on this, e.g. 'Essential hypertension%'
);

-- 1 row per lab result / vital sign
CREATE TABLE observations (
    obs_time     TIMESTAMPTZ,
    patient_id   TEXT,
    encounter_id TEXT,
    category     TEXT,                 -- laboratory, vital-signs, ...
    code         TEXT,                 -- LOINC code, e.g. 33914-3 = eGFR
    description  TEXT,
    value        TEXT,                 -- text because some values are words, cast to number later
    units        TEXT,
    type         TEXT                  -- 'numeric' if value is a number
);

-- 1 row per prescription
CREATE TABLE medications (
    start_time         TIMESTAMPTZ,
    stop_time          TIMESTAMPTZ,    -- empty if still taking it
    patient_id         TEXT,
    payer_id           TEXT,
    encounter_id       TEXT,
    code               TEXT,
    description        TEXT,           -- e.g. 'lisinopril 10 MG Oral Tablet'
    base_cost          DOUBLE PRECISION,
    payer_coverage     DOUBLE PRECISION,
    dispenses          DOUBLE PRECISION,
    total_cost         DOUBLE PRECISION,
    reason_code        TEXT,
    reason_description TEXT
);
