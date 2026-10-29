# checks for the survival part (step 5), run with: pytest
# these tests don't need the database, they use a few made-up patients where the answer is easy to see

import numpy as np
import pandas as pd

import survival_utils as su


def make_rows(*rows):
    # each row: (index_date, first_dx_date, death_date), None = never happened
    df = pd.DataFrame(rows, columns=["index_date", "first_dx_date", "death_date"])
    return df.apply(pd.to_datetime)


# ---------- time and event ----------

def test_diagnosis_is_an_event():
    df = make_rows(("2021-01-01", "2021-04-01", None))
    out = su.add_time_and_event(df, close_date="2024-01-01")
    assert out["event"][0] == 1
    assert out["time_days"][0] == 90          # 31 + 28 + 31 days


def test_death_before_diagnosis_is_censored():
    df = make_rows(("2021-01-01", "2022-06-01", "2021-06-01"))
    out = su.add_time_and_event(df, close_date="2024-01-01")
    assert out["event"][0] == 0
    assert out["time_days"][0] == 151         # up to the death date


def test_patient_without_diagnosis_is_censored_at_the_close_date():
    df = make_rows(("2023-01-01", None, None))
    out = su.add_time_and_event(df, close_date="2024-01-01")
    assert out["event"][0] == 0
    assert out["time_days"][0] == 365


def test_nothing_after_the_close_date_is_used():
    # training rows must not know what happened in the test years
    df = make_rows(("2023-01-01", "2024-03-01", None),     # diagnosed after the close date
                   ("2023-01-01", "2024-01-01", None),     # diagnosed ON the close date
                   ("2023-01-01", "2023-12-31", None))     # diagnosed the day before
    out = su.add_time_and_event(df, close_date="2024-01-01")
    assert list(out["event"]) == [0, 0, 1]
    assert list(out["time_days"]) == [365, 365, 364]


def test_follow_up_never_runs_past_the_close_date_or_the_cap():
    rng = np.random.default_rng(1)
    n = 500
    index = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, n), unit="D")
    dx = index + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D")
    dx = dx.where(rng.random(n) < 0.5)                       # half of them never get it
    death = index + pd.to_timedelta(rng.integers(1, 365 * 8, n), unit="D")
    death = death.where(rng.random(n) < 0.3)
    df = pd.DataFrame({"index_date": index, "first_dx_date": dx, "death_date": death})

    out = su.add_time_and_event(df, close_date="2024-01-01")
    end = out["index_date"] + pd.to_timedelta(out["time_days"], unit="D")
    assert (end <= pd.Timestamp("2024-01-01")).all()
    assert (out["time_days"] <= su.MAX_DAYS).all()
    assert (out["time_days"] >= 0).all()
    # events only when the diagnosis is inside the window
    assert (out.loc[out["event"] == 1, "first_dx_date"] < pd.Timestamp("2024-01-01")).all()


def test_follow_up_is_capped_at_3_years():
    # 2019 snapshot, diagnosed in 2023: more than 3 years later, so not an event
    df = make_rows(("2019-01-01", "2023-01-01", None))
    out = su.add_time_and_event(df, close_date="2024-01-01")
    assert out["event"][0] == 0
    assert out["time_days"][0] == su.MAX_DAYS


# ---------- person-period table ----------

def test_person_period_rows():
    df = pd.DataFrame({
        "time_days": [100, 100, 0, 10, su.MAX_DAYS],
        "event":     [1,   0,   1, 0,  0],
    })
    long = su.to_person_period(df)
    # 100 days = quarter 1 (a quarter is 91.3 days)
    # row 0: diagnosed on day 100  -> quarters 0 and 1, outcome 0 then 1
    # row 1: censored on day 100   -> quarter 0 only
    # row 2: diagnosed on day 0    -> quarter 0, outcome 1
    # row 3: censored on day 10    -> no complete quarter, no rows
    # row 4: censored at the cap   -> 12 quarters, all 0
    assert len(long) == 2 + 1 + 1 + 0 + 12
    first = long[(long["time_days"] == 100) & (long["event"] == 1)]
    assert list(first["interval"]) == [0, 1]
    assert list(first["outcome"]) == [0, 1]
    day_zero = long[long["time_days"] == 0]
    assert list(day_zero["interval"]) == [0]
    assert list(day_zero["outcome"]) == [1]
    capped = long[long["time_days"] == su.MAX_DAYS]
    assert list(capped["interval"]) == list(range(12))
    assert capped["outcome"].sum() == 0


def test_person_period_keeps_the_event_count():
    # every event row gives exactly one outcome = 1, censored rows give none
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"time_days": rng.integers(0, su.MAX_DAYS, 300), "event": rng.integers(0, 2, 300)})
    long = su.to_person_period(df)
    assert long["outcome"].sum() == df["event"].sum()


def test_hazards_to_survival():
    hazards = np.array([[0.1, 0.1, 0.2]])
    survival = su.hazards_to_survival(hazards)
    assert np.allclose(survival, [[0.9, 0.81, 0.648]])


# ---------- Kaplan-Meier ----------

def test_kaplan_meier_by_hand():
    # times 1, 2, 3, 4 with diagnosis at 1 and 3 (2 and 4 are censored)
    # t=1: 4 at risk, 1 event -> 3/4 = 0.75
    # t=3: 2 at risk (times 3 and 4), 1 event -> 0.75 * 1/2 = 0.375
    km = su.kaplan_meier([1, 2, 3, 4], [1, 0, 1, 0])
    assert list(km["time"]) == [0, 1, 3]
    assert np.allclose(km["survival"], [1, 0.75, 0.375])
    assert su.km_at(km, 2.5)[0] == 0.75
    assert su.km_at(km, 0.5)[0] == 1.0
    assert su.km_at(km, 10)[0] == 0.375


def test_kaplan_meier_without_censoring_is_one_minus_share():
    km = su.kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert np.allclose(km["survival"], [1, 0.75, 0.5, 0.25, 0.0])


# ---------- C-index ----------

def brute_force_c(time, event, risk):
    # the same thing with 2 loops over all pairs, slow but obviously correct
    right, pairs = 0.0, 0
    for i in range(len(time)):
        for j in range(len(time)):
            if event[i] and time[i] < time[j]:
                pairs += 1
                right += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return right / pairs


def test_c_index_matches_the_slow_version():
    rng = np.random.default_rng(3)
    time = rng.integers(1, 50, 80)          # integer times, so there are ties
    event = rng.integers(0, 2, 80)
    risk = rng.integers(0, 10, 80)          # ties in the scores too
    assert np.isclose(su.concordance_index(time, event, risk), brute_force_c(time, event, risk))


def test_c_index_perfect_and_reversed():
    time = np.array([1, 2, 3, 4, 5])
    event = np.array([1, 1, 1, 1, 0])
    assert su.concordance_index(time, event, risk=[5, 4, 3, 2, 1]) == 1.0
    assert su.concordance_index(time, event, risk=[1, 2, 3, 4, 5]) == 0.0
    assert su.concordance_index(time, event, risk=[1, 1, 1, 1, 1]) == 0.5


def test_c_index_tau_ignores_late_diagnoses():
    time = np.array([1, 2, 10])
    event = np.array([1, 1, 1])
    risk = np.array([3, 2, 1])
    # with tau = 5 the diagnosis at time 10 is not an event, but pairs with the first 2 patients still count
    assert su.concordance_index(time, event, risk, tau=5) == 1.0
