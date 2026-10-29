# helper functions for the survival analysis (step 5)
# only pandas and numpy, no database, so the tests can run anywhere

import numpy as np
import pandas as pd

# time is cut into quarters (intervals) of a year
INTERVAL_DAYS = 365.25 / 4
N_INTERVALS = 12                       # follow each patient for at most 12 quarters = 3 years
MAX_DAYS = 1096                        # 3 years in days (12 * 91.3 = 1095.75, rounded up)


def add_time_and_event(df, close_date, max_days=MAX_DAYS):
    # adds 3 columns to the cohort rows:
    #   time_days     days from the index date until the follow-up ends
    #   event         1 if the follow-up ended with the kidney disease diagnosis, otherwise 0 ("censored")
    #   possible_days days we could have followed this patient at most (needed to pick rows for the evaluation)
    #
    # the follow-up of a patient stops at the first of:
    #   - the diagnosis                      -> event = 1
    #   - death                              -> event = 0 (we don't know if he would have been diagnosed)
    #   - close_date                         -> event = 0 (we only look at data before this date)
    #   - max_days after the index date      -> event = 0
    # a diagnosis ON the close date is not counted, the close date itself is outside the window
    #
    # df needs the columns index_date, first_dx_date and death_date (dates, empty if it never happened)
    df = df.copy()

    # last day of follow-up for each row = the earlier of close_date and "index date + max_days"
    cap = df["index_date"] + pd.to_timedelta(max_days, unit="D")
    close = pd.Series(pd.Timestamp(close_date), index=df.index).clip(upper=cap)

    # first thing that happens (min skips empty dates)
    stop = pd.concat([df["first_dx_date"], df["death_date"], close], axis=1).min(axis=1)

    # the diagnosis counts only when it is the first thing that happens and before the close
    df["event"] = (df["first_dx_date"].notna()
                   & (df["first_dx_date"] == stop)
                   & (df["first_dx_date"] < close)).astype(int)
    df["time_days"] = (stop - df["index_date"]).dt.days
    df["possible_days"] = (close - df["index_date"]).dt.days
    return df


def to_person_period(df, n_intervals=N_INTERVALS):
    # one row per patient per quarter that the patient was followed ("person-period" table)
    # outcome = 1 in the quarter of the diagnosis, 0 in all the quarters before
    #
    # a quarter is used when the patient was followed through the whole quarter,
    # or when the diagnosis happened in it. a patient who is censored half way through
    # a quarter does not count for that quarter.
    #
    # example: diagnosed on day 100 -> quarter 0 (outcome 0), quarter 1 (outcome 1)
    #          censored on day 100  -> quarter 0 (outcome 0)
    # df needs the columns time_days and event (from add_time_and_event)
    time_days = df["time_days"].to_numpy()
    event = df["event"].to_numpy() == 1

    # quarter in which the follow-up ended (0 = first quarter)
    last = np.floor(time_days / INTERVAL_DAYS).astype(int)
    last = np.minimum(last, n_intervals)
    n_rows = last + event      # event: the quarter of the diagnosis counts as well (True adds 1)

    # repeat every row n_rows times and number the repeats 0, 1, 2, ...
    position = np.repeat(np.arange(len(df)), n_rows)
    first_of_group = np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
    interval = np.arange(n_rows.sum()) - first_of_group

    long = df.iloc[position].reset_index(drop=True)
    long["interval"] = interval
    long["outcome"] = (event[position] & (interval == last[position])).astype(int)
    return long


def hazards_to_survival(hazards):
    # hazards: array with 1 row per patient and 1 column per quarter =
    #          chance of the diagnosis in that quarter, given no diagnosis before
    # survival after k quarters = (1 - h0) * (1 - h1) * ... * (1 - h(k-1))
    # = chance of still having no diagnosis after k quarters
    return np.cumprod(1 - hazards, axis=1)


def kaplan_meier(time, event):
    # Kaplan-Meier estimate of the chance of "no diagnosis yet" over time
    # returns a table with the survival after each diagnosis time and its standard error (Greenwood)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)

    order = np.argsort(time)
    time, event = time[order], event[order]

    # the times at which at least 1 diagnosis happened, and how many
    event_times, n_events = np.unique(time[event == 1], return_counts=True)

    # patients still followed just before each of those times (time >= that time)
    n_at_risk = len(time) - np.searchsorted(time, event_times, side="left")

    survival = np.cumprod(1 - n_events / n_at_risk)

    # Greenwood: variance = S^2 * sum( d / (n * (n - d)) ). the last step can have n = d, skip the error there
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.where(n_at_risk > n_events, n_events / (n_at_risk * (n_at_risk - n_events)), 0.0)
    se = survival * np.sqrt(np.cumsum(steps))

    return pd.DataFrame({"time": np.r_[0.0, event_times],
                         "survival": np.r_[1.0, survival],
                         "se": np.r_[0.0, se]})


def km_at(km, t):
    # survival and standard error at time t = the last row with time <= t
    row = np.searchsorted(km["time"].to_numpy(), t, side="right") - 1
    return km["survival"].iloc[row], km["se"].iloc[row]


def concordance_index(time, event, risk, tau=None):
    # Harrell's C-index: take 2 patients, one gets the diagnosis first, the other one is still
    # without diagnosis at that time. how often did the first patient have the higher risk score?
    # 0.5 = random, 1.0 = perfect
    # a pair is only used when we know who was first, so the earlier patient must have had the diagnosis
    # (pairs with the same time are skipped)
    # tau: only diagnoses up to this time count, later ones are treated as "no diagnosis"
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk, dtype=float)

    if tau is not None:
        event = event & (time <= tau)

    n_pairs = 0
    n_right = 0.0
    for i in np.flatnonzero(event):
        still_free = time > time[i]                  # no diagnosis yet when patient i got his
        n_pairs += still_free.sum()
        n_right += (risk[i] > risk[still_free]).sum()
        n_right += 0.5 * (risk[i] == risk[still_free]).sum()   # equal scores count half
    return n_right / n_pairs


def recall_at_top(y_true, scores, fraction):
    # same as in step 3: sort by score (highest first), take the top fraction,
    # return what share of all positives is in there
    ranked = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(scores)})
    ranked = ranked.sort_values("score", ascending=False)
    top = ranked.head(int(len(ranked) * fraction))
    return top["y"].sum() / ranked["y"].sum()
