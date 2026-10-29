# step 5: survival analysis = WHEN will the kidney disease be diagnosed?
#
# step 3 asked: "will this patient be diagnosed in the next 12 months?" (yes or no).
# here every snapshot row is followed over time, until the diagnosis happens.
#
# - event: the kidney disease diagnosis
# - censored: we stop following a patient before any diagnosis. all we know is that the patient
#   was free of the disease until then. reasons: death, the end of the data, or the 3-year limit
# - the training rows are only followed until 2024-01-01, the start of the test years,
#   so the training data never knows what happens in the test period (same idea as in step 3)
#
# model: discrete-time survival model.
#   time is cut into 12 quarters and a patient gets 1 row per quarter that he was followed
#   (outcome 0 = no diagnosis in that quarter, 1 = diagnosis in that quarter).
#   an ordinary classifier on that table predicts the chance of a diagnosis in each quarter (the "hazard").
#   (1 - hazard) multiplied over the quarters = chance of still having no diagnosis = the survival curve.
#   no class_weight here (step 3 used it), because the hazards have to stay real probabilities.
#
# results: reports/survival_results.csv, survival_calibration.csv and 3 pictures, runs in MLflow

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import settings
import survival_utils as su

MONTH_DAYS = 365.25 / 12
HORIZON_12 = 4 * su.INTERVAL_DAYS      # 12 months = 4 quarters, in days
HORIZON_24 = 8 * su.INTERVAL_DAYS      # 24 months = 8 quarters

# not model inputs: ids, the step 3 label, the 2 date columns for the tests, and the survival columns
NOT_FEATURES = ["patient_id", "index_date", "label", "last_visit_date", "last_lab_date",
                "first_dx_date", "death_date", "time_days", "event", "possible_days"]

# features table + first kidney diagnosis + date of death
QUERY = """
    SELECT f.*, k.first_dx_date, p.death_date
    FROM features f
    JOIN patients p ON p.patient_id = f.patient_id
    LEFT JOIN first_kidney_dx k ON k.patient_id = f.patient_id
    ORDER BY f.patient_id, f.index_date
"""


# ---------- model input / output ----------

def make_X(patient_features, interval, as_dummies):
    # patient_features: 1 row for every row of the person-period table, interval: its quarter (0-11)
    X = patient_features.reset_index(drop=True)
    if as_dummies:
        # logistic regression: 1 yes/no column per quarter, so every quarter gets its own baseline chance
        quarter = pd.Series(pd.Categorical(interval, categories=range(su.N_INTERVALS)))
        X = pd.concat([X, pd.get_dummies(quarter, prefix="q").astype(int)], axis=1)
    else:
        # gradient boosting can work with the quarter number as it is
        X["interval"] = interval
    return X


def predict_survival(model, rows, feature_columns, as_dummies):
    # chance of "no diagnosis yet" after 1, 2, ... 12 quarters, for every row
    n = len(rows)
    # every patient 12 times, once for every quarter
    repeated = pd.DataFrame(np.repeat(rows[feature_columns].astype(float).to_numpy(), su.N_INTERVALS, axis=0),
                            columns=feature_columns)
    interval = np.tile(np.arange(su.N_INTERVALS), n)
    X = make_X(repeated, interval, as_dummies)
    hazards = model.predict_proba(X)[:, 1].reshape(n, su.N_INTERVALS)
    return su.hazards_to_survival(hazards)


# ---------- evaluation ----------

def evaluate(test, survival):
    risk12 = 1 - survival[:, 3]       # after 4 quarters
    risk24 = 1 - survival[:, 7]       # after 8 quarters
    m = {}

    # 12 months: all test rows and the label from the SQL = the same question and the same
    # cases as in step 3, so the numbers can be compared. a death counts as "not diagnosed".
    y12 = test["label"].to_numpy()
    m["pr_auc_12m"] = average_precision_score(y12, risk12)
    m["roc_auc_12m"] = roc_auc_score(y12, risk12)
    m["recall_top_5pct_12m"] = su.recall_at_top(y12, risk12, 0.05)
    m["recall_top_10pct_12m"] = su.recall_at_top(y12, risk12, 0.10)
    m["n_cases_12m"] = int(y12.sum())

    # 24 months: only rows that can be followed for 24 months. the 2025 snapshot cannot,
    # the data ends on 2026-10-01 (21 months later). using it would mean cases but no controls.
    ok = (test["possible_days"] >= HORIZON_24).to_numpy()
    y24 = ((test["event"] == 1) & (test["time_days"] < HORIZON_24)).to_numpy()
    m["pr_auc_24m"] = average_precision_score(y24[ok], risk24[ok])
    m["roc_auc_24m"] = roc_auc_score(y24[ok], risk24[ok])
    m["recall_top_10pct_24m"] = su.recall_at_top(y24[ok], risk24[ok], 0.10)
    m["n_cases_24m"] = int(y24[ok].sum())

    # C-index: uses the time of the diagnosis too, and handles censored patients (all test rows)
    m["c_index_24m"] = su.concordance_index(test["time_days"], test["event"], risk24, tau=HORIZON_24)
    return m, risk12, risk24


def calibration_table(test, risk12):
    # predicted 12-month risk vs what really happened, for groups of patients ordered by risk
    # observed risk = 1 - Kaplan-Meier survival at 12 months (deaths are handled as censored)
    rank = pd.Series(risk12).rank(pct=True).to_numpy()      # 0 to 1, 1 = highest risk
    names = ["bottom 50%", "50-80%", "80-90%", "90-95%", "top 5%"]
    group = pd.cut(rank, bins=[0, 0.5, 0.8, 0.9, 0.95, 1.0], labels=names, include_lowest=True)

    rows = []
    for name in names:
        in_group = np.asarray(group == name)
        part = test[in_group]
        km = su.kaplan_meier(part["time_days"], part["event"])
        survival, se = su.km_at(km, HORIZON_12)
        rows.append({"group": name, "patients": len(part),
                     "predicted_pct": 100 * risk12[in_group].mean(),
                     "observed_pct": 100 * (1 - survival),
                     "ci_95_pct": 100 * 1.96 * se})
    return pd.DataFrame(rows)


# ---------- pictures ----------

def draw_curve(ax, time_days, event, x_end, label):
    # share of patients diagnosed so far (= 1 - Kaplan-Meier survival) in %, with a 95% band
    km = su.kaplan_meier(time_days, event)
    low = np.clip(km["survival"] - 1.96 * km["se"], 0, 1)
    high = np.clip(km["survival"] + 1.96 * km["se"], 0, 1)

    # one extra point at the end, so the line runs to the edge of the plot
    x = np.r_[km["time"].to_numpy() / MONTH_DAYS, x_end]
    line = ax.step(x, 100 * (1 - np.r_[km["survival"], km["survival"].iloc[-1]]), where="post", label=label)[0]
    ax.fill_between(x, 100 * (1 - np.r_[high, high.iloc[-1]]), 100 * (1 - np.r_[low, low.iloc[-1]]),
                    step="post", alpha=0.15, color=line.get_color())


def plot_hypertension(train, path):
    # the classic first look: Kaplan-Meier curves for 2 groups
    # only the first snapshot (1 Jan 2019), so every patient is in there once
    first = train[train["index_date"] == train["index_date"].min()]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for flag, name in [(1, "high blood pressure"), (0, "no high blood pressure")]:
        part = first[first["has_hypertension"] == flag]
        draw_curve(ax, part["time_days"], part["event"], 36,
                   "{} (n={:,}, {} diagnosed)".format(name, len(part), part["event"].sum()))
    ax.set_xlim(0, 36)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Months since 1 January 2019")
    ax.set_ylabel("% diagnosed with kidney disease so far")
    ax.set_title("Time to diagnosis, by blood pressure")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_risk_groups(test, risk24, path):
    # Kaplan-Meier curves for patients split by the predicted risk
    # only the first test snapshot (1 Jan 2024): every patient once, and 24 months of follow-up for all
    first = (test["index_date"] == test["index_date"].min()).to_numpy()
    part = test[first]
    rank = pd.Series(risk24[first]).rank(pct=True).to_numpy()

    groups = [("top 10% risk", rank > 0.9), ("next 40%", (rank > 0.5) & (rank <= 0.9)), ("bottom 50%", rank <= 0.5)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, in_group in groups:
        sub = part[in_group]
        draw_curve(ax, sub["time_days"], sub["event"], 24,
                   "{} (n={:,}, {} diagnosed)".format(name, len(sub), sub["event"].sum()))
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Months since 1 January 2024")
    ax.set_ylabel("% diagnosed with kidney disease so far")
    ax.set_title("Time to diagnosis by predicted risk (test year 2024)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(table, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # the interval can't go below 0%, so the lower arm is cut there
    below = np.minimum(table["ci_95_pct"], table["observed_pct"])
    ax.errorbar(table["predicted_pct"], table["observed_pct"], yerr=[below, table["ci_95_pct"]],
                fmt="o", capsize=3, label="risk groups, with 95% interval")
    top = max(table["predicted_pct"].max(), table["observed_pct"].max()) * 1.3
    ax.plot([0, top], [0, top], linestyle="--", color="grey", label="perfect")
    for _, row in table.iterrows():
        ax.annotate(row["group"], (row["predicted_pct"], row["observed_pct"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted risk of a diagnosis within 12 months (%)")
    ax.set_ylabel("Observed (%)")
    ax.set_title("Calibration, test years 2024-2025")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- main ----------

def main():
    reports = settings.REPORTS_DIR
    os.makedirs(reports, exist_ok=True)

    df = settings.query_to_df(QUERY)
    for col in ["index_date", "first_dx_date", "death_date"]:
        df[col] = pd.to_datetime(df[col])
    feature_columns = [col for col in df.columns if col not in NOT_FEATURES]

    # --- time and event. train: follow-up closes at the start of the test years. test: at the end of the data ---
    train = su.add_time_and_event(df[df["index_date"] < settings.TEST_FROM], close_date=settings.TEST_FROM)
    test = su.add_time_and_event(df[df["index_date"] >= settings.TEST_FROM], close_date=settings.DATA_END)

    for name, part in [("Train", train), ("Test", test)]:
        end = part["index_date"] + pd.to_timedelta(part["time_days"], unit="D")
        died = (part["event"] == 0) & part["death_date"].notna() & (part["death_date"] <= end)
        print("{}: {:,} rows, {} diagnosed, {} died before a diagnosis, {:,} followed to the end of the window"
              .format(name, len(part), part["event"].sum(), died.sum(), len(part) - part["event"].sum() - died.sum()))

    plot_hypertension(train, os.path.join(reports, "km_hypertension.png"))

    # --- person-period table for training ---
    long = su.to_person_period(train)
    print("Person-period table: {:,} rows, {} with a diagnosis".format(len(long), long["outcome"].sum()))

    # logistic regression needs no missing values and scaled inputs (same as step 3):
    # missing labs -> median + a 0/1 "was missing" column, StandardScaler for the size of the numbers
    models = {
        "discrete_time_logistic": (make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                 StandardScaler(), LogisticRegression(max_iter=2000)), True),
        # min_samples_leaf=200 because the table has many more rows than in step 3. not tuned.
        "discrete_time_gradient_boosting": (HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, min_samples_leaf=200, random_state=42), False),
    }

    results = {}
    risks = {}
    for name, (model, as_dummies) in models.items():
        print("\nTraining", name)
        X = make_X(long[feature_columns], long["interval"].to_numpy(), as_dummies)
        model.fit(X, long["outcome"])

        survival = predict_survival(model, test, feature_columns, as_dummies)
        results[name], risk12, risk24 = evaluate(test, survival)
        risks[name] = (risk12, risk24)
        for key, value in results[name].items():
            print("  {:22s} {:.3f}".format(key, value))

    table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    table.to_csv(os.path.join(reports, "survival_results.csv"), index=False)

    # best model = highest PR-AUC at 12 months (same rule as step 3)
    best = table.sort_values("pr_auc_12m", ascending=False).iloc[0]["model"]
    print("\nBest model:", best)
    risk12, risk24 = risks[best]

    calibration = calibration_table(test, risk12)
    calibration.to_csv(os.path.join(reports, "survival_calibration.csv"), index=False)
    print("\nCalibration at 12 months ({}):".format(best))
    print(calibration.round(2).to_string(index=False))

    plot_risk_groups(test, risk24, os.path.join(reports, "km_risk_groups.png"))
    plot_calibration(calibration, os.path.join(reports, "calibration_12m.png"))

    # --- MLflow: 1 run per model, the pictures go to the run of the best model ---
    mlflow.set_tracking_uri(settings.MLFLOW_URI)
    mlflow.set_experiment("ckd_survival")
    pictures = ["km_hypertension.png", "km_risk_groups.png", "calibration_12m.png"]
    for name in models:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "n_features": len(feature_columns),
                               "quarters": su.N_INTERVALS, "n_train_rows": len(train),
                               "n_person_period_rows": len(long), "n_test_rows": len(test)})
            mlflow.log_metrics({key: float(value) for key, value in results[name].items()})
            if name == best:
                for picture in pictures:
                    mlflow.log_artifact(os.path.join(reports, picture))


if __name__ == "__main__":
    main()
