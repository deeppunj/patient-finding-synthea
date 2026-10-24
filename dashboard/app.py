# streamlit dashboard, reads the csv files from reports/ (run run_all.py first)
# start with: streamlit run dashboard/app.py   -> opens http://localhost:8501
# streamlit reruns the whole script every time the slider moves
#
# charts are drawn with altair (comes with streamlit) instead of showing the matplotlib pngs,
# so they follow the light/dark theme, resize with the window and show values on hover
#
# it's in its own folder with its own requirements.txt, so streamlit cloud only installs
# streamlit + pandas + altair (and not the whole conda environment.yml of the project)

import os

import altair as alt
import pandas as pd
import streamlit as st

# reports/ is one folder up from this file
reports = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")

# readable names for the feature chart
NICE_NAMES = {
    "age": "Age",
    "male": "Male",
    "n_visits": "Number of visits",
    "n_emergency_visits": "Emergency visits",
    "has_hypertension": "High blood pressure",
    "has_diabetes": "Type 2 diabetes",
    "has_prediabetes": "Prediabetes",
    "has_obesity": "Obesity",
    "has_heart_disease": "Heart disease",
    "last_egfr": "eGFR (latest)",
    "min_egfr": "eGFR (lowest)",
    "last_creatinine": "Creatinine",
    "last_urea": "Urea",
    "last_albumin_creatinine_ratio": "Urine albumin/creatinine",
    "last_glucose": "Glucose",
    "last_hba1c": "HbA1c",
    "last_systolic_bp": "Systolic blood pressure",
    "last_bmi": "BMI",
    "n_medications": "Number of medicines",
    "on_blood_pressure_medicine": "On blood pressure medicine",
    "on_diabetes_medicine": "On diabetes medicine",
    "on_nsaid_painkiller": "On NSAID painkiller",
}

st.set_page_config(page_title="Kidney disease patient finding", layout="wide")
st.title("Finding adults with undiagnosed kidney disease")
st.caption("Synthetic data (Synthea). Model trained on 2019-2023, tested on 2024-2025.")

# --- load the results from step 3 and 4 ---
scores = pd.read_csv(os.path.join(reports, "test_scores.csv"))        # sorted, highest risk first
models = pd.read_csv(os.path.join(reports, "model_comparison.csv"))
importance = pd.read_csv(os.path.join(reports, "feature_importance.csv"))
delays = pd.read_csv(os.path.join(reports, "diagnostic_delay.csv"))

n_patients = len(scores)
n_cases = int(scores["label"].sum())
best = models.iloc[0]

# --- slider: how many patients can be reviewed ---
pct = st.slider("Review the top ... % of patients (highest risk first)", min_value=1, max_value=30, value=10)

n_review = int(n_patients * pct / 100)
top = scores.head(n_review)
found = int(top["label"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients to review", f"{n_review:,}")
col2.metric("Future kidney patients found", f"{found} of {n_cases}")
col3.metric("Share found", f"{100 * found / n_cases:.0f}%")
col4.metric("Random review would find", f"about {pct}%")

st.divider()

# ---------------------------------------------------------------
# row 1: cases found curve + diagnostic delay
# ---------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Future patients found per review effort")

    # % of cases found after reviewing 0%, 1%, ..., 100% of the patients
    # (101 points is enough for a smooth line, no need to plot all 13k patients)
    curve = pd.DataFrame({"reviewed": range(0, 101)})
    curve["found"] = [100 * scores.head(int(n_patients * p / 100))["label"].sum() / n_cases
                      for p in curve["reviewed"]]
    curve["random"] = curve["reviewed"]

    line = alt.Chart(curve).mark_line(strokeWidth=3).encode(
        x=alt.X("reviewed:Q", title="% of patients reviewed"),
        y=alt.Y("found:Q", title="% of future kidney patients found"),
        tooltip=[alt.Tooltip("reviewed:Q", title="% reviewed"),
                 alt.Tooltip("found:Q", title="% found", format=".0f")],
    )
    diagonal = alt.Chart(curve).mark_line(strokeDash=[5, 5], color="grey").encode(
        x="reviewed:Q", y="random:Q",
    )
    # vertical line where the slider is
    marker = alt.Chart(pd.DataFrame({"reviewed": [pct]})).mark_rule(color="red").encode(x="reviewed:Q")

    st.altair_chart(line + diagonal + marker)
    st.caption("Grey dashed line = reviewing patients at random. Red line = the slider.")

with right:
    st.subheader("Diagnostic delay")

    # only patients who had an abnormal kidney test before the diagnosis
    delay_months = delays["delay_days"].dropna() / 30.44
    delay_df = pd.DataFrame({"months": delay_months})

    hist = alt.Chart(delay_df).mark_bar().encode(
        x=alt.X("months:Q", bin=alt.Bin(maxbins=20), title="Months from first abnormal kidney test to diagnosis"),
        y=alt.Y("count():Q", title="Patients"),
        tooltip=[alt.Tooltip("count():Q", title="Patients")],
    )
    st.altair_chart(hist)
    st.caption(f"{len(delay_months)} of {len(delays)} newly diagnosed adults had an abnormal test first. "
               f"Median delay: {delay_months.median():.0f} months.")

st.divider()

# ---------------------------------------------------------------
# row 2: feature importance + model comparison
# ---------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("What drives the risk score")

    # importance as % of the best model's pr-auc, easier to read than 0.0019 etc.
    imp = importance.head(10).copy()
    imp["name"] = imp["feature"].map(NICE_NAMES).fillna(imp["feature"])
    imp["percent"] = 100 * imp["importance"] / best["pr_auc"]

    bars = alt.Chart(imp).mark_bar().encode(
        x=alt.X("percent:Q", title="% of model performance lost when the feature is shuffled"),
        y=alt.Y("name:N", sort="-x", title=None,   # sort="-x": biggest bar on top
                axis=alt.Axis(labelLimit=300)),     # default cuts long names off
        tooltip=[alt.Tooltip("name:N", title="Feature"),
                 alt.Tooltip("percent:Q", title="% lost", format=".1f")],
    )
    st.altair_chart(bars)

with right:
    st.subheader("Model comparison (test years)")
    table = models.rename(columns={
        "model": "Model",
        "pr_auc": "PR-AUC",
        "pr_auc_random_baseline": "PR-AUC random guess",
        "roc_auc": "ROC-AUC",
        "recall_top_5pct": "Found in top 5%",
    })
    table["Model"] = table["Model"].str.replace("_", " ").str.capitalize()
    table["Found in top 5%"] = (100 * table["Found in top 5%"]).round(0).astype(int).astype(str) + "%"
    # 3 decimals everywhere, otherwise 0.030 shows as 0.03 next to 0.028
    st.dataframe(table, hide_index=True, column_config={
        "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "PR-AUC random guess": st.column_config.NumberColumn(format="%.3f"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("PR-AUC is the main metric because only 0.5% of patients are positive.")

st.divider()

# ---------------------------------------------------------------
# row 3: the review list
# ---------------------------------------------------------------
st.subheader(f"Patients to review ({n_review:,}), highest risk first")

# show the rank instead of the raw score.
# class_weight="balanced" pushes all probabilities up, so the top patients all get ~1.00.
# the order is still right (that's what pr-auc measures), but the score is not a real probability
# (would need calibration for that). rank = position in the sorted list as % of all test patients
top = top.copy()
top["risk_rank"] = ["top {:.1f}%".format(100 * (i + 1) / n_patients) for i in range(len(top))]
top = top.drop(columns=["risk_score"])

review = top.rename(columns={
    "patient_id": "Patient",
    "index_date": "Snapshot",
    "label": "Diagnosed within 12 months",
    "age": "Age",
    "has_hypertension": "High blood pressure",
    "last_egfr": "eGFR",
    "last_creatinine": "Creatinine",
    "last_systolic_bp": "Systolic BP",
    "risk_rank": "Risk rank",
})
review["High blood pressure"] = review["High blood pressure"].map({1: "yes", 0: "no"})
review["Diagnosed within 12 months"] = review["Diagnosed within 12 months"].map({1: "yes", 0: "no"})

st.dataframe(
    review,
    hide_index=True,
    column_config={
        "eGFR": st.column_config.NumberColumn(format="%.0f"),
        "Creatinine": st.column_config.NumberColumn(format="%.1f"),
        "Systolic BP": st.column_config.NumberColumn(format="%.0f"),
    },
)
st.caption("Risk rank = position in the list of all test patients, sorted by risk score. "
           "The raw scores are only used for ranking: class weighting makes them too high to read as probabilities. "
           "'Diagnosed within 12 months' is only known here because this is test data.")
