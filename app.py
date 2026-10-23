# streamlit dashboard, reads the files from reports/ (run run_all.py first)
# start with: streamlit run app.py   -> opens http://localhost:8501
# streamlit reruns the whole script every time the slider moves

import os

import pandas as pd
import streamlit as st

import settings

reports = settings.REPORTS_DIR

st.set_page_config(page_title="Kidney disease patient finding", layout="wide")
st.title("Finding adults with undiagnosed kidney disease")
st.caption("Synthetic data (Synthea). Test years 2024-2025.")

# test_scores.csv is sorted, highest risk first
scores = pd.read_csv(os.path.join(reports, "test_scores.csv"))
models = pd.read_csv(os.path.join(reports, "model_comparison.csv"))
n_cases = int(scores["label"].sum())

# --- slider: how many patients can be reviewed ---
pct = st.slider("Review the top ... % of patients", min_value=1, max_value=30, value=5)

n_review = int(len(scores) * pct / 100)
top = scores.head(n_review)
found = int(top["label"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Patients to review", n_review)
col2.metric("Future kidney patients found", "{} of {}".format(found, n_cases))
col3.metric("Random review would find", "about {}".format(round(n_cases * pct / 100)))

# --- charts from step 3 and 4 ---
left, right = st.columns(2)
left.subheader("Patients found per review effort")
left.image(os.path.join(reports, "cases_found_curve.png"))
right.subheader("Diagnostic delay")
right.image(os.path.join(reports, "diagnostic_delay.png"))

left, right = st.columns(2)
left.subheader("What drives the risk score")
left.image(os.path.join(reports, "feature_importance.png"))
right.subheader("Model comparison")
right.dataframe(models.round(3), hide_index=True)

# --- review list ---
st.subheader("Patients to review, highest risk first")
st.caption("label = 1: diagnosed within 12 months (in real use this would not be known yet)")
st.dataframe(top.round(3), hide_index=True)
