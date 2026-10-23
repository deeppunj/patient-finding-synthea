# step 4: insights + summary
#
# 1. diagnostic delay: time between first abnormal kidney test and the diagnosis
# 2. cases found vs % of patients reviewed (from the step 3 scores)
# 3. reports/summary.md for non-technical readers

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import settings


def main():
    reports = settings.REPORTS_DIR

    # --- 1. diagnostic delay ---
    settings.run_sql_file("4_diagnostic_delay.sql")
    journey = settings.query_to_df("SELECT * FROM diagnostic_delay")
    journey.to_csv(os.path.join(reports, "diagnostic_delay.csv"), index=False)  # for the dashboard

    n_new = len(journey)

    # delay_days is empty when there was no abnormal test before the diagnosis -> dropna
    # 30.44 = average days per month
    delay_months = journey["delay_days"].dropna() / 30.44

    n_warned = len(delay_months)
    pct_warned = 100 * n_warned / n_new
    median_delay = delay_months.median()
    pct_hypertension = 100 * journey["had_hypertension"].mean()  # mean of 0/1 = share

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delay_months, bins=20, color="steelblue")
    ax.axvline(median_delay, color="navy", linestyle="--", label="median {:.0f} months".format(median_delay))
    ax.set_xlabel("Months from first abnormal kidney test to diagnosis")
    ax.set_ylabel("Patients")
    ax.set_title("Diagnostic delay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(reports, "diagnostic_delay.png"), dpi=120)
    plt.close(fig)

    # --- 2. cases found vs review effort ---
    # test_scores.csv is already sorted, highest risk first
    scores = pd.read_csv(os.path.join(reports, "test_scores.csv"))
    n_cases = int(scores["label"].sum())

    # going down the sorted list: % reviewed so far vs % of cases found so far
    reviewed_pct = 100 * np.arange(1, len(scores) + 1) / len(scores)
    found_pct = 100 * scores["label"].cumsum() / n_cases

    # diagonal = what random review would give
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(reviewed_pct, found_pct, label="review by risk score")
    ax.plot([0, 100], [0, 100], linestyle="--", color="grey", label="review at random")
    ax.set_xlabel("% of patients reviewed (highest risk first)")
    ax.set_ylabel("% of future kidney patients found")
    ax.set_title("Future patients found per review effort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(reports, "cases_found_curve.png"), dpi=120)
    plt.close(fig)

    # same numbers for top 5/10/20% as table rows for the summary
    table_rows = []
    for pct in [5, 10, 20]:
        n_review = int(len(scores) * pct / 100)
        found = int(scores.head(n_review)["label"].sum())
        table_rows.append("| Top {}% ({} patients) | {} of {} ({:.0f}%) | about {}% |".format(
            pct, n_review, found, n_cases, 100 * found / n_cases, pct))

    # --- 3. summary ---
    models = pd.read_csv(os.path.join(reports, "model_comparison.csv"))
    best = models.iloc[0]
    top_features = pd.read_csv(os.path.join(reports, "feature_importance.csv"))["feature"].head(5)

    # chr(10) = newline, to put the table rows under each other
    text = f"""# Finding adults with undiagnosed kidney disease: summary

*Synthetic data (Synthea). The method is the point, not the exact numbers.*

**Question:** can routine records flag adults who will be diagnosed with chronic kidney disease
in the next 12 months, so they can be tested and treated earlier?

## Patient journey
- {n_new} adults were newly diagnosed in 2019-2025. {pct_hypertension:.0f}% already had high blood pressure.
- {pct_warned:.0f}% ({n_warned} patients) had an abnormal kidney test (eGFR below 60 or raised creatinine)
  a median of {median_delay:.0f} months before their diagnosis.

## How well the model finds them
Only {n_cases} of {len(scores):,} test records ({100 * n_cases / len(scores):.1f}%) were diagnosed in the next 12 months.

| If we review the... | Future kidney patients found | Random review would find |
|---|---|---|
{chr(10).join(table_rows)}

Best model: {best['model']}, PR-AUC {best['pr_auc']:.3f} against {best['pr_auc_random_baseline']:.3f} for a random guess.

Strongest signals: {', '.join(top_features)}.

## Suggestions
1. Start with patients who already have an abnormal kidney test but no diagnosis: a simple rule finds them.
2. Use the risk score to prioritise patients with high blood pressure for kidney testing.
3. Check the approach on real health records first: synthetic data is cleaner than real data.
"""
    with open(os.path.join(reports, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
