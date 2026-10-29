# Finding adults with undiagnosed kidney disease: summary

*Synthetic data (Synthea). The method is the point, not the exact numbers.*

**Question:** can routine records flag adults who will be diagnosed with chronic kidney disease
in the next 12 months, so they can be tested and treated earlier?

## Patient journey
- 265 adults were newly diagnosed in 2019-2025. 86% already had high blood pressure.
- 42% (110 patients) had an abnormal kidney test (eGFR below 60 or raised creatinine)
  a median of 55 months before their diagnosis.

## How well the model finds them
Only 75 of 13,769 test records (0.5%) were diagnosed in the next 12 months.

| If we review the... | Future kidney patients found | Random review would find |
|---|---|---|
| Top 5% (688 patients) | 20 of 75 (27%) | about 5% |
| Top 10% (1376 patients) | 37 of 75 (49%) | about 10% |
| Top 20% (2753 patients) | 65 of 75 (87%) | about 20% |

Best model: logistic_regression, PR-AUC 0.030 against 0.005 for a random guess.

Strongest signals: has_hypertension, last_albumin_creatinine_ratio, n_medications, last_systolic_bp, min_egfr.

## Suggestions
1. Start with patients who already have an abnormal kidney test but no diagnosis: a simple rule finds them.
2. Use the risk score to prioritise patients with high blood pressure for kidney testing.
3. Check the approach on real health records first: synthetic data is cleaner than real data.
