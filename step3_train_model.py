# step 3: train 2 models, evaluate on 2024-2025, log runs to mlflow
#
# models: logistic regression (baseline) and gradient boosting
# metrics: pr-auc (main one, because positives are only ~0.5%), roc-auc,
#          and recall in the top 5% (share of future cases found if you review the top 5%)

import os

import matplotlib
matplotlib.use("Agg")  # save plots to file, no window
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import settings

# not model inputs: ids, the label itself, and the 2 date columns that are only there for the tests
NOT_FEATURES = ["patient_id", "index_date", "label", "last_visit_date", "last_lab_date"]


def recall_at_top(y_true, scores, fraction):
    # sort by score (highest first), take the top fraction,
    # and return what share of all positives is in there
    ranked = pd.DataFrame({"y": y_true, "score": scores}).sort_values("score", ascending=False)
    top = ranked.head(int(len(ranked) * fraction))
    return top["y"].sum() / ranked["y"].sum()


def main():
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)

    # --- load data ---
    # ORDER BY so the row order is always the same
    # (without it gradient boosting gave slightly different results each run)
    df = settings.query_to_df("SELECT * FROM features ORDER BY patient_id, index_date")
    df["index_date"] = pd.to_datetime(df["index_date"])

    feature_columns = [col for col in df.columns if col not in NOT_FEATURES]

    # --- train/test split by time ---
    # train on 2019-2023 snapshots, test on 2024-2025
    # (a random split would mix future and past)
    train = df[df["index_date"] < settings.TEST_FROM]
    test = df[df["index_date"] >= settings.TEST_FROM]

    X_train, y_train = train[feature_columns], train["label"]
    X_test, y_test = test[feature_columns], test["label"]

    print("Train:", len(train), "rows,", y_train.sum(), "positives")
    print("Test: ", len(test), "rows,", y_test.sum(), "positives")

    # --- models ---
    # class_weight="balanced" because of the class imbalance (~0.5% positives)
    models = {
        # logistic regression needs no missing values and scaled inputs:
        # - missing labs -> median, add_indicator adds a 0/1 "was missing" column
        #   (not being tested says something too)
        # - StandardScaler so big numbers (age) don't dominate small ones (creatinine)
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        # gradient boosting handles missing values itself
        # min_samples_leaf=50 to avoid learning from very small groups
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.05, max_iter=300,
            min_samples_leaf=50, random_state=42,
        ),
    }

    mlflow.set_tracking_uri(settings.MLFLOW_URI)
    mlflow.set_experiment("ckd_patient_finding")

    results = []
    scores = {}

    # --- train + evaluate each model, 1 mlflow run per model ---
    for name, model in models.items():
        print("\nTraining", name)
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            # probability of class 1
            scores[name] = model.predict_proba(X_test)[:, 1]

            # random guess pr-auc = share of positives, so compare against that
            metrics = {
                "pr_auc": average_precision_score(y_test, scores[name]),
                "pr_auc_random_baseline": y_test.mean(),
                "roc_auc": roc_auc_score(y_test, scores[name]),
                "recall_top_5pct": recall_at_top(y_test.values, scores[name], 0.05),
            }
            for key, value in metrics.items():
                print("  {:24s} {:.3f}".format(key, value))

            mlflow.log_params({"model": name, "n_features": len(feature_columns),
                               "n_train": len(train), "n_test": len(test)})
            mlflow.log_metrics(metrics)
            #mlflow.sklearn.log_model(model, name="model")
            # newer mlflow saves models with skops and checks the types inside.
            # these 2 are flagged by default but fine here, because we made the model ourselves
            trusted = ["numpy.dtype", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted)

        results.append({"model": name, **metrics})

    # best model = highest pr-auc
    results = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    results.to_csv(os.path.join(settings.REPORTS_DIR, "model_comparison.csv"), index=False)
    best_name = results.iloc[0]["model"]
    best_model = models[best_name]
    print("\nBest model:", best_name)

    # --- precision-recall curves ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in models:
        precision, recall, _ = precision_recall_curve(y_test, scores[name])
        ax.plot(recall, precision, label=name)
    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="random guess")
    ax.set_ylim(0, 0.2)  # first few patients give a spike to 1.0, cut it off so the rest is visible
    ax.set_xlabel("Recall (share of future kidney patients found)")
    ax.set_ylabel("Precision (share of flagged who are real cases)")
    ax.set_title("Precision-recall on the test years")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(settings.REPORTS_DIR, "pr_curves.png"), dpi=120)
    plt.close(fig)

    # --- save test patients with risk score (used in step 4 and the dashboard) ---
    scored = test[["patient_id", "index_date", "label", "age", "has_hypertension",
                   "last_egfr", "last_creatinine", "last_systolic_bp"]].copy()
    scored["risk_score"] = scores[best_name]
    scored = scored.sort_values("risk_score", ascending=False)
    scored.to_csv(os.path.join(settings.REPORTS_DIR, "test_scores.csv"), index=False)

    # --- feature importance ---
    # permutation importance: shuffle 1 column, see how much pr-auc drops
    # n_repeats=5 and take the mean, 1 shuffle is too noisy
    print("Calculating feature importance...")
    result = permutation_importance(best_model, X_test, y_test, scoring="average_precision",
                                    n_repeats=5, random_state=42)
    importance = pd.DataFrame({"feature": feature_columns, "importance": result.importances_mean})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(os.path.join(settings.REPORTS_DIR, "feature_importance.csv"), index=False)

    # top 10, reversed so the biggest bar is on top
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("What drives the risk score")
    fig.tight_layout()
    fig.savefig(os.path.join(settings.REPORTS_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
