"""Train Logistic Regression and Decision Tree models, then save the best one."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import prepare_train_test_data

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "model_metrics.json"


def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculate standard classification metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["Fail", "Pass"], output_dict=True),
    }


def train_models() -> dict:
    """Train both models and persist all artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, cleaned_df = prepare_train_test_data()

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_leaf=12),
    }

    metrics = {
        "dataset": {
            "total_records": int(len(cleaned_df)),
            "training_records": int(len(x_train)),
            "testing_records": int(len(x_test)),
            "pass_records": int((cleaned_df["result"] == "Pass").sum()),
            "fail_records": int((cleaned_df["result"] == "Fail").sum()),
        },
        "models": {},
        "selection_criteria": "The best model is selected by highest F1 score. Accuracy is used as a tie-breaker.",
    }

    model_paths = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
        "Decision Tree": MODELS_DIR / "decision_tree.pkl",
    }

    trained_models = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics["models"][model_name] = evaluate_predictions(y_test, predictions)
        joblib.dump(model, model_paths[model_name])
        trained_models[model_name] = model

    selected_model_name = max(
        metrics["models"],
        key=lambda name: (metrics["models"][name]["f1_score"], metrics["models"][name]["accuracy"]),
    )
    metrics["selected_model"] = selected_model_name
    metrics["selected_model_reason"] = (
        f"{selected_model_name} achieved the strongest F1 score on the held-out test set, "
        "with accuracy used only if the F1 score tied."
    )

    joblib.dump(trained_models[selected_model_name], MODELS_DIR / "best_model.pkl")

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Training complete.")
    for model_name, model_metrics in metrics["models"].items():
        print(
            f"{model_name}: accuracy={model_metrics['accuracy']}, "
            f"precision={model_metrics['precision']}, recall={model_metrics['recall']}, "
            f"f1={model_metrics['f1_score']}"
        )
    print(f"Selected model: {selected_model_name}")
    return metrics


if __name__ == "__main__":
    train_models()
