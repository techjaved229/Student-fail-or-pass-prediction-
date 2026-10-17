"""Generate evaluation charts from the trained model and actual dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.preprocessing import load_dataset, prepare_train_test_data

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
BEST_MODEL_PATH = Path("models/best_model.pkl")


def save_pass_fail_distribution(df) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="result", hue="result", palette={"Pass": "#2e7d32", "Fail": "#c62828"}, legend=False)
    ax.set_title("Pass vs Fail Distribution")
    ax.set_xlabel("Result")
    ax.set_ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pass_fail_distribution.png", dpi=160)
    plt.close()


def save_distribution(df, column: str, title: str, xlabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data=df, x=column, hue="result", kde=True, bins=25, palette={"Pass": "#2e7d32", "Fail": "#c62828"})
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=160)
    plt.close()


def save_confusion_matrix() -> None:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Best model not found. Run train_model.py first.")

    _, x_test, _, y_test, _ = prepare_train_test_data()
    model = joblib.load(BEST_MODEL_PATH)
    predictions = model.predict(x_test)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        confusion_matrix(y_test, predictions),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
    )
    ax.set_title("Confusion Matrix for Selected Model")
    ax.set_xlabel("Predicted Result")
    ax.set_ylabel("Actual Result")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def generate_evaluation_reports() -> None:
    """Generate all required plots from actual project data and model results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics file not found. Run train_model.py first.")

    df = load_dataset()
    save_confusion_matrix()
    save_pass_fail_distribution(df)
    save_distribution(df, "attendance", "Attendance Distribution by Result", "Attendance (%)", "attendance_distribution.png")
    save_distribution(df, "internal_marks", "Internal Marks Distribution by Result", "Internal Marks (0-50)", "marks_distribution.png")
    save_distribution(df, "test_score", "Test Score Distribution by Result", "Test Score (0-100)", "test_score_distribution.png")

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    print("Evaluation charts generated.")
    print(f"Selected model: {metrics['selected_model']}")


if __name__ == "__main__":
    generate_evaluation_reports()
