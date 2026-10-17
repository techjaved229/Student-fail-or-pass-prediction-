"""Prediction module for single-student pass/fail inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from utils.preprocessing import FEATURE_COLUMNS
from utils.validation import validate_student_inputs

MODEL_PATH = Path("models/best_model.pkl")


def load_model(path: Path = MODEL_PATH):
    """Load the trained best model."""
    if not path.exists():
        raise FileNotFoundError("Best model not found. Run train_model.py first.")
    return joblib.load(path)


def predict_student(attendance, internal_marks, test_score) -> dict:
    """Validate inputs and return model prediction with class probabilities."""
    attendance, internal_marks, test_score = validate_student_inputs(attendance, internal_marks, test_score)
    model = load_model()

    input_df = pd.DataFrame(
        [[attendance, internal_marks, test_score]],
        columns=FEATURE_COLUMNS,
    )
    predicted_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    fail_probability = round(float(probabilities[0] * 100), 2)
    pass_probability = round(float(probabilities[1] * 100), 2)

    return {
        "prediction": "PASS" if predicted_class == 1 else "FAIL",
        "pass_probability": pass_probability,
        "fail_probability": fail_probability,
    }


if __name__ == "__main__":
    print(predict_student(85, 42, 78))
