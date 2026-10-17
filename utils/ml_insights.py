"""Advanced ML insight helpers for the Streamlit application."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import auc, roc_curve

from utils.preprocessing import FEATURE_COLUMNS, prepare_train_test_data
from utils.validation import validate_student_inputs

DEFAULT_MODEL_PATH = Path("models/best_model.pkl")
LOGISTIC_MODEL_PATH = Path("models/logistic_regression.pkl")
TREE_MODEL_PATH = Path("models/decision_tree.pkl")


def load_trained_model(path: Path = DEFAULT_MODEL_PATH):
    """Load a trained model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def model_prediction(attendance: float, internal_marks: float, test_score: float, model=None) -> dict:
    """Return prediction, probabilities, confidence, and risk values from the model."""
    attendance, internal_marks, test_score = validate_student_inputs(attendance, internal_marks, test_score)
    model = model or load_trained_model()
    input_df = pd.DataFrame([[attendance, internal_marks, test_score]], columns=FEATURE_COLUMNS)
    prediction_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    fail_probability = round(float(probabilities[0] * 100), 2)
    pass_probability = round(float(probabilities[1] * 100), 2)
    risk_score = round(fail_probability, 2)
    return {
        "prediction": "PASS" if prediction_class == 1 else "FAIL",
        "pass_probability": pass_probability,
        "fail_probability": fail_probability,
        "confidence": round(max(pass_probability, fail_probability), 2),
        "risk_score": risk_score,
        "risk_category": categorize_risk(risk_score),
    }


def categorize_risk(risk_score: float) -> str:
    """Categorize risk from model-derived failure probability."""
    if risk_score < 35:
        return "Low Risk"
    if risk_score < 65:
        return "Moderate Risk"
    return "High Risk"


def recommendations_for_inputs(attendance: float, internal_marks: float, test_score: float) -> list[str]:
    """Generate deterministic recommendations from the actual input values."""
    attendance, internal_marks, test_score = validate_student_inputs(attendance, internal_marks, test_score)
    recommendations: list[str] = []
    if attendance < 75:
        recommendations.append("Improve attendance to strengthen academic consistency.")
    if internal_marks < 30:
        recommendations.append("Focus on internal assessments, assignments, and class tests.")
    if test_score < 55:
        recommendations.append("Additional test preparation may help improve exam performance.")
    if attendance >= 85 and internal_marks >= 38 and test_score >= 75:
        recommendations.append("Current academic indicators are strong. Maintain the same study routine.")
    if not recommendations:
        recommendations.append("Academic indicators are acceptable, but regular revision and attendance should be maintained.")
    return recommendations


def feature_based_explanation(attendance: float, internal_marks: float, test_score: float, model, model_name: str) -> list[str]:
    """Explain the prediction using model-derived importance where available plus input-level signals."""
    explanations: list[str] = []
    importances = get_feature_importance(model, model_name)
    if importances:
        top_feature = max(importances, key=lambda item: abs(item["value"]))
        explanations.append(
            f"{top_feature['feature']} has the strongest model-derived influence in the selected model."
        )
    else:
        explanations.append("This explanation uses feature values because model-level importance is unavailable.")

    if attendance < 75:
        explanations.append("Attendance is below the usual academic consistency target.")
    elif attendance >= 85:
        explanations.append("Attendance is strong and supports a positive academic outcome.")
    else:
        explanations.append("Attendance is moderate and should be monitored.")

    if internal_marks < 30:
        explanations.append("Internal marks indicate weak continuous assessment performance.")
    elif internal_marks >= 38:
        explanations.append("Internal marks indicate strong continuous assessment performance.")
    else:
        explanations.append("Internal marks indicate moderate academic performance.")

    if test_score < 55:
        explanations.append("Test score indicates that exam preparation needs improvement.")
    elif test_score >= 75:
        explanations.append("Test score indicates strong exam performance.")
    else:
        explanations.append("Test score indicates moderate exam readiness.")
    return explanations


def get_feature_importance(model, model_name: str) -> list[dict]:
    """Return actual model-derived feature importance or coefficients."""
    if hasattr(model, "feature_importances_"):
        return [
            {"feature": feature, "value": round(float(value), 4), "type": "importance"}
            for feature, value in zip(FEATURE_COLUMNS, model.feature_importances_)
        ]

    classifier = getattr(model, "named_steps", {}).get("classifier") if hasattr(model, "named_steps") else None
    if classifier is not None and hasattr(classifier, "coef_"):
        return [
            {"feature": feature, "value": round(float(value), 4), "type": "coefficient"}
            for feature, value in zip(FEATURE_COLUMNS, classifier.coef_[0])
        ]
    return []


def predict_batch(uploaded_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Validate and score a batch CSV without changing the main dataset."""
    required_columns = {"student_id", "student_name", *FEATURE_COLUMNS}
    missing = required_columns - set(uploaded_df.columns)
    if missing:
        raise ValueError(f"Uploaded CSV missing required columns: {sorted(missing)}")

    batch_df = uploaded_df.copy()
    for column in FEATURE_COLUMNS:
        batch_df[column] = pd.to_numeric(batch_df[column], errors="coerce")
    if batch_df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Attendance, internal marks, and test score must be numeric.")
    if not batch_df["attendance"].between(0, 100).all():
        raise ValueError("Attendance values must be between 0 and 100.")
    if not batch_df["internal_marks"].between(0, 50).all():
        raise ValueError("Internal marks values must be between 0 and 50.")
    if not batch_df["test_score"].between(0, 100).all():
        raise ValueError("Test score values must be between 0 and 100.")

    model = model or load_trained_model()
    probabilities = model.predict_proba(batch_df[FEATURE_COLUMNS])
    predictions = model.predict(batch_df[FEATURE_COLUMNS])
    batch_df["prediction"] = ["PASS" if int(value) == 1 else "FAIL" for value in predictions]
    batch_df["pass_probability"] = (probabilities[:, 1] * 100).round(2)
    batch_df["fail_probability"] = (probabilities[:, 0] * 100).round(2)
    batch_df["risk_score"] = batch_df["fail_probability"].round(2)
    batch_df["risk_category"] = batch_df["risk_score"].apply(categorize_risk)
    return batch_df


def score_dataset(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Attach model predictions, probabilities, and risk categories to a dataset."""
    model = model or load_trained_model()
    scored = df.copy()
    probabilities = model.predict_proba(scored[FEATURE_COLUMNS])
    predictions = model.predict(scored[FEATURE_COLUMNS])
    scored["model_prediction"] = ["PASS" if int(value) == 1 else "FAIL" for value in predictions]
    scored["pass_probability"] = (probabilities[:, 1] * 100).round(2)
    scored["fail_probability"] = (probabilities[:, 0] * 100).round(2)
    scored["risk_score"] = scored["fail_probability"].round(2)
    scored["risk_category"] = scored["risk_score"].apply(categorize_risk)
    return scored


def get_roc_data(model=None) -> dict:
    """Calculate ROC curve points and ROC-AUC from the held-out test set."""
    model = model or load_trained_model()
    _, x_test, _, y_test, _ = prepare_train_test_data()
    probabilities = model.predict_proba(x_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probabilities)
    return {
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "thresholds": thresholds,
        "roc_auc": round(float(auc(false_positive_rate, true_positive_rate)), 4),
    }


def create_student_report_pdf(student: dict, recommendations: list[str], model_used: str) -> bytes:
    """Create a PDF prediction report using actual prediction data."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Student Pass/Fail Prediction Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 16),
    ]

    rows = [
        ["Student Name", student.get("student_name", "")],
        ["Student ID", student.get("student_id", "")],
        ["Attendance", f"{student.get('attendance', 0):.1f}%"],
        ["Internal Marks", f"{student.get('internal_marks', 0):.1f}/50"],
        ["Test Score", f"{student.get('test_score', 0):.1f}/100"],
        ["Prediction", student.get("prediction", "")],
        ["Pass Probability", f"{student.get('pass_probability', 0):.2f}%"],
        ["Fail Probability", f"{student.get('fail_probability', 0):.2f}%"],
        ["Risk Score", f"{student.get('risk_score', 0):.2f}/100"],
        ["Risk Category", student.get("risk_category", "")],
        ["Model Used", model_used],
    ]
    table = Table(rows, colWidths=[150, 300])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Recommendations", styles["Heading2"]))
    for item in recommendations:
        elements.append(Paragraph(f"- {item}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            "Note: This prediction is an estimate based on the trained dataset and model. "
            "It should not be treated as a definitive academic decision.",
            styles["Italic"],
        )
    )
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()
