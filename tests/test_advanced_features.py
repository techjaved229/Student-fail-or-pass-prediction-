import pandas as pd

from utils.ml_insights import (
    categorize_risk,
    create_student_report_pdf,
    get_feature_importance,
    load_trained_model,
    model_prediction,
    predict_batch,
    recommendations_for_inputs,
)


def test_risk_score_uses_failure_probability():
    result = model_prediction(85, 42, 78)
    assert result["risk_score"] == result["fail_probability"]
    assert result["risk_category"] in {"Low Risk", "Moderate Risk", "High Risk"}


def test_risk_category_boundaries():
    assert categorize_risk(10) == "Low Risk"
    assert categorize_risk(40) == "Moderate Risk"
    assert categorize_risk(80) == "High Risk"


def test_batch_prediction_valid_csv_shape():
    batch = pd.DataFrame(
        [
            {"student_id": "B001", "student_name": "Test Student", "attendance": 80, "internal_marks": 40, "test_score": 75},
            {"student_id": "B002", "student_name": "Risk Student", "attendance": 45, "internal_marks": 18, "test_score": 35},
        ]
    )
    result = predict_batch(batch)
    assert {"prediction", "pass_probability", "fail_probability", "risk_score", "risk_category"}.issubset(result.columns)
    assert len(result) == 2


def test_batch_prediction_rejects_invalid_columns():
    batch = pd.DataFrame([{"student_id": "B001", "attendance": 80}])
    try:
        predict_batch(batch)
    except ValueError as exc:
        assert "missing required columns" in str(exc).lower()
    else:
        raise AssertionError("Invalid batch CSV should raise ValueError")


def test_what_if_simulation_changes_are_model_based():
    current = model_prediction(62, 27, 48)
    simulated = model_prediction(85, 42, 78)
    assert current["prediction"] in {"PASS", "FAIL"}
    assert simulated["prediction"] in {"PASS", "FAIL"}
    assert 0 <= simulated["fail_probability"] <= 100


def test_feature_importance_available_for_best_model():
    model = load_trained_model()
    importance = get_feature_importance(model, "Best Model")
    assert len(importance) == 3
    assert {item["feature"] for item in importance} == {"attendance", "internal_marks", "test_score"}


def test_recommendations_are_input_based():
    recommendations = recommendations_for_inputs(50, 20, 40)
    assert any("attendance" in item.lower() for item in recommendations)
    assert any("internal" in item.lower() for item in recommendations)
    assert any("test" in item.lower() for item in recommendations)


def test_pdf_report_generation():
    student = {
        "student_name": "Test Student",
        "student_id": "T001",
        "attendance": 80,
        "internal_marks": 40,
        "test_score": 75,
        "prediction": "PASS",
        "pass_probability": 90.0,
        "fail_probability": 10.0,
        "risk_score": 10.0,
        "risk_category": "Low Risk",
    }
    pdf_bytes = create_student_report_pdf(student, ["Maintain performance."], "Decision Tree")
    assert pdf_bytes.startswith(b"%PDF")
