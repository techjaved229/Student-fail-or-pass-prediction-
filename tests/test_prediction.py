from pathlib import Path

import pytest

from predict import load_model, predict_student


def test_valid_prediction():
    result = predict_student(85, 42, 78)
    assert result["prediction"] in {"PASS", "FAIL"}
    assert 0 <= result["pass_probability"] <= 100
    assert 0 <= result["fail_probability"] <= 100


def test_attendance_boundary_values():
    low = predict_student(0, 25, 50)
    high = predict_student(100, 25, 50)
    assert low["prediction"] in {"PASS", "FAIL"}
    assert high["prediction"] in {"PASS", "FAIL"}


def test_invalid_attendance():
    with pytest.raises(ValueError):
        predict_student(101, 25, 50)


def test_invalid_marks():
    with pytest.raises(ValueError):
        predict_student(75, 51, 50)


def test_model_loading():
    assert Path("models/best_model.pkl").exists()
    model = load_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
