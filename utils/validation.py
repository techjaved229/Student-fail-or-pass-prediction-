"""Input validation helpers used by training, prediction, tests, and UI."""

from __future__ import annotations


FEATURE_RANGES = {
    "attendance": (0, 100),
    "internal_marks": (0, 50),
    "test_score": (0, 100),
}


def validate_numeric_value(name: str, value: float) -> float:
    """Validate a single feature value and return it as a float."""
    if name not in FEATURE_RANGES:
        raise ValueError(f"Unknown feature: {name}")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name.replace('_', ' ').title()} must be numeric.") from exc

    minimum, maximum = FEATURE_RANGES[name]
    if numeric_value < minimum or numeric_value > maximum:
        label = name.replace("_", " ").title()
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")

    return numeric_value


def validate_student_inputs(attendance: float, internal_marks: float, test_score: float) -> tuple[float, float, float]:
    """Validate the three model features."""
    return (
        validate_numeric_value("attendance", attendance),
        validate_numeric_value("internal_marks", internal_marks),
        validate_numeric_value("test_score", test_score),
    )
