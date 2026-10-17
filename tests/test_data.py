from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/students.csv")
REQUIRED_COLUMNS = {"student_id", "student_name", "attendance", "internal_marks", "test_score", "result"}


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_required_columns_exist():
    df = pd.read_csv(DATA_PATH)
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_dataset_contains_at_least_500_records():
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 500


def test_no_unexpected_target_values():
    df = pd.read_csv(DATA_PATH)
    assert set(df["result"].unique()).issubset({"Pass", "Fail"})


def test_numeric_columns_contain_valid_values():
    df = pd.read_csv(DATA_PATH)
    assert df["attendance"].between(0, 100).all()
    assert df["internal_marks"].between(0, 50).all()
    assert df["test_score"].between(0, 100).all()
