"""Data loading, cleaning, validation, encoding, and train/test splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/students.csv")
FEATURE_COLUMNS = ["attendance", "internal_marks", "test_score"]
TARGET_COLUMN = "result"
TARGET_MAPPING = {"Fail": 0, "Pass": 1}
REVERSE_TARGET_MAPPING = {value: key for key, value in TARGET_MAPPING.items()}
REQUIRED_COLUMNS = ["student_id", "student_name", *FEATURE_COLUMNS, TARGET_COLUMN]


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the student dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"\ufeffstudent_id": "student_id", "bstudent_id": "student_id"})
    return df


def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return missing value counts for each column."""
    return df.isna().sum()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric missing values with medians and text values with modes."""
    cleaned = df.copy()
    for column in FEATURE_COLUMNS:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in ["student_id", "student_name", TARGET_COLUMN]:
        if cleaned[column].isna().any():
            mode_value = cleaned[column].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(fill_value)
    return cleaned


def detect_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows."""
    return int(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def validate_data_types(df: pd.DataFrame) -> None:
    """Validate required columns, numeric ranges, and target values."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be numeric.")

    if not df["attendance"].between(0, 100).all():
        raise ValueError("attendance values must be between 0 and 100.")
    if not df["internal_marks"].between(0, 50).all():
        raise ValueError("internal_marks values must be between 0 and 50.")
    if not df["test_score"].between(0, 100).all():
        raise ValueError("test_score values must be between 0 and 100.")

    invalid_targets = set(df[TARGET_COLUMN].unique()) - set(TARGET_MAPPING)
    if invalid_targets:
        raise ValueError(f"Unexpected target values found: {invalid_targets}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning and validation steps."""
    cleaned = handle_missing_values(df)
    cleaned = remove_duplicates(cleaned)
    validate_data_types(cleaned)
    return cleaned


def select_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model features and encode Pass/Fail labels."""
    features = df[FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].map(TARGET_MAPPING)
    return features, target


def prepare_train_test_data(
    path: str | Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load, clean, encode, and split data without data leakage."""
    raw_df = load_dataset(path)
    cleaned_df = clean_dataset(raw_df)
    features, target = select_features_and_target(cleaned_df)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return x_train, x_test, y_train, y_test, cleaned_df
