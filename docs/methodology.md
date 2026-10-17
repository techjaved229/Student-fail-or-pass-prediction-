# Methodology

## Workflow

```text
Dataset
↓
Data Cleaning
↓
Preprocessing
↓
Feature Selection
↓
Train/Test Split
↓
Model Training
↓
Evaluation
↓
Best Model
↓
Prediction
↓
Web Application
```

## Dataset

The dataset is generated using `generate_dataset.py`. It contains student attendance, internal marks, test score, and the final pass/fail result.

## Data Cleaning

The preprocessing module checks for missing values, handles missing values, detects duplicate rows, removes duplicates, and validates column data types.

## Preprocessing

The target variable `result` is encoded as:

- `Fail` = 0
- `Pass` = 1

## Feature Selection

The selected input features are:

- Attendance percentage
- Internal marks
- Test score

## Train/Test Split

The dataset is split into training and testing data using `random_state=42`. Stratification is used so both pass and fail classes are represented fairly in both sets.

## Model Training

Two models are trained:

- Logistic Regression
- Decision Tree Classifier

## Evaluation

Both models are evaluated using accuracy, precision, recall, F1 score, confusion matrix, and classification report.

## Best Model

The model with the highest F1 score is selected. Accuracy is used as a tie-breaker.

## Prediction

The `predict_student()` function validates user input, loads the selected model, makes a prediction, and returns pass/fail probabilities.

## Web Application

The Streamlit application provides dashboard, prediction, dataset, model performance, analytics, and about project pages.
