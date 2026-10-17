# Student Pass/Fail Prediction Using Machine Learning

A complete college-submission-ready machine learning project that predicts whether a student is likely to **Pass** or **Fail** using attendance percentage, internal marks, and test score.

## Features

- Generates a realistic synthetic dataset with 650 student records.
- Cleans and validates data before training.
- Trains Logistic Regression and Decision Tree classifiers.
- Selects the best model using F1 score, with accuracy as a tie-breaker.
- Saves trained models using Joblib.
- Produces real evaluation metrics and charts.
- Provides a professional Streamlit dashboard with prediction, dataset, performance, analytics, and project information pages.
- Adds advanced student risk scoring, student profiles, batch prediction, what-if simulation, session prediction history, feature importance, ROC-AUC, and PDF report export.
- Includes pytest test cases for data and prediction behavior.

## Technology Stack

Python 3, Pandas, NumPy, Scikit-learn, Joblib, Streamlit, Plotly, Matplotlib, Seaborn, and Pytest.

## Project Structure

```text
student-pass-fail-prediction/
├── app.py
├── train_model.py
├── predict.py
├── generate_dataset.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── data/students.csv
├── models/
├── reports/
├── utils/
├── tests/
└── docs/
```

## Dataset

The dataset is saved as `data/students.csv` and contains:

- `student_id`
- `student_name`
- `attendance`
- `internal_marks`
- `test_score`
- `result`

The pass/fail result is generated using a transparent academic scoring rule with controlled random noise, making the classification realistic without hardcoding predictions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Workflow

```bash
python3 generate_dataset.py
python3 train_model.py
python3 evaluate_model.py
pytest
python3 -c "from predict import predict_student; print(predict_student(85,42,78))"
```

## Run Streamlit App

```bash
streamlit run app.py
```

## Advanced Application Pages

- Dashboard
- Student Prediction
- Student Profiles
- Risk Analysis
- Batch Prediction
- What-If Simulator
- Dataset
- Model Performance
- Analytics
- Prediction History
- About Project

Risk score is calculated from the trained model's failure probability. It is grouped as Low Risk, Moderate Risk, or High Risk for academic support analysis.

## Example Prediction

```python
from predict import predict_student

print(predict_student(85, 42, 78))
```

The prediction result and probabilities come from the trained `models/best_model.pkl` model.

## Screenshots

Add screenshots of the dashboard, prediction page, dataset page, analytics page, and model performance page after running the Streamlit app.

## Limitations

- The dataset is synthetic, although generated with realistic variation.
- Only three academic features are used.
- Predictions are estimates and should not replace official academic decisions.

## Future Improvements

- Train on real institutional academic data.
- Add additional features such as assignment scores, study hours, and previous semester results.
- Include explainable AI methods for better interpretation.
- Deploy the dashboard on an internal college server.
