# Student Pass/Fail Prediction Using Machine Learning

## 1. Cover Page

**Project Title:** Student Pass/Fail Prediction Using Machine Learning  
**Submitted By:** ____________________  
**Roll Number:** ____________________  
**Department:** ____________________  
**College:** ____________________  
**Academic Year:** ____________________

## 2. Certificate

This is to certify that the project titled **Student Pass/Fail Prediction Using Machine Learning** has been successfully completed by ____________________ under the guidance of ____________________.

## 3. Declaration

I declare that this project work is my original work and has been completed for academic submission.

## 4. Acknowledgement

I sincerely thank my guide, teachers, department, and institution for their support and guidance throughout this project.

## 5. Abstract

This project predicts whether a student is likely to pass or fail based on attendance percentage, internal marks, and test score. It uses a generated dataset with realistic academic variation, trains Logistic Regression and Decision Tree models, evaluates both models using standard classification metrics, and provides a Streamlit web application for demonstration.

## 6. Introduction

Academic performance prediction is a useful application of machine learning. It can help educators identify students who may require support before final examinations or official results.

## 7. Problem Statement

Many institutions evaluate student performance after examinations, which can delay academic intervention. This project addresses the problem by predicting likely pass/fail outcomes using available academic indicators.

## 8. Objectives

- Generate a realistic student dataset.
- Train Logistic Regression and Decision Tree classifiers.
- Compare models using real evaluation metrics.
- Save the selected model for prediction.
- Build a professional Streamlit application.
- Document the complete workflow for college submission and viva.

## 9. Existing System

Traditional academic monitoring depends heavily on manual review of attendance and marks. Such systems may not provide fast predictive insight.

## 10. Proposed System

The proposed system uses machine learning to predict pass/fail outcomes from attendance, internal marks, and test score. The model is integrated into a dashboard for prediction and analysis.

## 11. Technologies Used

Python 3, Pandas, NumPy, Scikit-learn, Joblib, Streamlit, Plotly, Matplotlib, Seaborn, and Pytest.

## 12. Hardware Requirements

- Processor: Dual-core or higher
- RAM: 4 GB minimum
- Storage: 500 MB free space

## 13. Software Requirements

- Python 3.10 or higher
- Web browser
- Required Python packages from `requirements.txt`

## 14. Dataset Description

The dataset is stored in `data/students.csv` and contains student ID, student name, attendance, internal marks, test score, and result. The dataset is generated using a reproducible random seed.

The result generation rule uses:

```text
academic_score = attendance * 0.30 + internal_marks * 1.00 + test_score * 0.45
result = Pass if academic_score + random_noise >= 84 else Fail
```

This creates a meaningful and realistic classification problem with controlled variation.

## 15. Data Preprocessing

The preprocessing workflow includes CSV loading, missing value detection, missing value handling, duplicate detection, duplicate removal, data type validation, target encoding, and train/test splitting.

## 16. Feature Selection

The features used for model training are attendance, internal marks, and test score. These are directly related to academic performance.

## 17. Machine Learning Methodology

The project follows a supervised classification approach because the target labels are known as Pass or Fail.

## 18. Logistic Regression

Logistic Regression estimates the probability that a record belongs to a particular class. It is simple, efficient, and suitable for binary classification.

## 19. Decision Tree

Decision Tree Classifier creates rule-based splits in the data. It is easy to understand and can model non-linear relationships.

## 20. Model Training

The dataset is split into training and testing sets using `random_state=42`. Logistic Regression and Decision Tree models are trained on the training data.

## 21. Model Evaluation

The models are evaluated using accuracy, precision, recall, F1 score, confusion matrix, and classification report.

## 22. Results

After training and evaluation, the Decision Tree Classifier was selected as the best model because it achieved the highest F1 score on the held-out test data.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.38% | 87.23% | 92.13% | 89.62% |
| Decision Tree | 88.46% | 90.22% | 93.26% | 91.71% |

The selected model is saved as `models/best_model.pkl`. Full metrics are stored in `reports/model_metrics.json`.

## 23. Screenshots

Add screenshots of:

- Dashboard page
- Prediction page
- Dataset page
- Model performance page
- Analytics page

## 24. Advantages

- Easy to understand.
- Uses real trained models.
- Provides a complete dashboard.
- Supports local execution.
- Includes tests and documentation.

## 25. Limitations

- Synthetic dataset.
- Limited to three features.
- Predictions are estimates and should not replace official decisions.

## 26. Future Scope

- Use real academic data.
- Add more student performance indicators.
- Improve explainability.
- Deploy the application for institutional use.

## 27. Conclusion

The project successfully demonstrates a complete machine learning workflow from dataset generation to model training, evaluation, prediction, and web application deployment.

## 28. References

- Python Documentation: https://docs.python.org/3/
- Pandas Documentation: https://pandas.pydata.org/docs/
- NumPy Documentation: https://numpy.org/doc/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Streamlit Documentation: https://docs.streamlit.io/
- Plotly Documentation: https://plotly.com/python/
