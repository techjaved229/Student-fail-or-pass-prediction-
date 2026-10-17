# Viva Questions and Answers

1. **What is Machine Learning?**  
Machine Learning is a branch of artificial intelligence where systems learn patterns from data and make predictions or decisions.

2. **What is supervised learning?**  
Supervised learning uses labelled data where both input features and correct output labels are available during training.

3. **What is classification?**  
Classification is a supervised learning task where the output is a category, such as Pass or Fail.

4. **What is Logistic Regression?**  
Logistic Regression is a classification algorithm that predicts the probability of a class using a logistic function.

5. **What is a Decision Tree?**  
A Decision Tree is a model that splits data into branches based on feature values and makes decisions using rule-like paths.

6. **Why use attendance?**  
Attendance reflects student participation and regularity, which often affects academic performance.

7. **Why use internal marks?**  
Internal marks show continuous assessment performance before the final result.

8. **Why use test scores?**  
Test scores directly measure academic understanding and exam readiness.

9. **What is training data?**  
Training data is the part of the dataset used by the model to learn patterns.

10. **What is testing data?**  
Testing data is unseen data used to evaluate how well the model performs.

11. **What is overfitting?**  
Overfitting occurs when a model learns training data too closely and performs poorly on new data.

12. **What is underfitting?**  
Underfitting occurs when a model is too simple to learn important patterns in the data.

13. **What is accuracy?**  
Accuracy is the percentage of total predictions that were correct.

14. **What is precision?**  
Precision tells how many predicted positive cases were actually positive.

15. **What is recall?**  
Recall tells how many actual positive cases were correctly identified.

16. **What is F1 score?**  
F1 score is the harmonic mean of precision and recall.

17. **What is a confusion matrix?**  
A confusion matrix shows correct and incorrect predictions for each class.

18. **What is a feature?**  
A feature is an input variable used by the model, such as attendance.

19. **What is a target variable?**  
The target variable is the value the model tries to predict, such as result.

20. **Why use random_state?**  
`random_state` makes random operations reproducible.

21. **What is data leakage?**  
Data leakage happens when information from outside the training data improperly influences model training.

22. **Why use Streamlit?**  
Streamlit makes it easy to build interactive Python-based web applications for data science projects.

23. **Why use Joblib?**  
Joblib is used to save and load trained machine learning models efficiently.

24. **What are the limitations of this project?**  
The dataset is synthetic and uses only three academic features.

25. **What improvements can be made?**  
Real student data, more features, model explainability, and deployment can improve the project.

26. **Why compare two models?**  
Model comparison helps select the algorithm that performs better on test data.

27. **Why is F1 score useful?**  
F1 score balances precision and recall, making it useful when both false passes and false fails matter.

28. **What is target encoding in this project?**  
It converts `Fail` to 0 and `Pass` to 1 for model training.

29. **What is train/test split?**  
It divides data into one part for training and another part for evaluation.

30. **Why validate user input?**  
Validation prevents impossible values, such as attendance above 100, from reaching the model.

31. **What is the selected model?**  
The selected model is the one saved as `models/best_model.pkl` after training.

32. **Does the application hardcode predictions?**  
No. Predictions are produced by the trained machine learning model.
