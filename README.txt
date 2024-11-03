# ChurnInsight: Predictive Analytics for Banking

## Problem
Customer churn poses a significant challenge for financial institutions, leading to revenue loss and increased acquisition costs. Predicting which customers are likely to leave can help banks implement retention strategies and improve customer satisfaction.

## Solution
This project leverages machine learning techniques to predict customer churn based on historical data. By analyzing customer behavior and characteristics, the model identifies at-risk customers, enabling proactive measures to enhance retention.

## Dataset
The dataset used for this project is derived from the Bank Churn dataset available on Kaggle, which contains customer data, including demographic information and account details. The dataset consists of features such as account balance, transaction history, and customer demographics.

- **Data Source**: [Kaggle Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)

## Model
The project employs the following machine learning models for classification:
1. **Logistic Regression**: A simple linear model for binary classification.
2. **Random Forest Classifier**: An ensemble model that enhances accuracy through bagging.
3. **XGBoost Classifier**: A powerful gradient boosting model that improves prediction accuracy.

### Hyperparameters Used
- **Logistic Regression**: `max_iter=1000`
- **Random Forest Classifier**:
  - `n_estimators=[100, 200]`
  - `max_depth=[None, 10, 20]`
  - `min_samples_split=[2, 5]`
  - `class_weight=['balanced', None]`
- **XGBoost Classifier**:
  - `n_estimators=[100, 200]`
  - `max_depth=[3, 5]`
  - `learning_rate=[0.01, 0.1]`
  - `subsample=[0.6, 0.8]`
  - `colsample_bytree=[0.6, 0.8]`
  - `gamma=[0, 0.1]`
  - `scale_pos_weight=[1]`

## Evaluation Score
The model performance is evaluated using the ROC-AUC score, which measures the model's ability to distinguish between churn and non-churn customers. The final results from the XGBoost classifier showed:
- **ROC-AUC Score**: `0.79` (average for churn prediction)

## Citation
Walter Reade and Ashley Chow. Binary Classification with a Bank Churn Dataset. [Kaggle](https://kaggle.com/competitions/playground-series-s4e1), 2024. Kaggle.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
