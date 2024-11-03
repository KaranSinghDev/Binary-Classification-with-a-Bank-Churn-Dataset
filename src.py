import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
from xgboost import XGBClassifier

# Load the datasets
train_data = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Binary Classification with a Bank Churn Dataset\train.csv")
test_data = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Binary Classification with a Bank Churn Dataset\test.csv")

# Drop unnecessary columns
X_train = train_data.drop(columns=['id', 'CustomerId', 'Surname', 'Exited'])
y_train = train_data['Exited']
X_test = test_data.drop(columns=['id', 'CustomerId', 'Surname'])

# Identify categorical and numerical columns
categorical_cols = ['Geography', 'Gender']
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Create preprocessing pipelines for both numeric and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for both types of data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_processed = preprocessor.transform(X_test)

# Splitting the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_split, y_train_split)

# Hyperparameter tuning for RandomForest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],  # Allow deeper trees
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=rf_param_grid,
                               scoring='f1', cv=5, verbose=2, n_jobs=-1)
rf_grid_search.fit(X_train_resampled, y_train_resampled)

# Use the best RandomForest model
best_rf_model = rf_grid_search.best_estimator_

# Evaluate using cross-validation
rf_cv_scores = cross_val_score(best_rf_model, X_train_processed, y_train, cv=5, scoring='f1')
print("Mean cross-validation F1 score (RandomForest): ", rf_cv_scores.mean())

# Make predictions on the validation set
y_val_pred_rf = best_rf_model.predict(X_val_split)
print("\nValidation Classification Report (RandomForest):")
print(classification_report(y_val_split, y_val_pred_rf))

# XGBoost Model Training
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Evaluate XGBoost with cross-validation
xgb_cv_scores = cross_val_score(xgb_model, X_train_processed, y_train, cv=5, scoring='f1')
print("Mean cross-validation F1 score (XGBoost): ", xgb_cv_scores.mean())

# Make predictions on the validation set
y_val_pred_xgb = xgb_model.predict(X_val_split)
print("\nValidation Classification Report (XGBoost):")
print(classification_report(y_val_split, y_val_pred_xgb))

# Select the best performing model based on validation F1 score
if xgb_cv_scores.mean() > rf_cv_scores.mean():
    best_model = xgb_model
else:
    best_model = best_rf_model

# Make predictions on the test data (probabilities for class 1)
y_test_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_data['CustomerId'],  # Use the CustomerId from the test dataset
    'Exited': y_test_pred_proba
})

submission.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Binary Classification with a Bank Churn Dataset\submission.csv", index=False)
print("Submission file created successfully!")
