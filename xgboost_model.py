import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the preprocessed data
df = pd.read_csv('D:\Customer churn prediction\preprocessed_customer_churn.csv')

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into 60% training and 40% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the XGBoost model
xgb_model = XGBClassifier(random_state=42)

# Train the model on the resampled data
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original, untouched test data
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print(f"XGBoost Model Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))