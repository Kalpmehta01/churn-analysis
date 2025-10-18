import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the preprocessed data
df = pd.read_csv('D:\Customer churn prediction\preprocessed_customer_churn.csv')

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into 60% training and 40% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Check the training data distribution before SMOTE
print("Class distribution before SMOTE:")
print(y_train.value_counts())

# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the training data distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(y_train_resampled.value_counts())

# Train the Random Forest model on the resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original, untouched test data
y_pred_smote = rf_model_smote.predict(X_test)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"\nModel Accuracy after SMOTE: {accuracy_smote:.4f}")
print("\nClassification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))