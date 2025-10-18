import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed data
df = pd.read_csv('D:\Customer churn prediction\preprocessed_customer_churn.csv')

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into 60% training and 40% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# --- Hyperparameter Tuning with Grid Search ---
# Define the parameter grid to search through
# We've slightly adjusted the range to be more efficient.
param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Set up the Grid Search to find the best parameters on the TRAINING data
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the TRAINING data
grid_search.fit(X_train, y_train)

# --- Evaluate the Tuned Model on the unseen TESTING data ---
# Get the best model found by the grid search
best_model = grid_search.best_estimator_

# Make predictions on the unseen test data with the best model
tuned_predictions = best_model.predict(X_test)

# Calculate the new accuracy score
tuned_accuracy = accuracy_score(y_test, tuned_predictions)

# Print the results
print("\n--- Hyperparameter Tuning Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Tuned Model Accuracy: {tuned_accuracy:.4f}")