import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dictionary with the accuracy scores of each model
model_accuracies = {
    'Logistic Regression': 0.8016,
    'Random Forest': 0.7846,
    'XGBoost (Final)': 0.7984 
}

# Convert the dictionary to a pandas Series for easy plotting
accuracies = pd.Series(model_accuracies)

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies.index, y=accuracies.values, palette='coolwarm')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.ylim(0.75, 0.82) # Set y-axis limits for better comparison
plt.show()