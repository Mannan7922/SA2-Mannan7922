# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 1: Load the dataset
# The dataset is assumed to be in CSV format. Here, we load it using pandas.
data = pd.read_excel(r'C:\Users\Dell\OneDrive\Documents\GitHub\SA2-Mannan7922\creditcard.xlsb', engine='pyxlsb')

# Step 2: Explore the dataset
# Initial overview of the dataset to understand its structure and key features.
print("Dataset Overview:")
print(data.head())  # Display the first few rows of the dataset

print("\nSummary:")
print(data.info())  # Print information about columns, data types, and missing values

print("\nClass Distribution:")
print(data['Class'].value_counts())  # Display the count of each class (e.g., fraud and non-fraud)

# Step 3: Preprocess the data
# Separate the features (X) from the target variable (y)
X = data.drop(columns=['Class'])  # Features include all columns except 'Class'
y = data['Class']  # Target variable is 'Class'

# Standardize the features to bring them to a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets, maintaining class distribution with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Train the models
# Logistic Regression: Simple, interpretable linear model for binary classification
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Decision Tree Classifier: Non-linear model that uses hierarchical splits
decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
decision_tree_model.fit(X_train, y_train)

# Step 5: Evaluate the models
# Function to calculate common evaluation metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)  # Model predictions
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# Evaluate Logistic Regression
logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
print("Logistic Regression Evaluation:")
print(f"Accuracy: {logistic_metrics[0]:.4f}")
print(f"Precision: {logistic_metrics[1]:.4f}")
print(f"Recall: {logistic_metrics[2]:.4f}")
print(f"F1-Score: {logistic_metrics[3]:.4f}\n")

# Evaluate Decision Tree
decision_tree_metrics = evaluate_model(decision_tree_model, X_test, y_test)
print("Decision Tree Evaluation:")
print(f"Accuracy: {decision_tree_metrics[0]:.4f}")
print(f"Precision: {decision_tree_metrics[1]:.4f}")
print(f"Recall: {decision_tree_metrics[2]:.4f}")
print(f"F1-Score: {decision_tree_metrics[3]:.4f}\n")

# Step 6: Generate and display classification reports
# Detailed breakdown of metrics per class
print("Logistic Regression Classification Report:")
print(classification_report(y_test, logistic_model.predict(X_test)))

print("Decision Tree Classification Report:")
print(classification_report(y_test, decision_tree_model.predict(X_test)))

# Step 7: Visualizing Model Performance
# Metrics labels for better interpretation
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Logistic Regression Performance Visualization
plt.figure(figsize=(8, 5))
plt.bar(metrics_labels, logistic_metrics, color='blue', alpha=0.7)
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Logistic Regression Performance")
plt.ylim(0, 1)
plt.show()

# Decision Tree Performance Visualization
plt.figure(figsize=(8, 5))
plt.bar(metrics_labels, decision_tree_metrics, color='green', alpha=0.7)
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Decision Tree Performance")
plt.ylim(0, 1)
plt.show()

# Step 8: Justifications and Observations
# Explaining choices and observations for better interpretability
print("\nModel Selection Justifications:")
print("Logistic Regression was chosen because it is a simple, efficient, and interpretable model.")
print("Decision Tree was chosen for its ability to capture non-linear patterns and generate interpretable decision rules.")

print("\nEvaluation Metrics Justification:")
print("Accuracy measures the overall correctness of predictions.")
print("Precision is important for minimizing false positives, especially for imbalanced datasets.")
print("Recall evaluates the ability to identify all actual positive cases.")
print("F1-Score balances precision and recall, which is especially relevant for imbalanced datasets.")

print("\nResults and Observations:")
if logistic_metrics[3] > decision_tree_metrics[3]:
    print("Logistic Regression performed better based on F1-Score, indicating its suitability for this dataset.")
else:
    print("Decision Tree performed better based on F1-Score, capturing more complex relationships in the data.")
