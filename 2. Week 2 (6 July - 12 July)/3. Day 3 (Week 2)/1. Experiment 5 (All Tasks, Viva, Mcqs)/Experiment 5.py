"""
EXPERIMENT 05 - Lab Tasks
Supervised Learning - Regression & Classification
Date: 08-07-2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.preprocessing import StandardScaler

print("="*60)
print("EXPERIMENT 05 - LAB TASKS")
print("="*60)


# ============================================================
# TASK 1: LinearRegression on Real Dataset
# ============================================================

print("\n" + "="*60)
print("TASK 1: Linear Regression on Diabetes Dataset")
print("="*60)

# Load diabetes dataset (real dataset)
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Dataset: Diabetes (442 samples, 10 features)")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"\nModel Coefficient: {lr_model.coef_[0]:.2f}")
print(f"Model Intercept: {lr_model.intercept_:.2f}")


# ============================================================
# TASK 2: LogisticRegression Classifier
# ============================================================

print("\n" + "="*60)
print("TASK 2: Logistic Regression Classifier")
print("="*60)

# Create a simple classification dataset
X_clf, y_clf = make_classification(n_samples=200, n_features=4, n_informative=3,
                                   n_redundant=1, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Train model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sample prediction
sample = X_test[0].reshape(1, -1)
sample_pred = log_model.predict(sample)[0]

print(f"Accuracy: {accuracy:.4f}")
print(f"Sample features: {X_test[0]}")
print(f"Sample prediction: Class {sample_pred}")
print(f"Actual class: {y_test[0]}")


# ============================================================
# TASK 3: Compare KNN with different k values
# ============================================================

print("\n" + "="*60)
print("TASK 3: KNN Comparison on Iris Dataset")
print("="*60)

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare different k values
k_values = [1, 3, 5, 7]
results = []

print("\nKNN Accuracy Comparison:")
print("-" * 40)
print("k-value | Accuracy")
print("-" * 40)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    results.append(acc)
    print(f"   {k}    |   {acc:.4f}")

print("-" * 40)

# Find best k
best_k = k_values[np.argmax(results)]
print(f"\nBest k value: {best_k} with accuracy {max(results):.4f}")


# ============================================================
# TASK 4: Decision Tree - Feature Importance
# ============================================================

print("\n" + "="*60)
print("TASK 4: Decision Tree - Feature Importance")
print("="*60)

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Get feature importances
importances = dt.feature_importances_

# Display results
print("\nFeature Importances:")
print("-" * 40)
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

print("-" * 40)

# Find top 2 features
sorted_idx = np.argsort(importances)[::-1]
top1 = feature_names[sorted_idx[0]]
top2 = feature_names[sorted_idx[1]]

print(f"\nThe two most important features are:")
print(f"1. {top1} (importance: {importances[sorted_idx[0]]:.4f})")
print(f"2. {top2} (importance: {importances[sorted_idx[1]]:.4f})")

# Model accuracy
accuracy = dt.score(X_test, y_test)
print(f"\nDecision Tree Accuracy: {accuracy:.4f}")


# ============================================================
# TASK 5: Difference between Regression and Classification
# ============================================================

print("\n" + "="*60)
print("TASK 5: Regression vs Classification - Explanation")
print("="*60)

print("""
REGRESSION:
- Predicts CONTINUOUS numerical values
- Output is a number (e.g., price, temperature, score)
- Examples: House price prediction, weather forecasting, stock price prediction
- Evaluation metrics: R², RMSE, MAE

CLASSIFICATION:
- Predicts DISCRETE categories or classes
- Output is a label (e.g., spam/not spam, cat/dog)
- Examples: Email spam detection, disease diagnosis, image recognition
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

KEY DIFFERENCE:
Regression predicts numbers, Classification predicts categories.
The choice depends on what type of output we want to predict.
""")


# ============================================================
# BONUS: Compare all models on Iris dataset
# ============================================================

print("\n" + "="*60)
print("BONUS: Model Comparison on Iris Dataset")
print("="*60)

# Load Iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
}

print("\nModel Comparison:")
print("-" * 50)
print("Model                | Accuracy")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name:<20} | {acc:.4f}")

print("-" * 50)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
✓ Task 1: Linear Regression trained on Diabetes dataset (R² and RMSE reported)
✓ Task 2: Logistic Regression classifier trained and evaluated (accuracy and sample prediction)
✓ Task 3: KNN compared with k=1, 3, 5, 7 on Iris dataset
✓ Task 4: Decision Tree trained and top 2 features identified
✓ Task 5: Regression vs Classification explained with examples
""")

print("\n" + "="*60)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*60)