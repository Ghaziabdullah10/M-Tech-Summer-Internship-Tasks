
import numpy as np
from sklearn.linear_model import LinearRegression

print("="*60)
print("EXPERIMENT 04 - LAB TASKS")
print("="*60)

# ============================================================
# TASK 1: Train LinearRegression and predict for 7 and 8 hours
# ============================================================

print("\n" + "="*60)
print("TASK 1: Linear Regression - Hours vs Marks")
print("="*60)

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([35, 50, 65, 80, 95])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
pred_7 = model.predict([[7]])[0]
pred_8 = model.predict([[8]])[0]

print("Training Data:")
print("Hours Studied:", X.flatten())
print("Marks Obtained:", y)
print("\nModel Learned:")
print(f"Slope (coef_): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print("\nPredictions:")
print(f"Predicted marks for 7 hours: {pred_7:.2f}")
print(f"Predicted marks for 8 hours: {pred_8:.2f}")

# Verify the pattern (each hour adds 15 marks)
print("\nPattern discovered: Each additional study hour adds 15 marks!")


# ============================================================
# TASK 2: ML Workflow Steps (One Sentence Each)
# ============================================================

print("\n" + "="*60)
print("TASK 2: Machine Learning Workflow Steps")
print("="*60)

steps = [
    "1. Collect Data: Gather relevant data from various sources like databases, APIs, or files.",
    "2. Clean & Preprocess: Handle missing values, remove duplicates, scale features, and encode categorical variables.",
    "3. Split Data: Divide the dataset into training set (to learn) and test set (to evaluate).",
    "4. Train Model: Apply a machine learning algorithm to the training data to learn patterns.",
    "5. Evaluate Model: Measure performance on the test set using appropriate metrics like accuracy or R².",
    "6. Tune Hyperparameters: Adjust model settings to improve performance and reduce errors.",
    "7. Deploy Model: Make the trained model available for real-world use through APIs or applications."
]

for step in steps:
    print(step)


# ============================================================
# TASK 3: Classification and Regression Examples
# ============================================================

print("\n" + "="*60)
print("TASK 3: Real-World Examples")
print("="*60)

print("\nCLASSIFICATION PROBLEMS (Predict Categories):")
print("-" * 40)
classification_examples = [
    "1. Email Spam Detection: Classify emails as SPAM or NOT SPAM",
    "2. Disease Diagnosis: Predict if a patient has a disease (YES/NO)",
    "3. Image Recognition: Identify what object is in an image (CAT/DOG/CAR)",
    "4. Customer Churn: Predict if a customer will leave or stay (CHURN/STAY)",
    "5. Credit Card Fraud: Detect if a transaction is FRAUD or LEGITIMATE"
]
for ex in classification_examples:
    print(ex)

print("\nREGRESSION PROBLEMS (Predict Numbers):")
print("-" * 40)
regression_examples = [
    "1. House Price Prediction: Predict the selling price of a house",
    "2. Weather Forecasting: Predict tomorrow's temperature",
    "3. Stock Market: Predict future stock prices",
    "4. Salary Prediction: Estimate employee salary based on experience",
    "5. Sales Forecasting: Predict next month's product sales"
]
for ex in regression_examples:
    print(ex)


# ============================================================
# TASK 4: Dataset Features and Labels
# ============================================================

print("\n" + "="*60)
print("TASK 4: Dataset - Features and Label")
print("="*60)

print("\nDATASET 1: Iris Flower Dataset")
print("-" * 40)
print("Features (Input Variables):")
print("  - Sepal Length (cm)")
print("  - Sepal Width (cm)")
print("  - Petal Length (cm)")
print("  - Petal Width (cm)")
print("Label (Target Variable):")
print("  - Species (Setosa, Versicolor, Virginica)")
print("Type: Classification Problem")

print("\nDATASET 2: House Price Dataset")
print("-" * 40)
print("Features (Input Variables):")
print("  - Area (square feet)")
print("  - Number of Bedrooms")
print("  - Number of Bathrooms")
print("  - Location")
print("  - Year Built")
print("Label (Target Variable):")
print("  - House Price ($)")
print("Type: Regression Problem")

print("\nDATASET 3: Titanic Passenger Survival")
print("-" * 40)
print("Features (Input Variables):")
print("  - Passenger Class (1st/2nd/3rd)")
print("  - Age")
print("  - Gender")
print("  - Fare Paid")
print("  - Embarked Port")
print("Label (Target Variable):")
print("  - Survived (Yes/No)")
print("Type: Classification Problem")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
✓ Task 1: Linear Regression model trained and predictions made
✓ Task 2: ML Workflow steps listed with one sentence each
✓ Task 3: Classification and Regression examples provided
✓ Task 4: Dataset features and labels identified
""")

print("\n" + "="*60)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*60)