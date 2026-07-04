"""
train_model.py
---------------
This module trains and compares five different Machine Learning
classification algorithms on the employee dataset:

    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. K-Nearest Neighbors (KNN)
    5. Support Vector Machine (SVM)

It evaluates each model using Accuracy, Precision, Recall and F1-Score,
automatically selects the best-performing model, and saves the following
artifacts (using joblib) inside the models/ folder so the GUI application
can load them instantly without retraining:

    - best_model.pkl        : the best trained model
    - scaler.pkl            : the fitted StandardScaler
    - encoders.pkl          : dictionary of fitted LabelEncoders (features)
    - target_encoder.pkl    : fitted LabelEncoder for the target column
    - feature_columns.pkl   : ordered list of feature column names
    - model_info.pkl        : dictionary with the winning model's name & metrics

It also saves visual reports inside the reports/ folder:
    - model_comparison.png  : bar chart comparing all 5 algorithms
    - confusion_matrix.png  : confusion matrix heat-map of the best model
    - evaluation_report.txt : plain text summary of all metrics
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend suitable for saving files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from preprocess import full_training_pipeline

# ---------------------------------------------------------------------------
# Project folder paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "employee_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def get_candidate_models():
    """
    Return a dictionary of all the candidate Machine Learning models that
    will be trained and compared. Each model uses reasonable default
    hyperparameters suitable for a small/medium sized HR dataset.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=10
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }


def evaluate_model(model, X_test, y_test):
    """
    Compute Accuracy, Precision, Recall and F1-Score for a trained model on
    the held-out test set. Returns a dictionary of the four metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics, y_pred


def save_comparison_chart(results, save_path):
    """
    Create and save a grouped bar chart comparing Accuracy, Precision,
    Recall and F1-Score across all trained models.
    """
    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metric_names):
        values = [results[name][metric] for name in model_names]
        plt.bar(x + (i - 1.5) * width, values, width, label=metric.capitalize(),
                color=colors[i])

    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison - Employee Attrition Predictor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_confusion_matrix_chart(y_test, y_pred, model_name, save_path):
    """
    Create and save a confusion matrix heat-map for the best performing model.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Attrition", "Attrition"],
        yticklabels=["No Attrition", "Attrition"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def write_text_report(results, best_model_name, save_path):
    """
    Write a plain-text summary report of all model evaluation metrics.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EMPLOYEE ATTRITION PREDICTOR - MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    for name, metrics in results.items():
        lines.append(f"Model: {name}")
        lines.append(f"    Accuracy  : {metrics['accuracy']:.4f}")
        lines.append(f"    Precision : {metrics['precision']:.4f}")
        lines.append(f"    Recall    : {metrics['recall']:.4f}")
        lines.append(f"    F1-Score  : {metrics['f1_score']:.4f}")
        lines.append("")
    lines.append("-" * 70)
    lines.append(f"BEST MODEL SELECTED: {best_model_name}")
    lines.append("-" * 70)

    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def train_and_save(dataset_path=DATASET_PATH, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR):
    """
    Run the complete training pipeline:
        1. Preprocess the data.
        2. Train all 5 candidate models.
        3. Evaluate each model on the test set.
        4. Select the best model (highest F1-Score).
        5. Save the best model + all preprocessing artifacts with joblib.
        6. Save comparison charts and a text report.

    Returns
    -------
    dict
        Summary information about the training run, including the name of
        the best model and its evaluation metrics.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("[train_model] Starting preprocessing pipeline...")
    data = full_training_pipeline(dataset_path)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    models = get_candidate_models()
    results = {}
    trained_models = {}
    predictions = {}

    print("[train_model] Training and evaluating candidate models...")
    for name, model in models.items():
        print(f"    -> Training {name} ...")
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model
        predictions[name] = y_pred
        print(f"       Accuracy={metrics['accuracy']:.4f}  "
              f"F1-Score={metrics['f1_score']:.4f}")

    # Select the best model based on F1-Score (a balanced metric that
    # accounts for both false positives and false negatives, which is
    # important in an HR attrition context).
    best_model_name = max(results, key=lambda name: results[name]["f1_score"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    best_predictions = predictions[best_model_name]

    print(f"[train_model] Best model selected: {best_model_name}")

    # ------------------------------------------------------------------
    # Save all artifacts using joblib
    # ------------------------------------------------------------------
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(data["scaler"], os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(data["encoders"], os.path.join(models_dir, "encoders.pkl"))
    joblib.dump(data["target_encoder"], os.path.join(models_dir, "target_encoder.pkl"))
    joblib.dump(data["feature_columns"], os.path.join(models_dir, "feature_columns.pkl"))

    model_info = {
        "model_name": best_model_name,
        "metrics": best_metrics,
        "all_results": results
    }
    joblib.dump(model_info, os.path.join(models_dir, "model_info.pkl"))

    # ------------------------------------------------------------------
    # Save visual + text reports
    # ------------------------------------------------------------------
    save_comparison_chart(results, os.path.join(reports_dir, "model_comparison.png"))
    save_confusion_matrix_chart(
        y_test, best_predictions, best_model_name,
        os.path.join(reports_dir, "confusion_matrix.png")
    )
    write_text_report(results, best_model_name, os.path.join(reports_dir, "evaluation_report.txt"))

    print("[train_model] Training complete. All artifacts saved successfully.")
    return model_info


if __name__ == "__main__":
    # Allow this script to be run directly: python train_model.py
    info = train_and_save()
    print("\nFinal Summary:")
    print(f"Best Model : {info['model_name']}")
    print(f"Metrics    : {info['metrics']}")
