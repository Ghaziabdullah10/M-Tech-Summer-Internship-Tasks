"""
train_model.py
--------------
Loads dataset/crop_data.csv, cleans it, scales the features, trains a
K-Nearest Neighbors classifier (with a small grid search for the best K),
evaluates it, and saves the model + scaler + label encoder to /models.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "dataset/crop_data.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_and_clean(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Loaded {before} rows -> {len(df)} rows after cleaning "
          f"(duplicates/missing removed)")
    return df


def main():
    df = load_and_clean(DATA_PATH)

    X = df[FEATURES].values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- find the best K (simple grid search over odd K values) ---
    best_k, best_score = 3, -1
    for k in range(3, 16, 2):
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_k, best_score = k, mean_score
    print(f"Best K found: {best_k} (CV accuracy: {best_score:.4f})")

    # --- train the final model ---
    model = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # --- save artifacts ---
    joblib.dump(model, "models/crop_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    print("Saved: models/crop_model.pkl, models/scaler.pkl, models/encoder.pkl")


if __name__ == "__main__":
    main()
