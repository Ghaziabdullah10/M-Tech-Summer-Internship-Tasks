"""
preprocess.py
-------------
This module contains all the Data Preprocessing logic for the
Employee Attrition Predictor project.

Responsibilities of this module:
    1. Load the raw dataset from a CSV file.
    2. Clean the data (handle missing values, remove duplicate rows).
    3. Encode categorical (text) columns into numeric values.
    4. Scale numeric features so that all features contribute equally.
    5. Split the data into training and testing sets.

The same encoders and scaler that are FIT during training are reused
(without re-fitting) whenever the GUI needs to transform new employee
information entered by the user, which guarantees that training data and
new prediction data are processed in exactly the same way.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Column definitions used throughout the whole project.
# Keeping these names in one place avoids typing mistakes elsewhere.
# ---------------------------------------------------------------------------

# Columns that hold text categories and must be Label-Encoded
CATEGORICAL_COLUMNS = [
    "Gender", "Department", "JobRole", "BusinessTravel",
    "OverTime", "MaritalStatus"
]

# Columns that are already numeric (including ordinal ratings such as
# Education, JobSatisfaction, etc. which are stored as integers 1-5)
NUMERIC_COLUMNS = [
    "Age", "MonthlyIncome", "Education", "JobSatisfaction",
    "EnvironmentSatisfaction", "WorkLifeBalance", "YearsAtCompany",
    "TotalWorkingYears", "DistanceFromHome", "PerformanceRating"
]

# Full ordered list of feature columns fed into the Machine Learning models.
# The ORDER here must always match the order used when building a prediction
# row inside predictor.py, otherwise the model will misinterpret the inputs.
FEATURE_COLUMNS = [
    "Age", "Gender", "Department", "JobRole", "MonthlyIncome", "Education",
    "BusinessTravel", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "OverTime", "YearsAtCompany", "TotalWorkingYears",
    "DistanceFromHome", "PerformanceRating", "MaritalStatus"
]

TARGET_COLUMN = "Attrition"


def load_dataset(csv_path):
    """
    Load the employee dataset from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV file.

    Returns
    -------
    pandas.DataFrame
        The raw, unprocessed dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{csv_path}'. "
            f"Please run generate_dataset.py first, or place a valid "
            f"employee_data.csv file inside the dataset/ folder."
        )
    df = pd.read_csv(csv_path)
    return df


def clean_data(df):
    """
    Clean the raw dataset:
        - Remove exact duplicate rows.
        - Fill missing numeric values with the column median.
        - Fill missing categorical values with the column mode (most frequent).
        - Drop rows where the target column (Attrition) itself is missing,
          because such rows cannot be used for supervised learning.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw dataset.

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset with no missing values and no duplicate rows.
    """
    df = df.copy()

    # Drop rows that have no target label at all
    df = df.dropna(subset=[TARGET_COLUMN])

    # Remove duplicate rows (keep the first occurrence)
    before_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before_rows - len(df)

    # Fill missing values for numeric columns using the median (robust to outliers)
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Fill missing values for categorical columns using the most frequent value
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    # Ensure numeric columns really are numeric types after filling
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    print(f"[preprocess] Removed {removed_duplicates} duplicate rows during cleaning.")
    return df.reset_index(drop=True)


def encode_features(df, encoders=None, fit=True):
    """
    Convert categorical text columns into numeric values using LabelEncoder.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset (already cleaned) containing the categorical columns.
    encoders : dict or None
        Dictionary of already-fitted {column_name: LabelEncoder}. Required
        when fit=False (i.e. when transforming new prediction data).
    fit : bool
        If True, a NEW LabelEncoder is fitted for each categorical column
        (used during training). If False, the provided encoders are reused
        to transform new data without re-fitting (used during prediction).

    Returns
    -------
    (pandas.DataFrame, dict)
        The dataframe with categorical columns replaced by numeric codes,
        and the dictionary of encoders used (so they can be saved/reused).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue

        if fit:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
        else:
            encoder = encoders[col]
            # Handle unseen categories gracefully by mapping them to the
            # most common (first) known class instead of crashing.
            known_classes = set(encoder.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    return df, encoders


def encode_target(df, target_encoder=None, fit=True):
    """
    Encode the target column (Attrition: Yes/No) into 1/0.

    Returns
    -------
    (pandas.Series, LabelEncoder)
    """
    df = df.copy()
    if fit:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df[TARGET_COLUMN].astype(str))
    else:
        y = target_encoder.transform(df[TARGET_COLUMN].astype(str))
    return y, target_encoder


def scale_features(X, scaler=None, fit=True):
    """
    Scale numeric feature values using StandardScaler so that every feature
    has a mean of 0 and a standard deviation of 1. This step is important
    for distance-based algorithms such as KNN and SVM.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to be scaled.
    scaler : sklearn.preprocessing.StandardScaler or None
        An already-fitted scaler (required when fit=False).
    fit : bool
        Whether to fit a new scaler (training) or reuse an existing one
        (prediction).

    Returns
    -------
    (numpy.ndarray, StandardScaler)
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and testing sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def full_training_pipeline(csv_path):
    """
    Convenience function that runs the ENTIRE preprocessing pipeline used
    during model training, from loading the raw CSV all the way to a
    train/test split ready to be fed into Machine Learning models.

    Returns
    -------
    dict
        A dictionary containing everything train_model.py needs:
        X_train, X_test, y_train, y_test, encoders, target_encoder,
        scaler and the feature column order.
    """
    # Step 1: Load raw data
    raw_df = load_dataset(csv_path)

    # Step 2: Clean data (missing values + duplicates)
    clean_df = clean_data(raw_df)

    # Step 3: Encode categorical feature columns
    encoded_df, encoders = encode_features(clean_df, fit=True)

    # Step 4: Encode the target column (Attrition -> 1/0)
    y, target_encoder = encode_target(encoded_df, fit=True)

    # Step 5: Build the feature matrix in the correct column order
    X = encoded_df[FEATURE_COLUMNS]

    # Step 6: Scale numeric features
    X_scaled, scaler = scale_features(X, fit=True)

    # Step 7: Train/test split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "clean_df": clean_df
    }
