"""
predictor.py
------------
This module is responsible for:
    1. Loading the trained Machine Learning model and all preprocessing
       artifacts (scaler, encoders, feature column order) that were saved
       by train_model.py.
    2. Converting a single new employee's information (entered by the user
       in the GUI) into the exact numeric format the model expects.
    3. Making a prediction (Attrition: Yes/No) together with a confidence
       percentage.

The GUI (gui.py) should only ever talk to this module for predictions - it
should never touch the raw model files directly. This separation keeps the
project organised and easy to maintain.
"""

import os
import joblib
import numpy as np
import pandas as pd

from preprocess import FEATURE_COLUMNS


class ModelNotFoundError(Exception):
    """Raised when the trained model files cannot be found on disk."""
    pass


class AttritionPredictor:
    """
    A convenient wrapper class around the trained model and all the
    preprocessing objects required to transform new employee data.
    """

    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.encoders = None
        self.target_encoder = None
        self.feature_columns = FEATURE_COLUMNS
        self.model_info = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load the trained model and preprocessing objects from the models/
        folder using joblib. Raises a clear, descriptive error if any file
        is missing so the GUI can show a helpful message to the user.
        """
        required_files = {
            "model": "best_model.pkl",
            "scaler": "scaler.pkl",
            "encoders": "encoders.pkl",
            "target_encoder": "target_encoder.pkl",
            "feature_columns": "feature_columns.pkl",
            "model_info": "model_info.pkl",
        }

        for attr_name, filename in required_files.items():
            file_path = os.path.join(self.models_dir, filename)
            if not os.path.exists(file_path):
                raise ModelNotFoundError(
                    f"Required model file '{filename}' was not found in "
                    f"'{self.models_dir}'. Please train the model first by "
                    f"running train_model.py (or use the 'Train Model' menu "
                    f"option in the application)."
                )
            setattr(self, attr_name, joblib.load(file_path))

    def predict(self, employee_data):
        """
        Predict whether an employee is likely to leave the company.

        Parameters
        ----------
        employee_data : dict
            Dictionary containing one value for every column in
            FEATURE_COLUMNS. Example keys: 'Age', 'Gender', 'Department', ...

        Returns
        -------
        (str, float)
            A tuple of (label, confidence_percentage) where label is
            "Likely to Leave" or "Not Likely to Leave", and confidence is a
            percentage value between 0 and 100.
        """
        # Build a single-row DataFrame in the exact column order used during training
        row = {col: employee_data[col] for col in self.feature_columns}
        # dtype=object avoids pandas' newer strict per-column dtype inference
        # (e.g. a column being locked to a 'string' dtype), which would
        # otherwise raise a TypeError when we later overwrite a text value
        # with its encoded integer equivalent below.
        df = pd.DataFrame([row], dtype=object)

        # Encode categorical columns using the SAVED encoders (do NOT re-fit)
        for col, encoder in self.encoders.items():
            if col in df.columns:
                value = str(df.at[0, col])
                known_classes = set(encoder.classes_)
                if value not in known_classes:
                    # Fall back gracefully to the first known class instead
                    # of crashing, in case of an unexpected input value.
                    value = encoder.classes_[0]
                df.at[0, col] = encoder.transform([value])[0]

        # Ensure every column is numeric before scaling
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Scale features using the SAVED scaler (do NOT re-fit).
        # Passing a DataFrame (instead of a bare numpy array) keeps the
        # feature names consistent with how the scaler was originally
        # fitted, avoiding sklearn's "missing feature names" warning.
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        # Make the prediction
        prediction = self.model.predict(X_scaled)[0]

        # Get confidence (probability) if the model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities)) * 100
        else:
            # Fallback for models without probability estimates
            confidence = 100.0

        label_text = self.target_encoder.inverse_transform([prediction])[0]
        result_label = "Likely to Leave" if label_text == "Yes" else "Not Likely to Leave"

        return result_label, round(confidence, 2)

    def get_model_name(self):
        """Return the name of the currently loaded best model."""
        if self.model_info:
            return self.model_info.get("model_name", "Unknown Model")
        return "Unknown Model"

    def get_model_metrics(self):
        """Return the evaluation metrics dictionary of the loaded model."""
        if self.model_info:
            return self.model_info.get("metrics", {})
        return {}
