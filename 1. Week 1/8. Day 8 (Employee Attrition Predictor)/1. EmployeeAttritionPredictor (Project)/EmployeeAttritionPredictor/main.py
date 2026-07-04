"""
main.py
-------
Entry point for the Employee Attrition Predictor desktop application.

This script:
    1. Checks whether a trained Machine Learning model already exists.
    2. If not, automatically trains one (so the application always works
       the very first time it is run, straight after generating/placing
       the dataset).
    3. Launches the Tkinter GUI.

To run the whole project, simply execute this file:
    python main.py

Project: Employee Attrition Predictor
Student : Ghazi Muhammad Abdullah
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Ensure the project's own folder is on the Python path so that the local
# modules (gui, predictor, preprocess, train_model) can always be imported,
# regardless of which folder the script is launched from in PyCharm.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "employee_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")


def ensure_dataset_exists():
    """
    Make sure a dataset CSV file is available. If it is missing, generate a
    synthetic one automatically using generate_dataset.py so the project can
    run without any manual setup steps.
    """
    if not os.path.exists(DATASET_PATH):
        print("[main] Dataset not found. Generating a synthetic dataset...")
        from generate_dataset import main as generate_main
        generate_main()


def ensure_model_exists():
    """
    Make sure a trained model is available. If it is missing, train one
    automatically by calling train_model.train_and_save().
    """
    if not os.path.exists(MODEL_PATH):
        print("[main] No trained model found. Training a new model now. "
              "This may take a short while...")
        from train_model import train_and_save
        train_and_save()
        print("[main] Model training complete.")


def main():
    """Main entry point: prepare data/model, then launch the GUI."""
    try:
        ensure_dataset_exists()
        ensure_model_exists()
    except Exception as e:
        # If anything goes wrong during automatic setup, still try to open
        # the GUI so the user can see a friendly error and use the
        # Model > Train / Retrain Model menu option manually.
        print(f"[main] Warning: automatic setup failed: {e}")

    from gui import EmployeeAttritionApp

    root = tk.Tk()
    try:
        app = EmployeeAttritionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_exit)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"The application encountered a fatal error:\n{e}")
        raise


if __name__ == "__main__":
    main()
