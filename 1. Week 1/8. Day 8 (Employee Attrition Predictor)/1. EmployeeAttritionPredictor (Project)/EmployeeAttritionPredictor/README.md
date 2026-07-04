# Employee Attrition Predictor

**A Machine Learning based Python Desktop Application**

| | |
|---|---|
| **Student Name** | Ghazi Muhammad Abdullah |
| **Project Title** | Employee Attrition Predictor |
| **Course** | Artificial Intelligence (University Project) |
| **Language** | Python 3 |
| **GUI Framework** | Tkinter (ttk) |
| **ML Library** | scikit-learn |

---

## 1. Project Overview

The **Employee Attrition Predictor** is a complete, self-contained Python
desktop application that predicts whether an employee is **likely to leave**
a company based on information such as age, department, monthly income, job
satisfaction, overtime status, and more.

The project demonstrates a full, real-world Machine Learning workflow:

1. Data loading and cleaning
2. Missing value handling and duplicate removal
3. Feature encoding and scaling
4. Training and comparing **5 different classification algorithms**
5. Automatically selecting the best-performing model
6. Saving/loading the model with **Joblib**
7. A professional **Tkinter** GUI for interactive predictions
8. Prediction history logging, searching, and CSV export
9. Visual performance reports (bar charts, confusion matrix)

This is a pure **desktop** application. It does **not** use any web
technology (no Flask/Django/HTML/JavaScript) - everything runs locally as a
native Tkinter window.

---

## 2. Machine Learning Algorithms Compared

The application trains and evaluates the following five algorithms on the
employee dataset, then automatically keeps the best one:

| # | Algorithm |
|---|-----------|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | Random Forest |
| 4 | K-Nearest Neighbors (KNN) |
| 5 | Support Vector Machine (SVM) |

Each model is evaluated using **Accuracy, Precision, Recall, and F1-Score**.
The model with the highest **F1-Score** on the held-out test set is selected
as the "best model" and saved for use inside the GUI.

---

## 3. Project Structure

```
EmployeeAttritionPredictor/
│
├── dataset/
│   └── employee_data.csv          # Employee dataset (synthetic, see Section 6)
│
├── models/                        # Created automatically after training
│   ├── best_model.pkl             # The winning ML model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── encoders.pkl               # Fitted LabelEncoders (per categorical column)
│   ├── target_encoder.pkl         # Fitted LabelEncoder for the target (Attrition)
│   ├── feature_columns.pkl        # Ordered list of feature column names
│   └── model_info.pkl             # Best model name + evaluation metrics
│
├── reports/                       # Created automatically after training
│   ├── model_comparison.png       # Bar chart comparing all 5 algorithms
│   ├── confusion_matrix.png       # Confusion matrix of the best model
│   ├── evaluation_report.txt      # Text summary of all metrics
│   └── prediction_history.csv     # Saved history of predictions made in the GUI
│
├── assets/
│   └── README.txt                 # Notes on adding a custom logo image
│
├── screenshots/
│   └── README.txt                 # Suggested screenshots for your report
│
├── generate_dataset.py            # Generates the synthetic employee dataset
├── preprocess.py                  # Data cleaning, encoding, scaling pipeline
├── train_model.py                 # Trains & compares all 5 ML models, saves the best
├── predictor.py                   # Loads the saved model and makes predictions
├── gui.py                         # Full Tkinter GUI application
├── main.py                        # Application entry point (run this file)
├── requirements.txt                # Python library dependencies
└── README.md                      # This file
```

---

## 4. How to Run the Project (PyCharm on Windows)

### Step 1 - Open the project in PyCharm
1. Open **PyCharm**.
2. Click **File > Open...** and select the `EmployeeAttritionPredictor` folder.
3. Wait for PyCharm to index the project.

### Step 2 - Set up a Python interpreter
1. Go to **File > Settings > Project: EmployeeAttritionPredictor > Python Interpreter**.
2. Make sure a Python 3.9+ interpreter is selected (create a virtual
   environment if you don't already have one).

### Step 3 - Install the required libraries
Open the PyCharm **Terminal** (bottom toolbar) and run:

```bash
pip install -r requirements.txt
```

> **Note about Tkinter:** Tkinter comes pre-installed with the standard
> Python installer on Windows, so you normally do not need to install it
> separately. If you ever see `ModuleNotFoundError: No module named 'tkinter'`
> on Windows, simply re-run the official Python installer and make sure the
> **"tcl/tk and IDLE"** option is checked.

### Step 4 - Run the application
Right-click **`main.py`** in the Project panel and choose **Run 'main'**
(or open `main.py` and press `Shift + F10`).

The very first time you run the project:
- If `dataset/employee_data.csv` does not exist, it will be generated automatically.
- If no trained model exists yet in `models/`, all 5 algorithms will be
  trained automatically (this takes a few seconds) and the best one will be
  saved before the GUI window opens.

On every subsequent run, the saved model is loaded instantly and the GUI
opens immediately.

---

## 5. Using the Application

### Prediction Tab
1. Fill in the employee's details using the text boxes, dropdown menus,
   spin boxes, and radio buttons provided.
2. Click **Predict**. The app will validate your inputs, run the trained
   model, and display:
   - **Likely to Leave** (shown in red) or **Not Likely to Leave** (shown in green)
   - A **confidence percentage**
3. Click **Reset** to clear the form, or **Exit** to close the application.

### History Tab
- Every prediction you make is automatically saved (with a timestamp) to
  `reports/prediction_history.csv`.
- Use the **search box** to filter past predictions by any value (name of a
  department, job role, Yes/No, etc.).
- Use **Clear History** to erase all saved predictions, or use
  **File > Export History to CSV...** from the menu bar to save a copy
  elsewhere on your computer.

### Graphs Tab
- **View Model Comparison Chart** - shows a bar chart comparing Accuracy,
  Precision, Recall, and F1-Score across all 5 trained algorithms.
- **View Confusion Matrix** - shows the confusion matrix of the winning model.
- **View Evaluation Report (Text)** - shows a plain-text summary of all metrics.

### Menu Bar
- **File** - Export history, Exit
- **Model** - Train/Retrain the model, Reload the saved model, View model info
- **View** - Quickly jump between the three tabs
- **Help** - Help guide and About dialog (project & student information)

---

## 6. About the Dataset

`dataset/employee_data.csv` is a **synthetic** dataset generated by
`generate_dataset.py`. It contains 1,500+ employee records with realistic,
logically-consistent relationships (for example, employees who work
overtime, have low job/environment satisfaction, or a low salary have a
higher simulated probability of leaving), plus a handful of intentionally
inserted missing values and duplicate rows so the data-cleaning code in
`preprocess.py` has genuine work to do.

**Columns included:**
`Age, Gender, Department, JobRole, MonthlyIncome, Education, BusinessTravel,
JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, OverTime,
YearsAtCompany, TotalWorkingYears, DistanceFromHome, PerformanceRating,
MaritalStatus, Attrition`

### Using a real-world dataset instead
If you would prefer to train on a real dataset such as the publicly
available **IBM HR Analytics Employee Attrition & Performance** dataset:

1. Download it from Kaggle (search "IBM HR Analytics Employee Attrition Dataset").
2. Rename the downloaded file to `employee_data.csv`.
3. Make sure it contains (at minimum) the same column names listed above -
   rename columns if needed to match.
4. Place it inside the `dataset/` folder, replacing the existing file.
5. Delete the `models/` folder contents (or just run **Model > Train /
   Retrain Model** from the GUI menu) so the app retrains on the new data.

No other code changes are required - `preprocess.py`, `train_model.py`,
`predictor.py`, and `gui.py` all work with any dataset that has these column
names.

---

## 7. Regenerating or Retraining

You can also run these steps individually and directly from the terminal,
if you would rather not rely on the automatic setup inside `main.py`:

```bash
# Step 1: (Re)generate the synthetic dataset
python generate_dataset.py

# Step 2: Train all 5 models and save the best one
python train_model.py

# Step 3: Launch the application
python main.py
```

---

## 8. Error Handling & Input Validation

The application validates every input field before making a prediction, for example:
- Age must be a whole number between 18 and 65.
- Monthly Income must be a positive number.
- Total Working Years cannot be less than Years At Company.
- Every dropdown/field must have a value selected before predicting.

Friendly error message boxes are shown whenever validation fails, and all
file operations (reading/writing the dataset, model files, or history CSV)
are wrapped in `try/except` blocks with clear error messages instead of the
program crashing.

---

## 9. Code Quality Notes

- Written entirely in **Python**, following **PEP 8** style guidelines.
- Every module and function includes **docstrings and inline comments**
  explaining what it does and why.
- No placeholder or "TODO" functions - every feature described in this
  README is fully implemented and working.
- The project is organised into clearly separated modules
  (`preprocess.py`, `train_model.py`, `predictor.py`, `gui.py`, `main.py`)
  following good software engineering practice, rather than one giant script.

---

## 10. Requirements

See `requirements.txt`. Summary of libraries used:

- `tkinter` / `ttk` (standard library - GUI)
- `pandas`, `numpy` (data handling)
- `scikit-learn` (Machine Learning models & preprocessing)
- `matplotlib`, `seaborn` (charts and visual reports)
- `joblib` (saving/loading trained models)
- `pillow` (displaying chart images inside the GUI)
- `os`, `csv`, `datetime` (standard library - file handling & timestamps)

---

## 11. Credits

Developed by **Ghazi Muhammad Abdullah** as a university
Artificial Intelligence course project, demonstrating Python programming,
Machine Learning, GUI development with Tkinter, file handling, and data
analysis.
