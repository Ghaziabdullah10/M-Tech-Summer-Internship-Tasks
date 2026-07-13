"""
generate_dataset.py
--------------------
Utility script used ONLY during project setup to create a realistic, synthetic
employee dataset (dataset/employee_data.csv) for the Employee Attrition
Predictor project.

Why a synthetic dataset?
Publicly available HR attrition datasets (such as the well-known IBM HR
Analytics Employee Attrition dataset) can also be used instead of this file.
Simply replace 'dataset/employee_data.csv' with a real dataset that has the
same column names, and the rest of the project (preprocess.py, train_model.py,
predictor.py, gui.py) will work without any changes.

This script builds 1500 employee records using randomised but logically
consistent rules so that the trained Machine Learning models produce
meaningful, explainable results (e.g. employees who work overtime, have low
job satisfaction, or low income are more likely to leave).

Run this file once (python generate_dataset.py) to (re)create the dataset.
"""

import numpy as np
import pandas as pd
import os

# Fix the random seed so the dataset is reproducible every time it is generated
np.random.seed(42)

# Number of synthetic employee records to generate
NUM_RECORDS = 1500

# ---------------------------------------------------------------------------
# Define possible categorical values for each column
# ---------------------------------------------------------------------------
genders = ["Male", "Female"]
departments = ["Sales", "Research & Development", "Human Resources"]
job_roles = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]
education_levels = ["Below College", "College", "Bachelor", "Master", "Doctor"]
business_travels = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
marital_statuses = ["Single", "Married", "Divorced"]
overtime_options = ["Yes", "No"]

# Mapping of education label -> numeric ordinal value (1 to 5)
education_map = {label: idx + 1 for idx, label in enumerate(education_levels)}


def generate_employee_records(n):
    """Generate n synthetic employee records as a list of dictionaries."""
    records = []
    for _ in range(n):
        age = int(np.clip(np.random.normal(37, 9), 18, 60))
        gender = np.random.choice(genders)
        department = np.random.choice(departments, p=[0.45, 0.45, 0.10])
        job_role = np.random.choice(job_roles)
        monthly_income = int(np.clip(np.random.normal(6500, 4000), 1000, 20000))
        education_label = np.random.choice(education_levels, p=[0.10, 0.25, 0.35, 0.20, 0.10])
        education = education_map[education_label]
        business_travel = np.random.choice(business_travels, p=[0.10, 0.65, 0.25])
        job_satisfaction = np.random.randint(1, 5)          # 1 (Low) - 4 (Very High)
        environment_satisfaction = np.random.randint(1, 5)  # 1 (Low) - 4 (Very High)
        work_life_balance = np.random.randint(1, 5)         # 1 (Bad) - 4 (Best)
        overtime = np.random.choice(overtime_options, p=[0.30, 0.70])
        years_at_company = int(np.clip(np.random.exponential(5), 0, 40))
        total_working_years = int(np.clip(years_at_company + np.random.randint(0, 10), 0, 40))
        distance_from_home = int(np.clip(np.random.exponential(8), 1, 30))
        performance_rating = np.random.choice([3, 4], p=[0.85, 0.15])  # 3 (Excellent) - 4 (Outstanding)
        marital_status = np.random.choice(marital_statuses, p=[0.35, 0.50, 0.15])

        # ------------------------------------------------------------------
        # Build a probability of attrition (leaving the company) based on a
        # weighted combination of realistic risk factors. This keeps the
        # dataset logically consistent instead of pure random noise, which
        # allows the ML models to learn genuine patterns.
        # ------------------------------------------------------------------
        risk_score = 0.0
        risk_score += 0.20 if overtime == "Yes" else 0.0
        risk_score += (4 - job_satisfaction) * 0.07
        risk_score += (4 - environment_satisfaction) * 0.06
        risk_score += (4 - work_life_balance) * 0.06
        risk_score += 0.15 if monthly_income < 3000 else 0.0
        risk_score += 0.10 if age < 25 else 0.0
        risk_score += 0.10 if years_at_company < 2 else 0.0
        risk_score += 0.08 if business_travel == "Travel_Frequently" else 0.0
        risk_score += 0.05 if distance_from_home > 20 else 0.0
        risk_score += 0.05 if marital_status == "Single" else 0.0

        # Add a bit of random noise so the relationship is not perfectly linear
        risk_score += np.random.normal(0, 0.08)
        risk_score = np.clip(risk_score, 0, 1)

        attrition = "Yes" if np.random.rand() < risk_score else "No"

        records.append({
            "Age": age,
            "Gender": gender,
            "Department": department,
            "JobRole": job_role,
            "MonthlyIncome": monthly_income,
            "Education": education,
            "BusinessTravel": business_travel,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": environment_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "OverTime": overtime,
            "YearsAtCompany": years_at_company,
            "TotalWorkingYears": total_working_years,
            "DistanceFromHome": distance_from_home,
            "PerformanceRating": performance_rating,
            "MaritalStatus": marital_status,
            "Attrition": attrition
        })
    return records


def main():
    """Generate the dataset and save it as a CSV file inside dataset/ folder."""
    records = generate_employee_records(NUM_RECORDS)
    df = pd.DataFrame(records)

    # Intentionally introduce a handful of missing values and duplicate rows
    # so that the data-cleaning code in preprocess.py has real work to do.
    missing_indices = np.random.choice(df.index, size=25, replace=False)
    for idx in missing_indices:
        col = np.random.choice(["MonthlyIncome", "TotalWorkingYears", "DistanceFromHome"])
        df.loc[idx, col] = np.nan

    duplicate_rows = df.sample(10, random_state=1)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "employee_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully with {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
