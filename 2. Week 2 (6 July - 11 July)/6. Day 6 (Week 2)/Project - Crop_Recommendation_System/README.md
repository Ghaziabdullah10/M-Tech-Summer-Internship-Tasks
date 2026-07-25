# 🌾 Crop Recommendation System (v2)

A simple, clean desktop app that recommends the best crop to plant based on
soil nutrients (N, P, K), temperature, humidity, soil pH, and rainfall —
using a **K-Nearest Neighbors (KNN)** classifier and a **PyQt5** GUI.

**Student:** Ghazi Muhammad Abdullah
**Stack:** Python 3 + Scikit-learn + PyQt5

---

## 📁 Project Structure

```
Crop_Recommendation_System/
├── dataset/
│   └── crop_data.csv        # training data (22 crops)
├── models/
│   ├── crop_model.pkl       # trained KNN model
│   ├── scaler.pkl           # StandardScaler
│   └── encoder.pkl          # LabelEncoder
├── generate_dataset.py      # creates the synthetic dataset
├── train_model.py           # trains + evaluates + saves the model
├── main.py                  # the PyQt5 GUI app (run this)
├── requirements.txt
└── README.md
```

## ⚙️ Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional — already done) Regenerate the dataset:
   ```
   python3 generate_dataset.py
   ```

3. (Optional — already done) Train the model:
   ```
   python3 train_model.py
   ```
   This prints the accuracy and saves `models/crop_model.pkl`,
   `models/scaler.pkl`, and `models/encoder.pkl`.

4. Run the app:
   ```
   python3 main.py
   ```

## 🖥️ Using the App

1. Enter Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH,
   and Rainfall — or click **"Load Sample"** to autofill example values.
2. Click **"Recommend Crop"**.
3. The app shows:
   - The recommended crop
   - Confidence percentage
   - Top 3 alternative crops
   - Growing season, water requirement, and a growing tip

Click **"Clear"** to reset all fields.

## 🧠 Model Details

- **Algorithm:** K-Nearest Neighbors (weighted, best K auto-selected via 5-fold cross-validation)
- **Features:** N, P, K, temperature, humidity, pH, rainfall (scaled with `StandardScaler`)
- **Crops covered:** 22 crops (rice, maize, cotton, banana, mango, coffee, etc.)
- **Test Accuracy:** ~98%

## 📝 Notes

- The dataset is synthetically generated from realistic agronomic ranges
  for each crop (see `generate_dataset.py`), since it's meant for learning
  purposes. You can swap in the real Kaggle "Crop Recommendation Dataset"
  by replacing `dataset/crop_data.csv` with the same column names
  (`N, P, K, temperature, humidity, ph, rainfall, label`) and re-running
  `train_model.py`.
- All inputs are validated (range + type checked) before prediction, so
  the app won't crash on bad input — it shows a friendly warning instead.
