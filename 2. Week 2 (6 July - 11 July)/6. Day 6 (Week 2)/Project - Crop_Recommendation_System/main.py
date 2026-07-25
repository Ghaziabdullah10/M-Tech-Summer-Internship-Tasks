"""
main.py
-------
Crop Recommendation System (v2) - Desktop GUI
A clean, modern PyQt5 application that loads a trained KNN model and
recommends the best crop from soil/weather inputs.

Run with:  python3 main.py
"""

import sys
import os
import datetime
import joblib
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox,
    QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

MODEL_DIR = "models"

# Friendly growing-tip / info lookup for each crop (kept small & simple)
CROP_INFO = {
    "rice": ("Kharif (Summer)", "High", "Flood-tolerant fields, keep soil moist."),
    "maize": ("Kharif (Summer)", "Medium", "Needs well-drained loamy soil."),
    "chickpea": ("Rabi (Winter)", "Low", "Avoid waterlogging, prefers cool weather."),
    "kidneybeans": ("Rabi (Winter)", "Medium", "Needs moderate rainfall, cool climate."),
    "pigeonpeas": ("Kharif (Summer)", "Low", "Drought tolerant, deep well-drained soil."),
    "mothbeans": ("Kharif (Summer)", "Low", "Thrives in arid, sandy soils."),
    "mungbean": ("Kharif (Summer)", "Medium", "Short duration crop, warm climate."),
    "blackgram": ("Kharif (Summer)", "Medium", "Needs humid conditions, avoid frost."),
    "lentil": ("Rabi (Winter)", "Low", "Cool season crop, well-drained soil."),
    "pomegranate": ("Year-round", "Medium", "Tolerates drought once established."),
    "banana": ("Year-round", "High", "Needs rich soil and consistent watering."),
    "mango": ("Summer", "Medium", "Dry spell before flowering improves yield."),
    "grapes": ("Spring", "Medium", "Needs trellising and good drainage."),
    "watermelon": ("Summer", "Medium", "Warm sandy loam soil, full sun."),
    "muskmelon": ("Summer", "Medium", "Warm dry climate, sandy soil."),
    "apple": ("Winter/Spring", "Medium", "Needs a cold dormant period."),
    "orange": ("Winter", "Medium", "Well-drained soil, moderate irrigation."),
    "papaya": ("Year-round", "High", "Fast growing, frost sensitive."),
    "coconut": ("Year-round", "High", "Coastal/tropical climate preferred."),
    "cotton": ("Kharif (Summer)", "Medium", "Needs long frost-free period."),
    "jute": ("Kharif (Summer)", "High", "High humidity and rainfall required."),
    "coffee": ("Year-round", "Medium", "Shade-grown, cool tropical climate."),
}

# (label, key, unit, min, max, placeholder)
FIELDS = [
    ("Nitrogen (N)",   "N",           "mg/kg", 0, 150, "e.g. 90"),
    ("Phosphorus (P)", "P",           "mg/kg", 0, 150, "e.g. 42"),
    ("Potassium (K)",  "K",           "mg/kg", 0, 210, "e.g. 43"),
    ("Temperature",    "temperature", "°C",    0, 50,  "e.g. 21"),
    ("Humidity",       "humidity",    "%",     0, 100, "e.g. 82"),
    ("Soil pH",        "ph",          "pH",    0, 14,  "e.g. 6.5"),
    ("Rainfall",       "rainfall",    "mm",    0, 300, "e.g. 203"),
]

SAMPLE_DATA = {"N": 90, "P": 42, "K": 43, "temperature": 21,
               "humidity": 82, "ph": 6.5, "rainfall": 203}


class CropRecommendationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop Recommendation System (v2)")
        self.resize(1000, 680)
        self.setMinimumSize(900, 620)

        self.model = None
        self.scaler = None
        self.encoder = None
        self.inputs = {}

        self._load_model()
        self._build_ui()
        self._apply_styles()

    # ---------- model loading ----------
    def _load_model(self):
        try:
            self.model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            self.encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
        except Exception as e:
            self.model = None
            self._model_error = str(e)

    # ---------- UI construction ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(20)
        body.addWidget(self._build_input_panel(), 5)
        body.addWidget(self._build_output_panel(), 4)

        body_widget = QWidget()
        body_widget.setLayout(body)
        root.addWidget(body_widget, 1)

        root.addWidget(self._build_footer())

    def _build_header(self):
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(80)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)

        logo = QLabel("🌾")
        logo.setStyleSheet("font-size: 34px;")

        title_box = QVBoxLayout()
        title = QLabel("Crop Recommendation System")
        title.setObjectName("title")
        subtitle = QLabel("AI-powered farming decision support (KNN)")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_box.setSpacing(2)

        layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addLayout(title_box)
        layout.addStretch()

        self.datetime_label = QLabel(self._current_datetime())
        self.datetime_label.setObjectName("datetime")
        layout.addWidget(self.datetime_label)

        return header

    def _current_datetime(self):
        return datetime.datetime.now().strftime("%A, %d %B %Y  •  %I:%M %p")

    def _build_input_panel(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Soil & Weather Data")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        for i, (label_text, key, unit, lo, hi, placeholder) in enumerate(FIELDS):
            row, col = divmod(i, 2)
            box = QVBoxLayout()
            lbl = QLabel(f"{label_text} ({unit})")
            lbl.setObjectName("fieldLabel")
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            field.setToolTip(f"Valid range: {lo} - {hi} {unit}")
            field.setObjectName("inputField")
            box.addWidget(lbl)
            box.addWidget(field)
            box.setSpacing(4)
            wrapper = QWidget()
            wrapper.setLayout(box)
            grid.addWidget(wrapper, row, col)
            self.inputs[key] = (field, lo, hi, label_text)

        layout.addLayout(grid)
        layout.addStretch()

        btn_row = QHBoxLayout()
        recommend_btn = QPushButton("🌱  Recommend Crop")
        recommend_btn.setObjectName("primaryBtn")
        recommend_btn.clicked.connect(self.recommend_crop)

        sample_btn = QPushButton("Load Sample")
        sample_btn.setObjectName("secondaryBtn")
        sample_btn.clicked.connect(self.load_sample)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.clicked.connect(self.clear_inputs)

        btn_row.addWidget(recommend_btn, 2)
        btn_row.addWidget(sample_btn, 1)
        btn_row.addWidget(clear_btn, 1)
        layout.addLayout(btn_row)

        return card

    def _build_output_panel(self):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        title = QLabel("Recommendation")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        self.result_badge = QLabel("Enter data and click\n\"Recommend Crop\"")
        self.result_badge.setObjectName("resultBadge")
        self.result_badge.setAlignment(Qt.AlignCenter)
        self.result_badge.setWordWrap(True)
        self.result_badge.setMinimumHeight(110)
        layout.addWidget(self.result_badge)

        conf_label = QLabel("Confidence")
        conf_label.setObjectName("smallLabel")
        layout.addWidget(conf_label)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setObjectName("confidenceBar")
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        layout.addWidget(self.confidence_bar)

        self.details_label = QLabel("")
        self.details_label.setObjectName("detailsLabel")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        alt_title = QLabel("Alternative Suitable Crops")
        alt_title.setObjectName("smallLabel")
        layout.addWidget(alt_title)
        self.alt_label = QLabel("—")
        self.alt_label.setObjectName("altLabel")
        self.alt_label.setWordWrap(True)
        layout.addWidget(self.alt_label)

        layout.addStretch()
        return card

    def _build_footer(self):
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(36)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 0, 20, 0)
        label = QLabel("Developed by Ghazi Muhammad Abdullah  •  PyQt5 + Scikit-learn (KNN)")
        label.setObjectName("footerLabel")
        layout.addWidget(label)
        layout.addStretch()
        return footer

    # ---------- actions ----------
    def load_sample(self):
        for key, value in SAMPLE_DATA.items():
            field, *_ = self.inputs[key]
            field.setText(str(value))

    def clear_inputs(self):
        for field, *_ in self.inputs.values():
            field.clear()
        self.result_badge.setText('Enter data and click\n"Recommend Crop"')
        self.result_badge.setStyleSheet("")
        self.confidence_bar.setValue(0)
        self.details_label.setText("")
        self.alt_label.setText("—")

    def _validate_inputs(self):
        values = {}
        for key, (field, lo, hi, label_text) in self.inputs.items():
            text = field.text().strip()
            if text == "":
                raise ValueError(f"Please enter a value for {label_text}.")
            try:
                val = float(text)
            except ValueError:
                raise ValueError(f"{label_text} must be a number.")
            if val < lo or val > hi:
                raise ValueError(f"{label_text} must be between {lo} and {hi}.")
            values[key] = val
        return values

    def recommend_crop(self):
        if self.model is None:
            QMessageBox.critical(
                self, "Model Not Found",
                "Trained model files were not found in the 'models' folder.\n"
                "Please run 'python3 train_model.py' first."
            )
            return

        try:
            values = self._validate_inputs()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        x = np.array([[values[k] for k in order]])
        x_scaled = self.scaler.transform(x)

        pred = self.model.predict(x_scaled)[0]
        proba = self.model.predict_proba(x_scaled)[0]
        crop = self.encoder.inverse_transform([pred])[0]
        confidence = proba[pred] * 100

        # top-3 alternatives (excluding the top pick)
        top_idx = np.argsort(proba)[::-1]
        alternatives = []
        for idx in top_idx:
            name = self.encoder.inverse_transform([idx])[0]
            if name != crop and len(alternatives) < 3:
                alternatives.append(f"{name.title()} ({proba[idx]*100:.1f}%)")

        season, water, tip = CROP_INFO.get(crop, ("N/A", "N/A", "No tip available."))

        self.result_badge.setText(f"✅  {crop.title()}")
        self.result_badge.setStyleSheet(
            "background-color: #e6f4ea; color: #1b5e20; "
            "font-size: 26px; font-weight: 700; border-radius: 12px;"
        )
        self.confidence_bar.setValue(int(confidence))
        self.confidence_bar.setFormat(f"{confidence:.1f}%")

        self.details_label.setText(
            f"<b>Growing Season:</b> {season}<br>"
            f"<b>Water Requirement:</b> {water}<br>"
            f"<b>Growing Tip:</b> {tip}<br>"
            f"<b>Algorithm:</b> K-Nearest Neighbors<br>"
            f"<b>Prediction Time:</b> {datetime.datetime.now().strftime('%I:%M:%S %p')}"
        )
        self.alt_label.setText(", ".join(alternatives) if alternatives else "—")

    # ---------- styling ----------
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f4f8f4; }

            #header {
                background-color: #2e7d32;
            }
            #title { color: white; font-size: 20px; font-weight: 700; }
            #subtitle { color: #d7ecd9; font-size: 12px; }
            #datetime { color: #e8f5e9; font-size: 12px; }

            #footer { background-color: #1b5e20; }
            #footerLabel { color: #d7ecd9; font-size: 11px; }

            #card {
                background-color: white;
                border-radius: 14px;
                border: 1px solid #dfe7df;
            }
            #cardTitle {
                font-size: 16px;
                font-weight: 700;
                color: #1b5e20;
                padding-bottom: 4px;
            }
            #fieldLabel {
                font-size: 12px;
                color: #33413a;
                font-weight: 600;
            }
            #inputField {
                padding: 8px;
                border: 1px solid #c8d6c9;
                border-radius: 8px;
                font-size: 13px;
                background-color: #fbfdfb;
            }
            #inputField:focus {
                border: 1px solid #2e7d32;
                background-color: white;
            }

            #primaryBtn {
                background-color: #2e7d32;
                color: white;
                font-weight: 700;
                font-size: 13px;
                border-radius: 10px;
                padding: 10px;
            }
            #primaryBtn:hover { background-color: #1b5e20; }

            #secondaryBtn {
                background-color: #eef3ee;
                color: #2e7d32;
                font-weight: 600;
                font-size: 12px;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #c8d6c9;
            }
            #secondaryBtn:hover { background-color: #e0ebe0; }

            #resultBadge {
                background-color: #f1f5f1;
                border-radius: 12px;
                font-size: 15px;
                color: #607064;
                padding: 10px;
            }
            #smallLabel {
                font-size: 12px;
                font-weight: 700;
                color: #33413a;
                margin-top: 6px;
            }
            #detailsLabel {
                font-size: 12px;
                color: #33413a;
                background-color: #fafcfa;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #eef1ee;
            }
            #altLabel {
                font-size: 12px;
                color: #2e7d32;
                font-weight: 600;
            }
            #confidenceBar {
                border: 1px solid #c8d6c9;
                border-radius: 8px;
                text-align: center;
                height: 22px;
                background-color: #f1f5f1;
            }
            #confidenceBar::chunk {
                background-color: #43a047;
                border-radius: 8px;
            }
        """)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = CropRecommendationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
