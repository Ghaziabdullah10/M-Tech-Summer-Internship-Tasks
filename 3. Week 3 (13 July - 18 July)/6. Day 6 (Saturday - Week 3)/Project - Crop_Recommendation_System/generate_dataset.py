"""
generate_dataset.py
--------------------
Generates a realistic synthetic crop recommendation dataset
(N, P, K, temperature, humidity, ph, rainfall -> crop label).

Each crop has its own typical parameter range (based on real agronomic
knowledge). Random samples are drawn from those ranges with a bit of
noise so the dataset behaves like real-world sensor data.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# crop: (N range, P range, K range, temp range, humidity range, ph range, rainfall range)
CROPS = {
    "rice":        ((60, 100), (30, 60), (30, 50), (20, 27), (75, 90), (5.5, 7.0), (150, 250)),
    "maize":       ((60, 100), (35, 65), (15, 25), (18, 27), (55, 75), (5.8, 7.2), (60, 110)),
    "chickpea":    ((20, 45),  (55, 80), (75, 95), (17, 25), (14, 25), (6.0, 8.0), (60, 100)),
    "kidneybeans": ((15, 30),  (55, 80), (15, 25), (15, 22), (18, 25), (5.5, 6.5), (60, 110)),
    "pigeonpeas":  ((15, 40),  (55, 80), (15, 25), (18, 32), (30, 65), (5.5, 7.0), (100, 200)),
    "mothbeans":   ((15, 30),  (35, 60), (15, 25), (24, 34), (40, 65), (4.5, 6.5), (40, 65)),
    "mungbean":    ((15, 30),  (35, 60), (15, 25), (25, 32), (75, 90), (6.0, 7.5), (40, 65)),
    "blackgram":   ((30, 55),  (55, 80), (15, 25), (25, 35), (60, 75), (6.0, 7.5), (60, 90)),
    "lentil":      ((15, 30),  (55, 80), (15, 25), (18, 27), (60, 75), (5.5, 7.0), (40, 70)),
    "pomegranate": ((10, 25),  (10, 25), (30, 45), (18, 25), (85, 95), (5.5, 7.0), (35, 110)),
    "banana":      ((90, 120), (70, 95), (45, 55), (25, 31), (75, 85), (5.5, 6.8), (90, 150)),
    "mango":       ((15, 30),  (15, 30), (25, 40), (27, 37), (45, 55), (5.5, 7.0), (60, 100)),
    "grapes":      ((15, 30),  (120,145),(190,205),(15, 25), (80, 85), (6.0, 7.0), (60, 80)),
    "watermelon":  ((90,110),  (10, 25), (45, 55), (24, 32), (55, 70), (6.0, 7.0), (40, 60)),
    "muskmelon":   ((90,110),  (10, 25), (45, 55), (25, 32), (85, 95), (6.0, 7.0), (25, 40)),
    "apple":       ((15, 30),  (120,145),(190,205),(20, 24), (85, 95), (5.5, 6.8), (100, 130)),
    "orange":      ((10, 25),  (10, 25), (10, 25), (22, 32), (85, 95), (6.0, 7.5), (100, 130)),
    "papaya":      ((30, 70),  (45, 70), (45, 60), (23, 35), (85, 95), (6.0, 7.0), (40, 150)),
    "coconut":     ((15, 30),  (10, 25), (25, 40), (25, 32), (85, 95), (5.5, 7.0), (130, 230)),
    "cotton":      ((100,140), (35, 60), (15, 25), (22, 30), (70, 85), (5.5, 7.0), (60, 100)),
    "jute":        ((60, 100), (35, 60), (35, 45), (23, 28), (70, 90), (6.0, 7.0), (150, 200)),
    "coffee":      ((90,120),  (15, 30), (25, 40), (18, 28), (50, 70), (6.0, 7.0), (150, 200)),
}

ROWS_PER_CROP = 100
records = []

for crop, (n_r, p_r, k_r, t_r, h_r, ph_r, r_r) in CROPS.items():
    for _ in range(ROWS_PER_CROP):
        n = np.random.uniform(*n_r)
        p = np.random.uniform(*p_r)
        k = np.random.uniform(*k_r)
        t = np.random.uniform(*t_r)
        h = np.random.uniform(*h_r)
        ph = np.random.uniform(*ph_r)
        rain = np.random.uniform(*r_r)
        records.append([round(n, 1), round(p, 1), round(k, 1), round(t, 2),
                         round(h, 2), round(ph, 2), round(rain, 2), crop])

df = pd.DataFrame(records, columns=[
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"
])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv("dataset/crop_data.csv", index=False)

print(f"Dataset generated: {len(df)} rows, {df['label'].nunique()} crops")
print(df.head())
