# Handwritten Digit CNN (v2)

Digitizing forms needs accuracy. This app gives you a lightweight GUI
where you draw a digit with your mouse and a CNN predicts it.

**Tech:** Tkinter (GUI, built into Python) + a small CNN implemented in
**pure NumPy** — no TensorFlow, no PyTorch, no internet connection
required.

## Why no TensorFlow?

The previous version used TensorFlow, which failed to install on some
systems (in particular, TensorFlow does not yet publish wheels for
every new Python release — e.g. Python 3.14 — so `pip install
tensorflow` can fail with "no matching distribution found"). This
version replaces it with a small CNN written from scratch in NumPy, so
there's nothing version-sensitive to install: if `numpy` and `Pillow`
work on your machine (they work almost everywhere), this project works.

The training data (a balanced ~15,000-image subset of real MNIST) ships
inside the `data/` folder, so training doesn't need internet access
either.

## Project structure

```
DigitCNN_Project/
├── run.py                 # <- start here: python run.py
├── check_setup.py         # <- run this FIRST if something breaks
├── requirements.txt
├── README.md
├── data/
│   └── mnist_subset.npz    # bundled training data (no download needed)
├── models/                 # trained weights get cached here automatically
└── src/
    ├── __init__.py
    ├── numpy_nn.py         # the CNN itself (Conv2D, ReLU, MaxPool, Dense...)
    ├── model.py            # loads data / trains / caches / loads the model
    ├── gui.py              # Tkinter canvas + main window
    └── main.py             # wires model + GUI together
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Only two packages are needed: `numpy` and `Pillow`. Both install
reliably on virtually any Python version (3.8 through 3.14+).

## Run

```bash
python run.py
```

- **First run:** no saved weights exist yet, so the app trains the CNN
  on the bundled dataset. This takes well under a minute on a normal
  CPU and reaches roughly 95-97% test accuracy, then saves the weights
  to `models/cnn_weights.npz`.
- **Every run after that:** the saved weights load instantly (under a
  tenth of a second) and the GUI opens right away.

## Using the app

1. Draw a digit (0-9) on the black canvas with your mouse.
2. Click **Predict** to see the predicted digit and the model's
   confidence.
3. Click **Clear** to erase the canvas and draw again.

## Troubleshooting

Start with:

```bash
python check_setup.py
```

This checks your Python version, tkinter, numpy, Pillow, and the
bundled dataset individually, and tells you exactly what to fix.

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'tkinter'` (Linux only) | `sudo apt-get install python3-tk` |
| `ModuleNotFoundError: No module named 'PIL'` | `pip install Pillow` |
| `ModuleNotFoundError: No module named 'numpy'` | `pip install numpy` |
| "Bundled dataset not found" | Make sure you extracted the **whole** zip, including the `data/` folder, not just the `.py` files |
| Want to retrain from scratch | Delete `models/cnn_weights.npz` and run `python run.py` again |
| App opens but window looks tiny/huge | Display-scaling quirk of Tkinter on some systems — resize the window, it won't affect functionality |

## Notes

- This was verified end-to-end before delivery: real training run
  (96.5% test accuracy in ~40 seconds), real weight save/load, and a
  full simulated draw → predict → clear cycle in the actual GUI code.
- Want higher accuracy later? `src/model.py` and `src/numpy_nn.py` are
  small and readable — it's easy to add more filters, another conv
  layer, or swap in the full 60,000-image MNIST set.
