"""
model.py
--------
Loads the bundled MNIST subset (data/mnist_subset.npz - ships with this
project, no internet required) and trains/loads the pure-NumPy CNN
defined in numpy_nn.py.
"""

import os
import numpy as np

from numpy_nn import SimpleCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "mnist_subset.npz")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_weights.npz")


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Bundled dataset not found at: {DATA_PATH}\n"
            "Make sure the 'data' folder was extracted alongside 'src'."
        )
    data = np.load(DATA_PATH)
    X_train = data["X_train"].astype("float32") / 255.0
    y_train = data["y_train"].astype("int64")
    X_test = data["X_test"].astype("float32") / 255.0
    y_test = data["y_test"].astype("int64")

    # add channel dimension -> (N, 28, 28, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, y_train, X_test, y_test


def train_and_save_model():
    print("No saved model found - training a small CNN (pure NumPy, no")
    print("TensorFlow/PyTorch needed) on the bundled MNIST subset...")
    print("This uses data already included in this project, so no")
    print("internet connection is required. Expect ~30-90 seconds.\n")

    X_train, y_train, X_test, y_test = load_dataset()

    model = SimpleCNN(seed=42)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test,
              epochs=6, batch_size=64, lr=0.08)

    final_acc = model.evaluate(X_test, y_test)
    print(f"\nTraining finished. Test accuracy: {final_acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}\n")
    return model


def get_model():
    """Load the cached model, or train + cache one if it doesn't exist yet."""
    model = SimpleCNN(seed=42)
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading saved model...")
            model.load(MODEL_PATH)
            return model
        except Exception as e:
            print(f"Could not load saved model ({e}). Retraining...\n")
            os.remove(MODEL_PATH)
            return train_and_save_model()
    return train_and_save_model()
