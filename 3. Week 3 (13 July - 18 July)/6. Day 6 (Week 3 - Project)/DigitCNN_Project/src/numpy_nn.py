"""
numpy_nn.py
-----------
A tiny, dependency-free CNN implementation using nothing but NumPy.

Why not TensorFlow/PyTorch? Those are large packages that frequently
fail to install on brand-new Python versions, on machines without a
matching pre-built wheel, or on restricted networks. NumPy has none of
those problems and is already required for the rest of this project,
so this keeps the whole app genuinely lightweight and dependency-free.

Layers implemented: Conv2D, ReLU, MaxPool2D, Flatten, Dense.
Includes a softmax + cross-entropy loss and a simple SGD training loop.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Helper: im2col (turns convolution into one big matrix multiply)
# --------------------------------------------------------------------------- #
def im2col(X, kh, kw, stride=1):
    """X: (N, H, W, C) -> cols: (N*out_h*out_w, kh*kw*C)"""
    N, H, W, C = X.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    shape = (N, out_h, out_w, kh, kw, C)
    strides = (
        X.strides[0],
        stride * X.strides[1],
        stride * X.strides[2],
        X.strides[1],
        X.strides[2],
        X.strides[3],
    )
    patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    cols = patches.reshape(N * out_h * out_w, kh * kw * C).copy()
    return cols, out_h, out_w


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class Conv2D:
    def __init__(self, num_filters, kernel_size, in_channels, seed=None):
        rng = np.random.default_rng(seed)
        self.kh = self.kw = kernel_size
        self.num_filters = num_filters
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.W = rng.standard_normal(
            (num_filters, kernel_size, kernel_size, in_channels)
        ) * scale
        self.b = np.zeros(num_filters)

    def forward(self, X):
        N = X.shape[0]
        cols, out_h, out_w = im2col(X, self.kh, self.kw)
        self.cols = cols
        self.out_h, self.out_w = out_h, out_w
        W_col = self.W.reshape(self.num_filters, -1).T
        out = cols @ W_col + self.b
        return out.reshape(N, out_h, out_w, self.num_filters)

    def backward(self, dout, lr):
        N = dout.shape[0]
        dout_flat = dout.reshape(-1, self.num_filters)
        dW = (self.cols.T @ dout_flat).T.reshape(self.W.shape)
        db = dout_flat.sum(axis=0)
        self.W -= lr * dW / N
        self.b -= lr * db / N
        return None  # this is the first layer, no need to propagate further


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dout, lr=None):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, X):
        N, H, W, C = X.shape
        s = self.stride
        out_h, out_w = H // s, W // s
        X_crop = X[:, : out_h * s, : out_w * s, :]
        X_reshaped = X_crop.reshape(N, out_h, s, out_w, s, C)
        out = X_reshaped.max(axis=(2, 4))

        self.X_shape = X.shape
        self.out_h, self.out_w = out_h, out_w
        out_expanded = out[:, :, None, :, None, :]
        mask = (X_reshaped == out_expanded)
        counts = mask.sum(axis=(2, 4), keepdims=True)
        counts[counts == 0] = 1
        self.mask = mask
        self.counts = counts
        return out

    def backward(self, dout, lr=None):
        N, out_h, out_w, C = dout.shape
        s = self.stride
        dout_expanded = dout[:, :, None, :, None, :]
        dX_reshaped = self.mask * dout_expanded / self.counts
        dX = np.zeros(self.X_shape)
        dX[:, : out_h * s, : out_w * s, :] = dX_reshaped.reshape(
            N, out_h * s, out_w * s, C
        )
        return dX


class Flatten:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout, lr=None):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_dim, out_dim, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((in_dim, out_dim)) * scale
        self.b = np.zeros(out_dim)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout, lr):
        N = self.X.shape[0]
        dW = self.X.T @ dout
        db = dout.sum(axis=0)
        dX = dout @ self.W.T
        self.W -= lr * dW / N
        self.b -= lr * db / N
        return dX


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs, labels):
    N = probs.shape[0]
    clipped = np.clip(probs[np.arange(N), labels], 1e-12, 1.0)
    return -np.log(clipped).mean()


# --------------------------------------------------------------------------- #
# The full model
# --------------------------------------------------------------------------- #
class SimpleCNN:
    """Conv -> ReLU -> MaxPool -> Flatten -> Dense -> ReLU -> Dense -> Softmax"""

    def __init__(self, seed=42):
        self.conv = Conv2D(num_filters=8, kernel_size=3, in_channels=1, seed=seed)
        self.relu1 = ReLU()
        self.pool = MaxPool2D(size=2, stride=2)
        self.flatten = Flatten()
        self.dense1 = Dense(13 * 13 * 8, 64, seed=seed)
        self.relu2 = ReLU()
        self.dense2 = Dense(64, 10, seed=seed)

    def forward(self, X):
        out = self.conv.forward(X)
        out = self.relu1.forward(out)
        out = self.pool.forward(out)
        out = self.flatten.forward(out)
        out = self.dense1.forward(out)
        out = self.relu2.forward(out)
        out = self.dense2.forward(out)
        return softmax(out)

    def backward(self, probs, labels, lr):
        N = probs.shape[0]
        onehot = np.zeros_like(probs)
        onehot[np.arange(N), labels] = 1
        dout = (probs - onehot) / N * N  # gradient of softmax+CE wrt logits

        dout = self.dense2.backward(dout, lr)
        dout = self.relu2.backward(dout)
        dout = self.dense1.backward(dout, lr)
        dout = self.flatten.backward(dout)
        dout = self.pool.backward(dout)
        dout = self.relu1.backward(dout)
        self.conv.backward(dout, lr)

    def predict_proba(self, X):
        return self.forward(X)

    def fit(self, X, y, X_val=None, y_val=None, epochs=5, batch_size=64,
            lr=0.05, verbose=True, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            X_shuf, y_shuf = X[order], y[order]
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                probs = self.forward(xb)
                total_loss += cross_entropy(probs, yb)
                n_batches += 1
                self.backward(probs, yb, lr)

            if verbose:
                msg = f"Epoch {epoch}/{epochs} - loss: {total_loss / n_batches:.4f}"
                if X_val is not None:
                    acc = self.evaluate(X_val, y_val)
                    msg += f" - val_acc: {acc * 100:.2f}%"
                print(msg)

    def evaluate(self, X, y, batch_size=256):
        correct = 0
        for start in range(0, X.shape[0], batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            preds = self.forward(xb).argmax(axis=1)
            correct += (preds == yb).sum()
        return correct / X.shape[0]

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        np.savez_compressed(
            path,
            conv_W=self.conv.W, conv_b=self.conv.b,
            dense1_W=self.dense1.W, dense1_b=self.dense1.b,
            dense2_W=self.dense2.W, dense2_b=self.dense2.b,
        )

    def load(self, path):
        data = np.load(path)
        self.conv.W, self.conv.b = data["conv_W"], data["conv_b"]
        self.dense1.W, self.dense1.b = data["dense1_W"], data["dense1_b"]
        self.dense2.W, self.dense2.b = data["dense2_W"], data["dense2_b"]
