"""Tải và tiền xử lý dữ liệu (MNIST, Fashion-MNIST).  Phụ trách: Uyên

Quy ước đầu ra: X float32 shape (N, 784) trong [0, 1]; y int64 shape (N,).
"""

import numpy as np

from .utils import SEED

DATA_DIR = "data"


def load_raw(name: str = "mnist"):
    """Tải dữ liệu gốc. Trả về (X_train, y_train, X_test, y_test), X uint8 shape (N, 28, 28)."""
    from torchvision import datasets

    cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[name]
    train = cls(DATA_DIR, train=True, download=True)
    test = cls(DATA_DIR, train=False, download=True)
    return (
        train.data.numpy(),
        train.targets.numpy(),
        test.data.numpy(),
        test.targets.numpy(),
    )


def normalize(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float32) / 255.0


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def train_val_split(X, y, val_size: int = 10_000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def load_dataset(name: str = "mnist"):
    """Hàm chính cho mọi thí nghiệm.

    Trả về dict với các khoá: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    X_tr, y_tr, X_te, y_te = load_raw(name)
    X_tr, X_te = flatten(normalize(X_tr)), flatten(normalize(X_te))
    X_tr, y_tr, X_val, y_val = train_val_split(X_tr, y_tr)
    return {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_te,
        "y_test": y_te,
    }
