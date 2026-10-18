"""Softmax (multinomial logistic) regression huấn luyện bằng mini-batch GD.  Phụ trách: Uyên"""

import numpy as np

from .base import BaseClassifier
from .nn_utils import cross_entropy, softmax, softmax_cross_entropy_grad


class LogisticRegression(BaseClassifier):
    name = "logistic_regression"

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 20,
        batch_size: int = 128,
        l2: float = 0.0,
        n_classes: int = 10,
        seed: int = 42,
    ):
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.l2, self.n_classes, self.seed = l2, n_classes, seed

    def loss_and_grad(self, X, y):
        """Trả về (loss, dW, db) của cross-entropy trung bình trên batch (X, y)."""
        P = softmax(X @ self.W + self.b)
        loss = cross_entropy(P, y) + 0.5 * self.l2 * np.sum(self.W**2)
        G = softmax_cross_entropy_grad(P, y)
        dW = X.T @ G + self.l2 * self.W
        db = G.sum(axis=0)
        return loss, dW, db

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X_train.shape
        self.W = rng.normal(0.0, 0.01, size=(d, self.n_classes)).astype(X_train.dtype)
        self.b = np.zeros(self.n_classes, dtype=X_train.dtype)
        self.history = {"train_loss": [], "val_acc": []}

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                loss, dW, db = self.loss_and_grad(X_train[batch], y_train[batch])
                self.W -= self.lr * dW
                self.b -= self.lr * db
                total_loss += loss * len(batch)
            self.history["train_loss"].append(total_loss / n)
            if X_val is not None:
                self.history["val_acc"].append(float(np.mean(self.predict(X_val) == y_val)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return softmax(X @ self.W + self.b)
