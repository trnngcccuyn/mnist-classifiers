"""Softmax (multinomial logistic) regression huấn luyện bằng mini-batch GD.  Phụ trách: Uyên

Ký hiệu: X là batch N mẫu (N, d), Y là nhãn dạng one-hot (N, K), λ là hệ số L2.
Mô hình là MLP không có lớp ẩn, nên dùng chung ký hiệu với src/models/mlp.py.

Mô hình:
    Z = X @ W + b                     W shape (d, K), b shape (K,)
    P = softmax(Z)                    (trừ max mỗi hàng trước khi lấy exp)

Hàm mất mát (cross-entropy trung bình + L2, không phạt bias):
    J = -(1/N) * tổng_i log P[i, y_i] + (λ/2) * ||W||_F^2

Gradient (loss_and_grad):
    Δ     = dJ/dZ = (P - Y) / N
    dJ/dW = X^T @ Δ + λ W
    dJ/db = tổng theo hàng của Δ

Cập nhật: mini-batch GD, W <- W - lr * dJ/dW, b <- b - lr * dJ/db.
Hàm mất mát lồi theo (W, b) nên GD với lr đủ nhỏ hội tụ về nghiệm tối ưu toàn cục; vì vậy
kết quả có thể đối chiếu với sklearn.linear_model.LogisticRegression (cùng hàm mục tiêu).
"""

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
        """Trả về (J, dW, db) của cross-entropy trung bình + L2 trên batch (X, y)."""
        P = softmax(X @ self.W + self.b)
        loss = cross_entropy(P, y) + 0.5 * self.l2 * np.sum(self.W**2)
        G = softmax_cross_entropy_grad(P, y)  # Δ = (P - Y) / N
        dW = X.T @ G + self.l2 * self.W
        db = G.sum(axis=0)
        return loss, dW, db

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X_train.shape
        self.W = rng.normal(0.0, 0.01, size=(d, self.n_classes)).astype(X_train.dtype)
        self.b = np.zeros(self.n_classes, dtype=X_train.dtype)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                loss, dW, db = self.loss_and_grad(X_train[batch], y_train[batch])
                self.W -= self.lr * dW
                self.b -= self.lr * db
                total_loss += loss * len(batch)
            # float(): loss cộng thêm số hạng L2 (np.float32) nên ép về float Python
            self.history["train_loss"].append(float(total_loss / n))
            if X_val is not None:
                P_val = self.predict_proba(X_val)
                self.history["val_loss"].append(cross_entropy(P_val, y_val))
                self.history["val_acc"].append(float(np.mean(P_val.argmax(axis=1) == y_val)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return softmax(X @ self.W + self.b)

    def weight_images(self) -> np.ndarray:
        """Trọng số mỗi lớp dưới dạng ảnh (K, 28, 28), để vẽ và giải thích mô hình."""
        return self.W.T.reshape(self.n_classes, 28, 28)
