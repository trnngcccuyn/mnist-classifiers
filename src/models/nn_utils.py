"""Hàm dùng chung cho các mô hình học bằng gradient (Logistic Regression, MLP)."""

import numpy as np


def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax theo từng hàng, trừ max để tránh tràn số khi lấy exp."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy trung bình. P: xác suất (n, k); y: nhãn nguyên (n,)."""
    n = len(y)
    return float(-np.mean(np.log(P[np.arange(n), y] + 1e-12)))


def softmax_cross_entropy_grad(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Đạo hàm của cross_entropy(softmax(Z), y) theo Z: (P - Y_onehot) / n."""
    n = len(y)
    G = P.copy()
    G[np.arange(n), y] -= 1
    return G / n
