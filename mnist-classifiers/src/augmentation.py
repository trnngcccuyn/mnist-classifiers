"""Tăng cường dữ liệu: xoay, dịch, nhiễu.  Phụ trách: Tiên

Đầu vào/ra: ảnh shape (N, 28, 28) float32 trong [0, 1].
"""

import numpy as np


def rotate(X: np.ndarray, max_deg: float = 15.0, rng=None) -> np.ndarray:
    # TODO (Tiên)
    raise NotImplementedError


def shift(X: np.ndarray, max_px: int = 2, rng=None) -> np.ndarray:
    # TODO (Tiên)
    raise NotImplementedError


def add_noise(X: np.ndarray, std: float = 0.1, rng=None) -> np.ndarray:
    # TODO (Tiên): cộng nhiễu Gauss rồi clip về [0, 1]
    raise NotImplementedError


def augment(X: np.ndarray, y: np.ndarray, rng=None):
    """Trả về (X_aug, y_aug) gồm dữ liệu gốc + bản đã biến đổi."""
    # TODO (Tiên)
    raise NotImplementedError
