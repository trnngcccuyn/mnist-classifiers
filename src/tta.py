"""Test-time augmentation (TTA).  Phụ trách: Uyên

Thay vì dự đoán một lần trên ảnh gốc, dự đoán trên ảnh gốc và n_views bản biến đổi ngẫu nhiên
của nó, rồi lấy trung bình xác suất:
    P_tta(x) = (P(x) + sum_k P(T_k(x))) / (n_views + 1)
Giống soft voting, nhưng các "thành viên" là các phiên bản biến đổi của cùng một ảnh, đưa qua
cùng một mô hình.

Phép biến đổi lúc test: get_tta_augment(dataset), tức pipeline train của get_augment(dataset)
(xoay ±10°, dịch ±2 px, Fashion thêm lật ngang) nhưng bỏ nhiễu: chỉ dùng biến đổi hình học
mà ảnh thật có thể gặp. TTA chỉ có ích khi mô hình đã quen với các biến đổi này (train có
augment). Mô hình train không augment thì các bản biến đổi là ảnh lạ với nó (slide 14 B6b).
"""

import numpy as np

from .augmentation import Augment, get_augment


def get_tta_augment(dataset: str = "mnist") -> Augment:
    """Biến đổi dùng lúc test: như get_augment(dataset) nhưng không cộng nhiễu."""
    aug = get_augment(dataset)
    return Augment(max_deg=aug.max_deg, max_px=aug.max_px, noise_std=0.0, flip_p=aug.flip_p)


def tta_views_proba(model, X, n_views: int, augment, rng) -> np.ndarray:
    """Xác suất của model trên ảnh gốc và n_views bản biến đổi.

    X: (N, 784). Trả về (n_views + 1, N, n_classes); phần tử 0 là ảnh gốc. Trung bình k + 1
    phần tử đầu là kết quả TTA với k bản biến đổi (k = 0 là không TTA).
    """
    X = np.asarray(X, dtype=np.float32)
    X_img = X.reshape(len(X), 28, 28)
    out = [model.predict_proba(X)]
    for _ in range(n_views):
        out.append(model.predict_proba(augment(X_img, rng).reshape(len(X), -1)))
    return np.stack(out)


def tta_predict_proba(model, X, n_views: int = 8, augment=None, rng=None) -> np.ndarray:
    """Xác suất TTA (N, n_classes): trung bình trên ảnh gốc và n_views bản biến đổi."""
    augment = augment or get_tta_augment("mnist")
    rng = rng or np.random.default_rng()
    return tta_views_proba(model, X, n_views, augment, rng).mean(axis=0)
