"""Các độ đo đánh giá, tự cài bằng NumPy.  Phụ trách: Uyên

Đối chiếu với sklearn.metrics trong tests/test_against_sklearn.py.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    """Ma trận (n_classes, n_classes); hàng = nhãn thật, cột = nhãn dự đoán."""
    # TODO
    raise NotImplementedError


def accuracy(y_true, y_pred) -> float:
    # TODO
    raise NotImplementedError


def precision_recall_f1_macro(y_true, y_pred, n_classes: int = 10):
    """Trả về (precision, recall, f1) lấy trung bình macro trên các lớp."""
    # TODO: tính từ confusion matrix; chú ý trường hợp chia cho 0
    raise NotImplementedError


def evaluate(y_true, y_pred, n_classes: int = 10) -> dict:
    """Gom mọi độ đo vào một dict để ghi ra bảng kết quả."""
    p, r, f1 = precision_recall_f1_macro(y_true, y_pred, n_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }
