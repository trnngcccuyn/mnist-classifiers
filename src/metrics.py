"""Các độ đo đánh giá, tự cài bằng NumPy.  Phụ trách: Uyên

Đối chiếu với sklearn.metrics trong tests/test_against_sklearn.py.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int = 10) -> np.ndarray:
    """Ma trận (n_classes, n_classes); hàng = nhãn thật, cột = nhãn dự đoán."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1_macro(y_true, y_pred, n_classes: int = 10):
    """Trả về (precision, recall, f1) lấy trung bình macro trên các lớp."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    n_pred = cm.sum(axis=0)  # tổng cột: số mẫu được dự đoán là lớp c
    n_true = cm.sum(axis=1)  # tổng hàng: số mẫu thật sự thuộc lớp c
    precision = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    recall = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


def evaluate(y_true, y_pred, n_classes: int = 10) -> dict:
    """Gom mọi độ đo vào một dict để ghi ra bảng kết quả."""
    p, r, f1 = precision_recall_f1_macro(y_true, y_pred, n_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }
