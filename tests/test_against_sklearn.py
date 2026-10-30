"""Đối chiếu cài đặt của nhóm với scikit-learn.  Phụ trách: Tiên

Chạy:  pytest tests/
Gỡ dòng skip của từng test khi phần tương ứng đã cài xong.
"""

import pytest


def test_metrics_match_sklearn():
    # So sánh accuracy / precision / recall / F1 macro / confusion matrix với sklearn.metrics
    import numpy as np
    from sklearn import metrics as sk

    from src.metrics import accuracy, confusion_matrix, precision_recall_f1_macro

    rng = np.random.default_rng(0)
    labels = np.arange(10)
    y_true = rng.integers(0, 10, 2000)
    # Lần hai: mô hình không bao giờ dự đoán lớp 7, 8, 9 (kiểm tra chia cho 0)
    for y_pred in (rng.integers(0, 10, 2000), rng.integers(0, 7, 2000)):
        assert np.array_equal(
            confusion_matrix(y_true, y_pred),
            sk.confusion_matrix(y_true, y_pred, labels=labels),
        )
        assert np.isclose(accuracy(y_true, y_pred), sk.accuracy_score(y_true, y_pred))
        expected = sk.precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )[:3]
        assert np.allclose(precision_recall_f1_macro(y_true, y_pred), expected)


@pytest.mark.skip(reason="TODO: chưa cài Logistic Regression")
def test_logistic_regression_close_to_sklearn(): ...
