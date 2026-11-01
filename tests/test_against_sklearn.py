"""Đối chiếu cài đặt của nhóm với scikit-learn.  Phụ trách: Tiên

Chạy:  pytest tests/
Test đối chiếu Random Forest nằm trong tests/test_random_forest.py.
"""


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


def test_logistic_regression_close_to_sklearn():
    """Cùng hàm mục tiêu thì GD của nhóm phải hội tụ về đúng nghiệm của sklearn.

    sklearn tối thiểu C * tổng_i CE_i + 0.5 ||W||^2 (không phạt intercept); nhóm tối thiểu
    (1/N) * tổng_i CE_i + (λ/2) ||W||^2. Chia hàm của sklearn cho C*N: hai hàm cùng nghiệm
    khi λ = 1 / (C * N). Hàm lồi chặt theo W nên nghiệm W là duy nhất.
    Bias của softmax chỉ xác định tới một hằng số cộng chung cho mọi lớp, nên so sau khi trừ
    trung bình.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression

    from src.models.logistic_regression import LogisticRegression

    rng = np.random.default_rng(0)
    n, d, k, lam = 300, 5, 3, 0.01
    centers = rng.normal(scale=2.0, size=(k, d))
    y = rng.integers(0, k, size=n)
    X = centers[y] + rng.normal(size=(n, d))

    # Full-batch GD, nhiều epoch để hội tụ hẳn (khoảng 1 giây)
    ours = LogisticRegression(n_classes=k, l2=lam, lr=1.0, epochs=5000, batch_size=n).fit(X, y)
    ref = SkLogisticRegression(C=1 / (lam * n), tol=1e-12, max_iter=10_000).fit(X, y)

    assert np.allclose(ours.W, ref.coef_.T, atol=1e-4)
    assert np.allclose(ours.b - ours.b.mean(), ref.intercept_ - ref.intercept_.mean(), atol=1e-4)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-4)
    assert np.array_equal(ours.predict(X), ref.predict(X))
