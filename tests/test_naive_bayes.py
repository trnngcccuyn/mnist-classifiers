"""Đối chiếu Gaussian Naive Bayes tự cài với scikit-learn."""

import numpy as np
from sklearn.naive_bayes import GaussianNB

from src.models.naive_bayes import GaussianNaiveBayes


def test_naive_bayes_close_to_sklearn():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 8))
    X_train[:, 0] = 0  # Đặc trưng hằng số cần được làm trơn.
    y_train = np.repeat([0, 1, 2, 3], 50)
    X_test = rng.normal(size=(30, 8))
    X_test[:, 0] = 0

    for smoothing in (1e-4, 1e-2, 1e-1):
        ours = GaussianNaiveBayes(var_smoothing=smoothing).fit(X_train, y_train)
        reference = GaussianNB(var_smoothing=smoothing).fit(X_train, y_train)

        assert np.allclose(ours.theta_, reference.theta_)
        assert np.allclose(ours.var_, reference.var_)
        assert np.allclose(ours.predict_proba(X_test), reference.predict_proba(X_test))
        assert np.array_equal(ours.predict(X_test), reference.predict(X_test))
