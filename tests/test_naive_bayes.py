"""Đối chiếu Gaussian và Bernoulli Naive Bayes tự cài với scikit-learn."""

import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from src.models.naive_bayes import BernoulliNaiveBayes, GaussianNaiveBayes


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


def test_bernoulli_naive_bayes_close_to_sklearn():
    rng = np.random.default_rng(42)
    X_train = rng.uniform(size=(200, 8))
    X_train[:, 0] = 0  # Pixel luôn tắt: alpha phải giữ log p hữu hạn.
    y_train = np.repeat([0, 1, 2, 3], 50)
    X_test = rng.uniform(size=(30, 8))

    for alpha in (0.01, 0.5, 1.0):
        for threshold in (0.3, 0.5, 0.7):
            ours = BernoulliNaiveBayes(alpha=alpha, binarize=threshold).fit(X_train, y_train)
            reference = BernoulliNB(alpha=alpha, binarize=threshold).fit(X_train, y_train)

            assert np.all(np.isfinite(ours.feature_log_prob_))
            assert np.allclose(ours.feature_log_prob_, reference.feature_log_prob_)
            assert np.allclose(ours.predict_proba(X_test), reference.predict_proba(X_test))
            assert np.array_equal(ours.predict(X_test), reference.predict(X_test))
