"""Gaussian và Bernoulli Naive Bayes tự cài bằng NumPy.  Phụ trách: Tiên

Cả hai mô hình chọn lớp c có log P(c) + log P(x | c) lớn nhất (giả định các pixel độc lập
khi đã biết lớp). Xác suất hậu nghiệm = softmax của log_joint theo các lớp.

Gaussian, với lớp c và pixel j:
    prior[c]       = N_c / N
    mean[c, j]     = trung bình x[:, j] của các mẫu lớp c
    var[c, j]      = phương sai x[:, j] của các mẫu lớp c + epsilon
    epsilon        = var_smoothing * (phương sai lớn nhất của một pixel trên toàn bộ train)
    log_joint[c]   = log prior[c]
                     - 0.5 * tổng_j [ log(2*pi*var[c, j]) + (x[j] - mean[c, j])**2 / var[c, j] ]
epsilon tránh chia cho 0 với pixel luôn bằng 0 (viền ảnh).

Bernoulli, với x[j] = 1 nếu pixel j > binarize, ngược lại 0:
    p[c, j]        = (số mẫu lớp c có x[j] = 1 + alpha) / (N_c + 2 * alpha)
    log_joint[c]   = log prior[c] + tổng_j [ x[j] * log p[c, j] + (1 - x[j]) * log(1 - p[c, j]) ]
alpha > 0 (làm trơn Laplace) tránh log(0) với pixel luôn tắt hoặc luôn bật.
"""

import numpy as np

from .base import BaseClassifier
from .nn_utils import softmax


class GaussianNaiveBayes(BaseClassifier):
    name = "naive_bayes"

    def __init__(self, var_smoothing: float = 0.05):
        self.var_smoothing = var_smoothing

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)
        self.classes_ = np.unique(y)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()

        self.class_log_prior_ = np.array([np.log(np.mean(y == c)) for c in self.classes_])
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + self.epsilon_
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_joint = np.zeros((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            mean, var = self.theta_[c], self.var_[c]
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
            log_joint[:, c] = self.class_log_prior_[c] + log_likelihood
        return softmax(log_joint)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BernoulliNaiveBayes(BaseClassifier):
    name = "bernoulli_naive_bayes"

    def __init__(self, alpha: float = 1.0, binarize: float = 0.5):
        self.alpha = alpha
        self.binarize = binarize

    def _binarize(self, X):
        return (np.asarray(X) > self.binarize).astype(np.float64)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = self._binarize(X_train)
        y = np.asarray(y_train)
        self.classes_ = np.unique(y)

        self.class_log_prior_ = np.array([np.log(np.mean(y == c)) for c in self.classes_])
        p = np.array(
            [
                (X[y == c].sum(axis=0) + self.alpha) / (np.sum(y == c) + 2 * self.alpha)
                for c in self.classes_
            ]
        )
        self.feature_log_prob_ = np.log(p)  # log p[c, j]
        self.feature_log_prob_off_ = np.log(1 - p)  # log (1 - p[c, j])
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = self._binarize(X)
        # Nhân ma trận = tổng theo j cho mọi mẫu và mọi lớp cùng lúc.
        log_joint = (
            self.class_log_prior_
            + X @ self.feature_log_prob_.T
            + (1 - X) @ self.feature_log_prob_off_.T
        )
        return softmax(log_joint)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
