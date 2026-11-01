"""Gaussian và Bernoulli Naive Bayes triển khai trực tiếp bằng các phép toán NumPy.

Phụ trách theo phân công: Tiên.

Gaussian, công thức cho lớp c, đặc trưng j:
    prior[c] = N_c / N
    mean[c, j] = tổng x[i, j] / N_c, với y[i] = c
    variance[c, j] = tổng (x[i, j] - mean[c, j])**2 / N_c
    log_joint[c] = log(prior[c]) - 0.5 * tổng_j (
        log(2*pi*variance[c, j]) + (x[j] - mean[c, j])**2 / variance[c, j]
    )
Phương sai trong log_joint đã được cộng epsilon để làm trơn.

Bernoulli, với x[j] = 1 nếu pixel j lớn hơn ngưỡng binarize, ngược lại 0:
    p[c, j] = (số mẫu lớp c có x[j] = 1 + alpha) / (N_c + 2*alpha)
    log_joint[c] = log(prior[c]) + tổng_j (
        x[j] * log(p[c, j]) + (1 - x[j]) * log(1 - p[c, j])
    )
alpha > 0 làm trơn Laplace, tránh log(0) với pixel luôn tắt hoặc luôn bật.
"""

import numpy as np

from .base import BaseClassifier


def _posterior_from_log_joint(log_joint):
    """Chuẩn hóa exp(log_joint) thành posterior; mỗi hàng có tổng bằng 1."""
    # Trừ max mỗi hàng trước khi exp để không bị tràn số.
    scores = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    return scores / scores.sum(axis=1, keepdims=True)


class GaussianNaiveBayes(BaseClassifier):
    name = "naive_bayes"

    def __init__(self, var_smoothing: float = 0.05):
        if var_smoothing <= 0:
            raise ValueError("var_smoothing phải lớn hơn 0")
        self.var_smoothing = var_smoothing

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Ước lượng tham số từ train; X_val/y_val chỉ giữ tương thích interface."""
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X_train phải có dạng (N, D) và y_train có dạng (N,)")

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))
        self.theta_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        # Phương sai quần thể (ddof=0); epsilon chung giữ cả pixel hằng số hữu hạn.
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ = np.vstack([X[y == c].var(axis=0) for c in self.classes_])
        self.var_ += self.epsilon_
        if np.any(self.var_ <= 0):
            raise ValueError("Dữ liệu có phương sai bằng 0; hãy chọn var_smoothing phù hợp")

        self._inverse_var = 1.0 / self.var_
        self._scaled_mean = self.theta_ * self._inverse_var
        # Khai triển (x - mean)**2 = x**2 - 2*x*mean + mean**2.
        # Gom các phần không phụ thuộc x để dùng lại lúc dự đoán.
        self._offset = self.class_log_prior_ - 0.5 * (
            np.log(2 * np.pi * self.var_).sum(axis=1)
            + (self.theta_ * self._scaled_mean).sum(axis=1)
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "theta_"):
            raise RuntimeError("Cần gọi fit trước predict_proba")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.theta_.shape[1]:
            raise ValueError("X phải có dạng (N, D) với D bằng số đặc trưng khi fit")

        proba = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        for start in range(0, len(X), 2048):
            batch = X[start : start + 2048]
            # Dùng nhân ma trận để tính công thức log_joint ở đầu file.
            log_joint = (
                self._offset
                + batch @ self._scaled_mean.T
                - 0.5 * (batch * batch) @ self._inverse_var.T
            )
            proba[start : start + len(batch)] = _posterior_from_log_joint(log_joint)
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class BernoulliNaiveBayes(BaseClassifier):
    name = "bernoulli_naive_bayes"

    def __init__(self, alpha: float = 1.0, binarize: float = 0.5):
        if alpha <= 0:
            raise ValueError("alpha phải lớn hơn 0")
        self.alpha = alpha
        self.binarize = binarize

    def _binarize(self, X):
        return (np.asarray(X, dtype=np.float64) > self.binarize).astype(np.float64)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Ước lượng tham số từ train; X_val/y_val chỉ giữ tương thích interface."""
        X = self._binarize(X_train)
        y = np.asarray(y_train)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X_train phải có dạng (N, D) và y_train có dạng (N,)")

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))
        on_counts = np.vstack([X[y == c].sum(axis=0) for c in self.classes_])
        p = (on_counts + self.alpha) / (counts[:, None] + 2 * self.alpha)
        self.feature_log_prob_ = np.log(p)
        log_off = np.log1p(-p)
        # Khai triển log_joint = offset + x @ (log p - log(1 - p)).T.
        # Gom các phần không phụ thuộc x để dùng lại lúc dự đoán.
        self._log_odds = self.feature_log_prob_ - log_off
        self._offset = self.class_log_prior_ + log_off.sum(axis=1)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "feature_log_prob_"):
            raise RuntimeError("Cần gọi fit trước predict_proba")
        X = self._binarize(X)
        if X.ndim != 2 or X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X phải có dạng (N, D) với D bằng số đặc trưng khi fit")
        return _posterior_from_log_joint(self._offset + X @ self._log_odds.T)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
