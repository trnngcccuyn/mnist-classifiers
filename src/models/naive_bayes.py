"""Gaussian Naive Bayes triển khai trực tiếp bằng các phép toán NumPy.

Phụ trách theo phân công: Tiên.

Công thức cho lớp c, đặc trưng j:
    prior[c] = N_c / N
    mean[c, j] = tổng x[i, j] / N_c, với y[i] = c
    variance[c, j] = tổng (x[i, j] - mean[c, j])**2 / N_c
    log_joint[c] = log(prior[c]) - 0.5 * tổng_j (
        log(2*pi*variance[c, j]) + (x[j] - mean[c, j])**2 / variance[c, j]
    )
Phương sai trong log_joint đã được cộng epsilon để làm trơn.
"""

import numpy as np

from .base import BaseClassifier


class GaussianNaiveBayes(BaseClassifier):
    name = "naive_bayes"

    def __init__(self, var_smoothing: float = 1e-2):
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
            log_joint -= log_joint.max(axis=1, keepdims=True)
            # Chuẩn hóa exp(log_joint) thành posterior; mỗi hàng có tổng bằng 1.
            scores = np.exp(log_joint)
            proba[start : start + len(batch)] = scores / scores.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
