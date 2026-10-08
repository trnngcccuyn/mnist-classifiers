"""Gaussian Naive Bayes tự cài bằng NumPy.  Phụ trách: Tiên"""

import numpy as np

from .base import BaseClassifier


class GaussianNaiveBayes(BaseClassifier):
    name = "naive_bayes"

    def __init__(self, var_smoothing: float = 1e-2):
        self.var_smoothing = var_smoothing

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # TODO (Tiên): ước lượng prior, mean, variance cho từng lớp
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        # TODO (Tiên): tính log-posterior (dùng log để tránh tràn số), rồi chuẩn hoá
        raise NotImplementedError
