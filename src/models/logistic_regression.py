"""Softmax (multinomial logistic) regression huấn luyện bằng mini-batch GD.  Phụ trách: Uyên"""

import numpy as np

from .base import BaseClassifier


class LogisticRegression(BaseClassifier):
    name = "logistic_regression"

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 20,
        batch_size: int = 128,
        l2: float = 0.0,
        seed: int = 42,
    ):
        self.lr, self.epochs, self.batch_size, self.l2, self.seed = lr, epochs, batch_size, l2, seed
        self.history = {"train_loss": [], "val_acc": []}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # TODO (Uyên): khởi tạo W, b; lặp mini-batch; gradient của cross-entropy; cập nhật
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        # TODO (Uyên): softmax(XW + b)
        raise NotImplementedError
