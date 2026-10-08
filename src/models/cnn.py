"""CNN đơn giản (phần nâng cao), so sánh với MLP.  Phụ trách: Uyên

Nhận X phẳng (N, 784) như các mô hình khác và tự reshape về (N, 1, 28, 28) bên trong.
"""

import numpy as np

from .base import BaseClassifier


class SimpleCNN(BaseClassifier):
    name = "cnn"

    def __init__(self, lr: float = 1e-3, epochs: int = 10, batch_size: int = 128, seed: int = 42):
        self.lr, self.epochs, self.batch_size, self.seed = lr, epochs, batch_size, seed
        self.history = {"train_loss": [], "val_acc": []}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # TODO (Uyên): ví dụ Conv-ReLU-Pool x2 -> Linear
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        # TODO (Uyên)
        raise NotImplementedError
