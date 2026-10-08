"""MLP huấn luyện bằng backpropagation.  Phụ trách: Uyên

Đề cho phép dùng PyTorch/TensorFlow cho phần khung. Hãy thống nhất trong nhóm (và hỏi
giảng viên nếu chưa rõ) mức độ được dùng autograd so với tự viết backward.
"""

import numpy as np

from .base import BaseClassifier


class MLP(BaseClassifier):
    name = "mlp"

    def __init__(
        self,
        hidden_sizes=(256, 128),
        lr: float = 0.01,
        epochs: int = 20,
        batch_size: int = 128,
        seed: int = 42,
    ):
        self.hidden_sizes, self.lr, self.epochs = hidden_sizes, lr, epochs
        self.batch_size, self.seed = batch_size, seed
        self.history = {"train_loss": [], "val_acc": []}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # TODO (Uyên): forward, loss, backward, cập nhật trọng số
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        # TODO (Uyên)
        raise NotImplementedError
