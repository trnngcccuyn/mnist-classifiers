"""MLP tự viết bằng NumPy, huấn luyện bằng backpropagation + mini-batch GD.  Phụ trách: Uyên"""

import numpy as np

from .base import BaseClassifier
from .nn_utils import cross_entropy, softmax, softmax_cross_entropy_grad


class MLP(BaseClassifier):
    name = "mlp"

    def __init__(
        self,
        hidden_sizes=(256, 128),
        lr: float = 0.1,
        epochs: int = 20,
        batch_size: int = 128,
        l2: float = 0.0,
        n_classes: int = 10,
        seed: int = 42,
    ):
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.l2, self.n_classes, self.seed = l2, n_classes, seed

    def _init_params(self, n_features, dtype, rng):
        """Khởi tạo He cho W (hợp với ReLU), b bằng 0."""
        sizes = [n_features, *self.hidden_sizes, self.n_classes]
        self.Ws, self.bs = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            self.Ws.append(W.astype(dtype))
            self.bs.append(np.zeros(fan_out, dtype=dtype))

    def _forward(self, X):
        """Trả về (P, cache). cache[l] là đầu vào của lớp l (cache[0] = X)."""
        cache = [X]
        A = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            A = np.maximum(A @ W + b, 0)  # ReLU
            cache.append(A)
        P = softmax(A @ self.Ws[-1] + self.bs[-1])
        return P, cache

    def loss_and_grads(self, X, y):
        """Trả về (loss, dWs, dbs) bằng lan truyền ngược."""
        P, cache = self._forward(X)
        loss = cross_entropy(P, y) + 0.5 * self.l2 * sum(np.sum(W**2) for W in self.Ws)

        dWs = [None] * len(self.Ws)
        dbs = [None] * len(self.bs)
        G = softmax_cross_entropy_grad(P, y)  # dL/dZ của lớp cuối
        for layer in reversed(range(len(self.Ws))):
            A_prev = cache[layer]
            dWs[layer] = A_prev.T @ G + self.l2 * self.Ws[layer]
            dbs[layer] = G.sum(axis=0)
            if layer > 0:
                # Lùi về lớp trước: nhân với W rồi với đạo hàm ReLU (1 nếu đầu ra > 0)
                G = (G @ self.Ws[layer].T) * (A_prev > 0)
        return loss, dWs, dbs

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X_train.shape
        self._init_params(d, X_train.dtype, rng)
        self.history = {"train_loss": [], "val_acc": []}

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                loss, dWs, dbs = self.loss_and_grads(X_train[batch], y_train[batch])
                for W, b, dW, db in zip(self.Ws, self.bs, dWs, dbs):
                    W -= self.lr * dW
                    b -= self.lr * db
                total_loss += loss * len(batch)
            self.history["train_loss"].append(total_loss / n)
            if X_val is not None:
                self.history["val_acc"].append(float(np.mean(self.predict(X_val) == y_val)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X)[0]

    def n_params(self) -> int:
        """Số tham số, dùng khi so sánh với CNN."""
        return int(sum(W.size + b.size for W, b in zip(self.Ws, self.bs)))
