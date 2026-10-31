"""MLP tự viết bằng NumPy, huấn luyện bằng backpropagation + mini-batch SGD.  Phụ trách: Uyên

Ký hiệu theo bài giảng/lab Tuần 4: L lớp tuyến tính, A^(0) = X (batch N mẫu xếp thành hàng),
Y là nhãn dạng one-hot, f là hàm kích hoạt, λ là hệ số L2.
Khác lab ở cách đặt chiều: lab dùng W^(l) shape (k_l, k_{l-1}) và Z = A W^T + b; ở đây W_l
shape (k_{l-1}, k_l) và Z = A W + b. Hai cách tương đương (W ở đây là W^T của lab).

Feedforward (_forward), lưu lại mọi A^(l) để dùng cho backward:
    Z^(l) = A^(l-1) @ W^(l) + b^(l)
    A^(l) = f(Z^(l))                               với l = 1, ..., L-1
    P     = softmax(Z^(L))                         (trừ max mỗi hàng trước khi lấy exp)

Hàm mất mát (cross-entropy trung bình + L2, không phạt bias):
    J = -(1/N) * tổng_i log P[i, y_i] + (λ/2) * tổng_l ||W^(l)||_F^2

Backpropagation (loss_and_grads):
    BP1: Δ^(L)   = dJ/dZ^(L) = (P - Y) / N
    BP2: Δ^(l-1) = (Δ^(l) @ W^(l)^T) ⊙ f'(Z^(l-1))
    BP3: dJ/db^(l) = tổng theo hàng của Δ^(l)
    BP4: dJ/dW^(l) = A^(l-1)^T @ Δ^(l) + λ W^(l)
Đạo hàm kích hoạt viết theo A = f(Z) đã lưu trong cache, không cần lưu Z:
    ReLU:    f'(Z) = 1[Z > 0] = 1[A > 0]           (A = max(Z, 0) nên A > 0 khi và chỉ khi Z > 0)
    sigmoid: f'(Z) = A (1 - A)
    tanh:    f'(Z) = 1 - A^2

Cập nhật: SGD + momentum (bài giảng trang 65), với momentum = 0 là mini-batch GD thường:
    ΔW(t) = α ΔW(t-1) - γ dJ/dW,   W(t+1) = W(t) + ΔW(t),   ΔW(0) = 0
(γ = lr, α = momentum; bias cập nhật giống hệt.)

Early stopping (patience không phải None): sau mỗi epoch tính cross-entropy trên validation;
dừng khi val loss không giảm trong `patience` epoch liên tiếp, rồi khôi phục trọng số của
epoch có val loss nhỏ nhất (self.best_epoch_, đánh số từ 1).

Khởi tạo: He (std = sqrt(2 / k_in)) cho ReLU; Xavier đều U(±sqrt(6 / (k_in + k_out))) cho
sigmoid/tanh; bias bằng 0.

Gradient được kiểm tra bằng đạo hàm số và PyTorch autograd trong tests/test_mlp.py.
"""

import numpy as np

from .base import BaseClassifier
from .nn_utils import cross_entropy, softmax, softmax_cross_entropy_grad

ACTIVATIONS = ("relu", "sigmoid", "tanh")


class MLP(BaseClassifier):
    name = "mlp"

    def __init__(
        self,
        hidden_sizes=(256, 128),
        lr: float = 0.1,
        epochs: int = 20,
        batch_size: int = 128,
        l2: float = 0.0,
        momentum: float = 0.0,
        activation: str = "relu",
        patience=None,
        n_classes: int = 10,
        seed: int = 42,
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation phải thuộc {ACTIVATIONS}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum phải nằm trong [0, 1)")
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.l2, self.momentum, self.activation = l2, momentum, activation
        self.patience, self.n_classes, self.seed = patience, n_classes, seed

    def _init_params(self, n_features, dtype, rng):
        """He cho ReLU, Xavier đều cho sigmoid/tanh; b bằng 0."""
        sizes = [n_features, *self.hidden_sizes, self.n_classes]
        self.Ws, self.bs = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            if self.activation == "relu":
                W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            else:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            self.Ws.append(W.astype(dtype))
            self.bs.append(np.zeros(fan_out, dtype=dtype))

    def _activate(self, Z):
        if self.activation == "relu":
            return np.maximum(Z, 0)
        if self.activation == "sigmoid":
            # = 1 / (1 + exp(-Z)), viết qua tanh để không tràn số khi Z âm lớn
            return 0.5 * (1.0 + np.tanh(0.5 * Z))
        return np.tanh(Z)

    def _activation_grad(self, A):
        """f'(Z) viết theo A = f(Z)."""
        if self.activation == "relu":
            return A > 0
        if self.activation == "sigmoid":
            return A * (1.0 - A)
        return 1.0 - A**2

    def _forward(self, X):
        """Trả về (P, cache). cache[l] là đầu vào của lớp tuyến tính l (cache[0] = X)."""
        cache = [X]
        A = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            A = self._activate(A @ W + b)
            cache.append(A)
        P = softmax(A @ self.Ws[-1] + self.bs[-1])
        return P, cache

    def loss_and_grads(self, X, y):
        """Trả về (J, dWs, dbs) bằng lan truyền ngược (BP1 đến BP4 ở đầu file)."""
        P, cache = self._forward(X)
        loss = cross_entropy(P, y) + 0.5 * self.l2 * sum(np.sum(W**2) for W in self.Ws)

        dWs = [None] * len(self.Ws)
        dbs = [None] * len(self.bs)
        G = softmax_cross_entropy_grad(P, y)  # BP1: Δ^(L) = (P - Y) / N
        for layer in reversed(range(len(self.Ws))):
            A_prev = cache[layer]
            dWs[layer] = A_prev.T @ G + self.l2 * self.Ws[layer]  # BP4
            dbs[layer] = G.sum(axis=0)  # BP3
            if layer > 0:
                # BP2: lùi về lớp trước qua W, rồi nhân với đạo hàm kích hoạt
                G = (G @ self.Ws[layer].T) * self._activation_grad(A_prev)
        return loss, dWs, dbs

    def _step(self, dWs, dbs):
        """Một bước SGD + momentum, cập nhật tại chỗ."""
        for W, b, vW, vb, dW, db in zip(self.Ws, self.bs, self._vWs, self._vbs, dWs, dbs):
            vW *= self.momentum
            vW -= self.lr * dW
            vb *= self.momentum
            vb -= self.lr * db
            W += vW
            b += vb

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.patience is not None and X_val is None:
            raise ValueError("Early stopping (patience) cần X_val, y_val")
        rng = np.random.default_rng(self.seed)
        n, d = X_train.shape
        self._init_params(d, X_train.dtype, rng)
        self._vWs = [np.zeros_like(W) for W in self.Ws]  # ΔW(0) = 0
        self._vbs = [np.zeros_like(b) for b in self.bs]
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_epoch_ = None
        best_loss, best_params, bad_epochs = np.inf, None, 0

        for epoch in range(1, self.epochs + 1):
            idx = rng.permutation(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                loss, dWs, dbs = self.loss_and_grads(X_train[batch], y_train[batch])
                self._step(dWs, dbs)
                total_loss += loss * len(batch)
            # float(): loss cộng thêm số hạng L2 (np.float32) nên ép về float Python
            self.history["train_loss"].append(float(total_loss / n))
            if not np.isfinite(self.history["train_loss"][-1]):
                break  # phân kỳ (lr quá lớn): dừng, không train tiếp vô ích
            if X_val is None:
                continue

            P_val = self.predict_proba(X_val)
            val_loss = cross_entropy(P_val, y_val)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(float(np.mean(P_val.argmax(axis=1) == y_val)))

            if self.patience is not None:
                if val_loss < best_loss:
                    best_loss, bad_epochs, self.best_epoch_ = val_loss, 0, epoch
                    best_params = ([W.copy() for W in self.Ws], [b.copy() for b in self.bs])
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break

        if best_params is not None:
            self.Ws, self.bs = best_params
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X)[0]

    def n_params(self) -> int:
        """Số tham số, dùng khi so sánh với CNN."""
        return int(sum(W.size + b.size for W, b in zip(self.Ws, self.bs)))
