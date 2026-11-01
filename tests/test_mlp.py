"""Kiểm tra MLP: backprop khớp đạo hàm số và PyTorch autograd, SGD + momentum đúng công thức,
early stopping khôi phục đúng trọng số, học thuộc được tập nhỏ."""

import numpy as np
import pytest

from src.models.mlp import ACTIVATIONS, MLP
from src.models.nn_utils import cross_entropy


def numerical_grad(f, param, eps=1e-6):
    grad = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + eps
        plus = f()
        param[i] = old - eps
        minus = f()
        param[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def small_model(activation, seed=0, n_in=6, hidden=(5, 4), n_classes=4):
    """MLP nhỏ float64, bias khác 0 để kiểm tra cả gradient của b."""
    rng = np.random.default_rng(seed)
    model = MLP(hidden_sizes=hidden, n_classes=n_classes, l2=0.01, activation=activation)
    model._init_params(n_in, np.float64, rng)
    for b in model.bs:
        b[:] = rng.normal(size=b.shape)
    return model, rng


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_backprop_matches_numerical(activation):
    model, rng = small_model(activation)
    X = rng.normal(size=(15, 6))
    y = rng.integers(0, 4, size=15)

    _, dWs, dbs = model.loss_and_grads(X, y)

    def loss():
        return model.loss_and_grads(X, y)[0]

    for W, dW in zip(model.Ws, dWs):
        assert np.allclose(dW, numerical_grad(loss, W), atol=1e-5)
    for b, db in zip(model.bs, dbs):
        assert np.allclose(db, numerical_grad(loss, b), atol=1e-5)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_backprop_matches_pytorch_autograd(activation):
    """Gradient tự tính bằng backprop phải khớp gradient PyTorch autograd trên cùng trọng số."""
    torch = pytest.importorskip("torch")
    act = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}[activation]

    model, rng = small_model(activation, seed=1, n_in=12, hidden=(16, 8), n_classes=10)
    X = rng.normal(size=(32, 12))
    y = rng.integers(0, 10, size=32)

    loss, dWs, dbs = model.loss_and_grads(X, y)

    # Dựng lại đúng mạng đó bằng PyTorch (float64) và để autograd tính đạo hàm
    Ws = [torch.tensor(W, requires_grad=True) for W in model.Ws]
    bs = [torch.tensor(b, requires_grad=True) for b in model.bs]
    A = torch.tensor(X)
    for W, b in zip(Ws[:-1], bs[:-1]):
        A = act(A @ W + b)
    logits = A @ Ws[-1] + bs[-1]
    torch_loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
    torch_loss = torch_loss + 0.5 * model.l2 * sum((W**2).sum() for W in Ws)
    torch_loss.backward()

    assert np.isclose(loss, torch_loss.item(), atol=1e-8)
    for dW, W in zip(dWs, Ws):
        assert np.allclose(dW, W.grad.numpy(), atol=1e-8)
    for db, b in zip(dbs, bs):
        assert np.allclose(db, b.grad.numpy(), atol=1e-8)


def test_momentum_step_matches_formula():
    """Cùng ví dụ ô kiểm tra 4.1 của lab Tuần 4: lr = 0.1, momentum = 0.9, gradient = 2."""
    model = MLP(lr=0.1, momentum=0.9)
    model.Ws, model.bs = [np.array([1.0])], [np.array([0.0])]
    model._vWs, model._vbs = [np.zeros(1)], [np.zeros(1)]
    grad = [np.array([2.0])]

    model._step(grad, [np.zeros(1)])  # ΔW = -0.2,                   W = 0.8
    assert np.allclose(model.Ws[0], 0.8)
    model._step(grad, [np.zeros(1)])  # ΔW = 0.9 * (-0.2) - 0.2 = -0.38, W = 0.42
    assert np.allclose(model.Ws[0], 0.42)


def test_zero_momentum_equals_plain_sgd():
    """momentum = 0 phải cho đúng W <- W - lr * dW (để kết quả cũ vẫn tái tạo được)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=64)
    model = MLP(hidden_sizes=(8,), n_classes=3, epochs=1, batch_size=64, lr=0.1)
    model._init_params(10, np.float32, np.random.default_rng(model.seed))
    _, dWs, dbs = model.loss_and_grads(X, y)
    expected = [W - model.lr * dW for W, dW in zip(model.Ws, dWs)]

    model.fit(X, y)  # 1 epoch, 1 batch: đúng một bước (thứ tự mẫu bị xáo nên chỉ sai số làm tròn)
    for W, W_exp in zip(model.Ws, expected):
        assert np.allclose(W, W_exp, rtol=0, atol=1e-6)


def test_early_stopping_restores_best_epoch():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(200, 10)).astype(np.float32)
    y_tr = rng.integers(0, 3, size=200)  # nhãn ngẫu nhiên: val loss sẽ tăng vì overfit
    X_va = rng.normal(size=(100, 10)).astype(np.float32)
    y_va = rng.integers(0, 3, size=100)
    model = MLP(hidden_sizes=(64,), n_classes=3, lr=0.1, momentum=0.9, epochs=50, patience=3)
    model.fit(X_tr, y_tr, X_va, y_va)

    val_loss = model.history["val_loss"]
    assert len(val_loss) < 50  # đã dừng sớm
    assert model.best_epoch_ == int(np.argmin(val_loss)) + 1
    assert len(val_loss) == model.best_epoch_ + model.patience
    # Trọng số sau fit là của epoch tốt nhất, không phải epoch cuối
    assert np.isclose(cross_entropy(model.predict_proba(X_va), y_va), min(val_loss))


def test_early_stopping_needs_validation():
    X = np.zeros((10, 4), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError):
        MLP(patience=2).fit(X, y)


def test_history_stores_python_floats():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=64)
    model = MLP(hidden_sizes=(8,), n_classes=3, epochs=2, l2=1e-3).fit(X, y, X, y)
    for key in ("train_loss", "val_loss", "val_acc"):
        assert all(type(v) is float for v in model.history[key])


def test_overfits_small_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 20)).astype(np.float32)
    y = rng.integers(0, 10, size=100)  # nhãn ngẫu nhiên: chỉ học thuộc mới đúng hết
    model = MLP(hidden_sizes=(64,), lr=0.1, epochs=300, batch_size=32).fit(X, y)
    assert np.mean(model.predict(X) == y) == 1.0
