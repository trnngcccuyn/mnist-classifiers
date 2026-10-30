"""Kiểm tra MLP: gradient lan truyền ngược khớp gradient số, và học thuộc được tập nhỏ."""

import numpy as np

from src.models.mlp import MLP


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


def test_backprop_matches_numerical():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 6))
    y = rng.integers(0, 4, size=15)
    model = MLP(hidden_sizes=(5, 4), n_classes=4, l2=0.01)
    model._init_params(6, np.float64, rng)
    for b in model.bs:
        b[:] = rng.normal(size=b.shape)  # b khác 0 để kiểm tra cả gradient của b

    _, dWs, dbs = model.loss_and_grads(X, y)

    def loss():
        return model.loss_and_grads(X, y)[0]

    for W, dW in zip(model.Ws, dWs):
        assert np.allclose(dW, numerical_grad(loss, W), atol=1e-5)
    for b, db in zip(model.bs, dbs):
        assert np.allclose(db, numerical_grad(loss, b), atol=1e-5)


def test_overfits_small_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 20)).astype(np.float32)
    y = rng.integers(0, 10, size=100)  # nhãn ngẫu nhiên: chỉ học thuộc mới đúng hết
    model = MLP(hidden_sizes=(64,), lr=0.1, epochs=300, batch_size=32).fit(X, y)
    assert np.mean(model.predict(X) == y) == 1.0
