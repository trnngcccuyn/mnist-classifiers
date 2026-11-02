"""Kiểm tra Logistic Regression: gradient giải tích khớp gradient số."""

import numpy as np

from src.models.logistic_regression import LogisticRegression


def test_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.integers(0, 3, size=20)
    model = LogisticRegression(n_classes=3, l2=0.1)
    model.W = rng.normal(size=(5, 3))
    model.b = rng.normal(size=3)

    _, dW, db = model.loss_and_grad(X, y)

    eps = 1e-6
    num_dW = np.zeros_like(model.W)
    for i in range(model.W.shape[0]):
        for j in range(model.W.shape[1]):
            model.W[i, j] += eps
            plus = model.loss_and_grad(X, y)[0]
            model.W[i, j] -= 2 * eps
            minus = model.loss_and_grad(X, y)[0]
            model.W[i, j] += eps
            num_dW[i, j] = (plus - minus) / (2 * eps)
    num_db = np.zeros_like(model.b)
    for j in range(model.b.shape[0]):
        model.b[j] += eps
        plus = model.loss_and_grad(X, y)[0]
        model.b[j] -= 2 * eps
        minus = model.loss_and_grad(X, y)[0]
        model.b[j] += eps
        num_db[j] = (plus - minus) / (2 * eps)

    assert np.allclose(dW, num_dW, atol=1e-6)
    assert np.allclose(db, num_db, atol=1e-6)


def test_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model = LogisticRegression(n_classes=2, epochs=10, lr=0.1).fit(X, y)
    assert model.history["train_loss"][-1] < model.history["train_loss"][0]
    assert np.mean(model.predict(X) == y) > 0.9


def test_history_stores_python_floats():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=64)
    model = LogisticRegression(n_classes=3, epochs=2, l2=1e-3).fit(X, y, X, y)
    for key in ("train_loss", "val_loss", "val_acc"):
        assert len(model.history[key]) == 2
        assert all(type(v) is float for v in model.history[key])
