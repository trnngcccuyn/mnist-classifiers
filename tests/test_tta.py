"""Kiểm tra TTA bằng một mô hình giả tính được tay (không cần train)."""

import numpy as np

from src.tta import get_tta_augment, tta_predict_proba, tta_views_proba


class FakeModel:
    """Xác suất 2 lớp từ độ sáng nửa trái so với nửa phải của ảnh: lật ngang là đổi lớp."""

    def predict_proba(self, X):
        img = np.asarray(X).reshape(len(X), 28, 28)
        left, right = img[:, :, :14].sum(axis=(1, 2)), img[:, :, 14:].sum(axis=(1, 2))
        p_left = (left + 1) / (left + right + 2)
        return np.stack([p_left, 1 - p_left], axis=1)


def data(n=20, seed=0):
    return np.random.default_rng(seed).random((n, 784), dtype=np.float32)


def test_views_shape_and_first_view_is_original():
    X, model = data(), FakeModel()
    V = tta_views_proba(model, X, 4, get_tta_augment("mnist"), np.random.default_rng(0))
    assert V.shape == (5, 20, 2)
    assert np.allclose(V[0], model.predict_proba(X))
    assert not np.allclose(V[1], V[0])  # bản biến đổi khác ảnh gốc


def test_zero_views_is_plain_prediction():
    X, model = data(), FakeModel()
    P = tta_predict_proba(model, X, n_views=0, rng=np.random.default_rng(0))
    assert np.allclose(P, model.predict_proba(X))


def test_tta_is_mean_of_views_and_reproducible():
    X, model, aug = data(), FakeModel(), get_tta_augment("mnist")
    V = tta_views_proba(model, X, 3, aug, np.random.default_rng(1))
    P = tta_predict_proba(model, X, n_views=3, augment=aug, rng=np.random.default_rng(1))
    assert np.allclose(P, V.mean(axis=0))
    assert np.allclose(P.sum(axis=1), 1.0)


def test_tta_augment_has_no_noise_and_keeps_dataset_flip_rule():
    mnist, fashion = get_tta_augment("mnist"), get_tta_augment("fashion_mnist")
    assert mnist.noise_std == 0.0 and fashion.noise_std == 0.0
    assert mnist.flip_p == 0.0 and fashion.flip_p == 0.5
    assert mnist.max_deg == 10.0 and mnist.max_px == 2
