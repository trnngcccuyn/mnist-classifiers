"""Kiểm tra augmentation: phép biến đổi đúng hình học (so với np.rot90 và dịch bằng tay), giữ
shape / kiểu / miền giá trị, tái tạo được theo seed, MNIST không bao giờ bị lật."""

import numpy as np
import pytest

from src.augmentation import (
    Augment,
    add_noise,
    affine_transform,
    get_augment,
    hflip,
    perturb,
    rotate,
    shift,
)


def images(n=6, seed=0):
    return np.random.default_rng(seed).random((n, 28, 28), dtype=np.float32)


def test_identity_transform_is_exact():
    X = images()
    assert np.array_equal(affine_transform(X, np.zeros(6), np.zeros((6, 2))), X)


@pytest.mark.parametrize("k, angle", [(1, 90.0), (2, 180.0), (-1, -90.0)])
def test_rotation_matches_rot90(k, angle):
    """Xoay quanh tâm (13.5, 13.5) một góc bội 90° đưa tâm pixel vào đúng tâm pixel: không nội
    suy, phải trùng np.rot90 (k > 0 là ngược chiều kim đồng hồ khi hiển thị)."""
    X = images()
    out = affine_transform(X, np.full(6, angle), np.zeros((6, 2)))
    assert np.allclose(out, np.rot90(X, k, axes=(1, 2)), atol=1e-5)


def test_integer_shift_matches_manual_shift_with_zero_fill():
    X = images()
    out = affine_transform(X, np.zeros(6), np.tile([1, -2], (6, 1)))  # xuống 1, sang trái 2
    expected = np.zeros_like(X)
    expected[:, 1:, :-2] = X[:, :-1, 2:]
    assert np.array_equal(out, expected)


def test_shift_out_of_image_gives_black_image():
    out = affine_transform(images(), np.zeros(6), np.full((6, 2), 30))
    assert np.all(out == 0)


def test_each_image_gets_its_own_parameters():
    X = np.repeat(images(1), 2, axis=0)
    out = affine_transform(X, [0.0, 90.0], np.zeros((2, 2)))
    assert np.array_equal(out[0], X[0])
    assert np.allclose(out[1], np.rot90(X[1]), atol=1e-5)


@pytest.mark.parametrize(
    "fn",
    [
        lambda X, rng: rotate(X, rng=rng),
        lambda X, rng: shift(X, rng=rng),
        lambda X, rng: add_noise(X, rng=rng),
        lambda X, rng: hflip(X, rng=rng),
        get_augment("mnist"),
        get_augment("fashion_mnist"),
    ],
    ids=["rotate", "shift", "add_noise", "hflip", "mnist", "fashion_mnist"],
)
def test_keeps_shape_dtype_range_and_does_not_modify_input(fn):
    X = images()
    X_copy = X.copy()
    out = fn(X, np.random.default_rng(0))
    assert out.shape == X.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.array_equal(X, X_copy)


def test_same_rng_seed_same_result():
    X = images()
    aug = get_augment("mnist")
    a = aug(X, np.random.default_rng(1))
    b = aug(X, np.random.default_rng(1))
    c = aug(X, np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def bright_pixel_positions(out):
    """Vị trí (hàng, cột) của pixel sáng nhất trong từng ảnh."""
    return np.array(np.divmod(out.reshape(len(out), -1).argmax(axis=1), 28)).T


def test_shift_stays_within_max_px():
    """Một chấm sáng ở (14, 14), dịch tối đa 2 px: phải rơi vào đủ 25 ô của hình vuông 5x5."""
    X = np.zeros((500, 28, 28), dtype=np.float32)
    X[:, 14, 14] = 1.0
    pos = bright_pixel_positions(shift(X, max_px=2, rng=np.random.default_rng(0)))
    assert np.all(np.abs(pos - 14) <= 2)
    assert len({tuple(p) for p in pos}) == 25


def test_mnist_never_flips_fashion_does():
    assert get_augment("mnist").flip_p == 0.0
    assert get_augment("fashion_mnist").flip_p == 0.5
    X = images()
    no_geom = dict(max_deg=0.0, max_px=0, noise_std=0.0)
    assert np.array_equal(Augment(**no_geom)(X, np.random.default_rng(0)), X)
    flipped = Augment(**no_geom, flip_p=1.0)(X, np.random.default_rng(0))
    assert np.array_equal(flipped, X[:, :, ::-1])


def test_get_augment_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        get_augment("cifar10")


def test_perturb_has_exact_magnitude():
    X = np.zeros((200, 28, 28), dtype=np.float32)
    X[:, 14, 14] = 1.0
    pos = bright_pixel_positions(perturb(X, "shift", 3, np.random.default_rng(0)))
    assert np.all(np.abs(pos - 14).sum(axis=1) == 3)  # đúng 3 px theo đúng một trục

    X = images()
    rot = perturb(X, "rotate", 90, np.random.default_rng(0))
    for img, r in zip(X, rot):
        assert np.allclose(r, np.rot90(img, 1), atol=1e-5) or np.allclose(
            r, np.rot90(img, -1), atol=1e-5
        )


def test_mlp_calls_augment_on_train_batches_only():
    from src.models.mlp import MLP

    rng = np.random.default_rng(0)
    X = rng.random((64, 784), dtype=np.float32)
    y = rng.integers(0, 10, size=64)
    shapes = []

    def augment(X_img, rng):
        shapes.append(X_img.shape)
        return X_img[:, ::-1, :].copy()

    MLP(hidden_sizes=(8,), epochs=1, batch_size=32, augment=augment).fit(X, y, X, y)
    assert shapes == [(32, 28, 28), (32, 28, 28)]
