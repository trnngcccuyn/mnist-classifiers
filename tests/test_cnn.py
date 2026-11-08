"""Kiểm tra SimpleCNN: đúng interface chung, đúng số tham số, tái tạo được theo seed,
học thuộc được tập nhỏ. Bỏ qua cả file nếu chưa cài PyTorch."""

import numpy as np
import pytest

pytest.importorskip("torch")

from src.models.cnn import ImprovedCNN, SimpleCNN  # noqa: E402


def random_data(n=64, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 784), dtype=np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, y


def test_n_params_matches_architecture():
    conv1 = 32 * 1 * 3 * 3 + 32
    conv2 = 64 * 32 * 3 * 3 + 64
    fc1 = 64 * 5 * 5 * 128 + 128
    fc2 = 128 * 10 + 10
    assert SimpleCNN().n_params() == conv1 + conv2 + fc1 + fc2 == 225_034


def test_predict_proba_shape_and_sums_to_one():
    X, y = random_data()
    model = SimpleCNN(epochs=1, batch_size=32, device="cpu").fit(X, y)
    P = model.predict_proba(X, batch_size=10)  # 64 không chia hết cho 10: kiểm tra batch cuối
    assert P.shape == (64, 10)
    assert P.dtype == np.float32
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(model.predict(X), P.argmax(axis=1))


def test_same_seed_same_result_different_seed_differs():
    X, y = random_data()
    kw = {"epochs": 2, "batch_size": 16, "device": "cpu"}
    P0 = SimpleCNN(seed=0, **kw).fit(X, y).predict_proba(X)
    P0_again = SimpleCNN(seed=0, **kw).fit(X, y).predict_proba(X)
    P1 = SimpleCNN(seed=1, **kw).fit(X, y).predict_proba(X)
    assert np.allclose(P0, P0_again, atol=1e-5)
    assert not np.allclose(P0, P1, atol=1e-3)


def test_history_has_one_python_float_per_epoch():
    X, y = random_data()
    model = SimpleCNN(epochs=3, batch_size=32, device="cpu").fit(X, y, X, y)
    for key in ("train_loss", "val_loss", "val_acc"):
        assert len(model.history[key]) == 3
        assert all(type(v) is float for v in model.history[key])


def test_no_validation_history_without_val_set():
    X, y = random_data()
    model = SimpleCNN(epochs=2, device="cpu").fit(X, y)
    assert len(model.history["train_loss"]) == 2
    assert model.history["val_acc"] == []


def test_augment_is_called_on_train_batches_only():
    X, y = random_data()
    shapes = []

    def augment(X_img, rng):
        shapes.append(X_img.shape)
        return X_img[:, ::-1, :].copy()  # biến đổi bất kỳ, giữ nguyên shape

    SimpleCNN(epochs=1, batch_size=32, augment=augment, device="cpu").fit(X, y, X, y)
    assert shapes == [(32, 28, 28), (32, 28, 28)]  # 2 batch train, không gọi trên val


def test_overfits_small_dataset():
    X, y = random_data(n=64)  # nhãn ngẫu nhiên: chỉ học thuộc mới đúng hết
    model = SimpleCNN(epochs=100, batch_size=32, device="cpu").fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.95


# ---------------------------------------------------------------- ImprovedCNN


def test_improved_n_params_matches_architecture():
    convs = 1 * 16 * 9 + 16 * 16 * 9 + 16 * 32 * 9 + 32 * 32 * 9
    fc = 32 * 7 * 7 * 128 + 128 + 128 * 10 + 10
    bn = 2 * (16 + 16 + 32 + 32)  # gamma + beta mỗi kênh
    conv_bias = 16 + 16 + 32 + 32
    assert ImprovedCNN().n_params() == convs + bn + fc == 218_586
    assert ImprovedCNN(batchnorm=False).n_params() == convs + conv_bias + fc == 218_490


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"batchnorm": False, "dropout": 0.0, "schedule": None},
        {"channels": (8, 16), "hidden": 32},
    ],
)
def test_improved_interface_and_reproducible(kw):
    X, y = random_data()
    common = {"epochs": 2, "batch_size": 16, "device": "cpu", **kw}
    m0 = ImprovedCNN(seed=0, **common).fit(X, y, X, y)
    P = m0.predict_proba(X, batch_size=10)
    assert P.shape == (64, 10) and P.dtype == np.float32
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    assert len(m0.history["val_acc"]) == 2
    P_again = ImprovedCNN(seed=0, **common).fit(X, y, X, y).predict_proba(X)
    assert np.allclose(P, P_again, atol=1e-5)


def test_improved_predictions_do_not_depend_on_batch_in_eval_mode():
    """BatchNorm và Dropout phải ở chế độ eval khi dự đoán: kết quả một ảnh không phụ thuộc
    vào các ảnh khác trong cùng batch, và gọi hai lần cho cùng kết quả."""
    X, y = random_data()
    m = ImprovedCNN(epochs=1, batch_size=16, device="cpu").fit(X, y)
    P_all = m.predict_proba(X, batch_size=64)
    P_one = np.concatenate([m.predict_proba(X[i : i + 1]) for i in range(5)])
    assert np.allclose(P_all[:5], P_one, atol=1e-5)
    assert np.allclose(P_all, m.predict_proba(X, batch_size=64), atol=1e-6)


def test_cosine_schedule_decays_lr_to_zero():
    X, y = random_data()
    m = ImprovedCNN(epochs=2, batch_size=16, device="cpu")
    m.fit(X, y)
    import torch

    opt = torch.optim.Adam(m.net_.parameters(), lr=1e-3)
    sched = m._make_scheduler(opt, total_steps=8)
    lrs = []
    for _ in range(8):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < 1e-3 and np.all(np.diff(lrs) <= 0) and lrs[-1] < 1e-9
    assert ImprovedCNN(schedule=None)._make_scheduler(opt, 8) is None


def test_improved_overfits_small_dataset():
    X, y = random_data(n=64)
    model = ImprovedCNN(epochs=60, batch_size=32, dropout=0.0, schedule=None, device="cpu")
    assert np.mean(model.fit(X, y).predict(X) == y) >= 0.9


def test_simple_cnn_has_no_scheduler():
    assert SimpleCNN()._make_scheduler(None, 10) is None
