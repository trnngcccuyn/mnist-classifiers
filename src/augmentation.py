"""Tăng cường dữ liệu: xoay, dịch, nhiễu, lật ngang.  Phụ trách: Uyên

Đầu vào/ra: ảnh shape (N, 28, 28) float32 trong [0, 1]. Mọi hàm ngẫu nhiên nhận rng
(np.random.Generator) để tái tạo được theo seed; không sửa mảng đầu vào.

Xoay và dịch tự cài bằng NumPy (affine_transform): với mỗi pixel đầu ra p, lấy giá trị ở vị trí
nguồn  q = R(-θ) (p - c - t) + c  bằng nội suy song tuyến (bilinear), c là tâm ảnh, t là vector
dịch; điểm rơi ra ngoài ảnh lấy giá trị 0 (nền đen). Xoay và dịch gộp vào một lần nội suy để ảnh
không bị mờ hai lần. Dịch một số nguyên pixel và xoay 90° là chính xác, không bị nội suy làm mờ.

Dùng khi train: get_augment(dataset) trả về một hàm (X_img, rng) -> X_img, truyền vào tham số
augment của MLP / SimpleCNN; hàm này được gọi trên từng batch train, mỗi epoch một phép biến đổi
ngẫu nhiên mới (như có thêm vô hạn ảnh), không áp lên validation / test.
- MNIST: xoay U(-10°, 10°), dịch nguyên U{-2..2} px mỗi trục, nhiễu Gauss σ = 0.1 cho một nửa số
  ảnh. KHÔNG lật: chữ số lật ngang không còn là chữ số đó (2, 3, 7 ...).
- Fashion-MNIST: như trên, thêm lật ngang với xác suất 0.5 (áo, giày lật vẫn là áo, giày).

Dùng khi đánh giá độ bền: perturb(X, kind, magnitude, rng) biến đổi với độ lớn cố định
(dịch đúng `magnitude` px theo một hướng ngẫu nhiên, xoay đúng ±`magnitude` độ).
"""

import numpy as np


def affine_transform(X: np.ndarray, angles_deg, shifts) -> np.ndarray:
    """Xoay quanh tâm ảnh rồi dịch, mỗi ảnh một tham số riêng.

    X: (N, H, W). angles_deg: (N,) độ, dương là ngược chiều kim đồng hồ khi hiển thị ảnh.
    shifts: (N, 2) số pixel dịch theo (hàng, cột): (+1, 0) là dịch xuống, (0, +1) là sang phải.
    """
    X = np.asarray(X)
    n, h, w = X.shape
    # float32 đủ chính xác cho ảnh 28x28 và nhanh hơn float64 khoảng 2 lần
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float32)).reshape(n, 1, 1)
    shifts = np.asarray(shifts, dtype=np.float32).reshape(n, 2)
    cy, cx = np.float32((h - 1) / 2), np.float32((w - 1) / 2)

    # Toạ độ (hàng, cột) của pixel đầu ra, đưa về gốc là tâm ảnh và bỏ phần dịch
    rows, cols = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    y = rows[None] - cy - shifts[:, 0, None, None]
    x = cols[None] - cx - shifts[:, 1, None, None]
    # Xoay ngược -θ để tìm điểm nguồn. Trục hàng hướng xuống nên xoay "ngược chiều kim đồng hồ
    # khi hiển thị" ứng với ma trận [[cos, -sin], [sin, cos]] trên (cột, -hàng).
    cos, sin = np.cos(angles), np.sin(angles)
    src_x = cos * x - sin * y + cx
    src_y = sin * x + cos * y + cy

    # Nội suy song tuyến. Đệm 2 pixel 0 quanh ảnh và kẹp toạ độ vào [-1, H] (hay [-1, W]): mọi
    # điểm ngoài ảnh đều đọc vào phần đệm nên ra 0, không cần xử lý riêng.
    pad = np.pad(X, ((0, 0), (2, 2), (2, 2)))
    src_y = np.clip(src_y, -1, h)
    src_x = np.clip(src_x, -1, w)
    y0, x0 = np.floor(src_y).astype(np.int64), np.floor(src_x).astype(np.int64)
    wy, wx = src_y - y0, src_x - x0
    idx = np.arange(n)[:, None, None]

    def at(yy, xx):
        return pad[idx, yy + 2, xx + 2]

    out = (
        at(y0, x0) * (1 - wy) * (1 - wx)
        + at(y0, x0 + 1) * (1 - wy) * wx
        + at(y0 + 1, x0) * wy * (1 - wx)
        + at(y0 + 1, x0 + 1) * wy * wx
    )
    return out.astype(X.dtype, copy=False)


def rotate(X: np.ndarray, max_deg: float = 10.0, rng=None) -> np.ndarray:
    """Xoay mỗi ảnh một góc ngẫu nhiên U(-max_deg, max_deg)."""
    rng = rng or np.random.default_rng()
    angles = rng.uniform(-max_deg, max_deg, size=len(X))
    return affine_transform(X, angles, np.zeros((len(X), 2)))


def shift(X: np.ndarray, max_px: int = 2, rng=None) -> np.ndarray:
    """Dịch mỗi ảnh một số nguyên pixel ngẫu nhiên trong [-max_px, max_px] theo mỗi trục."""
    rng = rng or np.random.default_rng()
    shifts = rng.integers(-max_px, max_px + 1, size=(len(X), 2))
    return affine_transform(X, np.zeros(len(X)), shifts)


def add_noise(X: np.ndarray, std: float = 0.1, p: float = 1.0, rng=None) -> np.ndarray:
    """Cộng nhiễu Gauss N(0, std²) vào một tỉ lệ p số ảnh rồi kẹp về [0, 1]."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, std, size=X.shape) * (rng.random(len(X)) < p)[:, None, None]
    return np.clip(X + noise, 0.0, 1.0).astype(X.dtype, copy=False)


def hflip(X: np.ndarray, p: float = 0.5, rng=None) -> np.ndarray:
    """Lật ngang một tỉ lệ p số ảnh."""
    rng = rng or np.random.default_rng()
    flip = rng.random(len(X)) < p
    out = X.copy()
    out[flip] = X[flip, :, ::-1]
    return out


class Augment:
    """Pipeline dùng khi train: xoay + dịch (một lần nội suy) -> nhiễu -> lật ngang.

    Gọi như hàm: aug(X_img, rng). Đặt tham số bằng 0 để tắt từng phép.
    """

    def __init__(self, max_deg=10.0, max_px=2, noise_std=0.1, noise_p=0.5, flip_p=0.0):
        self.max_deg, self.max_px = max_deg, max_px
        self.noise_std, self.noise_p, self.flip_p = noise_std, noise_p, flip_p

    def __call__(self, X: np.ndarray, rng) -> np.ndarray:
        n = len(X)
        angles = rng.uniform(-self.max_deg, self.max_deg, size=n)
        shifts = rng.integers(-self.max_px, self.max_px + 1, size=(n, 2))
        X = affine_transform(X, angles, shifts)
        if self.noise_std > 0 and self.noise_p > 0:
            X = add_noise(X, self.noise_std, self.noise_p, rng)
        if self.flip_p > 0:
            X = hflip(X, self.flip_p, rng)
        return X

    def __repr__(self):
        return (
            f"Augment(max_deg={self.max_deg}, max_px={self.max_px}, noise_std={self.noise_std}, "
            f"noise_p={self.noise_p}, flip_p={self.flip_p})"
        )


def get_augment(dataset: str = "mnist") -> Augment:
    """Pipeline augmentation cho từng bộ dữ liệu; chỉ Fashion-MNIST được lật ngang."""
    if dataset == "mnist":
        return Augment()
    if dataset == "fashion_mnist":
        return Augment(flip_p=0.5)
    raise ValueError(f"dataset không hợp lệ: {dataset}")


def perturb(X: np.ndarray, kind: str, magnitude: float, rng) -> np.ndarray:
    """Biến đổi với độ lớn cố định, để đo độ bền của mô hình trên ảnh test.

    kind = "shift": dịch đúng `magnitude` px (số nguyên) lên / xuống / trái / phải, ngẫu nhiên.
    kind = "rotate": xoay đúng +`magnitude` hoặc -`magnitude` độ, dấu ngẫu nhiên.
    """
    n = len(X)
    if kind == "shift":
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        shifts = directions[rng.integers(0, 4, size=n)] * int(magnitude)
        return affine_transform(X, np.zeros(n), shifts)
    if kind == "rotate":
        angles = rng.choice([-1.0, 1.0], size=n) * magnitude
        return affine_transform(X, angles, np.zeros((n, 2)))
    raise ValueError(f"kind phải là 'shift' hoặc 'rotate', không phải {kind!r}")
