"""Đưa ảnh vẽ tay về dạng giống MNIST trước khi đưa vào mạng.  Phụ trách: Uyên

MNIST gốc: chữ số trắng trên nền đen, được thu nhỏ cho vừa ô 20x20 (giữ tỉ lệ), rồi đặt vào
ảnh 28x28 sao cho tâm khối (center of mass) nằm giữa ảnh. Ảnh vẽ trên canvas thì khác hẳn
(nét đen trên nền trắng/trong suốt, to, lệch góc), nên nếu chỉ resize thẳng về 28x28 thì mô
hình đoán sai nhiều. Hàm ở đây làm lại đúng các bước trên. Chỉ dùng NumPy + Pillow.
"""

import numpy as np
from PIL import Image


def to_ink(img) -> np.ndarray:
    """Ảnh từ canvas (H, W), (H, W, 3) hoặc (H, W, 4), uint8 -> độ đậm của mực trong [0, 1].

    Mực = (1 - độ sáng) * độ phủ (alpha), nên dùng được cả khi canvas có nền trắng đặc
    lẫn khi nền trong suốt và chỉ có nét vẽ.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        gray, alpha = a.astype(np.float32), np.full(a.shape, 255.0, np.float32)
    else:
        rgb = a[..., :3].astype(np.float32)
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        alpha = a[..., 3].astype(np.float32) if a.shape[-1] == 4 else np.full(gray.shape, 255.0)
    return (1.0 - gray / 255.0) * (alpha / 255.0)


def to_mnist(ink: np.ndarray, threshold: float = 0.1) -> np.ndarray | None:
    """Độ đậm mực (H, W) trong [0, 1] -> ảnh 28x28 float32 trong [0, 1] kiểu MNIST.

    Trả None nếu canvas trống. Các bước: cắt khung bao quanh nét vẽ -> thu nhỏ cạnh dài về
    20 px (giữ tỉ lệ) -> đặt vào ảnh 28x28 -> dịch để tâm khối nằm ở giữa.
    """
    ink = np.asarray(ink, dtype=np.float32)
    rows = np.where((ink > threshold).any(axis=1))[0]
    cols = np.where((ink > threshold).any(axis=0))[0]
    if len(rows) == 0:
        return None
    crop = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )
    small = np.asarray(small, dtype=np.float32) / 255.0

    out = np.zeros((28, 28), np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    out[top : top + new_h, left : left + new_w] = small

    # Dịch theo tâm khối (làm tròn về số nguyên pixel), như cách tạo MNIST
    total = out.sum()
    if total > 0:
        ys, xs = np.indices(out.shape)
        cy, cx = (ys * out).sum() / total, (xs * out).sum() / total
        dy, dx = round(13.5 - cy), round(13.5 - cx)
        out = _shift(out, dy, dx)
    return np.clip(out, 0.0, 1.0)


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Dịch ảnh dy hàng, dx cột; phần trống điền 0 (không cuộn vòng như np.roll)."""
    out = np.zeros_like(img)
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out
