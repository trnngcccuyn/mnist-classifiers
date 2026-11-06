"""Tiện ích chung: cố định seed, đo thời gian, lưu xác suất dự đoán."""

import os
import random
import time

import numpy as np

SEED = 42
PROBS_DIR = "results/probs"


def set_seed(seed: int = SEED) -> None:
    """Cố định bộ sinh số ngẫu nhiên toàn cục của random, NumPy và PyTorch.

    Các mô hình NumPy (Logistic Regression, MLP, Random Forest) không dùng bộ sinh toàn cục:
    mỗi mô hình tự tạo np.random.default_rng(self.seed) từ tham số seed của nó, nên kết quả
    chỉ phụ thuộc vào seed truyền vào mô hình. Naive Bayes không có yếu tố ngẫu nhiên.
    Hàm này thực sự cần cho CNN (PyTorch: khởi tạo trọng số, xáo trộn batch).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class Timer:
    """Dùng: with Timer() as t: ...  -> t.elapsed (giây)."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


def save_probs(probs, dataset: str, model_name: str, split: str) -> str:
    """Lưu xác suất dự đoán (N, n_classes) ra results/probs/<dataset>/<model>_<split>.npy.

    Thứ tự mẫu đúng như load_dataset() trả về; dùng cho phần ensemble mà không phải train lại.
    """
    path = os.path.join(PROBS_DIR, dataset, f"{model_name}_{split}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(probs, dtype=np.float32))
    return path


def load_probs(dataset: str, model_name: str, split: str) -> np.ndarray:
    return np.load(os.path.join(PROBS_DIR, dataset, f"{model_name}_{split}.npy"))
