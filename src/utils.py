"""Tiện ích chung: cố định seed, đo thời gian."""

import random
import time

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
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


PROBS_DIR = "results/probs"


def save_probs(probs, dataset: str, model_name: str, split: str) -> str:
    """Lưu xác suất dự đoán (N, n_classes) ra results/probs/<dataset>/<model>_<split>.npy.

    Thứ tự mẫu đúng như load_dataset() trả về; dùng cho phần ensemble mà không phải train lại.
    """
    import os

    path = os.path.join(PROBS_DIR, dataset, f"{model_name}_{split}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(probs, dtype=np.float32))
    return path


def load_probs(dataset: str, model_name: str, split: str) -> np.ndarray:
    import os

    return np.load(os.path.join(PROBS_DIR, dataset, f"{model_name}_{split}.npy"))
