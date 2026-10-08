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
