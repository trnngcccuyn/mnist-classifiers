"""Tiện ích chung: cố định seed, đo thời gian, ghi cấu hình máy, lưu xác suất dự đoán."""

import os
import platform
import random
import subprocess
import sys
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


def _cpu_name() -> str:
    """Tên CPU, ví dụ "Apple M2" hay "Intel(R) Core(TM) i7-12700H"."""
    try:
        if platform.system() == "Darwin":
            command = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.run(command, capture_output=True, text=True).stdout.strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb():
    """Dung lượng RAM (GB); None nếu hệ điều hành không cho đọc (Windows)."""
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):
        return None


def machine_info() -> dict:
    """Cấu hình máy đang chạy, ghi kèm kết quả để biết thời gian được đo trên máy nào.

    Không ghi tên máy (hostname) để không lộ thông tin cá nhân khi commit.
    """
    info = {
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


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
