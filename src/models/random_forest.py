"""Decision Tree và Random Forest tự cài bằng NumPy.  Phụ trách: Tiên

Decision Tree (CART, tiêu chí Gini):
    Gini(nút) = 1 - tổng_c p_c**2, với p_c là tỉ lệ mẫu lớp c trong nút.
    Tại mỗi nút, thử các cặp (pixel j, ngưỡng t): mẫu có x[j] <= t sang trái, còn lại sang phải.
    Chọn cặp có Gini sau chia nhỏ nhất:
        Gini_chia = (N_trái * Gini(trái) + N_phải * Gini(phải)) / N
    Dừng (thành lá) khi nút chỉ còn một lớp, có ít hơn min_samples_split mẫu, đạt max_depth,
    hoặc không phép chia nào làm Gini giảm. Lá lưu tỉ lệ các lớp, dùng làm xác suất dự đoán.

Để chạy đủ nhanh trên MNIST, mỗi nút chỉ xét:
    - max_features = "sqrt": sqrt(784) = 28 pixel chọn ngẫu nhiên;
    - max_thresholds = 16 ngưỡng cách đều trong các trung điểm giữa hai giá trị liền kề
      (max_thresholds = None: xét mọi trung điểm).

Random Forest:
    Mỗi cây học trên một mẫu bootstrap (rút N mẫu có hoàn lại từ N mẫu train).
    Xác suất của rừng = trung bình xác suất của các cây; nhãn = lớp có xác suất lớn nhất.

Các cây được train song song bằng joblib (chỉ để nhanh hơn, không đổi thuật toán). Mẫu bootstrap
và seed của từng cây được rút trước, theo thứ tự, nên:
    - kết quả giống hệt nhau với mọi n_jobs;
    - k cây đầu của rừng chính là rừng k cây cùng seed, nên thí nghiệm số cây chỉ cần train
      rừng lớn nhất một lần rồi gọi predict_proba(X, n_trees=k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from .base import BaseClassifier


def gini(counts):
    """Gini của một nút từ số mẫu mỗi lớp; counts có thể là ma trận (mỗi hàng một nút)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1)
    return np.where(total[..., 0] > 0, 1.0 - np.sum(p**2, axis=-1), 0.0)


@dataclass
class Node:
    proba: np.ndarray  # tỉ lệ các lớp trong nút
    feature: int | None = None  # None nghĩa là lá
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = 12,
        min_samples_split: int = 10,
        max_features: str | None = None,
        max_thresholds: int | None = 16,
        seed=42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.seed = seed

    def fit(self, X, y, classes=None):
        """classes: danh sách nhãn của cả rừng, để mọi cây có cùng thứ tự cột xác suất
        (mẫu bootstrap có thể thiếu một lớp)."""
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y) if classes is None else np.asarray(classes)
        y = np.searchsorted(self.classes_, y)  # đổi nhãn thành chỉ số 0..K-1
        n_features = X.shape[1]
        self._n_try = n_features if self.max_features is None else int(np.sqrt(n_features))
        self._rng = np.random.default_rng(self.seed)
        self.root_ = self._grow(X, y, np.arange(len(y)), depth=0)
        return self

    def _grow(self, X, y, indices, depth):
        """Xây cây con cho các mẫu indices (đệ quy)."""
        counts = np.bincount(y[indices], minlength=len(self.classes_))
        node = Node(proba=counts / len(indices))
        if (
            np.count_nonzero(counts) == 1
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        split = self._best_split(X, y, indices, counts)
        if split is None:
            return node
        node.feature, node.threshold = split
        goes_left = X[indices, node.feature] <= node.threshold
        node.left = self._grow(X, y, indices[goes_left], depth + 1)
        node.right = self._grow(X, y, indices[~goes_left], depth + 1)
        return node

    def _thresholds(self, values):
        """Trung điểm giữa các giá trị khác nhau liền kề; lấy cách đều nếu quá max_thresholds."""
        unique = np.unique(values).astype(np.float64)
        midpoints = (unique[:-1] + unique[1:]) / 2
        if self.max_thresholds is not None and len(midpoints) > self.max_thresholds:
            picked = np.linspace(0, len(midpoints) - 1, self.max_thresholds).astype(int)
            midpoints = midpoints[picked]
        return midpoints

    def _best_split(self, X, y, indices, counts):
        """Trả về (pixel, ngưỡng) làm Gini giảm nhiều nhất, hoặc None nếu không giảm được."""
        n = len(indices)
        one_hot = np.eye(len(self.classes_), dtype=np.float32)[y[indices]]  # (n, K)
        best_gini, best_split = gini(counts), None

        features = self._rng.choice(X.shape[1], self._n_try, replace=False)
        for feature in features:
            values = X[indices, feature]
            thresholds = self._thresholds(values)
            if len(thresholds) == 0:
                continue  # pixel hằng số trong nút, không chia được
            # Hàng t: mẫu nào sang trái với ngưỡng t. Nhân với one-hot ra số mẫu mỗi lớp.
            goes_left = values[None, :] <= thresholds[:, None]  # (T, n)
            left_counts = goes_left @ one_hot  # (T, K)
            right_counts = counts - left_counts
            n_left = left_counts.sum(axis=1)
            split_gini = (n_left * gini(left_counts) + (n - n_left) * gini(right_counts)) / n

            t = int(np.argmin(split_gini))
            if split_gini[t] < best_gini - 1e-12:
                best_gini, best_split = split_gini[t], (int(feature), float(thresholds[t]))
        return best_split

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X)
        proba = np.zeros((len(X), len(self.classes_)))
        self._fill_proba(self.root_, X, np.arange(len(X)), proba)
        return proba

    def _fill_proba(self, node, X, indices, proba):
        """Đưa các mẫu indices đi xuống cây; tới lá thì ghi xác suất của lá."""
        if len(indices) == 0:
            return
        if node.feature is None:
            proba[indices] = node.proba
            return
        goes_left = X[indices, node.feature] <= node.threshold
        self._fill_proba(node.left, X, indices[goes_left], proba)
        self._fill_proba(node.right, X, indices[~goes_left], proba)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _fit_tree(X, y, bootstrap, classes, seed, params):
    """Train một cây trên mẫu bootstrap; để ở cấp module để joblib gửi sang tiến trình con."""
    return DecisionTree(seed=seed, **params).fit(X[bootstrap], y[bootstrap], classes)


class RandomForest(BaseClassifier):
    name = "random_forest"

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int | None = 24,
        max_features: str | None = "sqrt",
        min_samples_split: int = 10,
        max_thresholds: int | None = 16,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """X_val, y_val không dùng; chỉ giữ cho giống interface chung."""
        X, y = np.asarray(X_train), np.asarray(y_train)
        self.classes_ = np.unique(y)

        # Rút trước mẫu bootstrap và seed của từng cây, theo thứ tự.
        rng = np.random.default_rng(self.seed)
        draws = []
        for _ in range(self.n_trees):
            bootstrap = rng.integers(0, len(y), size=len(y))
            draws.append((bootstrap, rng.integers(0, 2**32)))

        params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "max_thresholds": self.max_thresholds,
        }
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(X, y, bootstrap, self.classes_, seed, params)
            for bootstrap, seed in draws
        )
        return self

    def predict_proba(self, X, n_trees: int | None = None) -> np.ndarray:
        """Trung bình xác suất của n_trees cây đầu; mặc định dùng cả rừng."""
        trees = self.trees if n_trees is None else self.trees[:n_trees]
        return sum(tree.predict_proba(X) for tree in trees) / len(trees)

    def predict(self, X, n_trees: int | None = None) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X, n_trees), axis=1)]
