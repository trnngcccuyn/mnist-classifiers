"""Decision Tree và Random Forest triển khai trực tiếp bằng NumPy. Phụ trách: Tiên.

Có sử dụng OpenAI Codex để hỗ trợ viết và rà soát mã; cần khai báo trong báo cáo.
Phần lõi tự tính Gini, chọn phép chia, xây cây và lấy mẫu bootstrap.
scikit-learn chỉ dùng trong tests/ để đối chiếu.

Mặc định mỗi nút của rừng xét sqrt(D) đặc trưng và tối đa 16 ngưỡng mỗi đặc trưng.
Đây là tìm kiếm ngưỡng xấp xỉ để giảm thời gian chạy MNIST. Đặt max_thresholds=None
để xét tất cả trung điểm giữa các giá trị khác nhau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseClassifier


def gini(counts: np.ndarray) -> float:
    """Gini = 1 - tổng p_c**2; counts là số mẫu theo từng lớp, không phải nhãn."""
    total = counts.sum()
    if total == 0:
        return 0.0
    return float(1.0 - np.sum((counts / total) ** 2))


@dataclass
class _Node:
    proba: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None


def _check_X(X, n_features=None):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X phải là ma trận (N, D) hữu hạn với D > 0")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Số đặc trưng phải bằng số đặc trưng khi fit")
    return X


class DecisionTree:
    """Cây phân loại dùng Gini và giới hạn số đặc trưng/ngưỡng được xét tại mỗi nút."""

    def __init__(
        self,
        max_depth: int | None = 12,
        min_samples_split: int = 10,
        max_features=None,
        rng=None,
        min_samples_leaf: int = 1,
        max_thresholds: int | None = 16,
    ):
        if max_depth is not None and (
            not isinstance(max_depth, (int, np.integer)) or max_depth < 0
        ):
            raise ValueError("max_depth phải là số nguyên >= 0 hoặc None")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError("min_samples_split phải là số nguyên >= 2")
        if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf phải là số nguyên >= 1")
        if max_thresholds is not None and (
            not isinstance(max_thresholds, (int, np.integer)) or max_thresholds < 1
        ):
            raise ValueError("max_thresholds phải là số nguyên >= 1 hoặc None")
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.max_features, self.rng = max_features, rng
        self.min_samples_leaf, self.max_thresholds = min_samples_leaf, max_thresholds

    def fit(self, X, y, *, classes=None):
        """Xây cây; classes giữ thứ tự cột xác suất chung giữa các cây bootstrap."""
        X = _check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError("y phải có dạng (N,) và train phải có ít nhất một mẫu")
        self.classes_ = np.unique(y) if classes is None else np.asarray(classes)
        if self.classes_.ndim != 1 or not np.array_equal(self.classes_, np.unique(self.classes_)):
            raise ValueError("classes phải là danh sách nhãn đã sắp xếp và không trùng")
        if not np.isin(y, self.classes_).all():
            raise ValueError("classes phải chứa mọi nhãn trong y")
        self.n_features_in_ = X.shape[1]
        self._n_features = self._resolve_max_features()
        self._rng = np.random.default_rng(42 if self.rng is None else self.rng)
        self._X, self._y = X, np.searchsorted(self.classes_, y)
        try:
            self.root_ = self._grow_tree(np.arange(len(y)), depth=0)
        finally:
            # Cây đã học chỉ giữ các nút, không giữ toàn bộ dữ liệu train.
            del self._X, self._y
        return self

    def _resolve_max_features(self):
        """Đổi cấu hình max_features thành số đặc trưng cần lấy tại mỗi nút."""
        if self.max_features is None:
            return self.n_features_in_
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_in_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_in_)))
        if isinstance(self.max_features, (int, np.integer)) and (
            1 <= self.max_features <= self.n_features_in_
        ):
            return int(self.max_features)
        raise ValueError("max_features phải là None, 'sqrt', 'log2' hoặc số nguyên 1..D")

    def _candidate_thresholds(self, values):
        """Lấy các trung điểm; giới hạn số ngưỡng theo max_thresholds nếu được đặt."""
        unique_values = np.unique(values).astype(np.float64)
        positions = np.arange(len(unique_values) - 1)
        if self.max_thresholds is not None and len(positions) > self.max_thresholds:
            positions = np.linspace(0, len(unique_values) - 2, self.max_thresholds, dtype=int)
        lower = unique_values[positions]
        upper = unique_values[positions + 1]
        return lower + (upper - lower) / 2

    def _split_impurities(self, values, labels, thresholds, counts):
        """Tính Gini sau chia cho mọi ngưỡng của một đặc trưng bằng histogram."""
        n_classes = len(self.classes_)
        # side='left' đảm bảo điều kiện nhánh trái là x <= threshold.
        bins = np.searchsorted(thresholds, values, side="left")
        histogram = np.bincount(
            bins * n_classes + labels, minlength=(len(thresholds) + 1) * n_classes
        ).reshape(-1, n_classes)
        left_counts = histogram.cumsum(axis=0)[:-1].astype(np.float64)
        right_counts = counts - left_counts
        n_left = left_counts.sum(axis=1)
        n_right = right_counts.sum(axis=1)
        valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)

        # (N_L * Gini_L + N_R * Gini_R) / N.
        impurities = (
            n_left
            - (left_counts * left_counts).sum(axis=1) / np.maximum(n_left, 1)
            + n_right
            - (right_counts * right_counts).sum(axis=1) / np.maximum(n_right, 1)
        ) / len(labels)
        impurities[~valid] = np.inf
        return impurities

    def _best_split(self, indices, counts):
        """Chọn phép chia giảm Gini nhiều nhất trong các đặc trưng/ngưỡng được xét."""
        labels = self._y[indices]
        features = self._rng.choice(self.n_features_in_, self._n_features, replace=False)
        best_impurity = gini(counts)
        best_split = None
        for feature in features:
            values = self._X[indices, feature]
            thresholds = self._candidate_thresholds(values)
            if len(thresholds) == 0:
                continue
            impurities = self._split_impurities(values, labels, thresholds, counts)
            position = int(np.argmin(impurities))
            if impurities[position] < best_impurity - 1e-12:
                best_impurity = impurities[position]
                best_split = int(feature), float(thresholds[position])
        return best_split

    def _grow_tree(self, indices, depth):
        counts = np.bincount(self._y[indices], minlength=len(self.classes_))
        node = _Node(proba=counts / len(indices))
        if (
            np.count_nonzero(counts) == 1
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node
        split = self._best_split(indices, counts)
        if split is None:
            return node
        node.feature, node.threshold = split
        mask = self._X[indices, node.feature] <= node.threshold
        node.left = self._grow_tree(indices[mask], depth + 1)
        node.right = self._grow_tree(indices[~mask], depth + 1)
        return node

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "root_"):
            raise RuntimeError("Cần gọi fit trước khi dự đoán")
        X = _check_X(X, self.n_features_in_)
        return self._predict_proba(X)

    def _predict_proba(self, X):
        """Duyệt cây; X đã được kiểm tra bởi cây hoặc rừng trước khi gọi."""
        proba = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            if node.feature is None:
                proba[indices] = node.proba
            else:
                mask = X[indices, node.feature] <= node.threshold
                stack.append((node.left, indices[mask]))
                stack.append((node.right, indices[~mask]))
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


class RandomForest(BaseClassifier):
    """Rừng bootstrap với xác suất dự đoán là trung bình xác suất của các cây."""

    name = "random_forest"

    def __init__(
        self,
        n_trees: int = 50,
        max_depth: int | None = 12,
        max_features="sqrt",
        seed: int = 42,
        min_samples_split: int = 10,
        min_samples_leaf: int = 1,
        max_thresholds: int | None = 16,
    ):
        if not isinstance(n_trees, (int, np.integer)) or n_trees < 1:
            raise ValueError("n_trees phải là số nguyên >= 1")
        self.n_trees, self.max_depth, self.max_features, self.seed = (
            n_trees,
            max_depth,
            max_features,
            seed,
        )
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.max_thresholds = max_thresholds
        self.trees = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Mỗi cây học từ N mẫu lấy có hoàn lại; validation không dùng để xây cây."""
        X = _check_X(X_train)
        y = np.asarray(y_train)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError("y_train phải có dạng (N,) và train phải có ít nhất một mẫu")
        self.classes_, self.n_features_in_ = np.unique(y), X.shape[1]
        rng = np.random.default_rng(self.seed)
        trees = []
        for _ in range(self.n_trees):
            bootstrap = rng.integers(0, len(y), size=len(y))
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                rng=np.random.default_rng(rng.integers(0, 2**32)),
                min_samples_leaf=self.min_samples_leaf,
                max_thresholds=self.max_thresholds,
            )
            tree.fit(X[bootstrap], y[bootstrap], classes=self.classes_)
            trees.append(tree)
        self.trees = trees
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Cần gọi fit trước khi dự đoán")
        X = _check_X(X, self.n_features_in_)
        proba = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for tree in self.trees:
            proba += tree._predict_proba(X)
        return proba / len(self.trees)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
