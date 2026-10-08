"""Random Forest tự cài bằng NumPy.  Phụ trách: Tiên

Gợi ý tăng tốc: mỗi nút chỉ xét sqrt(n_features) đặc trưng, chỉ thử một số ít ngưỡng,
giới hạn độ sâu, và phát triển/thử nghiệm trên tập con nhỏ trước.
(Nếu nhóm đổi sang AdaBoost/Gradient Boosting thì đổi tên file và lớp cho khớp.)
"""

import numpy as np

from .base import BaseClassifier


class DecisionTree:
    def __init__(
        self, max_depth: int = 12, min_samples_split: int = 10, max_features=None, rng=None
    ):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.max_features, self.rng = max_features, rng

    def fit(self, X, y):
        # TODO (Tiên): xây cây đệ quy, chọn phép tách theo Gini
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        # TODO (Tiên)
        raise NotImplementedError


class RandomForest(BaseClassifier):
    name = "random_forest"

    def __init__(self, n_trees: int = 50, max_depth: int = 12, max_features="sqrt", seed: int = 42):
        self.n_trees, self.max_depth, self.max_features, self.seed = (
            n_trees,
            max_depth,
            max_features,
            seed,
        )
        self.trees = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # TODO (Tiên): với mỗi cây, lấy mẫu bootstrap rồi huấn luyện DecisionTree
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        # TODO (Tiên): trung bình xác suất (hoặc bỏ phiếu) của các cây
        raise NotImplementedError
