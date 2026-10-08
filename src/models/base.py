"""Interface chung cho mọi bộ phân lớp, để các script thí nghiệm dùng thay thế cho nhau."""

import numpy as np


class BaseClassifier:
    name = "base"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Huấn luyện. Trả về self. Mô hình lặp nên lưu lịch sử loss/accuracy vào self.history."""
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        """Trả về xác suất shape (N, n_classes)."""
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
