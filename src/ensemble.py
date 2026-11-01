"""Kết hợp xác suất dự đoán của nhiều mô hình: soft/hard voting và đo diversity.

Phụ trách theo phân công: Tiên.

Mọi hàm nhận xác suất đã tính sẵn (mỗi mô hình một mảng (N, n_classes)), không huấn luyện lại.
Với trọng số w[m] đã chuẩn hóa tổng bằng 1:
    soft voting: proba[i] = tổng_m w[m] * P_m[i]
    hard voting: phiếu[i, c] = tổng_m w[m] * [argmax P_m[i] == c]
    tỉ lệ cùng sai(a, b) = #(a sai và b sai) / #(a sai hoặc b sai)
"""

import itertools

import numpy as np


def _stack(probs):
    P = np.stack([np.asarray(p, dtype=np.float64) for p in probs])
    if P.ndim != 3:
        raise ValueError("Mỗi phần tử của probs phải có dạng (N, n_classes) và cùng shape")
    return P


def _normalize_weights(n_models, weights):
    if weights is None:
        return np.full(n_models, 1.0 / n_models)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_models,) or np.any(w < 0) or w.sum() == 0:
        raise ValueError("weights phải gồm đúng n_models số không âm, tổng lớn hơn 0")
    return w / w.sum()


def soft_vote(probs, weights=None) -> np.ndarray:
    """Trung bình có trọng số của các ma trận xác suất; trả về shape (N, n_classes)."""
    P = _stack(probs)
    w = _normalize_weights(len(P), weights)
    return np.tensordot(w, P, axes=1)


def hard_vote(probs, weights=None) -> np.ndarray:
    """Mỗi mô hình bỏ một phiếu (có trọng số) cho lớp nó chọn; trả về nhãn shape (N,).

    Hòa phiếu thì chọn lớp có xác suất soft voting cao hơn trong các lớp đang hòa.
    """
    P = _stack(probs)
    w = _normalize_weights(len(P), weights)
    _, n_samples, n_classes = P.shape
    votes = np.zeros((n_samples, n_classes))
    rows = np.arange(n_samples)
    for weight, p in zip(w, P):
        votes[rows, p.argmax(axis=1)] += weight
    tied = votes >= votes.max(axis=1, keepdims=True) - 1e-12
    soft = np.tensordot(w, P, axes=1)
    return np.where(tied, soft, -np.inf).argmax(axis=1)


def co_errors(pred_a, pred_b, y) -> tuple[int, int]:
    """Trả về (số mẫu cả hai cùng sai, số mẫu có ít nhất một mô hình sai)."""
    wrong_a = np.asarray(pred_a) != np.asarray(y)
    wrong_b = np.asarray(pred_b) != np.asarray(y)
    return int((wrong_a & wrong_b).sum()), int((wrong_a | wrong_b).sum())


def search_weights(probs, y, grid=(0, 1, 2, 3)):
    """Thử mọi bộ trọng số lấy từ grid, chọn bộ có accuracy soft voting cao nhất trên (probs, y).

    Chỉ gọi trên tập validation. Trả về (trọng số đã chuẩn hóa, accuracy);
    nếu nhiều bộ cùng tốt nhất thì giữ bộ gặp đầu tiên.
    """
    P = _stack(probs)
    y = np.asarray(y)
    best_w, best_acc = None, -1.0
    for combo in itertools.product(grid, repeat=len(P)):
        if sum(combo) == 0:
            continue
        w = _normalize_weights(len(P), combo)
        acc = float(np.mean(np.tensordot(w, P, axes=1).argmax(axis=1) == y))
        if acc > best_acc:
            best_w, best_acc = w, acc
    return best_w, best_acc
