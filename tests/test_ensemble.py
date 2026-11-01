"""Kiểm tra các hàm ensemble trên dữ liệu nhỏ tính tay được."""

import numpy as np
import pytest

from src.ensemble import co_errors, hard_vote, search_weights, soft_vote

# argmax từng hàng: P1 -> [0, 1, 1], P2 -> [0, 0, 1], P3 -> [1, 1, 0]
P1 = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
P2 = np.array([[0.6, 0.4], [0.7, 0.3], [0.45, 0.55]])
P3 = np.array([[0.3, 0.7], [0.2, 0.8], [0.9, 0.1]])
Y = np.array([0, 1, 1])


def test_soft_vote_equal_weights_is_mean():
    assert np.allclose(soft_vote([P1, P2]), (P1 + P2) / 2)


def test_soft_vote_normalizes_weights():
    assert np.allclose(soft_vote([P1, P2], weights=[3, 1]), 0.75 * P1 + 0.25 * P2)
    assert np.allclose(soft_vote([P1, P2], weights=[1, 0]), P1)


def test_hard_vote_majority():
    assert np.array_equal(hard_vote([P1, P2, P3]), [0, 1, 1])


def test_hard_vote_breaks_ties_with_soft_vote():
    # Mẫu thứ hai hòa 1-1; soft voting cho lớp 0 là 0.55 > 0.45.
    assert np.array_equal(hard_vote([P1, P2]), [0, 0, 1])


def test_co_errors_counts():
    assert co_errors([0, 0, 0], [1, 0, 1], Y) == (1, 3)


def test_search_weights_finds_best_model():
    weights, acc = search_weights([P2, P1], Y, grid=(0, 1))
    assert acc == 1.0
    assert np.allclose(weights, [0, 1])


def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        soft_vote([P1, P2], weights=[1, 2, 3])
    with pytest.raises(ValueError):
        soft_vote([P1, P2], weights=[0, 0])
