"""Kiểm tra các hàm dùng chung trong nn_utils."""

import numpy as np

from src.models.nn_utils import cross_entropy, softmax


def test_softmax_rows_sum_to_one_and_stable():
    Z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    P = softmax(Z)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P[1], 1 / 3)  # không bị tràn số với giá trị lớn


def test_cross_entropy_of_uniform_prediction():
    P = np.full((4, 10), 0.1)
    y = np.array([0, 3, 5, 9])
    assert np.isclose(cross_entropy(P, y), np.log(10))
