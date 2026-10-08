"""Đối chiếu cài đặt của nhóm với scikit-learn.  Phụ trách: Tiên

Chạy:  pytest tests/
Gỡ dòng skip của từng test khi phần tương ứng đã cài xong.
"""

import pytest


@pytest.mark.skip(reason="TODO: chưa cài metrics")
def test_metrics_match_sklearn():
    # So sánh accuracy / precision / recall / F1 macro / confusion matrix với sklearn.metrics
    ...


@pytest.mark.skip(reason="TODO: chưa cài Naive Bayes")
def test_naive_bayes_close_to_sklearn():
    # sklearn.naive_bayes.GaussianNB: accuracy chênh không quá vài điểm %
    ...


@pytest.mark.skip(reason="TODO: chưa cài Logistic Regression")
def test_logistic_regression_close_to_sklearn(): ...


@pytest.mark.skip(reason="TODO: chưa cài Random Forest")
def test_random_forest_close_to_sklearn(): ...
