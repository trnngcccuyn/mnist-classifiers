"""Kiểm tra cây và rừng tự cài; scikit-learn chỉ làm kết quả tham chiếu."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models.random_forest import DecisionTree, RandomForest, gini


def test_gini():
    assert gini(np.array([4, 0])) == 0
    assert gini(np.array([2, 2])) == 0.5
    assert gini(np.array([0, 0])) == 0


def test_tree_matches_sklearn_on_unique_best_split():
    X = np.arange(8, dtype=np.float64).reshape(-1, 1)
    y = np.array([5, 5, 5, 9, 9, 9, 9, 9])
    query = np.array([[0.0], [2.5], [2.6], [6.0]])
    ours = DecisionTree(max_depth=1, min_samples_split=2, max_thresholds=None).fit(X, y)
    reference = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y)
    assert np.allclose(ours.predict_proba(query), reference.predict_proba(query))
    assert np.array_equal(ours.predict(query), reference.predict(query))


def test_constant_features_produce_class_distribution():
    X = np.zeros((6, 3))
    y = np.array([2, 2, 2, 2, 7, 7])
    model = DecisionTree(min_samples_split=2).fit(X, y)
    assert np.allclose(model.predict_proba(X), np.tile([2 / 3, 1 / 3], (6, 1)))
    assert np.array_equal(model.predict(X), np.full(6, 2))


def test_min_samples_leaf_constrains_split():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 0, 0, 1])
    model = DecisionTree(min_samples_split=2, min_samples_leaf=2).fit(X, y)
    assert np.allclose(model.predict_proba(X), [[1, 0], [1, 0], [0.5, 0.5], [0.5, 0.5]])


def test_bootstrap_preserves_columns_for_missing_classes():
    # Seed 0 lấy hai lần mẫu thứ hai: cây chỉ thấy lớp 20 trong bootstrap.
    X = np.array([[0.0], [1.0]])
    y = np.array([10, 20])
    model = RandomForest(n_trees=1, seed=0).fit(X, y)
    assert np.array_equal(model.classes_, [10, 20])
    assert np.allclose(model.predict_proba(X), [[0, 1], [0, 1]])
    assert np.array_equal(model.predict(X), [20, 20])


def test_forest_close_to_sklearn_on_held_out_data():
    X, y = make_blobs(n_samples=900, n_features=8, centers=3, cluster_std=2.5, random_state=42)
    X_train, X_test, y_train, y_test = X[:600], X[600:], y[:600], y[600:]
    ours = RandomForest(n_trees=15, max_depth=8, min_samples_split=2).fit(X_train, y_train)
    reference = RandomForestClassifier(
        n_estimators=15, max_depth=8, min_samples_split=2, max_features="sqrt", random_state=42
    ).fit(X_train, y_train)
    ours_acc = np.mean(ours.predict(X_test) == y_test)
    reference_acc = np.mean(reference.predict(X_test) == y_test)
    assert ours_acc >= 0.9
    # Bootstrap, chọn đặc trưng và giới hạn ngưỡng khác nhau: không so từng cây.
    assert abs(ours_acc - reference_acc) <= 0.05
    probabilities = ours.predict_proba(X_test)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1)
    expected = np.mean([tree.predict_proba(X_test) for tree in ours.trees], axis=0)
    assert np.allclose(probabilities, expected)


def test_seed_and_refit_are_reproducible():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForest(n_trees=4, max_depth=4, seed=7).fit(X, y)
    first = model.predict_proba(X)
    model.fit(X, y)
    assert len(model.trees) == 4
    assert np.array_equal(first, model.predict_proba(X))
    assert not hasattr(model.trees[0], "_X")


@pytest.mark.parametrize("model", [DecisionTree(), RandomForest(n_trees=2)])
def test_prediction_requires_fit(model):
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.zeros((1, 2)))


@pytest.mark.parametrize("model", [DecisionTree(), RandomForest(n_trees=2)])
def test_input_shape_validation(model):
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.array([0, 1]))
    model.fit(np.zeros((3, 2)), np.array([0, 1, 1]))
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, 3)))
    assert model.predict_proba(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_trees": 0}, {"max_depth": -1}, {"min_samples_leaf": 0}, {"max_thresholds": 0}],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        RandomForest(**kwargs).fit(np.zeros((3, 2)), np.array([0, 1, 1]))
