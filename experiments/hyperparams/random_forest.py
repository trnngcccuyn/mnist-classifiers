"""Thí nghiệm siêu tham số cho Random Forest.

Chạy: python -m experiments.hyperparams.random_forest
"""

from src.data import load_dataset
from src.models.random_forest import RandomForest

from .common import sweep

N_TREES = [10, 25, 50, 100]
MAX_DEPTHS = [4, 8, 12, 16]


def main():
    data = load_dataset("mnist")
    sweep(RandomForest, "n_trees", N_TREES, data)
    sweep(RandomForest, "max_depth", MAX_DEPTHS, data)


if __name__ == "__main__":
    main()
