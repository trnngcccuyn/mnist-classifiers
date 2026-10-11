"""Thí nghiệm siêu tham số cho Logistic Regression.  Phụ trách: Uyên

Chạy:  python -m experiments.hyperparams.logistic_regression
"""

from src.data import load_dataset
from src.models.logistic_regression import LogisticRegression

from .common import sweep

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]


def main():
    data = load_dataset("mnist")
    sweep(LogisticRegression, "lr", LEARNING_RATES, data)


if __name__ == "__main__":
    main()
