"""Thí nghiệm siêu tham số cho Gaussian Naive Bayes.

Chạy: python -m experiments.hyperparams.naive_bayes
"""

from src.data import load_dataset
from src.models.naive_bayes import GaussianNaiveBayes

from .common import sweep

VAR_SMOOTHINGS = [1e-4, 1e-3, 1e-2, 1e-1]


def main():
    data = load_dataset("mnist")
    sweep(GaussianNaiveBayes, "var_smoothing", VAR_SMOOTHINGS, data)


if __name__ == "__main__":
    main()
