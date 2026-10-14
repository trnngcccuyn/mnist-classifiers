"""Thí nghiệm siêu tham số cho Gaussian Naive Bayes.

Chạy: python -m experiments.hyperparams.naive_bayes
"""

from src.data import load_dataset
from src.metrics import evaluate
from src.models.naive_bayes import GaussianNaiveBayes

from .common import sweep

VAR_SMOOTHINGS = [1e-9, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def main():
    data = load_dataset("mnist")
    results = sweep(GaussianNaiveBayes, "var_smoothing", VAR_SMOOTHINGS, data)
    best = results.loc[results["val_acc"].idxmax()]
    smoothing = float(best["var_smoothing"])
    model = GaussianNaiveBayes(var_smoothing=smoothing).fit(data["X_train"], data["y_train"])
    test_metrics = evaluate(data["y_test"], model.predict(data["X_test"]))


if __name__ == "__main__":
    main()
