"""Thí nghiệm siêu tham số cho Gaussian Naive Bayes.

Chạy: python -m experiments.hyperparams.naive_bayes
"""

import pandas as pd

from src.data import load_dataset
from src.metrics import evaluate
from src.models.naive_bayes import GaussianNaiveBayes
from src.utils import Timer

from .common import sweep

VAR_SMOOTHINGS = [1e-9, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def main():
    data = load_dataset("mnist")
    results = sweep(GaussianNaiveBayes, "var_smoothing", VAR_SMOOTHINGS, data)
    best = results.loc[results["val_acc"].idxmax()]
    smoothing = float(best["var_smoothing"])
    model = GaussianNaiveBayes(var_smoothing=smoothing)
    with Timer() as fit_timer:
        model.fit(data["X_train"], data["y_train"])
    with Timer() as predict_timer:
        prediction = model.predict(data["X_test"])
    test_metrics = evaluate(data["y_test"], prediction)
    summary = {
        "var_smoothing": smoothing,
        "train_size": len(data["y_train"]),
        "val_accuracy": float(best["val_acc"]),
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "fit_time_s": fit_timer.elapsed,
        "test_predict_time_s": predict_timer.elapsed,
    }
    # sweep đã tạo sẵn results/tables.
    pd.DataFrame([summary]).to_csv("results/tables/mnist_naive_bayes_best.csv", index=False)
    print(f"Cấu hình được chọn và kết quả test: {summary}")


if __name__ == "__main__":
    main()
