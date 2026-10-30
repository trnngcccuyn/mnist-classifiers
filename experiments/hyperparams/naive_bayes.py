"""Thí nghiệm siêu tham số cho Gaussian và Bernoulli Naive Bayes.

Chạy: python -m experiments.hyperparams.naive_bayes --dataset mnist
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.metrics import evaluate
from src.models.naive_bayes import BernoulliNaiveBayes, GaussianNaiveBayes
from src.utils import Timer

from .common import sweep

VAR_SMOOTHINGS = [1e-9, 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0]
BINARIZE_THRESHOLDS = [0.05, 0.1, 0.3, 0.5, 0.7]
ALPHAS = [0.01, 0.1, 0.5, 1.0]


def _best(results, param_name):
    """Trả về giá trị tham số và val accuracy tốt nhất của một lần sweep."""
    row = results.loc[results["val_acc"].idxmax()]
    return float(row[param_name]), float(row["val_acc"])


def _evaluate_on_test(model_cls, params, data, val_accuracy):
    """Fit cấu hình được chọn và đánh giá test một lần, kèm thời gian thực hiện."""
    model = model_cls(**params)
    with Timer() as fit_timer:
        model.fit(data["X_train"], data["y_train"])
    with Timer() as predict_timer:
        prediction = model.predict(data["X_test"])
    test_metrics = evaluate(data["y_test"], prediction)
    row = {
        "model": model.name,
        "params": ", ".join(f"{key}={value:g}" for key, value in params.items()),
        "val_accuracy": val_accuracy,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "fit_time_s": fit_timer.elapsed,
        "test_predict_time_s": predict_timer.elapsed,
    }
    return model, row


def _plot_pixel_probs(model, dataset):
    """Vẽ xác suất pixel bật p[c, j] của từng lớp thành ảnh 28x28."""
    probs = np.exp(model.feature_log_prob_).reshape(-1, 28, 28)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for ax, image, label in zip(axes.flat, probs, model.classes_):
        shown = ax.imshow(image, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"lớp {label}")
        ax.axis("off")
    fig.colorbar(shown, ax=axes, shrink=0.8, label="P(pixel bật | lớp)")
    path = f"results/figures/{dataset}_bernoulli_naive_bayes_pixel_probs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    args = parser.parse_args()
    data = load_dataset(args.dataset)

    results = sweep(GaussianNaiveBayes, "var_smoothing", VAR_SMOOTHINGS, data, dataset=args.dataset)
    smoothing, gaussian_val = _best(results, "var_smoothing")

    # sweep chỉ quét một tham số mỗi lần: chọn ngưỡng với alpha=1, rồi chọn alpha.
    results = sweep(
        BernoulliNaiveBayes,
        "binarize",
        BINARIZE_THRESHOLDS,
        data,
        fixed={"alpha": 1.0},
        dataset=args.dataset,
    )
    threshold, _ = _best(results, "binarize")
    results = sweep(
        BernoulliNaiveBayes,
        "alpha",
        ALPHAS,
        data,
        fixed={"binarize": threshold},
        dataset=args.dataset,
    )
    alpha, bernoulli_val = _best(results, "alpha")

    _, gaussian_row = _evaluate_on_test(
        GaussianNaiveBayes, {"var_smoothing": smoothing}, data, gaussian_val
    )
    bernoulli, bernoulli_row = _evaluate_on_test(
        BernoulliNaiveBayes, {"alpha": alpha, "binarize": threshold}, data, bernoulli_val
    )
    _plot_pixel_probs(bernoulli, args.dataset)

    df = pd.DataFrame([gaussian_row, bernoulli_row])
    df.to_csv(f"results/tables/{args.dataset}_naive_bayes_variants.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
