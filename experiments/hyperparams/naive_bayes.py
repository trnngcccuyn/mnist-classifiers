"""Thí nghiệm siêu tham số cho Gaussian và Bernoulli Naive Bayes.  Phụ trách: Tiên

Chạy: python -m experiments.hyperparams.naive_bayes --dataset mnist

- Gaussian: sweep var_smoothing.
- Bernoulli: sweep chỉ quét một tham số mỗi lần, nên chọn ngưỡng binarize với alpha = 1,
  rồi chọn alpha với ngưỡng đó.
Cấu hình tốt nhất của mỗi biến thể (theo validation) được đánh giá trên test một lần.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.models.naive_bayes import BernoulliNaiveBayes, GaussianNaiveBayes
from src.plotting import save_figure

from .common import best, evaluate_on_test, sweep

VAR_SMOOTHINGS = [1e-9, 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0]
BINARIZE_THRESHOLDS = [0.05, 0.1, 0.3, 0.5, 0.7]
ALPHAS = [0.01, 0.1, 0.5, 1.0]


def plot_pixel_probs(model, dataset):
    """Vẽ xác suất pixel bật p[c, j] của từng lớp thành ảnh 28x28."""
    probs = np.exp(model.feature_log_prob_).reshape(-1, 28, 28)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for ax, image, label in zip(axes.flat, probs, model.classes_):
        shown = ax.imshow(image, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"lớp {label}")
        ax.axis("off")
    fig.colorbar(shown, ax=axes, shrink=0.8, label="P(pixel bật | lớp)")
    fig.suptitle("Bernoulli NB: xác suất pixel bật của từng lớp")
    save_figure(fig, f"{dataset}_bernoulli_naive_bayes_pixel_probs.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    args = parser.parse_args()
    data = load_dataset(args.dataset)

    gaussian = best(
        sweep(GaussianNaiveBayes, "var_smoothing", VAR_SMOOTHINGS, data, dataset=args.dataset)
    )
    threshold = best(
        sweep(
            BernoulliNaiveBayes,
            "binarize",
            BINARIZE_THRESHOLDS,
            data,
            fixed={"alpha": 1.0},
            dataset=args.dataset,
        )
    )["binarize"]
    bernoulli = best(
        sweep(
            BernoulliNaiveBayes,
            "alpha",
            ALPHAS,
            data,
            fixed={"binarize": float(threshold)},
            dataset=args.dataset,
        )
    )

    chosen = [
        (GaussianNaiveBayes, {"var_smoothing": float(gaussian["var_smoothing"])}, gaussian),
        (
            BernoulliNaiveBayes,
            {"alpha": float(bernoulli["alpha"]), "binarize": float(threshold)},
            bernoulli,
        ),
    ]
    rows = []
    for model_cls, params, sweep_row in chosen:
        model = model_cls(**params)
        rows.append(
            {
                "model": model.name,
                "params": ", ".join(f"{key}={value:g}" for key, value in params.items()),
                "val_accuracy": sweep_row["val_acc"],
                **evaluate_on_test(model, data),
            }
        )
    plot_pixel_probs(model, args.dataset)  # model cuối vòng lặp là Bernoulli

    df = pd.DataFrame(rows)
    df.to_csv(f"results/tables/{args.dataset}_naive_bayes_variants.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
