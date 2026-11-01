"""Thí nghiệm siêu tham số cho Logistic Regression.  Phụ trách: Uyên

Chạy:  python -m experiments.hyperparams.logistic_regression
       python -m experiments.hyperparams.logistic_regression --only l2 curve weights

Gồm (chọn theo validation, không dùng test):
- sweep learning rate và hệ số L2;
- learning curve: train loss, val loss, val acc theo từng epoch;
- ảnh trọng số của 10 lớp (khả năng giải thích của mô hình tuyến tính).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.models.logistic_regression import LogisticRegression
from src.utils import set_seed

from .common import sweep

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]
L2_VALUES = [0.0, 1e-4, 1e-3, 1e-2]
EXPERIMENTS = ["lr", "l2", "curve", "weights"]


def _fit_default(data):
    set_seed()
    return LogisticRegression().fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])


def learning_curve(model):
    """Vẽ train loss, val loss, val acc theo epoch từ model.history; lưu CSV + hình."""
    h = model.history
    df = pd.DataFrame({"epoch": np.arange(1, len(h["train_loss"]) + 1), **h})
    df.to_csv("results/tables/learning_curve_logistic_regression.csv", index=False)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(
        df["epoch"], df["train_loss"], marker="o", ms=3, label="train (cross-entropy + L2)"
    )
    ax_loss.plot(df["epoch"], df["val_loss"], marker="o", ms=3, label="validation (cross-entropy)")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Logistic Regression: loss")
    ax_loss.legend(fontsize=8)
    ax_acc.plot(df["epoch"], df["val_acc"], marker="o", ms=3, color="tab:orange")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_title("Logistic Regression: validation accuracy")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.savefig(
        "results/figures/learning_curve_logistic_regression.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"learning curve: val acc cuối = {df['val_acc'].iloc[-1]:.4f}")


def weight_images(model):
    """Ảnh trọng số W[:, c] của từng lớp: đỏ = pixel có mực làm tăng điểm lớp c, xanh = giảm."""
    images = model.weight_images()
    vmax = np.abs(images).max()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for c, ax in enumerate(axes.flat):
        im = ax.imshow(images[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"lớp {c}")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="trọng số")
    fig.suptitle("Logistic Regression: trọng số của từng lớp")
    fig.savefig("results/figures/logistic_regression_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    args = parser.parse_args()

    data = load_dataset("mnist")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    if "lr" in args.only:
        sweep(LogisticRegression, "lr", LEARNING_RATES, data)
    if "l2" in args.only:
        sweep(LogisticRegression, "l2", L2_VALUES, data)
    if "curve" in args.only or "weights" in args.only:
        model = _fit_default(data)  # train một lần, dùng cho cả hai hình
        if "curve" in args.only:
            learning_curve(model)
        if "weights" in args.only:
            weight_images(model)


if __name__ == "__main__":
    main()
