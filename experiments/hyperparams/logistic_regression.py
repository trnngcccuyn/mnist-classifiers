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

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.models.logistic_regression import LogisticRegression
from src.plotting import MODEL_COLORS, plot_class_images, plot_learning_curves
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
    plot_curve(df)
    print(f"learning curve: val acc cuối = {df['val_acc'].iloc[-1]:.4f}")


def plot_curve(curves):
    plot_learning_curves(
        curves,
        "learning_curve_logistic_regression.png",
        "Logistic Regression: learning curve",
        colors={None: MODEL_COLORS["logistic_regression"]},
    )


def weight_images(model):
    """Ảnh trọng số W[:, c] của từng lớp: đỏ = pixel có mực làm tăng điểm lớp c, xanh = giảm."""
    plot_class_images(
        model.weight_images(),
        "logistic_regression_weights.png",
        "Logistic Regression: trọng số của từng lớp",
        "trọng số",
        symmetric=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    args = parser.parse_args()

    data = load_dataset("mnist")
    os.makedirs("results/tables", exist_ok=True)
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
