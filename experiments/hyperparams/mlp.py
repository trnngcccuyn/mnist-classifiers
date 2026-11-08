"""Thí nghiệm siêu tham số cho MLP.  Phụ trách: Uyên

Chạy: python -m experiments.hyperparams.mlp
      python -m experiments.hyperparams.mlp --only momentum curve   # chỉ chạy một phần

Gồm (mỗi sweep chỉ đổi một yếu tố, chọn theo validation, không dùng test):
- số lớp / số nơ-ron, learning rate, hệ số L2, hàm kích hoạt (SGD thường, lr = 0.1);
- momentum, với lr = 0.05 như baseline lab Tuần 4 (lr 0.1 + momentum 0.9 cho lr hiệu dụng
  lr / (1 - momentum) = 1.0, quá lớn);
- learning curve: so SGD thường với SGD + momentum + early stopping theo từng epoch.
"""

import argparse
import os

import pandas as pd

from src.data import load_dataset
from src.metrics import accuracy
from src.models.mlp import ACTIVATIONS, MLP
from src.plotting import plot_learning_curves
from src.utils import Timer, set_seed

from .common import sweep

HIDDEN_SIZES = [(64,), (256,), (256, 128), (512, 256, 128)]
LEARNING_RATES = [0.001, 0.01, 0.1]
L2_VALUES = [0.0, 1e-5, 1e-4, 1e-3]
MOMENTA = [0.0, 0.5, 0.9]
MOMENTUM_LR = 0.05
EXPERIMENTS = ["hidden_sizes", "lr", "l2", "activation", "momentum", "curve"]

# Hai cấu hình so sánh trong learning curve
CURVE_CONFIGS = {
    "SGD (lr 0.1)": dict(),
    "momentum 0.9 + early stopping (lr 0.05)": dict(
        lr=MOMENTUM_LR, momentum=0.9, epochs=30, patience=5
    ),
}


def learning_curve(data, configs=CURVE_CONFIGS):
    """Train từng cấu hình, vẽ train loss, val loss, val acc theo epoch; lưu CSV + hình.

    Dùng luôn model.history mà fit đã lưu, không phải train lại từng epoch.
    """
    rows, summary, kept_epochs = [], [], {}
    for label, params in configs.items():
        set_seed()
        model = MLP(**params)
        with Timer() as t:
            model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        h = model.history
        epochs = list(range(1, len(h["train_loss"]) + 1))
        for e, tl, vl, va in zip(epochs, h["train_loss"], h["val_loss"], h["val_acc"]):
            rows.append(
                {"config": label, "epoch": e, "train_loss": tl, "val_loss": vl, "val_acc": va}
            )
        if model.best_epoch_ is not None:  # epoch được early stopping giữ lại
            kept_epochs[label] = model.best_epoch_

        summary.append(
            {
                "config": label,
                "epochs_run": len(epochs),
                "kept_epoch": model.best_epoch_ or epochs[-1],
                "train_acc": accuracy(data["y_train"], model.predict(data["X_train"])),
                "val_acc": accuracy(data["y_val"], model.predict(data["X_val"])),
                "fit_time_s": t.elapsed,
            }
        )
        print(summary[-1])

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tables/learning_curve_mlp.csv", index=False)
    pd.DataFrame(summary).to_csv("results/tables/learning_curve_mlp_summary.csv", index=False)
    plot_mlp_curves(pd.DataFrame(rows), kept_epochs)
    return pd.DataFrame(summary)


def plot_mlp_curves(curves, kept_epochs):
    """kept_epochs: {cấu hình: epoch early stopping giữ lại}, vẽ nét đứt."""
    plot_learning_curves(
        curves,
        "learning_curve_mlp.png",
        "MLP: SGD thường và SGD + momentum + early stopping",
        group="config",
        mark_epoch=kept_epochs,
        mark_label="early stopping giữ lại",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    args = parser.parse_args()

    data = load_dataset("mnist")
    if "hidden_sizes" in args.only:
        sweep(MLP, "hidden_sizes", HIDDEN_SIZES, data)
    if "lr" in args.only:
        sweep(MLP, "lr", LEARNING_RATES, data)
    if "l2" in args.only:
        sweep(MLP, "l2", L2_VALUES, data)
    if "activation" in args.only:
        sweep(MLP, "activation", list(ACTIVATIONS), data)
    if "momentum" in args.only:
        sweep(MLP, "momentum", MOMENTA, data, fixed={"lr": MOMENTUM_LR})
    if "curve" in args.only:
        learning_curve(data)


if __name__ == "__main__":
    main()
