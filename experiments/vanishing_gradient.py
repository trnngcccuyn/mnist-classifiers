"""Hàm kích hoạt và vanishing gradient: mạng nông (2 lớp ẩn) so với mạng sâu (6 lớp ẩn).
Phụ trách: Uyên

Chạy:  python -m experiments.vanishing_gradient
       python -m experiments.vanishing_gradient --only grad      # chỉ đo gradient (vài giây)

Gồm hai phần, mỗi phần cho ReLU / sigmoid / tanh (khởi tạo He cho ReLU, Xavier cho sigmoid/tanh):
1. grad: chuẩn Frobenius ||dJ/dW^(l)|| của từng lớp lúc khởi tạo, trên một batch 512 ảnh train,
   dùng chính loss_and_grads (backprop tự viết). Vanishing gradient thể hiện ở chỗ gradient
   các lớp đầu nhỏ hơn rất nhiều so với lớp cuối.
2. train: val accuracy theo epoch. Mạng nông chạy SHALLOW_EPOCHS epoch (dài hơn 20 epoch của
   sweep để thấy sigmoid chỉ hội tụ chậm chứ vẫn học được); mạng sâu chạy DEEP_EPOCHS epoch.

Kết quả: results/tables/vanishing_gradient_{grad_norms,curves}.csv và
         results/figures/vanishing_gradient_{grad_norms,curves}.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    PercentFormatter,
)

from src.data import load_dataset
from src.models.mlp import ACTIVATIONS, MLP
from src.plotting import LINE_COLORS, save_figure
from src.utils import SEED, Timer, set_seed

NETWORKS = {"2 lớp ẩn": (256, 128), "6 lớp ẩn": (128,) * 6}
SHALLOW_EPOCHS = 60
DEEP_EPOCHS = 20
LR = 0.1  # SGD thường như sweep activation, để so sánh công bằng giữa các hàm kích hoạt
COLORS = dict(zip(ACTIVATIONS, LINE_COLORS))


def grad_norms(data, n_samples=512):
    X, y = data["X_train"][:n_samples], data["y_train"][:n_samples]
    rows = []
    for net, hidden in NETWORKS.items():
        for act in ACTIVATIONS:
            model = MLP(hidden_sizes=hidden, activation=act)
            model._init_params(X.shape[1], X.dtype, np.random.default_rng(SEED))
            _, dWs, _ = model.loss_and_grads(X, y)
            for layer, dW in enumerate(dWs, start=1):
                rows.append(
                    {
                        "network": net,
                        "activation": act,
                        "layer": layer,
                        "grad_norm": float(np.linalg.norm(dW)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/vanishing_gradient_grad_norms.csv", index=False)
    plot_grad_norms(df)

    ratio = df.groupby(["network", "activation"], sort=False)["grad_norm"].agg(
        lambda g: g.iloc[0] / g.iloc[-1]
    )
    print("Tỉ lệ gradient lớp đầu / lớp cuối:")
    print(ratio.to_string())
    return df


def plot_grad_norms(df):
    # sharey: hai mạng dùng chung trục y nên so trực tiếp được độ lớn gradient
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    width = 0.27
    for ax, (net, sub) in zip(axes, df.groupby("network", sort=False)):
        for k, act in enumerate(ACTIVATIONS):
            s = sub[sub["activation"] == act]
            ax.bar(
                s["layer"] + (k - 1) * width, s["grad_norm"], width, label=act, color=COLORS[act]
            )
        ax.set_yscale("log")
        # Chỉ hiện các mốc 0.0001, 0.001, ..., 10 dạng số thường; bỏ nhãn phụ kiểu 6×10^-1
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("lớp l (1 = gần đầu vào)")
        ax.set_title(f"{net}: độ lớn gradient lúc khởi tạo")
        ax.set_xticks(sorted(sub["layer"].unique()))
        ax.grid(axis="y")
        ax.tick_params(axis="y", labelleft=True)  # sharey ẩn số ở hình phải; hiện lại cho dễ đọc
    axes[0].set_ylabel(
        "độ lớn gradient của lớp l\n" + r"$\|\partial J / \partial W^{(l)}\|$ (thang log)"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    save_figure(fig, "vanishing_gradient_grad_norms.png")


def train_curves(data):
    rows = []
    for net, hidden in NETWORKS.items():
        epochs = SHALLOW_EPOCHS if net == "2 lớp ẩn" else DEEP_EPOCHS
        for act in ACTIVATIONS:
            set_seed()
            model = MLP(hidden_sizes=hidden, activation=act, lr=LR, epochs=epochs)
            with Timer() as t:
                model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
            val_acc = model.history["val_acc"]
            for e, (tl, va) in enumerate(zip(model.history["train_loss"], val_acc), start=1):
                rows.append(
                    {"network": net, "activation": act, "epoch": e, "train_loss": tl, "val_acc": va}
                )
            print(f"{net}, {act}: val acc cuối = {val_acc[-1]:.4f} ({t.elapsed:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/vanishing_gradient_curves.csv", index=False)
    plot_train_curves(df)


def plot_train_curves(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (net, sub) in zip(axes, df.groupby("network", sort=False)):
        for act in ACTIVATIONS:
            s = sub[sub["activation"] == act]
            ax.plot(s["epoch"], s["val_acc"], label=act, color=COLORS[act])
        if net == "2 lớp ẩn":
            ax.axvline(20, linestyle="--", linewidth=1, color="#999999")
            ax.text(20.5, 0.15, "20 epoch\n(sweep activation)", fontsize=8, color="#666666")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy")
        ax.set_title(f"{net} (SGD, lr = {LR})")
        ax.grid()
        ax.legend(loc="center right")
    save_figure(fig, "vanishing_gradient_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=["grad", "train"], default=["grad", "train"])
    args = parser.parse_args()

    data = load_dataset("mnist")
    os.makedirs("results/tables", exist_ok=True)
    if "grad" in args.only:
        grad_norms(data)
    if "train" in args.only:
        train_curves(data)


if __name__ == "__main__":
    main()
