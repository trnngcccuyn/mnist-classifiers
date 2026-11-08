"""Ablation augmentation cho CNN trên Fashion-MNIST: phép nào giúp, phép nào hại.  Phụ trách: Tiên

Chạy:  python -m experiments.augmentation_ablation --seeds 0 1 2  # 15 epoch (~75 phút CPU)
       python -m experiments.augmentation_ablation                # chỉ seed 0

Bối cảnh: augmentation_exp cho thấy bộ get_augment("fashion_mnist") (xoay ±10°, dịch ±2 px,
nhiễu σ = 0.1 cho một nửa số ảnh, lật ngang) làm CNN giảm khoảng 3,9 điểm test acc. Ở đây bật
lần lượt từng phép để tách tác động.

Ngân sách train khác run_all: 15 epoch thay vì 10, vì learning curve trên Fashion-MNIST
(fashion_mnist_learning_curve_cnn.csv) cho thấy val acc còn tăng sau epoch 10, và ảnh có augment
cần train lâu hơn mới hội tụ. Vì vậy dòng "none" ở đây không phải con số CNN chính của báo cáo
(con số đó lấy từ run_all, 10 epoch). Bảng gộp ghi thêm val acc ở epoch RUN_ALL_EPOCHS để thấy
khoảng cách giữa các cấu hình thu hẹp lại khi train lâu hơn.

Các cấu hình (cộng dồn, cộng một cấu hình bỏ riêng phép lật):
  none                    không augment
  flip                    lật ngang p = 0.5
  flip+shift              + dịch ±2 px
  flip+shift+rotate       + xoay ±10°
  full                    + nhiễu (đúng get_augment("fashion_mnist"))
  full-flip               xoay + dịch + nhiễu (đúng get_augment("mnist"))
Dịch nguyên pixel và góc 0 trong affine_transform là chính xác, nên "flip" không bị nội suy làm mờ.

Script không chọn cấu hình nào: chỉ đo tác động của từng phép biến đổi. Cần nhiều seed vì
chênh lệch giữa các cấu hình gần nhau (vd none và flip) cỡ độ lệch chuẩn theo seed.

Kết quả: results/tables/fashion_mnist_augmentation_ablation{,_curves,_agg}.csv,
         results/figures/fashion_mnist_augmentation_ablation{,_curves}.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.augmentation import Augment
from src.data import load_dataset
from src.metrics import accuracy
from src.plotting import (
    FIGSIZE,
    INK,
    LINE_COLORS,
    TRAIN_COLOR,
    VAL_COLOR,
    plot_learning_curves,
    save_figure,
    set_percent_axis,
)
from src.utils import Timer, set_seed

DATASET = "fashion_mnist"
EPOCHS = 15
RUN_ALL_EPOCHS = MODEL_CONFIGS["cnn"][1]["epochs"]
CONFIGS = {
    "none": None,
    "flip": Augment(max_deg=0, max_px=0, noise_std=0, flip_p=0.5),
    "flip+shift": Augment(max_deg=0, max_px=2, noise_std=0, flip_p=0.5),
    "flip+shift+rotate": Augment(max_deg=10, max_px=2, noise_std=0, flip_p=0.5),
    "full": Augment(max_deg=10, max_px=2, noise_std=0.1, noise_p=0.5, flip_p=0.5),
    "full-flip": Augment(max_deg=10, max_px=2, noise_std=0.1, noise_p=0.5, flip_p=0.0),
}
# Ba cấu hình vẽ learning curve (LINE_COLORS chỉ đủ 3 màu tách biệt)
CURVE_CONFIGS = ("none", "full-flip", "full")
TABLES = "results/tables"


def run(d, seeds, configs, epochs):
    cls, params = MODEL_CONFIGS["cnn"]
    rows, curves = [], []
    for seed in seeds:
        for name in configs:
            set_seed(seed)
            model = cls(**{**params, "epochs": epochs}, augment=CONFIGS[name], seed=seed)
            with Timer() as t:
                model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
            row = {"config": name, "seed": seed, "fit_time_s": t.elapsed}
            for split in ("train", "val", "test"):
                row[f"{split}_acc"] = accuracy(d[f"y_{split}"], model.predict(d[f"X_{split}"]))
            rows.append(row)
            h = model.history
            for e, (vl, va) in enumerate(zip(h["val_loss"], h["val_acc"]), start=1):
                curves.append(
                    {"config": name, "seed": seed, "epoch": e, "val_loss": vl, "val_acc": va}
                )
            print(row, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(curves)


def aggregate(df, curves, configs):
    df = df.merge(
        curves.loc[curves["epoch"] == RUN_ALL_EPOCHS, ["config", "seed", "val_acc"]].rename(
            columns={"val_acc": "val_acc_ep10"}
        ),
        on=["config", "seed"],
        how="left",
    )
    g = df.groupby("config", sort=False)[
        ["train_acc", "val_acc", "test_acc", "val_acc_ep10", "fit_time_s"]
    ]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    agg.insert(0, "n_seeds", g.size())
    agg = agg.reindex(configs).reset_index()
    base = agg.loc[agg["config"] == "none", "test_acc_mean"].iloc[0]
    agg["test_acc_vs_none"] = agg["test_acc_mean"] - base
    agg["val_acc_vs_none"] = (
        agg["val_acc_mean"] - agg.loc[agg["config"] == "none", "val_acc_mean"].iloc[0]
    )
    agg["val_acc_ep10_vs_none"] = (
        agg["val_acc_ep10_mean"] - agg.loc[agg["config"] == "none", "val_acc_ep10_mean"].iloc[0]
    )
    return agg


def plot(agg, filename, epochs):
    """Train / val / test accuracy cuối cùng của từng cấu hình, ± 1 độ lệch chuẩn theo seed."""
    n = len(agg)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * 1.5, 0.55 * n + 1.5))
    ax.hlines(y, agg["test_acc_mean"], agg["train_acc_mean"], color="#cccccc", linewidth=3)
    for col, color, marker, label in [
        ("train_acc", TRAIN_COLOR, "o", "train"),
        ("val_acc", VAL_COLOR, "o", "validation"),
        ("test_acc", INK, "D", "test"),
    ]:
        ax.errorbar(
            agg[f"{col}_mean"],
            y,
            xerr=agg[f"{col}_std"].fillna(0),
            fmt=marker,
            color=color,
            markersize=7,
            capsize=3,
            label=label,
        )
    for yi, (_, r) in zip(y, agg.iterrows()):
        delta = "" if r["config"] == "none" else f" ({100 * r['test_acc_vs_none']:+.2f})"
        ax.annotate(
            f"test {r['test_acc_mean']:.2%}{delta}",
            (r["train_acc_mean"] + np.nan_to_num(r["train_acc_std"]), yi),  # sau thanh sai số
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y, agg["config"])
    ax.invert_yaxis()
    ax.set_xlim(right=agg["train_acc_mean"].max() + 0.035)  # chừa chỗ cho số ghi bên phải
    set_percent_axis(ax.xaxis)
    ax.set_xlabel("accuracy")
    ax.grid(axis="x")
    ax.legend(fontsize=9, loc="lower right")
    k = int(agg["n_seeds"].min())
    ax.set_title(
        f"Fashion-MNIST: ablation augmentation cho CNN ({epochs} epoch, {k} seed)\n"
        "trong ngoặc: chênh lệch test (điểm %) so với none",
        fontsize=11,
    )
    return save_figure(fig, filename)


def plot_curves(curves, filename, epochs):
    """Val loss, val acc theo epoch của none, full-flip, full; nét đứt: số epoch của run_all."""
    names = [c for c in CURVE_CONFIGS if c in set(curves["config"])]
    part = curves[curves["config"].isin(names)]
    return plot_learning_curves(
        part,
        filename,
        f"Fashion-MNIST: CNN có và không có augmentation "
        f"(trung bình {part['seed'].nunique()} seed, {epochs} epoch)",
        group="config",
        colors=dict(zip(names, LINE_COLORS)),
        mark_epoch=RUN_ALL_EPOCHS,
        mark_label="số epoch của run_all",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    if "none" not in args.configs:
        parser.error("cần có cấu hình none để so sánh")

    os.makedirs(TABLES, exist_ok=True)
    d = load_dataset(DATASET)
    df, curves = run(d, args.seeds, args.configs, args.epochs)
    agg = aggregate(df, curves, args.configs)

    df.to_csv(f"{TABLES}/{DATASET}_augmentation_ablation.csv", index=False)
    curves.to_csv(f"{TABLES}/{DATASET}_augmentation_ablation_curves.csv", index=False)
    agg.to_csv(f"{TABLES}/{DATASET}_augmentation_ablation_agg.csv", index=False)
    plot(agg, f"{DATASET}_augmentation_ablation.png", args.epochs)
    plot_curves(curves, f"{DATASET}_augmentation_ablation_curves.png", args.epochs)

    pd.set_option("display.width", 200)
    print("\n" + agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
