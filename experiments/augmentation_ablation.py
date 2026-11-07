"""Ablation augmentation cho CNN trên Fashion-MNIST: phép nào giúp, phép nào hại.  Phụ trách: Tiên

Chạy:  python -m experiments.augmentation_ablation  # seed 0, 15 epoch (~25 phút CPU)
       python -m experiments.augmentation_ablation --seeds 0 1 2

Bối cảnh: augmentation_exp cho thấy bộ get_augment("fashion_mnist") (xoay ±10°, dịch ±2 px,
nhiễu σ = 0.1 cho một nửa số ảnh, lật ngang) làm CNN giảm khoảng 3,9 điểm test acc. Ở đây bật
lần lượt từng phép để tách tác động. Dùng 15 epoch thay vì 10 của run_all, vì learning curve trên
Fashion-MNIST (fashion_mnist_learning_curve_cnn.csv) cho thấy val acc còn tăng đến khoảng epoch 15.

Các cấu hình (cộng dồn, cộng một cấu hình bỏ riêng phép lật):
  none                    không augment
  flip                    lật ngang p = 0.5
  flip+shift              + dịch ±2 px
  flip+shift+rotate       + xoay ±10°
  full                    + nhiễu (đúng get_augment("fashion_mnist"))
  full-flip               xoay + dịch + nhiễu (đúng get_augment("mnist"))
Dịch nguyên pixel và góc 0 trong affine_transform là chính xác, nên "flip" không bị nội suy làm mờ.

Chọn cấu hình theo val; test chỉ để báo cáo.

Kết quả: results/tables/fashion_mnist_augmentation_ablation{,_curves}.csv,
         results/figures/fashion_mnist_augmentation_ablation.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.augmentation import Augment
from src.data import load_dataset
from src.metrics import accuracy
from src.utils import Timer, set_seed

DATASET = "fashion_mnist"
EPOCHS = 15
CONFIGS = {
    "none": None,
    "flip": Augment(max_deg=0, max_px=0, noise_std=0, flip_p=0.5),
    "flip+shift": Augment(max_deg=0, max_px=2, noise_std=0, flip_p=0.5),
    "flip+shift+rotate": Augment(max_deg=10, max_px=2, noise_std=0, flip_p=0.5),
    "full": Augment(max_deg=10, max_px=2, noise_std=0.1, noise_p=0.5, flip_p=0.5),
    "full-flip": Augment(max_deg=10, max_px=2, noise_std=0.1, noise_p=0.5, flip_p=0.0),
}
TABLES, FIGURES = "results/tables", "results/figures"


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
            for e, va in enumerate(model.history["val_acc"], start=1):
                curves.append({"config": name, "seed": seed, "epoch": e, "val_acc": va})
            print(row, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(curves)


def aggregate(df, configs):
    g = df.groupby("config", sort=False)[["train_acc", "val_acc", "test_acc", "fit_time_s"]]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    agg.insert(0, "n_seeds", g.size())
    agg = agg.reindex(configs).reset_index()
    base = agg.loc[agg["config"] == "none", "test_acc_mean"].iloc[0]
    agg["test_acc_vs_none"] = agg["test_acc_mean"] - base
    return agg


def plot(agg, curves, path, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
    ax = axes[0]
    y = range(len(agg))[::-1]
    for col, c, lab in [
        ("train_acc_mean", "tab:gray", "train"),
        ("val_acc_mean", "tab:blue", "val"),
        ("test_acc_mean", "tab:orange", "test"),
    ]:
        ax.plot(100 * agg[col], y, "o", color=c, label=lab)
    for yi, v in zip(y, agg["test_acc_mean"]):
        ax.annotate(
            f"{100 * v:.2f}",
            (100 * v, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="tab:orange",
        )
    ax.set_yticks(list(y), agg["config"])
    ax.set_xlabel("accuracy (%)")
    ax.set_title(f"CNN {epochs} epoch, trung bình {int(agg['n_seeds'].iloc[0])} seed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    for name, g in curves.groupby("config", sort=False):
        m = g.groupby("epoch")["val_acc"].mean()
        ax.plot(m.index, 100 * m, marker="o", ms=3, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy (%)")
    ax.set_title("Val accuracy theo epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("Fashion-MNIST: ablation augmentation cho CNN")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    if "none" not in args.configs:
        parser.error("cần có cấu hình none để so sánh")

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = load_dataset(DATASET)
    df, curves = run(d, args.seeds, args.configs, args.epochs)
    agg = aggregate(df, args.configs)

    df.to_csv(f"{TABLES}/{DATASET}_augmentation_ablation.csv", index=False)
    curves.to_csv(f"{TABLES}/{DATASET}_augmentation_ablation_curves.csv", index=False)
    plot(agg, curves, f"{FIGURES}/{DATASET}_augmentation_ablation.png", args.epochs)

    pd.set_option("display.width", 200)
    print("\n" + agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
