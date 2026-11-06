"""So sánh CNN với MLP: accuracy, số tham số, thời gian huấn luyện/dự đoán.  Phụ trách: Uyên

Chạy:  python -m experiments.cnn_vs_mlp                      # seed 0 1 2 + thí nghiệm hoán vị
       python -m experiments.cnn_vs_mlp --seeds 0 --skip-permuted   # chạy thử nhanh

Cả hai mô hình dùng đúng cấu hình cuối trong experiments/run_all.py (MODEL_CONFIGS), cùng cách
chia train/val/test. Gồm ba phần:
1. Mỗi seed: val/test accuracy, số tham số, thời gian train (tổng và mỗi epoch), thời gian dự
   đoán 10.000 ảnh test; đường train loss / val acc theo epoch.
2. Lỗi trên test (seed đầu): bao nhiêu ảnh chỉ MLP sai, chỉ CNN sai, cả hai cùng sai; accuracy
   theo từng lớp.
3. Hoán vị pixel (seed đầu): xáo thứ tự 784 pixel bằng MỘT hoán vị cố định, áp cho mọi ảnh, rồi
   train lại cả hai. MLP coi 784 pixel là 784 đặc trưng rời nhau nên gần như không đổi; CNN dựa
   vào việc pixel kề nhau trong ảnh có liên quan với nhau (bộ lọc 3x3, max pooling), nên sẽ
   giảm. Đây là bằng chứng trực tiếp cho lập luận "trải phẳng ảnh làm mất thông tin pixel kề
   nhau" mà MLP phải tự học lại từ dữ liệu.

Kết quả: results/tables/cnn_vs_mlp{,_agg,_curves,_errors,_per_class,_permuted}.csv và
         results/figures/cnn_vs_mlp_curves.png
Thời gian chạy phụ thuộc máy (CPU/GPU, số nhân, máy đang bận) nên chỉ dùng để so tương đối giữa
hai mô hình chạy trên cùng một máy.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.data import load_dataset
from src.metrics import accuracy
from src.utils import SEED, Timer, set_seed

MODELS = ("mlp", "cnn")
COLORS = {"mlp": "tab:blue", "cnn": "tab:orange"}
TABLES, FIGURES = "results/tables", "results/figures"


def train_one(name, seed, X_tr, y_tr, X_val, y_val):
    cls, params = MODEL_CONFIGS[name]
    set_seed(seed)
    model = cls(**params, seed=seed)
    with Timer() as t_fit:
        model.fit(X_tr, y_tr, X_val, y_val)
    return model, t_fit.elapsed


def compare(d, seeds):
    """Phần 1. Trả về (bảng theo seed, bảng đường học, dự đoán test của seed đầu)."""
    rows, curves, first_preds = [], [], {}
    for seed in seeds:
        for name in MODELS:
            model, fit_time = train_one(
                name, seed, d["X_train"], d["y_train"], d["X_val"], d["y_val"]
            )
            with Timer() as t_pred:
                y_pred = model.predict(d["X_test"])
            n_epochs = len(model.history["train_loss"])
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "n_params": model.n_params(),
                    "epochs": n_epochs,
                    "val_acc": model.history["val_acc"][-1],
                    "test_acc": accuracy(d["y_test"], y_pred),
                    "fit_time_s": fit_time,
                    "time_per_epoch_s": fit_time / n_epochs,
                    "predict_time_test_s": t_pred.elapsed,
                }
            )
            for epoch, (tl, vl, va) in enumerate(
                zip(*(model.history[k] for k in ("train_loss", "val_loss", "val_acc"))), start=1
            ):
                curves.append(
                    {
                        "model": name,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": tl,
                        "val_loss": vl,
                        "val_acc": va,
                    }
                )
            if seed == seeds[0]:
                first_preds[name] = y_pred
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    return pd.DataFrame(rows), pd.DataFrame(curves), first_preds


def aggregate(df):
    cols = ["val_acc", "test_acc", "fit_time_s", "time_per_epoch_s", "predict_time_test_s"]
    g = df.groupby("model", sort=False)
    agg = g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std"))
    agg.insert(0, "n_params", g["n_params"].first())
    agg.insert(0, "n_seeds", g.size())
    return agg.reset_index()


def error_overlap(y_true, preds):
    """Phần 2: bảng chồng lấn lỗi và accuracy theo lớp của MLP và CNN."""
    mlp_wrong = preds["mlp"] != y_true
    cnn_wrong = preds["cnn"] != y_true
    errors = pd.DataFrame(
        {
            "case": ["cả hai đúng", "chỉ MLP sai", "chỉ CNN sai", "cả hai sai"],
            "n_images": [
                int(np.sum(~mlp_wrong & ~cnn_wrong)),
                int(np.sum(mlp_wrong & ~cnn_wrong)),
                int(np.sum(~mlp_wrong & cnn_wrong)),
                int(np.sum(mlp_wrong & cnn_wrong)),
            ],
        }
    )
    per_class = pd.DataFrame(
        {
            "class": np.arange(10),
            "n_test": np.bincount(y_true, minlength=10),
            **{
                f"{name}_acc": [float(np.mean(preds[name][y_true == c] == c)) for c in range(10)]
                for name in MODELS
            },
        }
    )
    return errors, per_class


def permuted_pixels(d, seed, original):
    """Phần 3: train lại MLP và CNN trên ảnh đã hoán vị pixel (cùng hoán vị cho mọi tập)."""
    perm = np.random.default_rng(SEED).permutation(d["X_train"].shape[1])
    X = {s: d[f"X_{s}"][:, perm] for s in ("train", "val", "test")}
    rows = []
    for name in MODELS:
        model, _ = train_one(name, seed, X["train"], d["y_train"], X["val"], d["y_val"])
        orig = original[(original["model"] == name) & (original["seed"] == seed)].iloc[0]
        rows.append(
            {
                "model": name,
                "seed": seed,
                "test_acc_original": float(orig["test_acc"]),
                "test_acc_permuted": accuracy(d["y_test"], model.predict(X["test"])),
            }
        )
        rows[-1]["drop"] = rows[-1]["test_acc_original"] - rows[-1]["test_acc_permuted"]
        print(rows[-1])
    return pd.DataFrame(rows)


def plot_curves(curves, path):
    """Train loss và val acc theo epoch: trung bình theo seed, dải mờ là ± 1 độ lệch chuẩn."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(
        axes, ("train_loss", "val_acc"), ("train loss (cross-entropy)", "validation accuracy")
    ):
        for name in MODELS:
            g = curves[curves["model"] == name].groupby("epoch")[metric]
            mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
            ax.plot(mean.index, mean, marker="o", ms=3, color=COLORS[name], label=name.upper())
            ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[name], alpha=0.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[1].set_ylim(0.97, 1.0)
    n_seeds = curves["seed"].nunique()
    fig.suptitle(f"MLP và CNN trên MNIST (trung bình {n_seeds} seed)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-permuted", action="store_true")
    args = parser.parse_args()

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = load_dataset("mnist")

    df, curves, preds = compare(d, args.seeds)
    df.to_csv(f"{TABLES}/cnn_vs_mlp.csv", index=False)
    curves.to_csv(f"{TABLES}/cnn_vs_mlp_curves.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(f"{TABLES}/cnn_vs_mlp_agg.csv", index=False)
    plot_curves(curves, f"{FIGURES}/cnn_vs_mlp_curves.png")

    errors, per_class = error_overlap(d["y_test"], preds)
    errors.to_csv(f"{TABLES}/cnn_vs_mlp_errors.csv", index=False)
    per_class.to_csv(f"{TABLES}/cnn_vs_mlp_per_class.csv", index=False)

    print("\n" + agg.to_string(index=False))
    print(f"\nLỗi trên test (seed {args.seeds[0]}):\n" + errors.to_string(index=False))
    print("\n" + per_class.to_string(index=False))

    if not args.skip_permuted:
        perm = permuted_pixels(d, args.seeds[0], df)
        perm.to_csv(f"{TABLES}/cnn_vs_mlp_permuted.csv", index=False)
        print("\nHoán vị pixel:\n" + perm.to_string(index=False))


if __name__ == "__main__":
    main()
