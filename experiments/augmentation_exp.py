"""Đánh giá tác động của data augmentation lên MLP và CNN.  Phụ trách: Uyên

Chạy:  python -m experiments.augmentation_exp                          # MNIST, seed 0 1 2
       python -m experiments.augmentation_exp --seeds 0                 # chạy thử nhanh
       python -m experiments.augmentation_exp --dataset fashion_mnist
       python -m experiments.augmentation_exp --only examples           # chỉ vẽ ảnh mẫu (vài giây)

Gồm hai phần:
1. examples: vẽ vài ảnh validation và các bản augment của chúng, để kiểm tra bằng mắt phép biến
   đổi hợp lý (chữ số vẫn đọc được, MNIST không bị lật).
2. train: mỗi seed, MLP và CNN (cấu hình cuối trong run_all.MODEL_CONFIGS), train có và không
   augmentation (src/augmentation.get_augment(dataset): xoay ±10°, dịch ±2 px, nhiễu; Fashion
   thêm lật ngang). Đánh giá trên:
   - val / test gốc;
   - test bị biến đổi với độ lớn cố định (độ bền): dịch đúng 1..4 px, xoay đúng ±10..45°.
     Mỗi độ lớn tạo một tập test cố định (rng theo SEED), mọi mô hình dùng chung tập đó.
     Dịch 3–4 px và xoay 20–45° vượt khỏi khoảng đã augment lúc train: đo khả năng tổng quát
     hoá ra ngoài khoảng đã thấy.

Cùng số epoch như khi không augment (để so sánh công bằng, cùng thời gian train). Bài toán có
augment khó hơn nên có thể cần nhiều epoch hơn: xem val acc epoch cuối trong *_curves.csv.
Mô hình không augment cùng seed cho đúng kết quả của run_all (kiểm tra tái tạo).

Kết quả (tiền tố <dataset>_ nếu không phải MNIST):
  results/tables/augmentation{,_agg,_robustness,_curves}.csv
  results/figures/augmentation_{examples,robustness}.png
  results/probs/<dataset>/{mlp,cnn}_aug_{val,test}.npy  (seed đầu, cho phần ensemble / TTA)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.augmentation import get_augment, perturb
from src.data import load_dataset
from src.metrics import accuracy
from src.utils import SEED, Timer, save_probs, set_seed

MODELS = ("mlp", "cnn")
PERTURBATIONS = {"shift": [1, 2, 3, 4], "rotate": [10, 20, 30, 45]}  # px, độ
STYLES = {  # (màu theo mô hình, nét theo augment)
    ("mlp", False): ("tab:blue", "--"),
    ("mlp", True): ("tab:blue", "-"),
    ("cnn", False): ("tab:orange", "--"),
    ("cnn", True): ("tab:orange", "-"),
}
TABLES, FIGURES = "results/tables", "results/figures"


def label(name, aug):
    return f"{name.upper()} + aug" if aug else name.upper()


def plot_examples(d, dataset, path, n_images=8, n_aug=5):
    rng = np.random.default_rng(SEED)
    X = d["X_val"][:n_images].reshape(-1, 28, 28)
    aug = get_augment(dataset)
    rows = [X] + [aug(X, rng) for _ in range(n_aug)]
    fig, axes = plt.subplots(len(rows), n_images, figsize=(n_images * 1.1, len(rows) * 1.15))
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel("gốc" if r == 0 else f"aug {r}", fontsize=9)
    fig.suptitle(f"{dataset}: {aug}", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def perturbed_test_sets(d):
    """{(kind, magnitude): X_test đã biến đổi, phẳng (N, 784)}, gồm cả (kind, 0) = test gốc."""
    X = d["X_test"].reshape(-1, 28, 28)
    sets = {}
    for k, (kind, mags) in enumerate(PERTURBATIONS.items()):
        sets[(kind, 0)] = d["X_test"]
        for i, m in enumerate(mags):
            rng = np.random.default_rng(SEED + 100 * k + i)  # mỗi tập một rng cố định
            sets[(kind, m)] = perturb(X, kind, m, rng).reshape(len(X), -1)
    return sets


def run(d, dataset, seeds, models):
    test_sets = perturbed_test_sets(d)
    rows, robust, curves = [], [], []
    for seed in seeds:
        for name in models:
            for aug in (False, True):
                cls, params = MODEL_CONFIGS[name]
                set_seed(seed)
                model = cls(**params, augment=get_augment(dataset) if aug else None, seed=seed)
                with Timer() as t:
                    model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
                P_val, P_test = model.predict_proba(d["X_val"]), model.predict_proba(d["X_test"])
                rows.append(
                    {
                        "model": name,
                        "augment": aug,
                        "seed": seed,
                        "train_acc": accuracy(d["y_train"], model.predict(d["X_train"])),
                        "val_acc": accuracy(d["y_val"], P_val.argmax(axis=1)),
                        "test_acc": accuracy(d["y_test"], P_test.argmax(axis=1)),
                        "fit_time_s": t.elapsed,
                    }
                )
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
                if aug and seed == seeds[0]:
                    save_probs(P_val, dataset, f"{name}_aug", "val")
                    save_probs(P_test, dataset, f"{name}_aug", "test")

                for (kind, mag), X in test_sets.items():
                    robust.append(
                        {
                            "model": name,
                            "augment": aug,
                            "seed": seed,
                            "perturbation": kind,
                            "magnitude": mag,
                            "test_acc": accuracy(d["y_test"], model.predict(X)),
                        }
                    )
                h = model.history
                for e, (tl, va) in enumerate(zip(h["train_loss"], h["val_acc"]), start=1):
                    curves.append(
                        {
                            "model": name,
                            "augment": aug,
                            "seed": seed,
                            "epoch": e,
                            "train_loss": tl,
                            "val_acc": va,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(robust), pd.DataFrame(curves)


def aggregate(df):
    cols = ["train_acc", "val_acc", "test_acc", "fit_time_s"]
    g = df.groupby(["model", "augment"], sort=False)
    agg = g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std"))
    agg.insert(0, "n_seeds", g.size())
    return agg.reset_index()


def plot_robustness(robust, path, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    xlabels = {"shift": "dịch (pixel)", "rotate": "xoay (độ)"}
    trained = {"shift": 2, "rotate": 10}  # khoảng đã augment lúc train
    for ax, (kind, sub) in zip(axes, robust.groupby("perturbation", sort=False)):
        for (name, aug), g in sub.groupby(["model", "augment"], sort=False):
            stats = g.groupby("magnitude")["test_acc"]
            mean, std = stats.mean(), stats.std(ddof=1).fillna(0.0)
            color, ls = STYLES[(name, aug)]
            ax.plot(mean.index, mean, marker="o", ms=4, color=color, ls=ls, label=label(name, aug))
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
        ax.axvline(trained[kind], color="gray", lw=1, ls=":", label="giới hạn augment lúc train")
        ax.set_xlabel(xlabels[kind])
        ax.set_title(f"Test bị {xlabels[kind].split()[0]}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[1].legend(fontsize=8, loc="lower left")
    n_seeds = robust["seed"].nunique()
    fig.suptitle(f"{dataset}: độ bền với ảnh test bị biến đổi (trung bình {n_seeds} seed)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--only", choices=["examples", "train"], default=None)
    args = parser.parse_args()

    prefix = "" if args.dataset == "mnist" else f"{args.dataset}_"
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = load_dataset(args.dataset)

    if args.only in (None, "examples"):
        plot_examples(d, args.dataset, f"{FIGURES}/{prefix}augmentation_examples.png")
    if args.only == "examples":
        return

    df, robust, curves = run(d, args.dataset, args.seeds, args.models)
    df.to_csv(f"{TABLES}/{prefix}augmentation.csv", index=False)
    robust.to_csv(f"{TABLES}/{prefix}augmentation_robustness.csv", index=False)
    curves.to_csv(f"{TABLES}/{prefix}augmentation_curves.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(f"{TABLES}/{prefix}augmentation_agg.csv", index=False)
    plot_robustness(robust, f"{FIGURES}/{prefix}augmentation_robustness.png", args.dataset)

    print("\n" + agg.round(4).to_string(index=False))
    table = robust.pivot_table(
        index=["perturbation", "magnitude"], columns=["model", "augment"], values="test_acc"
    )
    print("\nTest acc trên ảnh bị biến đổi (trung bình theo seed):\n" + table.round(4).to_string())


if __name__ == "__main__":
    main()
