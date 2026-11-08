"""CNN cải tiến: thêm từng kỹ thuật một và đo trên validation.  Phụ trách: Uyên

Chạy:  python -m experiments.cnn_improved                                  # 4 cấu hình, seed 0
       python -m experiments.cnn_improved --configs v2_full --seeds 0 1 2  # cấu hình chọn, 3 seed
       python -m experiments.cnn_improved --configs v2_full --seeds 0 1 2 3 4   # + ensemble 5 mạng

Mọi cấu hình đều train với augmentation (get_augment(dataset)), Adam lr 1e-3, batch 128,
15 epoch, 16/32 bộ lọc (~219 nghìn tham số, xấp xỉ SimpleCNN). Thêm lần lượt (src/models/cnn.py,
ImprovedCNN):
  v2_plain     2 conv mỗi khối, padding giữ kích thước, không BatchNorm, không Dropout, lr cố định
  v2_bn        + BatchNorm sau mỗi conv
  v2_bn_drop   + Dropout 0.3 trước lớp ra
  v2_full      + lịch lr cosine (giảm dần về 0 trong 15 epoch)
Mốc so sánh: SimpleCNN + aug cùng seed, đọc từ results/tables/seed_ensemble_members.csv.

Chọn cấu hình theo val acc. Một seed dao động ±0,1–0,2%, nên chênh lệch nhỏ hơn mức đó giữa
hai cấu hình là chưa phân biệt được; khi đó chọn cấu hình đơn giản hơn, rồi chạy thêm seed.

Mỗi lần train lưu vào results/cache/<dataset>/cnn_improved/<cấu hình>_seed<k>.{npz,pt}
(không commit): chạy lại thì đọc lại, chỉ train phần còn thiếu. Bảng kết quả luôn gồm MỌI lần
train đã có trong bộ nhớ đệm (cả các lần chạy trước). File .pt là trọng số mạng, dùng lại được
cho demo.

Kết quả (tiền tố <dataset>_ nếu không phải MNIST):
  results/tables/cnn_improved{,_agg,_curves}.csv, results/figures/cnn_improved_curves.png
  results/probs/<dataset>/cnn_<cấu hình>_{val,test}.npy (seed đầu) và
  cnn_<cấu hình>_ens<n>_{val,test}.npy (khi đã có từ 3 seed trở lên)
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.augmentation import get_augment
from src.data import load_dataset
from src.ensemble import soft_vote
from src.models.cnn import ImprovedCNN
from src.plotting import plot_learning_curves
from src.utils import Timer, save_probs, set_seed

COMMON = {"lr": 1e-3, "epochs": 15, "batch_size": 128, "channels": (16, 32), "hidden": 128}
CONFIGS = {
    "v2_plain": {"batchnorm": False, "dropout": 0.0, "schedule": None},
    "v2_bn": {"batchnorm": True, "dropout": 0.0, "schedule": None},
    "v2_bn_drop": {"batchnorm": True, "dropout": 0.3, "schedule": None},
    "v2_full": {"batchnorm": True, "dropout": 0.3, "schedule": "cosine"},
}
TABLES, FIGURES, CACHE = "results/tables", "results/figures", "results/cache"
# Tên hiển thị trong hình: mỗi cấu hình thêm một kỹ thuật so với cấu hình trước
CONFIG_LABELS = {
    "v2_plain": "v2_plain",
    "v2_bn": "+ BatchNorm",
    "v2_bn_drop": "+ Dropout 0.3",
    "v2_full": "+ lr cosine (v2_full)",
}
# 4 màu khác sắc cho 4 cấu hình (LINE_COLORS chỉ có 3 màu; xanh / cam để dành cho train /
# validation): olive, xanh lá, tím, đỏ rượu; v2_full (cấu hình được chọn) dùng màu đậm nhất.
# Kiểm tra như src/plotting.py (OKLab ΔE, Machado 2009): mọi cặp tách biệt với người nhìn bình
# thường (ΔE >= 18) và người mù màu đỏ/lục (ΔE >= 9); tương phản với nền trắng >= 3:1.
STEP_COLORS = ["#999933", "#117733", "#7b5bd6", "#882255"]


def acc(P, y):
    return float(np.mean(np.asarray(P).argmax(axis=1) == y))


def train_or_load(config, seed, d, dataset):
    base = os.path.join(CACHE, dataset, "cnn_improved", f"{config}_seed{seed}")
    if os.path.exists(base + ".npz"):
        print(f"đọc {base}.npz")
        z = np.load(base + ".npz")
        return {k: z[k] for k in z.files}

    set_seed(seed)
    model = ImprovedCNN(**COMMON, **CONFIGS[config], augment=get_augment(dataset), seed=seed)
    with Timer() as t:
        model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    out = {
        "val": model.predict_proba(d["X_val"]),
        "test": model.predict_proba(d["X_test"]),
        "train_acc": accuracy_train(model, d),
        "fit_time_s": t.elapsed,
        "n_params": model.n_params(),
        **{k: np.asarray(v) for k, v in model.history.items()},
    }
    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.savez_compressed(base + ".npz", **out)
    torch.save(model.net_.state_dict(), base + ".pt")
    print(
        f"{config} seed {seed}: {t.elapsed:.0f} s, val {acc(out['val'], d['y_val']):.4f}, "
        f"test {acc(out['test'], d['y_test']):.4f}"
    )
    return out


def load_all_cached(dataset):
    """{(cấu hình, seed): kết quả} cho mọi file trong bộ nhớ đệm, theo thứ tự CONFIGS."""
    folder = os.path.join(CACHE, dataset, "cnn_improved")
    runs = {}
    for config in CONFIGS:
        for name in sorted(os.listdir(folder)):
            if name.startswith(f"{config}_seed") and name.endswith(".npz"):
                seed = int(name[len(config) + 5 : -4])
                z = np.load(os.path.join(folder, name))
                runs[(config, seed)] = {k: z[k] for k in z.files}
    return runs


def accuracy_train(model, d):
    return float(np.mean(model.predict(d["X_train"]) == d["y_train"]))


def baseline_rows(seeds, dataset):
    """SimpleCNN + aug cùng seed, lấy từ kết quả seed ensemble (nếu đã chạy)."""
    prefix = "" if dataset == "mnist" else f"{dataset}_"
    path = f"{TABLES}/{prefix}seed_ensemble_members.csv"
    if not os.path.exists(path):
        print(f"(không có {path}: bỏ qua dòng mốc SimpleCNN + aug)")
        return []
    m = pd.read_csv(path)
    m = m[(m["variant"] == "cnn_aug") & (m["seed"].isin(seeds))]
    return [
        {
            "config": "simple_cnn_aug (mốc)",
            "seed": int(r.seed),
            "n_params": 225_034,
            "train_acc": np.nan,
            "val_acc": r.val_acc,
            "test_acc": r.test_acc,
            "fit_time_s": r.fit_time_s,
        }
        for r in m.itertuples()
    ]


def plot_curves(curves, dataset="mnist"):
    """Learning curve các cấu hình, chỉ dùng các seed mà MỌI cấu hình đều có.

    v2_full có thêm seed: lấy trung bình 3 seed cho nó và 1 seed cho các cấu hình khác thì
    đường v2_full mượt hơn hẳn, so sánh không công bằng. Kết quả đủ seed nằm ở bảng _agg.
    """
    common = set.intersection(*(set(g["seed"]) for _, g in curves.groupby("config")))
    part = curves[curves["seed"].isin(common)].copy()
    part["config"] = part["config"].map(CONFIG_LABELS)
    seeds = ", ".join(str(s) for s in sorted(common))
    prefix = "" if dataset == "mnist" else f"{dataset}_"
    return plot_learning_curves(
        part,
        f"{prefix}cnn_improved_curves.png",
        f"CNN cải tiến: thêm lần lượt từng kỹ thuật (seed {seeds})",
        group="config",
        colors=dict(zip(CONFIG_LABELS.values(), STEP_COLORS)),
        log_loss=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    prefix = "" if args.dataset == "mnist" else f"{args.dataset}_"
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = load_dataset(args.dataset)

    for c in args.configs:
        for s in args.seeds:
            train_or_load(c, s, d, args.dataset)
    runs = load_all_cached(args.dataset)  # mọi lần train đã có, kể cả từ lần chạy trước

    seeds_all = sorted({s for _, s in runs})
    rows, curves = baseline_rows(seeds_all, args.dataset), []
    for (config, seed), r in runs.items():
        rows.append(
            {
                "config": config,
                "seed": seed,
                "n_params": int(r["n_params"]),
                "train_acc": float(r["train_acc"]),
                "val_acc": acc(r["val"], d["y_val"]),
                "test_acc": acc(r["test"], d["y_test"]),
                "fit_time_s": float(r["fit_time_s"]),
            }
        )
        for e, (tl, vl, va) in enumerate(
            zip(r["train_loss"], r["val_loss"], r["val_acc"]), start=1
        ):
            curves.append(
                {
                    "config": config,
                    "seed": seed,
                    "epoch": e,
                    "train_loss": tl,
                    "val_loss": vl,
                    "val_acc": va,
                }
            )
    df, curves = pd.DataFrame(rows), pd.DataFrame(curves)

    cols = ["train_acc", "val_acc", "test_acc", "fit_time_s"]
    g = df.groupby("config", sort=False)
    agg = g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std"))
    agg.insert(0, "n_params", g["n_params"].first())
    agg.insert(0, "n_seeds", g.size())
    agg = agg.reset_index()

    # Ensemble mọi seed đã có của cùng cấu hình (khi có từ 3 seed)
    ens_rows = []
    for config in CONFIGS:
        seeds = sorted(s for c, s in runs if c == config)
        if not seeds:
            continue
        P_val = [runs[(config, s)]["val"] for s in seeds]
        P_test = [runs[(config, s)]["test"] for s in seeds]
        save_probs(P_val[0], args.dataset, f"cnn_{config}", "val")
        save_probs(P_test[0], args.dataset, f"cnn_{config}", "test")
        if len(seeds) >= 3:
            n = len(seeds)
            ens_val, ens_test = soft_vote(P_val), soft_vote(P_test)
            save_probs(ens_val, args.dataset, f"cnn_{config}_ens{n}", "val")
            save_probs(ens_test, args.dataset, f"cnn_{config}_ens{n}", "test")
            ens_rows.append(
                {
                    "config": f"{config} (ensemble {n} seed)",
                    "n_seeds": n,
                    "val_acc_mean": acc(ens_val, d["y_val"]),
                    "test_acc_mean": acc(ens_test, d["y_test"]),
                }
            )
    agg = pd.concat([agg, pd.DataFrame(ens_rows)], ignore_index=True)

    df.to_csv(f"{TABLES}/{prefix}cnn_improved.csv", index=False)
    agg.to_csv(f"{TABLES}/{prefix}cnn_improved_agg.csv", index=False)
    curves.to_csv(f"{TABLES}/{prefix}cnn_improved_curves.csv", index=False)
    plot_curves(curves, args.dataset)

    pd.set_option("display.width", 200)
    print("\n" + df.round(4).to_string(index=False))
    print("\n" + agg.round(4).to_string(index=False))
    last = curves.groupby(["config", "epoch"])["val_acc"].mean().unstack("config")
    print("\nVal acc 5 epoch cuối (trung bình theo seed):\n" + last.tail(5).round(4).to_string())


if __name__ == "__main__":
    main()
