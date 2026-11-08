"""Seed ensemble nhiều CNN và test-time augmentation (TTA).  Phụ trách: Uyên

Chạy:  python -m experiments.seed_ensemble_tta                # MNIST, 5 seed x 2 (~35 phút CPU)
       python -m experiments.seed_ensemble_tta --seeds 0 1 2  # 3 seed
       python -m experiments.seed_ensemble_tta --dataset fashion_mnist

Hai biến thể CNN (cấu hình cuối trong run_all.MODEL_CONFIGS["cnn"]):
  cnn      train không augment
  cnn_aug  train với src.augmentation.get_augment(dataset)
Mỗi biến thể train một mạng cho mỗi seed. Các mạng chỉ khác nhau ở seed (khởi tạo và thứ tự
batch). Với mỗi mạng, lưu xác suất val/test trên ảnh gốc và trên n_views bản biến đổi
(src.tta.get_tta_augment: xoay ±10°, dịch ±2 px, không nhiễu), mỗi seed một rng cố định.
Bộ nhớ đệm results/cache/<dataset>/<biến thể>_seed<k>.npz (không commit): chạy lại thì bỏ
qua phần train đã xong, nên có thể dừng giữa chừng rồi chạy tiếp.

Phân tích (không train thêm):
1. Seed ensemble (slide 11 B6b): soft voting m mạng, m = 1..n_seeds. Với mỗi m lấy trung bình
   trên MỌI tổ hợp m seed (không chỉ m seed đầu), kèm min / max, để đường cong không phụ thuộc
   vào việc seed nào được chọn trước. Có thêm hard voting để so sánh.
2. Cùng sai (3 seed đầu): số ảnh test có ít nhất một mạng sai, cả ba cùng sai, và ensemble sai.
   Ensemble chỉ sửa được những ảnh mà các mạng KHÔNG cùng sai.
3. TTA (slide 14 B6b): accuracy theo số bản biến đổi k = 0..n_views, có và không augment lúc
   train.
4. Bảng tổng hợp: 1 mạng, 1 mạng + TTA, n mạng, n mạng + TTA.
Accuracy test chỉ để báo cáo. Nếu cần chọn (ví dụ số bản TTA), chọn theo val.

Kết quả (tiền tố <dataset>_ nếu không phải MNIST):
  results/tables/seed_ensemble{,_members,_coerrors}.csv, tta.csv, seed_ensemble_tta_summary.csv
  results/figures/seed_ensemble_tta.png
  results/probs/<dataset>/<biến thể>_{tta,ens<n>,ens<n>_tta}_{val,test}.npy  (cho phần ensemble)
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.augmentation import get_augment
from src.data import load_dataset
from src.ensemble import hard_vote, soft_vote
from src.plotting import (
    DATASET_LABELS,
    LINE_COLORS,
    MODEL_COLORS,
    save_figure,
    set_percent_axis,
)
from src.tta import get_tta_augment, tta_views_proba
from src.utils import SEED, Timer, save_probs, set_seed

VARIANTS = ("cnn", "cnn_aug")
N_VIEWS = 8
COLORS = {"cnn": MODEL_COLORS["cnn"], "cnn_aug": LINE_COLORS[2]}  # không dùng xanh/cam train/val
LABELS = {"cnn": "CNN", "cnn_aug": "CNN + aug"}
TABLES, FIGURES, CACHE = "results/tables", "results/figures", "results/cache"


def acc(P, y):
    return float(np.mean(np.asarray(P).argmax(axis=-1) == y))


# ---------------------------------------------------------------- train (hoặc đọc bộ nhớ đệm)


def train_or_load(variant, seed, d, dataset, n_views):
    """Trả về dict: val / test = mảng (n_views + 1, N, C) xác suất trên ảnh gốc + bản biến đổi."""
    path = os.path.join(CACHE, dataset, f"{variant}_seed{seed}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if z["val"].shape[0] >= n_views + 1:
            print(f"đọc {path}")
            return {
                "val": z["val"][: n_views + 1],
                "test": z["test"][: n_views + 1],
                "fit_time_s": float(z["fit_time_s"]),
            }

    cls, params = MODEL_CONFIGS["cnn"]
    augment = get_augment(dataset) if variant == "cnn_aug" else None
    set_seed(seed)
    model = cls(**params, augment=augment, seed=seed)
    with Timer() as t:
        model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    tta_aug = get_tta_augment(dataset)
    out = {"fit_time_s": t.elapsed}
    for i, split in enumerate(("val", "test")):
        rng = np.random.default_rng([SEED, seed, i])  # cố định theo seed và tập dữ liệu
        out[split] = tta_views_proba(model, d[f"X_{split}"], n_views, tta_aug, rng)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(
        f"{variant} seed {seed}: {t.elapsed:.0f} s, val {acc(out['val'][0], d['y_val']):.4f}, "
        f"test {acc(out['test'][0], d['y_test']):.4f}"
    )
    return out


# ---------------------------------------------------------------- phân tích


def members_table(runs, d):
    rows = []
    for (variant, seed), r in runs.items():
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "val_acc": acc(r["val"][0], d["y_val"]),
                "test_acc": acc(r["test"][0], d["y_test"]),
                "val_acc_tta": acc(r["val"].mean(axis=0), d["y_val"]),
                "test_acc_tta": acc(r["test"].mean(axis=0), d["y_test"]),
                "fit_time_s": r["fit_time_s"],
            }
        )
    return pd.DataFrame(rows)


def tta_table(runs, d):
    """Accuracy khi lấy trung bình ảnh gốc + k bản biến đổi đầu tiên, k = 0..n_views."""
    rows = []
    for (variant, seed), r in runs.items():
        for split in ("val", "test"):
            cum = np.cumsum(r[split], axis=0)
            for k in range(len(cum)):
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "split": split,
                        "n_views": k,
                        "acc": acc(cum[k] / (k + 1), d[f"y_{split}"]),
                    }
                )
    return pd.DataFrame(rows)


def ensemble_table(runs, d, seeds):
    """Mỗi biến thể, mỗi số mạng m: thống kê trên mọi tổ hợp m seed."""
    rows = []
    for variant in VARIANTS:
        for m in range(1, len(seeds) + 1):
            stats = {k: [] for k in ("val", "test", "test_hard", "test_tta")}
            for combo in itertools.combinations(seeds, m):
                members = [runs[(variant, s)] for s in combo]
                stats["val"].append(acc(soft_vote([r["val"][0] for r in members]), d["y_val"]))
                stats["test"].append(acc(soft_vote([r["test"][0] for r in members]), d["y_test"]))
                pred = hard_vote([r["test"][0] for r in members])
                stats["test_hard"].append(float(np.mean(pred == d["y_test"])))
                P_tta = soft_vote([r["test"].mean(axis=0) for r in members])
                stats["test_tta"].append(acc(P_tta, d["y_test"]))
            rows.append(
                {
                    "variant": variant,
                    "n_models": m,
                    "n_subsets": len(stats["val"]),
                    "val_acc_mean": np.mean(stats["val"]),
                    "test_acc_mean": np.mean(stats["test"]),
                    "test_acc_min": np.min(stats["test"]),
                    "test_acc_max": np.max(stats["test"]),
                    "test_acc_hard_mean": np.mean(stats["test_hard"]),
                    "test_acc_tta_mean": np.mean(stats["test_tta"]),
                }
            )
    return pd.DataFrame(rows)


def coerror_table(runs, d, seeds):
    """3 seed đầu của mỗi biến thể: các mạng cùng sai đến mức nào trên test."""
    y = d["y_test"]
    rows = []
    for variant in VARIANTS:
        P = [runs[(variant, s)]["test"][0] for s in seeds[:3]]
        wrong = np.stack([p.argmax(axis=1) != y for p in P])
        n_any, n_all = int(wrong.any(axis=0).sum()), int(wrong.all(axis=0).sum())
        rows.append(
            {
                "variant": variant,
                "seeds": " ".join(map(str, seeds[:3])),
                "errors_each": " / ".join(str(int(w.sum())) for w in wrong),
                "n_any_wrong": n_any,
                "n_all_wrong": n_all,
                "pct_all_wrong_of_any": 100 * n_all / max(n_any, 1),
                "n_soft_vote_wrong": int(np.sum(soft_vote(P).argmax(axis=1) != y)),
                "n_hard_vote_wrong": int(np.sum(hard_vote(P) != y)),
            }
        )
    return pd.DataFrame(rows)


def summary_table(members, ens, n):
    rows = []
    for variant in VARIANTS:
        m = members[members["variant"] == variant]
        e = ens[(ens["variant"] == variant) & (ens["n_models"] == n)].iloc[0]
        configs = [
            ("1 mạng (trung bình seed)", m["test_acc"].mean(), m["test_acc"].std(ddof=1)),
            ("1 mạng + TTA", m["test_acc_tta"].mean(), m["test_acc_tta"].std(ddof=1)),
            (f"{n} mạng", e["test_acc_mean"], np.nan),
            (f"{n} mạng + TTA", e["test_acc_tta_mean"], np.nan),
        ]
        for config, mean, std in configs:
            rows.append(
                {"variant": variant, "config": config, "test_acc": mean, "test_acc_std": std}
            )
    return pd.DataFrame(rows)


def plot(ens, tta, filename, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax = axes[0]
    for variant in VARIANTS:
        e = ens[ens["variant"] == variant]
        c = COLORS[variant]
        ax.plot(e["n_models"], e["test_acc_mean"], marker="o", color=c, label=LABELS[variant])
        ax.fill_between(e["n_models"], e["test_acc_min"], e["test_acc_max"], color=c, alpha=0.15)
        label = f"{LABELS[variant]} + TTA"
        ax.plot(e["n_models"], e["test_acc_tta_mean"], "s--", ms=4, color=c, label=label)
    ax.set_xlabel("số mạng trong ensemble (soft voting)")
    ax.set_ylabel("test accuracy")
    set_percent_axis(ax.yaxis)
    ax.set_title("Seed ensemble (dải mờ: min–max trên mọi tổ hợp seed)", fontsize=11)
    ax.set_xticks(sorted(ens["n_models"].unique()))
    ax.legend(fontsize=9)
    ax.grid()

    ax = axes[1]
    t = tta[tta["split"] == "test"]
    for variant in VARIANTS:
        g = t[t["variant"] == variant].groupby("n_views")["acc"]
        mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
        c = COLORS[variant]
        ax.plot(mean.index, mean, marker="o", color=c, label=LABELS[variant])
        ax.fill_between(mean.index, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xlabel("số bản biến đổi lúc test (0 = không TTA)")
    ax.set_xticks(sorted(t["n_views"].unique()))
    ax.set_title("TTA, 1 mạng (trung bình ± độ lệch chuẩn theo seed)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid()
    fig.suptitle(f"{DATASET_LABELS[dataset]}: seed ensemble và TTA", y=1.02)
    return save_figure(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-views", type=int, default=N_VIEWS)
    args = parser.parse_args()
    if len(args.seeds) < 3:
        parser.error("cần ít nhất 3 seed (bảng cùng sai dùng 3 mạng)")

    prefix = "" if args.dataset == "mnist" else f"{args.dataset}_"
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    d = load_dataset(args.dataset)

    runs = {
        (v, s): train_or_load(v, s, d, args.dataset, args.n_views)
        for v in VARIANTS
        for s in args.seeds
    }
    n = len(args.seeds)
    members = members_table(runs, d)
    tta = tta_table(runs, d)
    ens = ensemble_table(runs, d, args.seeds)
    co = coerror_table(runs, d, args.seeds)
    summary = summary_table(members, ens, n)

    members.to_csv(f"{TABLES}/{prefix}seed_ensemble_members.csv", index=False)
    tta.to_csv(f"{TABLES}/{prefix}tta.csv", index=False)
    ens.to_csv(f"{TABLES}/{prefix}seed_ensemble.csv", index=False)
    co.to_csv(f"{TABLES}/{prefix}seed_ensemble_coerrors.csv", index=False)
    summary.to_csv(f"{TABLES}/{prefix}seed_ensemble_tta_summary.csv", index=False)
    plot(ens, tta, f"{prefix}seed_ensemble_tta.png", args.dataset)

    first = args.seeds[0]
    for v in VARIANTS:
        for split in ("val", "test"):
            P = [runs[(v, s)][split] for s in args.seeds]
            save_probs(P[0].mean(axis=0), args.dataset, f"{v}_tta", split)  # seed đầu
            save_probs(soft_vote([p[0] for p in P]), args.dataset, f"{v}_ens{n}", split)
            P_ens_tta = soft_vote([p.mean(axis=0) for p in P])
            save_probs(P_ens_tta, args.dataset, f"{v}_ens{n}_tta", split)
    print(f"đã lưu xác suất {{cnn,cnn_aug}}_{{tta,ens{n},ens{n}_tta}} (TTA đơn lẻ: seed {first})")

    pd.set_option("display.width", 200)
    print("\nTừng mạng:\n" + members.round(4).to_string(index=False))
    print("\nSeed ensemble:\n" + ens.round(4).to_string(index=False))
    print("\nCùng sai (3 seed đầu):\n" + co.round(2).to_string(index=False))
    t = tta.pivot_table(index="n_views", columns=["variant", "split"], values="acc")
    print("\nTTA (trung bình theo seed):\n" + t.round(4).to_string())
    print("\nTổng hợp:\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
