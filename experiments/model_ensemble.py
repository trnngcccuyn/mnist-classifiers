"""Ensemble các mô hình khác loại (NB, LR, MLP, RF, CNN) bằng soft/hard voting.  Phụ trách: Tiên

Chạy:  python -m experiments.model_ensemble                         # MNIST
       python -m experiments.model_ensemble --dataset fashion_mnist
       python -m experiments.model_ensemble --models logistic_regression mlp cnn

Không train lại: đọc xác suất val/test đã lưu bởi run_all (results/probs/<dataset>/<model>_*.npy,
seed đầu tiên), nên phải chạy `python -m experiments.run_all --dataset <dataset>` trước.
Khác với seed_ensemble_tta (nhiều CNN chỉ khác seed), ở đây các thành viên khác nhau về bản chất
nên hay sai ở những ảnh khác nhau, nhưng cũng chênh nhau nhiều về độ chính xác.

Phân tích:
1. Từng mô hình: accuracy val / test.
2. Cùng sai từng cặp trên test: #(cả hai sai) / #(ít nhất một sai). Thấp = đa dạng, ensemble
   có nhiều chỗ để sửa.
3. Mọi tập con >= 2 mô hình: soft và hard voting trọng số đều.
4. Bảng tổng hợp, mọi lựa chọn (tập con, trọng số) đều dựa trên VAL, test chỉ để báo cáo:
   mô hình đơn tốt nhất, tất cả (soft / hard), tất cả với trọng số tìm trên val
   (src.ensemble.search_weights), tập con soft voting tốt nhất theo val.

Kết quả (tiền tố <dataset>_ nếu không phải MNIST):
  results/tables/model_ensemble{_members,_coerrors,_subsets,_summary}.csv
  results/figures/model_ensemble.png
  results/probs/<dataset>/model_ensemble_{val,test}.npy  (ensemble trọng số tìm trên val)
"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.ensemble import co_errors, hard_vote, search_weights, soft_vote
from src.plotting import (
    DATASET_LABELS,
    INK,
    MODEL_LABELS,
    SEQUENTIAL_CMAP,
    VAL_COLOR,
    save_figure,
    set_percent_axis,
)
from src.utils import load_probs, save_probs

MODELS = (
    "naive_bayes",
    "bernoulli_naive_bayes",
    "logistic_regression",
    "mlp",
    "random_forest",
    "cnn",
)
# Tên viết tắt để ghép tên tập con ("LR+MLP+CNN"); nhãn trục dùng MODEL_LABELS
SHORT = {
    "naive_bayes": "GNB",
    "bernoulli_naive_bayes": "BNB",
    "logistic_regression": "LR",
    "mlp": "MLP",
    "random_forest": "RF",
    "cnn": "CNN",
}
TABLES = "results/tables"


def acc(P, y):
    return float(np.mean(np.asarray(P).argmax(axis=-1) == y))


def short(names):
    return "+".join(SHORT.get(n, n) for n in names)


def members_table(probs, d):
    return pd.DataFrame(
        [
            {
                "model": m,
                "val_acc": acc(p["val"], d["y_val"]),
                "test_acc": acc(p["test"], d["y_test"]),
            }
            for m, p in probs.items()
        ]
    )


def coerror_matrix(probs, d):
    """Ma trận (n_models, n_models): tỉ lệ cùng sai trên test; đường chéo là số lỗi của mô hình."""
    names = list(probs)
    y = d["y_test"]
    preds = {m: probs[m]["test"].argmax(axis=1) for m in names}
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                M[i, j] = np.sum(preds[a] != y)
            else:
                both, either = co_errors(preds[a], preds[b], y)
                M[i, j] = both / max(either, 1)
    return pd.DataFrame(M, index=names, columns=names)


def subsets_table(probs, d):
    names = list(probs)
    rows = []
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            row = {"models": short(combo), "n_models": k}
            for split in ("val", "test"):
                P = [probs[m][split] for m in combo]
                y = d[f"y_{split}"]
                row[f"{split}_acc_soft"] = acc(soft_vote(P), y)
                row[f"{split}_acc_hard"] = float(np.mean(hard_vote(P) == y))
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(probs, members, subsets, d, weights):
    names = list(probs)
    best = members.loc[members["val_acc"].idxmax()]
    full = subsets[subsets["n_models"] == len(names)].iloc[0]
    best_sub = subsets.loc[subsets["val_acc_soft"].idxmax()]
    P_w = {s: soft_vote([probs[m][s] for m in names], weights) for s in ("val", "test")}
    w_str = " ".join(f"{SHORT.get(m, m)}={w:.2f}" for m, w in zip(names, weights) if w > 0)
    rows = [
        ("mô hình đơn tốt nhất theo val", best["model"], best["val_acc"], best["test_acc"]),
        ("tất cả, soft voting", short(names), full["val_acc_soft"], full["test_acc_soft"]),
        ("tất cả, hard voting", short(names), full["val_acc_hard"], full["test_acc_hard"]),
        (
            "tất cả, trọng số tìm trên val",
            w_str,
            acc(P_w["val"], d["y_val"]),
            acc(P_w["test"], d["y_test"]),
        ),
        (
            "tập con soft voting tốt nhất theo val",
            best_sub["models"],
            best_sub["val_acc_soft"],
            best_sub["test_acc_soft"],
        ),
    ]
    cols = ["config", "models", "val_acc", "test_acc"]
    return pd.DataFrame(rows, columns=cols), P_w


def plot(co, summary, filename, dataset):
    names = list(co.index)
    labels = [MODEL_LABELS.get(m, m).replace(" ", "\n") for m in names]
    fig, (ax, ax_sum) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.2]}, layout="constrained"
    )

    M = co.to_numpy().copy()
    np.fill_diagonal(M, np.nan)
    shown = ax.imshow(M, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                text, color = f"{int(co.iloc[i, j])}", INK
            else:
                text, color = f"{M[i, j]:.2f}", "white" if M[i, j] > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(names)), labels, fontsize=9)
    ax.set_yticks(range(len(names)), labels, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Cùng sai trên test: #(cả hai sai) / #(ít nhất một sai)\n(đường chéo: số lỗi)", fontsize=11
    )
    fig.colorbar(shown, ax=ax, shrink=0.8, label="tỉ lệ cùng sai")

    # Chấm thay cho thanh: chênh lệch giữa các cấu hình chỉ vài phần mười điểm, đọc theo vị trí
    # trên trục hẹp, không theo độ dài thanh (thanh cắt trục sẽ phóng đại chênh lệch).
    y = np.arange(len(summary))
    ax_sum.hlines(y, summary["test_acc"], summary["val_acc"], color="#cccccc", linewidth=3)
    ax_sum.plot(summary["val_acc"], y, "o", color=VAL_COLOR, markersize=8, label="validation")
    ax_sum.plot(summary["test_acc"], y, "D", color=INK, markersize=7, label="test")
    right = summary[["val_acc", "test_acc"]].max(axis=1)
    for yi, (x, (_, r)) in enumerate(zip(right, summary.iterrows())):
        ax_sum.annotate(
            f"test {r['test_acc']:.2%}  ({MODEL_LABELS.get(r['models'], r['models'])})",
            (x, yi),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax_sum.set_yticks(y, summary["config"], fontsize=9)
    ax_sum.invert_yaxis()
    lo = min(summary["val_acc"].min(), summary["test_acc"].min())
    hi = max(summary["val_acc"].max(), summary["test_acc"].max())
    ax_sum.set_xlim(lo - 0.1 * (hi - lo), hi + 0.7 * (hi - lo))  # chừa chỗ cho số ghi bên phải
    # Chỗ trống bên phải chỉ để ghi số: bỏ các vạch quá 100%
    ax_sum.set_xticks([t for t in ax_sum.get_xticks() if t <= 1])
    set_percent_axis(ax_sum.xaxis)
    ax_sum.set_xlabel("accuracy")
    ax_sum.grid(axis="x")
    ax_sum.legend(fontsize=9, loc="lower left")
    ax_sum.set_title("Ensemble khác loại mô hình (mọi lựa chọn theo validation)", fontsize=11)
    fig.suptitle(f"{DATASET_LABELS[dataset]}: ensemble các mô hình")
    return save_figure(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--grid", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("cần ít nhất 2 mô hình")

    prefix = "" if args.dataset == "mnist" else f"{args.dataset}_"
    os.makedirs(TABLES, exist_ok=True)
    d = load_dataset(args.dataset)
    probs = {m: {s: load_probs(args.dataset, m, s) for s in ("val", "test")} for m in args.models}

    members = members_table(probs, d)
    co = coerror_matrix(probs, d)
    subsets = subsets_table(probs, d)
    weights, _ = search_weights([p["val"] for p in probs.values()], d["y_val"], args.grid)
    summary, P_w = summary_table(probs, members, subsets, d, weights)

    members.to_csv(f"{TABLES}/{prefix}model_ensemble_members.csv", index=False)
    co.to_csv(f"{TABLES}/{prefix}model_ensemble_coerrors.csv")
    subsets.to_csv(f"{TABLES}/{prefix}model_ensemble_subsets.csv", index=False)
    summary.to_csv(f"{TABLES}/{prefix}model_ensemble_summary.csv", index=False)
    plot(co, summary, f"{prefix}model_ensemble.png", args.dataset)
    for split, P in P_w.items():
        save_probs(P, args.dataset, "model_ensemble", split)

    pd.set_option("display.width", 200)
    print("Từng mô hình:\n" + members.round(4).to_string(index=False))
    print("\nCùng sai trên test (đường chéo: số lỗi):\n" + co.round(2).to_string())
    top = subsets.sort_values("val_acc_soft", ascending=False).head(10)
    print("\n10 tập con tốt nhất theo val (soft):\n" + top.round(4).to_string(index=False))
    print("\nTổng hợp:\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
