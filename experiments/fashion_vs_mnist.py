"""So sánh các mô hình trên MNIST và Fashion-MNIST.  Phụ trách: Tiên

Chạy:  python -m experiments.run_all --dataset mnist --seeds 0 1 2
       python -m experiments.run_all --dataset fashion_mnist --seeds 0 1 2
       python -m experiments.fashion_vs_mnist

Không train lại: đọc bảng results/tables/<dataset>_summary_agg.csv và ma trận nhầm lẫn trên test
(results/tables/fashion_mnist_<model>_confusion.csv, seed đầu tiên) mà run_all đã lưu.
Cấu hình mô hình giữ nguyên như MNIST (run_all.MODEL_CONFIGS), không chọn lại siêu tham số.

Phân tích:
1. Accuracy, F1 macro test (trung bình ± độ lệch chuẩn theo seed) trên hai tập; mức giảm
   accuracy khi chuyển sang Fashion-MNIST. Không so thời gian huấn luyện: hai tập được chạy
   trên hai máy khác nhau (xem results/tables/<dataset>_machine.json).
2. Recall từng lớp trên Fashion-MNIST của mỗi mô hình: lớp nào khó.
3. Các cặp lớp hay bị nhầm nhất của từng mô hình trên Fashion-MNIST.

Kết quả:
  results/tables/fashion_vs_mnist.csv, fashion_mnist_per_class_recall.csv,
  fashion_mnist_top_confusions.csv
  results/figures/fashion_vs_mnist.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.plotting import (
    INK,
    MODEL_COLORS,
    MODEL_LABELS,
    SEQUENTIAL_CMAP,
    save_figure,
    set_percent_axis,
)

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
DATASETS = ("mnist", "fashion_mnist")
DATASET_LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}
TABLES = "results/tables"


def compare_table(models):
    test = {}
    for ds in DATASETS:
        agg = pd.read_csv(f"{TABLES}/{ds}_summary_agg.csv")
        test[ds] = agg[agg["split"] == "test"].set_index("model")
    rows = []
    for m in models:
        row = {"model": m, "n_seeds": int(test["fashion_mnist"].loc[m, "n_seeds"])}
        for ds in DATASETS:
            t = test[ds].loc[m]
            row[f"{ds}_acc"] = t["accuracy_mean"]
            row[f"{ds}_acc_std"] = t["accuracy_std"]
            row[f"{ds}_f1"] = t["f1_mean"]
        row["acc_drop"] = row["mnist_acc"] - row["fashion_mnist_acc"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_confusions(models):
    return {m: pd.read_csv(f"{TABLES}/fashion_mnist_{m}_confusion.csv").to_numpy() for m in models}


def recall_table(cms):
    """Hàng = mô hình, cột = lớp: recall = đúng / số mẫu thật của lớp."""
    rows = {m: np.diag(cm) / cm.sum(axis=1) for m, cm in cms.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=CLASSES)


def top_confusions(cms, k=5):
    """k ô ngoài đường chéo lớn nhất (nhãn thật -> nhãn dự đoán) của từng mô hình."""
    rows = []
    for m, cm in cms.items():
        off = cm.astype(float).copy()
        np.fill_diagonal(off, -1)
        for idx in np.argsort(off, axis=None)[::-1][:k]:
            i, j = np.unravel_index(idx, cm.shape)
            rows.append(
                {
                    "model": m,
                    "true": CLASSES[i],
                    "pred": CLASSES[j],
                    "count": int(cm[i, j]),
                    "pct_of_true": 100 * cm[i, j] / cm[i].sum(),
                }
            )
    return pd.DataFrame(rows)


def spread(values, min_gap):
    """Vị trí y cho nhãn: nhãn nào cách nhãn ngay dưới chưa đủ min_gap thì đẩy lên."""
    order = np.argsort(values)
    pos = np.asarray(values, dtype=float).copy()
    for a, b in zip(order, order[1:]):
        pos[b] = max(pos[b], pos[a] + min_gap)
    return pos


def plot(cmp, recall, filename):
    fig, (ax, ax_heat) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.25]}, layout="constrained"
    )
    # Slope chart: mỗi mô hình một đoạn MNIST -> Fashion-MNIST. Đọc theo vị trí chứ không theo độ
    # dài thanh, nên trục y không cần bắt đầu từ 0 mà vẫn không phóng đại chênh lệch.
    for _, r in cmp.iterrows():
        color = MODEL_COLORS.get(r["model"], INK)
        ax.errorbar(
            [0, 1],
            [r["mnist_acc"], r["fashion_mnist_acc"]],
            yerr=[r["mnist_acc_std"], r["fashion_mnist_acc_std"]],
            color=color,
            marker="o",
            capsize=3,
        )
    gap = 0.012
    for x, ds, ha, dx in [(0, "mnist", "right", -0.06), (1, "fashion_mnist", "left", 0.06)]:
        for (_, r), y in zip(cmp.iterrows(), spread(cmp[f"{ds}_acc"].to_numpy(), gap)):
            text = f"{r[f'{ds}_acc']:.2%}"
            if ds == "fashion_mnist":
                text = f"{text}  {MODEL_LABELS.get(r['model'], r['model'])}"
            ax.text(x + dx, y, text, ha=ha, va="center", fontsize=9, color=MODEL_COLORS[r["model"]])
    ax.set_xticks([0, 1], [DATASET_LABELS[ds] for ds in DATASETS])
    ax.set_xlim(-0.45, 1.75)
    ax.set_ylabel("test accuracy")
    set_percent_axis(ax.yaxis)
    ax.grid(axis="y")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Test accuracy (trung bình ± độ lệch chuẩn theo seed)", fontsize=11)

    # Heatmap: lớp trên trục y (tên dài nằm ngang), mô hình trên trục x
    R = recall.to_numpy().T
    shown = ax_heat.imshow(R, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            color = "white" if R[i, j] > 0.6 else "black"
            ax_heat.text(j, i, f"{R[i, j]:.0%}", ha="center", va="center", fontsize=8, color=color)
    labels = [MODEL_LABELS.get(m, m).replace(" ", "\n") for m in recall.index]
    ax_heat.set_xticks(range(len(recall)), labels, fontsize=9)
    ax_heat.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_title("Fashion-MNIST: recall từng lớp trên test (seed đầu)", fontsize=11)
    fig.colorbar(shown, ax=ax_heat, shrink=0.8, label="recall", format="{x:.0%}")
    fig.suptitle("MNIST và Fashion-MNIST, cùng cấu hình mô hình")
    return save_figure(fig, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS))
    args = parser.parse_args()

    cmp = compare_table(args.models)
    cms = load_confusions(args.models)
    recall = recall_table(cms)
    top = top_confusions(cms)

    cmp.to_csv(f"{TABLES}/fashion_vs_mnist.csv", index=False)
    recall.to_csv(f"{TABLES}/fashion_mnist_per_class_recall.csv", index_label="model")
    top.to_csv(f"{TABLES}/fashion_mnist_top_confusions.csv", index=False)
    plot(cmp, recall, "fashion_vs_mnist.png")

    pd.set_option("display.width", 200)
    print("So sánh test:\n" + cmp.round(4).to_string(index=False))
    print("\nRecall từng lớp (Fashion-MNIST):\n" + recall.round(3).to_string())
    print("\nCặp nhầm nhiều nhất:\n" + top.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
