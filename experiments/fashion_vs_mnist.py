"""So sánh các mô hình trên MNIST và Fashion-MNIST.  Phụ trách: Tiên

Chạy:  python -m experiments.run_all --dataset mnist --seeds 0 1 2
       python -m experiments.run_all --dataset fashion_mnist --seeds 0 1 2
       python -m experiments.fashion_vs_mnist

Không train lại: đọc bảng results/tables/<dataset>_summary_agg.csv và ma trận nhầm lẫn trên test
(results/tables/fashion_mnist_<model>_confusion.csv, seed đầu tiên) mà run_all đã lưu.
Cấu hình mô hình giữ nguyên như MNIST (run_all.MODEL_CONFIGS), không chọn lại siêu tham số.

Phân tích:
1. Accuracy, F1 macro test (trung bình ± độ lệch chuẩn theo seed) và thời gian huấn luyện trên
   hai tập; mức giảm accuracy khi chuyển sang Fashion-MNIST.
2. Recall từng lớp trên Fashion-MNIST của mỗi mô hình: lớp nào khó.
3. Các cặp lớp hay bị nhầm nhất của từng mô hình trên Fashion-MNIST.

Kết quả:
  results/tables/fashion_vs_mnist.csv, fashion_mnist_per_class_recall.csv,
  fashion_mnist_top_confusions.csv
  results/figures/fashion_vs_mnist.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import MODEL_CONFIGS

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
SHORT = {
    "naive_bayes": "GNB",
    "bernoulli_naive_bayes": "BNB",
    "logistic_regression": "LR",
    "mlp": "MLP",
    "random_forest": "RF",
    "cnn": "CNN",
}
TABLES, FIGURES = "results/tables", "results/figures"


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
            row[f"{ds}_fit_time_s"] = t["fit_time_s_mean"]
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


def plot(cmp, recall, path):
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.4]}, layout="constrained"
    )
    ax = axes[0]
    x = np.arange(len(cmp))
    for k, (ds, label, c) in enumerate(
        [("mnist", "MNIST", "tab:blue"), ("fashion_mnist", "Fashion-MNIST", "tab:orange")]
    ):
        err = 100 * cmp[f"{ds}_acc_std"].fillna(0)
        ax.bar(x + (k - 0.5) * 0.4, 100 * cmp[f"{ds}_acc"], 0.4, yerr=err, color=c, label=label)
    for xi, (a, b) in enumerate(zip(cmp["mnist_acc"], cmp["fashion_mnist_acc"])):
        ax.text(xi + 0.2, 100 * b + 0.5, f"{100 * b:.1f}", ha="center", fontsize=7)
        ax.text(xi - 0.2, 100 * a + 0.5, f"{100 * a:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [SHORT.get(m, m) for m in cmp["model"]])
    ax.set_ylim(60, 101)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Test accuracy (trung bình ± độ lệch chuẩn theo seed)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    R = 100 * recall.to_numpy()
    im = ax.imshow(R, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(recall)), [SHORT.get(m, m) for m in recall.index])
    ax.set_title("Fashion-MNIST: recall từng lớp trên test (%, seed đầu)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("MNIST và Fashion-MNIST, cùng cấu hình mô hình")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS))
    args = parser.parse_args()
    os.makedirs(FIGURES, exist_ok=True)

    cmp = compare_table(args.models)
    cms = load_confusions(args.models)
    recall = recall_table(cms)
    top = top_confusions(cms)

    cmp.to_csv(f"{TABLES}/fashion_vs_mnist.csv", index=False)
    recall.to_csv(f"{TABLES}/fashion_mnist_per_class_recall.csv", index_label="model")
    top.to_csv(f"{TABLES}/fashion_mnist_top_confusions.csv", index=False)
    plot(cmp, recall, f"{FIGURES}/fashion_vs_mnist.png")

    pd.set_option("display.width", 200)
    print("So sánh test:\n" + cmp.round(4).to_string(index=False))
    print("\nRecall từng lớp (Fashion-MNIST):\n" + recall.round(3).to_string())
    print("\nCặp nhầm nhiều nhất:\n" + top.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
