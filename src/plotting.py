"""Kiểu vẽ chung cho mọi hình trong báo cáo, để các mô hình đối chiếu được với nhau.

Quy ước:
- cùng kích thước hình, cùng độ phân giải, lưu vào results/figures/;
- train luôn màu xanh, validation luôn màu cam; mỗi mô hình có một màu cố định (MODEL_COLORS)
  dùng chung ở mọi hình so sánh;
- tên mô hình và tên siêu tham số hiển thị theo MODEL_LABELS, PARAM_LABELS (tiếng Việt,
  giữ thuật ngữ quen thuộc như train, validation, accuracy);
- accuracy hiển thị theo phần trăm; điểm được chọn theo validation đánh dấu bằng ngôi sao.

Bảng màu đã kiểm tra bằng mô phỏng mù màu đỏ/lục (OKLab ΔE, Machado 2009): các màu đứng cạnh
nhau trong hình so sánh mô hình, và cặp MLP/CNN (vẽ chung trong learning curve), đều tách
biệt cả với người mù màu (ΔE >= 8) lẫn người nhìn bình thường (ΔE >= 15).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

FIGURES_DIR = "results/figures"
FIGSIZE = (6, 4)
DPI = 150
TRAIN_COLOR = "#2a78d6"  # xanh dương
VAL_COLOR = "#eb6834"  # cam
INK = "#333333"  # màu chữ: xám đậm, đỡ gắt hơn đen tuyền
# Thang một tông xanh, trắng -> xanh đậm, cho ma trận nhầm lẫn
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#999999",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "text.color": INK,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
    }
)

MODEL_LABELS = {
    "naive_bayes": "Gaussian NB",
    "bernoulli_naive_bayes": "Bernoulli NB",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "cnn": "CNN",
}
# Thứ tự khoá = thứ tự thanh trong hình so sánh; đổi thứ tự thì phải kiểm tra lại màu.
MODEL_COLORS = {
    "naive_bayes": "#6250d6",  # tím
    "bernoulli_naive_bayes": "#e87ba4",  # hồng
    "logistic_regression": "#008300",  # xanh lá
    "mlp": "#2a78d6",  # xanh dương
    "random_forest": "#1baf7a",  # xanh ngọc
    "cnn": "#eb6834",  # cam
}
PARAM_LABELS = {
    "var_smoothing": "var_smoothing",
    "binarize": "ngưỡng nhị phân hoá",
    "alpha": "alpha (làm trơn Laplace)",
    "lr": "learning rate",
    "l2": "hệ số L2 (λ)",
    "hidden_sizes": "số nơ-ron các lớp ẩn",
    "activation": "hàm kích hoạt",
    "momentum": "momentum",
    "n_trees": "số cây",
    "max_depth": "độ sâu tối đa",
}


def save_figure(fig, filename):
    """Lưu hình vào results/figures/filename rồi đóng hình."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sweep(df, model_name, param_name, filename):
    """Train và validation accuracy theo từng giá trị của param_name.

    df: mỗi hàng một giá trị thử, có các cột param_name, train_acc, val_acc.
    Các giá trị đặt cách đều trên trục x (có thể không phải số, ví dụ hàm kích hoạt).
    """
    labels = [str(value) for value in df[param_name]]
    x = np.arange(len(df))
    best = int(np.argmax(df["val_acc"]))
    best_val = df["val_acc"].iloc[best]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, df["train_acc"], marker="o", color=TRAIN_COLOR, label="train")
    ax.plot(x, df["val_acc"], marker="o", color=VAL_COLOR, label="validation")
    ax.plot(
        x[best],
        best_val,
        marker="*",
        markersize=16,
        color=VAL_COLOR,
        markeredgecolor="black",
        linestyle="none",
    )
    long_labels = len(labels) > 6 or max(len(label) for label in labels) > 8
    ax.set_xticks(
        x, labels, rotation=35 if long_labels else 0, ha="right" if long_labels else "center"
    )
    ax.set_xlabel(PARAM_LABELS.get(param_name, param_name))
    ax.set_ylabel("accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(
        f"{MODEL_LABELS.get(model_name, model_name)}: {PARAM_LABELS.get(param_name, param_name)}\n"
        f"★ chọn {labels[best]} (val {best_val:.2%})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid()
    return save_figure(fig, filename)


def plot_confusion(cm, model_name, filename):
    """Ma trận nhầm lẫn trên test: ô ghi số mẫu, màu là tỉ lệ theo hàng (theo nhãn thật)."""
    cm = np.asarray(cm)
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    accuracy = np.trace(cm) / cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    shown = ax.imshow(rates, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if rates[i, j] > 0.5 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xlabel("nhãn dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)}: test accuracy {accuracy:.2%}")
    fig.colorbar(shown, ax=ax, shrink=0.8, label="tỉ lệ theo hàng")
    return save_figure(fig, filename)


def plot_model_comparison(test_df, filename):
    """So sánh các mô hình trên test: accuracy (trái) và thời gian huấn luyện (phải).

    test_df: bảng gộp theo seed (results/tables/<dataset>_summary_agg.csv), chỉ các dòng
    split == "test". Thanh sai số là ± 1 độ lệch chuẩn theo seed (nếu có nhiều seed).
    """
    names = list(test_df["model"])
    labels = [MODEL_LABELS.get(name, name) for name in names]
    colors = [MODEL_COLORS.get(name, "tab:gray") for name in names]
    y = np.arange(len(names))

    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(12, 0.6 * len(names) + 1.5), sharey=True)
    ax_acc.barh(
        y,
        test_df["accuracy_mean"],
        xerr=test_df["accuracy_std"].fillna(0),
        color=colors,
        ecolor=INK,
    )
    for i, value in enumerate(test_df["accuracy_mean"]):
        ax_acc.text(value + 0.003, i, f"{value:.2%}", va="center", fontsize=9)
    ax_acc.set_xlim(test_df["accuracy_mean"].min() - 0.05, 1.03)
    ax_acc.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_acc.set_xlabel("test accuracy")

    ax_time.barh(y, test_df["fit_time_s_mean"], color=colors)
    for i, value in enumerate(test_df["fit_time_s_mean"]):
        ax_time.text(value * 1.1, i, f"{value:.1f} s", va="center", fontsize=9)
    ax_time.set_xscale("log")
    ax_time.set_xlim(right=test_df["fit_time_s_mean"].max() * 5)
    ax_time.set_xlabel("thời gian huấn luyện (giây, thang log)")

    ax_acc.set_yticks(y, labels)
    ax_acc.invert_yaxis()  # mô hình đầu bảng nằm trên cùng
    for ax in (ax_acc, ax_time):
        ax.grid(axis="x")
    fig.suptitle("So sánh các mô hình trên tập test")
    return save_figure(fig, filename)
