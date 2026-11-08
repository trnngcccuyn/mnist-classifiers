"""Kiểu vẽ chung cho mọi hình trong báo cáo, để các mô hình đối chiếu được với nhau.

Quy ước:
- cùng kích thước hình, cùng độ phân giải, lưu vào results/figures/;
- train luôn màu xanh, validation luôn màu cam, và hai màu này chỉ dùng cho train/validation;
  mỗi mô hình có một màu cố định (MODEL_COLORS) dùng chung ở mọi hình so sánh; các đường
  cấu hình, hàm kích hoạt... dùng LINE_COLORS, không trùng màu nào ở trên;
- tên mô hình và tên siêu tham số hiển thị theo MODEL_LABELS, PARAM_LABELS (tiếng Việt,
  giữ thuật ngữ quen thuộc như train, validation, accuracy);
- accuracy hiển thị theo phần trăm; điểm được chọn theo validation đánh dấu bằng ngôi sao.

Bảng màu đã kiểm tra bằng mô phỏng mù màu đỏ/lục (OKLab ΔE, Machado 2009): các màu đứng cạnh
nhau trong hình so sánh mô hình, và cặp MLP/CNN (vẽ chung trong learning curve), đều tách
biệt cả với người mù màu (ΔE >= 8) lẫn người nhìn bình thường (ΔE >= 15). Màu MLP, CNN và
LINE_COLORS cũng tách biệt với xanh train và cam validation theo cùng ngưỡng.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter

FIGURES_DIR = "results/figures"
FIGSIZE = (6, 4)
DPI = 150
TRAIN_COLOR = "#2a78d6"  # xanh dương
VAL_COLOR = "#eb6834"  # cam
# Màu cho các đường không phải mô hình (cấu hình, hàm kích hoạt...), dùng theo thứ tự.
# Tối đa 3 đường một hình: 3 màu này tách biệt từng đôi một, kể cả với người mù màu.
LINE_COLORS = ["#8e2c48", "#c9b200", "#a28ad8"]  # đỏ mận, vàng, tím nhạt
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
    "mlp": "#3b4a5a",  # xám xanh
    "random_forest": "#1baf7a",  # xanh ngọc
    "cnn": "#c2378e",  # hồng đậm
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


def set_percent_axis(axis):
    """Trục accuracy dạng phần trăm, số chữ số thập phân vừa đủ để phân biệt các vạch.

    PercentFormatter(decimals=1) cố định 1 chữ số nên trục hẹp (vd 83.36%-83.52%) ghi trùng
    "83.5%"; decimals=None thì thừa số 0 ("93.00%"). Ở đây số chữ số lấy theo khoảng cách
    giữa hai vạch: cách 0.02% thì ghi 2 chữ số, cách 1% thì không ghi phần thập phân.
    """

    def fmt(value, _):
        ticks = axis.get_majorticklocs()
        step = np.min(np.diff(ticks)) * 100 if len(ticks) > 1 else 1.0
        decimals = next((d for d in range(7) if abs(round(step, d) - step) < 1e-6 * step), 6)
        return f"{value * 100:.{decimals}f}%"

    axis.set_major_formatter(FuncFormatter(fmt))


def set_log_axis(axis, subs=(1,)):
    """Thang log, nhãn dạng số thường (0.01, 1, 100) thay cho 10^-2, 10^0, 10^2.

    Dùng chung cho mọi trục log để các hình ghi số cùng một kiểu. subs: các mốc có nhãn trong
    mỗi bậc 10; (1, 2, 5) cho trục hẹp, cần thêm mốc (0.01, 0.02, 0.05, 0.1...).
    """
    getattr(axis.axes, f"set_{axis.axis_name}scale")("log")
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def format_param_value(text):
    """Nhãn trục x cho một giá trị siêu tham số: số ghi theo :g ("0.0" -> "0", "1e-05"),
    bộ số ghi nối bằng gạch ("(512, 256, 128)" -> "512-256-128"), còn lại giữ nguyên."""
    if not is_ordered_value(text):
        return text
    parts = [part for part in text.strip("()[] ").split(",") if part.strip()]
    return "-".join(f"{float(part):g}" for part in parts)


def save_figure(fig, filename):
    """Lưu hình vào results/figures/filename rồi đóng hình."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def is_ordered_value(text):
    """True nếu text là số hoặc bộ số ("0.01", "1e-05", "(256, 128)"): giá trị có thứ tự."""
    parts = [part for part in text.strip("()[] ").split(",") if part.strip()]
    try:
        [float(part) for part in parts]
    except ValueError:
        return False
    return bool(parts)


def fit_tick_labels(fig, ax, pad_px=12):
    """Kéo rộng hình (giữ chiều cao) nếu nhãn trục x nằm ngang chồng lên nhau.

    Không xoay nhãn: đo độ rộng thật của từng nhãn, so với khoảng cách giữa hai vạch, rồi
    nhân chiều rộng hình theo tỉ lệ thiếu. Nhãn vừa sẵn thì giữ kích thước FIGSIZE.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [label.get_window_extent(renderer).width for label in ax.get_xticklabels()]
    if len(widths) < 2:
        return
    ticks = ax.transData.transform([(t, 0) for t in ax.get_xticks()])[:, 0]
    spacing = np.min(np.diff(ticks))
    needed = max((a + b) / 2 for a, b in zip(widths, widths[1:])) + pad_px
    if needed > spacing:
        width, height = fig.get_size_inches()
        fig.set_size_inches(width * needed / spacing, height)


def plot_sweep(df, model_name, param_name, filename):
    """Train và validation accuracy theo từng giá trị của param_name.

    df: mỗi hàng một giá trị thử, có các cột param_name, train_acc, val_acc.
    Các giá trị đặt cách đều trên trục x. Giá trị có thứ tự (số, hoặc bộ số như hidden_sizes)
    vẽ thành đường; giá trị phân loại (vd hàm kích hoạt) vẽ dạng dumbbell: chỉ có điểm, nối
    train với validation của cùng một giá trị, vì nối các giá trị với nhau sẽ gợi ra xu hướng
    không có thật.
    """
    values = [str(value) for value in df[param_name]]
    labels = [format_param_value(value) for value in values]
    x = np.arange(len(df))
    best = int(np.argmax(df["val_acc"]))
    best_val = df["val_acc"].iloc[best]
    categorical = not all(is_ordered_value(value) for value in values)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if categorical:
        ax.vlines(x, df["val_acc"], df["train_acc"], color="#cccccc", linewidth=3, zorder=1)
        ax.plot(x, df["train_acc"], "o", markersize=9, color=TRAIN_COLOR, label="train")
        ax.plot(x, df["val_acc"], "o", markersize=9, color=VAL_COLOR, label="validation")
        for xi, train, val in zip(x, df["train_acc"], df["val_acc"]):
            for value, color in ((train, TRAIN_COLOR), (val, VAL_COLOR)):
                ax.annotate(
                    f"{value:.2%}",
                    (xi, value),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        ax.set_xlim(-0.5, len(df) - 0.3)  # chừa chỗ cho số ghi bên phải điểm cuối
    else:
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
    ax.set_xticks(x, labels)
    fit_tick_labels(fig, ax)
    ax.set_xlabel(PARAM_LABELS.get(param_name, param_name))
    ax.set_ylabel("accuracy")
    set_percent_axis(ax.yaxis)
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
        ax_acc.text(value + 0.01, i, f"{value:.2%}", va="center", fontsize=9)
    # Thanh bắt đầu từ 0: cắt trục thì độ dài thanh phóng đại chênh lệch giữa các mô hình.
    # Chênh lệch nhỏ đọc bằng số ghi cạnh thanh.
    ax_acc.set_xlim(0, 1.15)
    ax_acc.set_xticks(np.arange(0, 1.01, 0.2))
    ax_acc.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_acc.set_xlabel("test accuracy")

    ax_time.barh(y, test_df["fit_time_s_mean"], color=colors)
    for i, value in enumerate(test_df["fit_time_s_mean"]):
        ax_time.text(value * 1.1, i, f"{value:.1f} s", va="center", fontsize=9)
    set_log_axis(ax_time.xaxis)
    ax_time.set_xlim(right=test_df["fit_time_s_mean"].max() * 5)
    ax_time.set_xlabel("thời gian huấn luyện (giây, thang log)")

    ax_acc.set_yticks(y, labels)
    ax_acc.invert_yaxis()  # mô hình đầu bảng nằm trên cùng
    for ax in (ax_acc, ax_time):
        ax.grid(axis="x")
    fig.suptitle("So sánh các mô hình trên tập test")
    return save_figure(fig, filename)


CURVE_PANELS = [
    ("train_loss", "train loss"),
    ("val_loss", "validation loss"),
    ("val_acc", "validation accuracy"),
]


def plot_learning_curves(
    curves,
    filename,
    title,
    group=None,
    colors=None,
    mark_epoch=None,
    mark_label="epoch được chọn",
    log_loss=False,
):
    """Learning curve: mỗi cột có trong bảng (train loss, val loss, val acc) là một ô.

    curves: bảng dài, mỗi hàng là một epoch, có cột "epoch" và các cột trong CURVE_PANELS.
    group: tên cột phân biệt các đường (cấu hình, mô hình, hàm kích hoạt); None = một đường.
        Nhiều hàng cùng (nhóm, epoch) là nhiều seed: vẽ trung bình, dải mờ ± 1 độ lệch chuẩn.
    colors: dict {giá trị nhóm: màu}; mặc định lấy LINE_COLORS theo thứ tự.
    mark_epoch: số epoch, hoặc dict {giá trị nhóm: epoch}, vẽ nét đứt (epoch được chọn hoặc
        epoch early stopping giữ lại).
    mark_label: chú thích nét đứt trong legend, ghi kèm số epoch ("<mark_label>: 6").
    log_loss: thang log cho các ô loss.
    """
    panels = [(col, label) for col, label in CURVE_PANELS if col in curves]
    groups = [None] if group is None else list(dict.fromkeys(curves[group]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    axes = axes[0]

    for k, name in enumerate(groups):
        part = curves if name is None else curves[curves[group] == name]
        color = colors[name] if colors else LINE_COLORS[k]
        label = None if name is None else MODEL_LABELS.get(name, name)
        epoch = mark_epoch.get(name) if isinstance(mark_epoch, dict) else mark_epoch
        for ax, (col, _) in zip(axes, panels):
            values = part.groupby("epoch")[col]
            mean, std = values.mean(), values.std(ddof=1).fillna(0.0)
            ax.plot(mean.index, mean, marker="o", markersize=3, color=color, label=label)
            if values.size().max() > 1:
                ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.2)
            if epoch is not None:
                # Chú thích nét đứt nằm trong legend của ô val acc, không ghi lên tiêu đề
                dashed_label = f"{mark_label}: {epoch}" if col == "val_acc" else None
                ax.axvline(epoch, linestyle="--", linewidth=1, color=color, label=dashed_label)

    for ax, (col, label) in zip(axes, panels):
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid()
        if col == "val_acc":
            set_percent_axis(ax.yaxis)
        elif log_loss:
            # Trục loss hẹp (chưa tới 2 bậc 10) nên cần thêm mốc 2, 5 giữa các bậc
            set_log_axis(ax.yaxis, subs=(1, 2, 5))
    if group is not None:
        axes[0].legend(fontsize=9)
    if mark_epoch is not None:
        # Chỉ chú thích nét đứt; tên các đường đã có trong legend ô đầu
        dashed = [line for line in axes[-1].get_lines() if line.get_linestyle() == "--"]
        axes[-1].legend(handles=dashed, fontsize=9, loc="lower right")
    fig.suptitle(title)
    return save_figure(fig, filename)


def plot_class_images(images, filename, title, colorbar_label, symmetric=False):
    """Lưới 2x5 ảnh 28x28, một ảnh cho mỗi lớp 0-9.

    symmetric=False: giá trị trong [0, 1], thang một tông xanh (xác suất).
    symmetric=True: giá trị âm/dương quanh 0, thang hai màu đỏ/xanh, trắng ở 0 (trọng số).
    """
    images = np.asarray(images).reshape(-1, 28, 28)
    if symmetric:
        vmax = np.abs(images).max()
        cmap, vmin = "RdBu_r", -vmax
    else:
        cmap, vmin, vmax = SEQUENTIAL_CMAP, 0, 1

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for c, (ax, image) in enumerate(zip(axes.flat, images)):
        shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"lớp {c}")
        ax.axis("off")
    fig.colorbar(shown, ax=axes, shrink=0.8, label=colorbar_label)
    fig.suptitle(title)
    return save_figure(fig, filename)
