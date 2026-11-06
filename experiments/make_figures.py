"""Vẽ lại các hình từ bảng CSV đã lưu, không phải train lại.

Chạy:  python -m experiments.make_figures --dataset mnist

Gồm: sweep siêu tham số, ma trận nhầm lẫn, so sánh mô hình, learning curve (LR, MLP, CNN,
MLP vs CNN) và vanishing gradient. Hai hình cần mô hình đã train (ảnh trọng số Logistic
Regression, xác suất pixel Bernoulli NB) được vẽ lại khi chạy script thí nghiệm tương ứng.
"""

import argparse
import glob
import os

import pandas as pd

from experiments import cnn_vs_mlp, vanishing_gradient
from experiments.hyperparams import cnn, logistic_regression, mlp
from src.plotting import MODEL_LABELS, plot_confusion, plot_model_comparison, plot_sweep

TABLES_DIR = "results/tables"


def read_table(name):
    """Đọc results/tables/<name>.csv; None nếu chưa có (thí nghiệm chưa chạy)."""
    path = f"{TABLES_DIR}/{name}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def mnist_only_figures():
    """Các hình chỉ có trên MNIST: learning curve và vanishing gradient."""
    curves = read_table("learning_curve_logistic_regression")
    if curves is not None:
        logistic_regression.plot_curve(curves)

    curves, summary = read_table("learning_curve_mlp"), read_table("learning_curve_mlp_summary")
    if curves is not None and summary is not None:
        # Chỉ đánh dấu cấu hình dừng sớm (epoch giữ lại < số epoch đã chạy)
        stopped = summary[summary["kept_epoch"] < summary["epochs_run"]]
        mlp.plot_mlp_curves(curves, dict(zip(stopped["config"], stopped["kept_epoch"])))

    curves = read_table("learning_curve_cnn")
    if curves is not None:
        cnn.plot(curves)

    curves = read_table("cnn_vs_mlp_curves")
    if curves is not None:
        cnn_vs_mlp.plot_curves(curves)

    grad_norms = read_table("vanishing_gradient_grad_norms")
    if grad_norms is not None:
        vanishing_gradient.plot_grad_norms(grad_norms)
    curves = read_table("vanishing_gradient_curves")
    if curves is not None:
        vanishing_gradient.plot_train_curves(curves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    args = parser.parse_args()
    prefix = "" if args.dataset == "mnist" else f"{args.dataset}_"

    # Sweep: hyperparam_<prefix><model>_<param>.csv; cột đầu tiên là tên siêu tham số.
    for path in sorted(glob.glob(f"{TABLES_DIR}/hyperparam_{prefix}*.csv")):
        name = os.path.basename(path)[len("hyperparam_") : -len(".csv")]
        if args.dataset == "mnist" and name.startswith("fashion_mnist_"):
            continue
        model_and_param = name[len(prefix) :]
        model = next(m for m in MODEL_LABELS if model_and_param.startswith(m + "_"))
        param = model_and_param[len(model) + 1 :]
        df = pd.read_csv(path, dtype={param: str})
        print(plot_sweep(df, model, param, f"hyperparam_{name}.png"))

    for model in MODEL_LABELS:
        path = f"{TABLES_DIR}/{args.dataset}_{model}_confusion.csv"
        if os.path.exists(path):
            cm = pd.read_csv(path).to_numpy()
            print(plot_confusion(cm, model, f"{args.dataset}_{model}_confusion.png"))

    path = f"{TABLES_DIR}/{args.dataset}_summary_agg.csv"
    if os.path.exists(path):
        agg = pd.read_csv(path)
        test = agg[agg["split"] == "test"]
        print(plot_model_comparison(test, f"{args.dataset}_model_comparison.png"))

    if args.dataset == "mnist":
        mnist_only_figures()


if __name__ == "__main__":
    main()
