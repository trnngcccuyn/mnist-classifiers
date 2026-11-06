"""Vẽ lại các hình dùng chung (sweep, ma trận nhầm lẫn, so sánh mô hình) từ bảng CSV đã lưu,
không phải train lại.

Chạy:  python -m experiments.make_figures --dataset mnist
"""

import argparse
import glob
import os

import pandas as pd

from src.plotting import MODEL_LABELS, plot_confusion, plot_model_comparison, plot_sweep

TABLES_DIR = "results/tables"


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


if __name__ == "__main__":
    main()
