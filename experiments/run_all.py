"""Huấn luyện + đánh giá các mô hình, lưu bảng so sánh.  Phụ trách: Uyên

Chạy:  python -m experiments.run_all --dataset mnist
"""

import argparse
import os

import pandas as pd

from src.data import load_dataset
from src.metrics import confusion_matrix, evaluate
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLP
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.random_forest import RandomForest
from src.utils import Timer, set_seed

MODELS = [GaussianNaiveBayes, LogisticRegression, MLP, RandomForest]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    args = parser.parse_args()

    set_seed()
    d = load_dataset(args.dataset)
    os.makedirs("results/tables", exist_ok=True)
    rows = []

    for cls in MODELS:
        model = cls()
        with Timer() as t_fit:
            model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
        for split in ("train", "val", "test"):
            X, y = d[f"X_{split}"], d[f"y_{split}"]
            with Timer() as t_pred:
                y_pred = model.predict(X)
            rows.append(
                {
                    "model": model.name,
                    "split": split,
                    **evaluate(y, y_pred),
                    "fit_time_s": t_fit.elapsed,
                    "predict_time_s": t_pred.elapsed,
                }
            )
            if split == "test":
                cm = confusion_matrix(y, y_pred)
                pd.DataFrame(cm).to_csv(
                    f"results/tables/{args.dataset}_{model.name}_confusion.csv", index=False
                )
        print(f"Xong {model.name}")

    df = pd.DataFrame(rows)
    df.to_csv(f"results/tables/{args.dataset}_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
