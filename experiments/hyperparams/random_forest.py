"""Thí nghiệm siêu tham số cho Random Forest.  Phụ trách: Tiên

Chạy: python -m experiments.hyperparams.random_forest
Chạy nhanh trên tập con:
python -m experiments.hyperparams.random_forest --train-size 5000 --n-trees 5 10 --max-depths 4 8

Chọn số cây trước (độ sâu cố định = --base-depth), rồi chọn độ sâu với số cây đó;
không phải grid search. Cấu hình được chọn được đánh giá trên test một lần.

Sweep số cây chỉ train rừng lớn nhất một lần rồi đánh giá k cây đầu (k cây đầu chính là rừng
k cây cùng seed). Vì vậy fit_time_s của sweep này là ước lượng (thời gian cả rừng * k / số cây),
không phải đo riêng cho từng k.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import load_dataset
from src.metrics import accuracy
from src.models.random_forest import RandomForest
from src.utils import SEED, Timer

from .common import best, evaluate_on_test, save_sweep, sweep

N_TREES = [10, 25, 50, 100]
MAX_DEPTHS = [4, 8, 12, 16, 20, 24]


def sweep_n_trees(data, n_trees_values, fixed):
    """Train rừng lớn nhất một lần rồi đánh giá theo k cây đầu."""
    model = RandomForest(n_trees=max(n_trees_values), **fixed)
    with Timer() as timer:
        model.fit(data["X_train"], data["y_train"])
    rows = []
    for k in n_trees_values:
        rows.append(
            {
                "n_trees": str(k),
                "train_acc": accuracy(data["y_train"], model.predict(data["X_train"], n_trees=k)),
                "val_acc": accuracy(data["y_val"], model.predict(data["X_val"], n_trees=k)),
                "fit_time_s": timer.elapsed * k / model.n_trees,  # ước lượng
            }
        )
        print(rows[-1])
    return save_sweep(pd.DataFrame(rows), RandomForest.name, "n_trees")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--n-trees", type=int, nargs="+", default=N_TREES)
    parser.add_argument("--max-depths", type=int, nargs="+", default=MAX_DEPTHS)
    parser.add_argument("--base-depth", type=int, default=12)
    parser.add_argument("--max-thresholds", type=int, default=16)
    args = parser.parse_args()

    data = load_dataset("mnist")
    # Train đã được xáo trộn theo seed 42 khi chia tập; giữ nguyên validation/test.
    data["X_train"] = data["X_train"][: args.train_size]
    data["y_train"] = data["y_train"][: args.train_size]

    fixed = {"seed": SEED, "max_thresholds": args.max_thresholds}
    trees_result = sweep_n_trees(data, args.n_trees, {**fixed, "max_depth": args.base_depth})
    n_trees = int(best(trees_result)["n_trees"])
    depths_result = sweep(
        RandomForest, "max_depth", args.max_depths, data, fixed={**fixed, "n_trees": n_trees}
    )
    chosen = best(depths_result)
    max_depth = int(chosen["max_depth"])

    tables = Path("results/tables")
    config = {
        "dataset": "mnist",
        "seed": SEED,
        "train_size": len(data["y_train"]),
        "val_size": len(data["y_val"]),
        "test_size": len(data["y_test"]),
        "n_trees_values": args.n_trees,
        "max_depth_values": args.max_depths,
        "base_depth": args.base_depth,
        "max_features": "sqrt",
        "max_thresholds": args.max_thresholds,
        "selection": "Chọn n_trees trước, rồi chọn max_depth; không phải grid search",
        "n_trees_fit_time": "Ước lượng: thời gian train cả rừng * k / số cây lớn nhất",
        "selected_n_trees": n_trees,
        "selected_max_depth": max_depth,
    }
    (tables / "random_forest_sweep_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model = RandomForest(n_trees=n_trees, max_depth=max_depth, **fixed)
    summary = {
        "n_trees": model.n_trees,
        "max_depth": model.max_depth,
        "max_thresholds": model.max_thresholds,
        "seed": model.seed,
        "n_jobs": model.n_jobs,
        "train_size": len(data["y_train"]),
        "val_accuracy": chosen["val_acc"],
        **evaluate_on_test(model, data),
    }
    pd.DataFrame([summary]).to_csv(tables / "mnist_random_forest_best.csv", index=False)
    print(f"Cấu hình được chọn và kết quả test: {summary}")


if __name__ == "__main__":
    main()
