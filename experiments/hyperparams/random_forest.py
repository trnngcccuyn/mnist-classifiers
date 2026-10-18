"""Thí nghiệm siêu tham số cho Random Forest.

Chạy: python -m experiments.hyperparams.random_forest
Chạy nhanh trên tập con:
python -m experiments.hyperparams.random_forest --train-size 5000 --n-trees 5 10 --max-depths 4 8
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import load_dataset
from src.metrics import confusion_matrix, evaluate
from src.models.random_forest import RandomForest
from src.utils import SEED, Timer

from .common import sweep

N_TREES = [10, 25, 50, 100]
MAX_DEPTHS = [4, 8, 12, 16]


def _parse_args():
    """Đọc và kiểm tra cấu hình chạy thí nghiệm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--n-trees", type=int, nargs="+", default=N_TREES)
    parser.add_argument("--max-depths", type=int, nargs="+", default=MAX_DEPTHS)
    parser.add_argument("--base-depth", type=int, default=12)
    parser.add_argument("--max-thresholds", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.train_size <= 50_000:
        parser.error("train-size phải nằm trong 1..50000")
    if any(n < 1 for n in args.n_trees) or any(d < 0 for d in args.max_depths):
        parser.error("n-trees phải >= 1 và max-depths phải >= 0")
    if args.base_depth < 0 or args.max_thresholds < 1:
        parser.error("base-depth phải >= 0 và max-thresholds phải >= 1")
    return args


def _select_model(data, args):
    """Chọn số cây trước, rồi chọn độ sâu bằng validation accuracy."""
    fixed = {"seed": SEED, "max_thresholds": args.max_thresholds}
    trees_result = sweep(
        RandomForest, "n_trees", args.n_trees, data, fixed={**fixed, "max_depth": args.base_depth}
    )
    best_n_trees = int(trees_result.loc[trees_result["val_acc"].idxmax(), "n_trees"])
    depths_result = sweep(
        RandomForest, "max_depth", args.max_depths, data, fixed={**fixed, "n_trees": best_n_trees}
    )
    best_depth = int(depths_result.loc[depths_result["val_acc"].idxmax(), "max_depth"])
    model = RandomForest(n_trees=best_n_trees, max_depth=best_depth, **fixed)
    return model, float(depths_result["val_acc"].max())


def _evaluate_model(model, data, val_accuracy):
    """Fit cấu hình được chọn và đánh giá test một lần, kèm thời gian thực hiện."""
    with Timer() as fit_timer:
        model.fit(data["X_train"], data["y_train"])
    with Timer() as predict_timer:
        prediction = model.predict(data["X_test"])
    test_metrics = evaluate(data["y_test"], prediction)
    summary = {
        "n_trees": model.n_trees,
        "max_depth": model.max_depth,
        "max_thresholds": model.max_thresholds,
        "seed": model.seed,
        "train_size": len(data["y_train"]),
        "val_accuracy": val_accuracy,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "fit_time_s": fit_timer.elapsed,
        "test_predict_time_s": predict_timer.elapsed,
    }
    return summary, prediction


def main():
    args = _parse_args()

    data = load_dataset("mnist")
    # Train đã được xáo trộn theo seed 42 khi chia tập; giữ nguyên validation/test.
    data["X_train"] = data["X_train"][: args.train_size]
    data["y_train"] = data["y_train"][: args.train_size]
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
    }
    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    print(f"Cấu hình: {config}", flush=True)

    model, val_accuracy = _select_model(data, args)
    config.update(selected_n_trees=model.n_trees, selected_max_depth=model.max_depth)
    (tables / "random_forest_sweep_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary, prediction = _evaluate_model(model, data, val_accuracy)
    pd.DataFrame([summary]).to_csv(tables / "mnist_random_forest_best.csv", index=False)
    pd.DataFrame(confusion_matrix(data["y_test"], prediction)).to_csv(
        tables / "mnist_random_forest_confusion.csv", index=False
    )
    print(f"Cấu hình được chọn và kết quả test: {summary}")


if __name__ == "__main__":
    main()
