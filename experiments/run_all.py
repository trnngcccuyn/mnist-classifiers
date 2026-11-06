"""Huấn luyện + đánh giá các mô hình với cấu hình đã chọn, lưu bảng so sánh.  Phụ trách: Uyên

Chạy:  python -m experiments.run_all --dataset mnist
       python -m experiments.run_all --dataset mnist --models logistic_regression mlp
       python -m experiments.run_all --dataset mnist --seeds 0 1 2

Mỗi mô hình lưu:
- một dòng cho mỗi (seed, tập train / val / test) trong results/tables/<dataset>_summary.csv
  (accuracy, precision, recall, F1 macro, thời gian huấn luyện và dự đoán);
- bảng gộp theo seed (trung bình, độ lệch chuẩn): results/tables/<dataset>_summary_agg.csv;
- ma trận nhầm lẫn trên test: results/tables/<dataset>_<model>_confusion.csv và hình
  results/figures/<dataset>_<model>_confusion.png;
- hình so sánh các mô hình trên test: results/figures/<dataset>_model_comparison.png;
- xác suất dự đoán trên val và test: results/probs/<dataset>/<model>_{val,test}.npy,
  để làm ensemble mà không phải train lại.
Ma trận nhầm lẫn và xác suất lấy từ seed đầu tiên trong --seeds.

Mô hình không có tham số seed (Naive Bayes, hoàn toàn tất định) chỉ chạy một lần.
Khi chạy một phần mô hình (--models), các dòng của mô hình khác trong summary được giữ nguyên.
"""

import argparse
import inspect
import os

import pandas as pd

from src.data import load_dataset
from src.metrics import confusion_matrix, evaluate
from src.models.cnn import SimpleCNN
from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLP
from src.models.naive_bayes import BernoulliNaiveBayes, GaussianNaiveBayes
from src.models.random_forest import RandomForest
from src.plotting import plot_confusion, plot_model_comparison
from src.utils import SEED, Timer, save_probs, set_seed

# Cấu hình cuối, chọn theo validation (xem experiments/hyperparams/ và results/tables/hyperparam_*):
MODEL_CONFIGS = {
    # var_smoothing 0.05: val 79.91%, cao nhất trong sweep mịn (PR #14)
    "naive_bayes": (GaussianNaiveBayes, {"var_smoothing": 0.05}),
    # alpha 0.01, binarize 0.1: val 83.52% (PR #15); đại diện NB trong bảng so sánh chính
    "bernoulli_naive_bayes": (BernoulliNaiveBayes, {"alpha": 0.01, "binarize": 0.1}),
    # lr 0.1: val 91.63% (lr 1.0 train cao hơn nhưng val thấp hơn); L2 chỉ làm giảm val acc
    "logistic_regression": (LogisticRegression, {"lr": 0.1}),
    # momentum 0.9, lr 0.05, 20 epoch: val 98.06% ± 0.09% (3 seed), cao nhất; không early
    # stopping vì dừng theo val loss làm mất ~0.4% val acc (xem learning_curve_mlp_summary.csv)
    "mlp": (MLP, {"hidden_sizes": (256, 128), "lr": 0.05, "momentum": 0.9, "epochs": 20}),
    # 100 cây, độ sâu 24: val 96.77% (PR #13); ghi rõ thay vì dựa vào mặc định của class
    "random_forest": (RandomForest, {"n_trees": 100, "max_depth": 24}),
    # Conv32-Conv64-FC128 (slide 12 B6b), Adam lr 1e-3, batch 128, 10 epoch: val 98.83% ± 0.14%
    # (3 seed); chạy 20 epoch chỉ thêm ≤ 0.1% val acc (learning_curve_cnn_summary.csv)
    "cnn": (SimpleCNN, {"lr": 1e-3, "epochs": 10, "batch_size": 128}),
}
METRICS = ["accuracy", "precision", "recall", "f1", "fit_time_s", "predict_time_s"]


def has_seed(cls) -> bool:
    return "seed" in inspect.signature(cls.__init__).parameters


def run_model(name, d, dataset, seed, save_outputs):
    cls, params = MODEL_CONFIGS[name]
    set_seed(seed)
    model = cls(**params, **({"seed": seed} if has_seed(cls) else {}))
    with Timer() as t_fit:
        model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"])

    rows = []
    for split in ("train", "val", "test"):
        X, y = d[f"X_{split}"], d[f"y_{split}"]
        with Timer() as t_pred:
            probs = model.predict_proba(X)
        y_pred = probs.argmax(axis=1)
        rows.append(
            {
                "model": name,
                "seed": seed if has_seed(cls) else None,
                "split": split,
                **evaluate(y, y_pred),
                "fit_time_s": t_fit.elapsed,
                "predict_time_s": t_pred.elapsed,
            }
        )
        if save_outputs and split in ("val", "test"):
            save_probs(probs, dataset, name, split)
        if save_outputs and split == "test":
            cm = confusion_matrix(y, y_pred)
            pd.DataFrame(cm).to_csv(f"results/tables/{dataset}_{name}_confusion.csv", index=False)
            plot_confusion(cm, name, f"{dataset}_{name}_confusion.png")
    print(f"Xong {name} (seed {rows[-1]['seed']}): test acc = {rows[-1]['accuracy']:.4f}")
    return rows


def aggregate(df):
    """Trung bình và độ lệch chuẩn (ddof=1) theo seed, cho từng (mô hình, tập)."""
    g = df.groupby(["model", "split"], sort=False)[METRICS]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    agg.insert(0, "n_seeds", g.size())
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_CONFIGS), default=list(MODEL_CONFIGS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    args = parser.parse_args()

    d = load_dataset(args.dataset)  # cách chia train/val/test luôn cố định (seed 42)
    os.makedirs("results/tables", exist_ok=True)
    rows = []
    for name in args.models:
        seeds = args.seeds if has_seed(MODEL_CONFIGS[name][0]) else args.seeds[:1]
        for i, seed in enumerate(seeds):
            rows += run_model(name, d, args.dataset, seed, save_outputs=(i == 0))

    path = f"results/tables/{args.dataset}_summary.csv"
    df = pd.DataFrame(rows)
    if os.path.exists(path):  # giữ kết quả của các mô hình không chạy lại lần này
        old = pd.read_csv(path)
        df = pd.concat([old[~old["model"].isin(args.models)], df], ignore_index=True)
    order = {name: i for i, name in enumerate(MODEL_CONFIGS)}
    df = df.sort_values("model", key=lambda s: s.map(order), kind="stable")
    df.to_csv(path, index=False)

    agg = aggregate(df)
    agg.to_csv(f"results/tables/{args.dataset}_summary_agg.csv", index=False)
    test = agg[agg["split"] == "test"]
    plot_model_comparison(test, f"{args.dataset}_model_comparison.png")
    print(
        test[["model", "n_seeds", "accuracy_mean", "accuracy_std", "fit_time_s_mean"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
