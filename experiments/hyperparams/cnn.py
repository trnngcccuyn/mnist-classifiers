"""Chọn số epoch cho CNN bằng learning curve.  Phụ trách: Uyên

Chạy: python -m experiments.hyperparams.cnn                 # 3 seed x 20 epoch (~15 phút CPU)
      python -m experiments.hyperparams.cnn --seeds 0       # 1 seed

Train SimpleCNN với cấu hình cuối trong experiments/run_all.py (MODEL_CONFIGS["cnn"]) nhưng chạy
MAX_EPOCHS epoch, đo train loss / val loss / val acc sau mỗi epoch (model.history). Learning
curve chính là một sweep theo số epoch mà chỉ cần train một lần.

Adam không có lịch giảm lr, nên với cùng seed thì 10 epoch đầu ở đây trùng khớp với lần chạy
10 epoch trong run_all / cnn_vs_mlp (dùng để kiểm tra tính tái tạo).

Quy tắc chọn (giống MLP): lấy số epoch cố định từ đó val acc không còn tăng, không lấy epoch có
val acc cao nhất, vì val acc dao động giữa các epoch cỡ ±0,2%, ngang với dao động theo seed;
chọn điểm cao nhất là chọn nhiễu và làm val acc báo cáo cao hơn thực tế. Không dùng val loss để
chọn: như ở MLP, val loss tăng lại sớm do mạng tự tin quá mức trong khi val acc vẫn giữ.

Kết quả: results/tables/learning_curve_cnn{,_summary}.csv, results/figures/learning_curve_cnn.png
"""

import argparse
import os

import pandas as pd

from experiments.run_all import MODEL_CONFIGS
from src.data import load_dataset
from src.plotting import MODEL_COLORS, plot_learning_curves
from src.utils import Timer, set_seed

MAX_EPOCHS = 20
CHOSEN_EPOCHS = MODEL_CONFIGS["cnn"][1]["epochs"]  # số epoch đang dùng trong run_all


def run_curves(data, seeds, max_epochs=MAX_EPOCHS):
    cls, params = MODEL_CONFIGS["cnn"]
    rows = []
    for seed in seeds:
        set_seed(seed)
        model = cls(**{**params, "epochs": max_epochs}, seed=seed)
        with Timer() as t:
            model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        h = model.history
        for e, (tl, vl, va) in enumerate(
            zip(h["train_loss"], h["val_loss"], h["val_acc"]), start=1
        ):
            rows.append({"seed": seed, "epoch": e, "train_loss": tl, "val_loss": vl, "val_acc": va})
        print(f"seed {seed}: {t.elapsed:.0f} s, val acc", [round(a, 4) for a in h["val_acc"]])
    return pd.DataFrame(rows)


def summarize(curves, chosen=CHOSEN_EPOCHS):
    """Mỗi seed một dòng, cộng dòng trung bình theo seed ("mean")."""
    rows = []
    for seed, g in curves.groupby("seed"):
        g = g.set_index("epoch")
        rows.append(
            {
                "seed": seed,
                "chosen_epochs": chosen,
                "val_acc_at_chosen": g.loc[chosen, "val_acc"],
                # trung bình 3 epoch cuối trước điểm chọn và mọi epoch sau đó: nếu xấp xỉ
                # nhau thì val acc đã phẳng, chạy thêm không được gì
                "val_acc_mean_before": g.loc[chosen - 2 : chosen, "val_acc"].mean(),
                "val_acc_mean_after": g.loc[chosen + 1 :, "val_acc"].mean(),
                "val_acc_max": g["val_acc"].max(),
                "epoch_val_acc_max": int(g["val_acc"].idxmax()),
                "epoch_val_loss_min": int(g["val_loss"].idxmin()),
            }
        )
    df = pd.DataFrame(rows)
    mean = df.drop(columns="seed").mean().to_frame().T
    mean.insert(0, "seed", "mean")
    return pd.concat([df, mean], ignore_index=True)


def plot(curves, chosen=CHOSEN_EPOCHS):
    """Trung bình theo seed, dải mờ ± 1 độ lệch chuẩn; nét đứt: số epoch đang dùng."""
    plot_learning_curves(
        curves,
        "learning_curve_cnn.png",
        f"CNN: learning curve (trung bình {curves['seed'].nunique()} seed)",
        colors={None: MODEL_COLORS["cnn"]},
        mark_epoch=chosen,
        log_loss=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    if args.epochs <= CHOSEN_EPOCHS:
        parser.error(f"--epochs phải lớn hơn {CHOSEN_EPOCHS} để thấy phần sau điểm chọn")

    data = load_dataset("mnist")
    curves = run_curves(data, args.seeds, args.epochs)
    summary = summarize(curves)

    os.makedirs("results/tables", exist_ok=True)
    curves.to_csv("results/tables/learning_curve_cnn.csv", index=False)
    summary.to_csv("results/tables/learning_curve_cnn_summary.csv", index=False)
    plot(curves)

    print("\n" + summary.round(4).to_string(index=False))
    per_epoch = curves.groupby("epoch")["val_acc"].agg(["mean", "std"])
    print("\nVal acc trung bình theo epoch:\n" + per_epoch.round(4).to_string())


if __name__ == "__main__":
    main()
