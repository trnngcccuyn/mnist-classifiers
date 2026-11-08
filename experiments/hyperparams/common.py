"""Hàm dùng chung cho các thí nghiệm siêu tham số."""

import os

import pandas as pd

from src.metrics import accuracy, evaluate
from src.plotting import plot_sweep
from src.utils import Timer, set_seed


def sweep(model_cls, param_name, values, data, fixed=None, dataset="mnist"):
    """Huấn luyện model_cls với từng giá trị của param_name, lưu bảng CSV và biểu đồ.

    fixed: dict các siêu tham số khác giữ cố định trong suốt thí nghiệm.
    dataset: khác "mnist" thì được thêm vào tên file, để không ghi đè kết quả MNIST.
    """
    rows = []
    for value in values:
        set_seed()
        model = model_cls(**{**(fixed or {}), param_name: value})
        with Timer() as t:
            model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        rows.append(
            {
                param_name: str(value),
                "train_acc": accuracy(data["y_train"], model.predict(data["X_train"])),
                "val_acc": accuracy(data["y_val"], model.predict(data["X_val"])),
                "fit_time_s": t.elapsed,
            }
        )
        print(rows[-1])
    return save_sweep(pd.DataFrame(rows), model_cls.name, param_name, dataset)


def save_sweep(df, model_name, param_name, dataset="mnist"):
    """Lưu bảng CSV và biểu đồ train/validation accuracy theo param_name."""
    name = f"{model_name}_{param_name}"
    if dataset != "mnist":
        name = f"{dataset}_{name}"
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv(f"results/tables/hyperparam_{name}.csv", index=False)
    plot_sweep(df, model_name, param_name, f"hyperparam_{name}.png")
    return df


def best(df):
    """Dòng có validation accuracy cao nhất của một lần sweep."""
    return df.loc[df["val_acc"].idxmax()]


def evaluate_on_test(model, data):
    """Fit cấu hình đã chọn trên train rồi đánh giá test đúng một lần, kèm thời gian."""
    with Timer() as fit_timer:
        model.fit(data["X_train"], data["y_train"])
    with Timer() as predict_timer:
        prediction = model.predict(data["X_test"])
    test_metrics = evaluate(data["y_test"], prediction)
    return {
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "fit_time_s": fit_timer.elapsed,
        "test_predict_time_s": predict_timer.elapsed,
    }
