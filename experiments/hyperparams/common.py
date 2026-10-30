"""Hàm dùng chung cho các thí nghiệm siêu tham số."""

import os

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import accuracy
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

    df = pd.DataFrame(rows)
    name = f"{model_cls.name}_{param_name}"
    if dataset != "mnist":
        name = f"{dataset}_{name}"
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    df.to_csv(f"results/tables/hyperparam_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(df))
    ax.plot(x, df["train_acc"], marker="o", label="train")
    ax.plot(x, df["val_acc"], marker="o", label="validation")
    ax.set_xticks(x, df[param_name])
    ax.set_xlabel(param_name)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{model_cls.name}: {param_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f"results/figures/hyperparam_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df
