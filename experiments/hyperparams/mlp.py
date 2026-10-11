"""Thí nghiệm siêu tham số cho MLP.

Chạy: python -m experiments.hyperparams.mlp
"""

from src.data import load_dataset
from src.models.mlp import MLP

from .common import sweep

HIDDEN_SIZES = [(64,), (256,), (256, 128), (512, 256, 128)]
LEARNING_RATES = [0.001, 0.01, 0.1]


def main():
    data = load_dataset("mnist")
    sweep(MLP, "hidden_sizes", HIDDEN_SIZES, data)
    sweep(MLP, "lr", LEARNING_RATES, data)


if __name__ == "__main__":
    main()
