# Handwritten Digit Recognition with Classical and Neural Classifiers

**English** | [Tiếng Việt](README.vi.md)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/core-NumPy-013243)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

Midterm project for the Pattern Recognition course. We implement four classifiers from
scratch (Naïve Bayes, Logistic Regression, MLP, Random Forest), train them on MNIST, and
compare them on accuracy, training time, prediction time and interpretability. Extensions:
a simple CNN, data augmentation, and Fashion-MNIST. scikit-learn is used only to
cross-check our results.

<!-- TODO: uncomment once the figure exists
![Model comparison](results/figures/model_comparison.png)
-->

## Team

| Full name | Student ID | Responsibilities |
|---|---|---|
| TODO (Uyên) | TODO | Data preprocessing, metrics, Logistic Regression, MLP, CNN |
| TODO (Tiên) | TODO | Naïve Bayes, Random Forest, data augmentation, Fashion-MNIST |

- **Course:** Pattern Recognition
- **Instructor:** TODO
- **Class / Semester:** TODO
- **University:** TODO

## Installation

Requires Python 3.10 or newer.

```bash
git clone <REPO-URL>
cd mnist-classifiers
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The datasets are downloaded automatically into `data/` on first run.

## Usage

```bash
# Train and evaluate the four required models; tables are saved to results/tables/
python -m experiments.run_all --dataset mnist

# Hyperparameter experiments; figures are saved to results/figures/
python -m experiments.hyperparams

# Extensions
python -m experiments.cnn_vs_mlp
python -m experiments.augmentation_exp
python -m experiments.run_all --dataset fashion_mnist

# Cross-check against scikit-learn
pytest
```

The main results can be reproduced with `notebooks/main_results.ipynb`.

## Results

MNIST, 50,000 train / 10,000 validation / 10,000 test, seed 42. TODO: fill in after running.

| Model | Train acc. | Val. acc. | Test acc. | Macro F1 (test) | Training time | Prediction time |
|---|---|---|---|---|---|---|
| Naïve Bayes | | | | | | |
| Logistic Regression | | | | | | |
| MLP | | | | | | |
| Random Forest | | | | | | |
| CNN | | | | | | |

## Project structure

```
mnist-classifiers/
├── src/
│   ├── data.py                 # loading, normalisation, flattening, splits
│   ├── metrics.py              # accuracy, precision, recall, F1, confusion matrix
│   ├── augmentation.py         # rotation, shift, noise
│   ├── utils.py                # seeding, timing
│   └── models/
│       ├── base.py             # shared fit / predict / predict_proba interface
│       ├── naive_bayes.py      # Gaussian Naïve Bayes
│       ├── logistic_regression.py  # softmax regression with mini-batch GD
│       ├── mlp.py              # MLP with backpropagation
│       ├── random_forest.py    # ensemble
│       └── cnn.py              # simple CNN (extension)
├── experiments/                # experiment scripts
├── notebooks/                  # notebook reproducing the main results
├── tests/                      # cross-checks against scikit-learn
├── results/                    # tables/ and figures/
├── report/                     # PDF report
├── slides/                     # presentation slides
└── docs/PLAN.md                # work split and schedule
```

## Report and slides

- Report: `report/report.pdf` (TODO)
- Slides: `slides/` (TODO)

## References

TODO
