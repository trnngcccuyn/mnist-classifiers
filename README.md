# Handwritten Digit Recognition with Classical and Neural Classifiers

**English** | [Tiếng Việt](README.vi.md)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/core-NumPy-013243)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

Midterm project for the Pattern Recognition course. We implement four classifiers from
scratch (Naïve Bayes, Logistic Regression, MLP, Random Forest), train them on MNIST, and
compare them on accuracy, training time, prediction time and interpretability. Extensions:
a CNN, data augmentation, seed ensembles and test-time augmentation (TTA), ensembles of
different models, an improved CNN, Fashion-MNIST, and a demo for hand-drawn digits.
scikit-learn is used only to cross-check our results.

![Pipeline overview](results/figures/pipeline_overview.png)

![Model comparison](results/figures/mnist_model_comparison.png)

## Team

| Full name | Student ID | Responsibilities |
|---|---|---|
| Tran Ngoc Cac Uyen | 24280030 | Data preprocessing, metrics, Logistic Regression, MLP, CNN and improved CNN, data augmentation, seed ensemble and TTA, notebook, demo |
| Le Minh Phuc Tien | 24280107 | Naïve Bayes, Random Forest, multi-model ensembles, Fashion-MNIST, pipeline figure |

- **Course:** Pattern Recognition (MTH10322)
- **Instructor:** Dr. Truong Phuoc Hung
- **University:** University of Science, VNU-HCM

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/trnngcccuyn/mnist-classifiers
cd mnist-classifiers
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The datasets are downloaded automatically into `data/` on first run.

## Usage

```bash
# Train and evaluate the four required models + CNN; tables are saved to results/tables/
python -m experiments.run_all --dataset mnist

# Hyperparameter experiments; figures are saved to results/figures/
python -m experiments.hyperparams.logistic_regression
python -m experiments.hyperparams.mlp
python -m experiments.hyperparams.cnn
python -m experiments.hyperparams.naive_bayes
python -m experiments.hyperparams.random_forest
python -m experiments.vanishing_gradient

# Extensions on MNIST
python -m experiments.cnn_vs_mlp
python -m experiments.augmentation_exp
python -m experiments.seed_ensemble_tta
python -m experiments.cnn_improved                                   # ablation, seed 0
python -m experiments.cnn_improved --configs v2_full --seeds 0 1 2   # final configuration, 3 seeds
python -m experiments.model_ensemble       # needs run_all first

# Fashion-MNIST
python -m experiments.run_all --dataset fashion_mnist
python -m experiments.fashion_vs_mnist
python -m experiments.hyperparams.cnn --dataset fashion_mnist --epochs 30
python -m experiments.augmentation_exp --dataset fashion_mnist
python -m experiments.augmentation_ablation --seeds 0 1 2
python -m experiments.seed_ensemble_tta --dataset fashion_mnist
python -m experiments.model_ensemble --dataset fashion_mnist

# Redraw the figures from the saved tables, without retraining
python -m experiments.make_figures --dataset mnist

# Tests (including cross-checks against scikit-learn)
pytest
```

Scripts that train CNNs are slow on a CPU (about 2–10 minutes per CNN; `seed_ensemble_tta`
and `cnn_improved` take about 30–60 minutes). These two scripts cache each training run in
`results/cache/` (not committed), so a rerun skips the parts that are already done.

`notebooks/main_results.ipynb` reproduces the main results in a few minutes: it retrains the
fast models to check them against the saved tables, and recomputes the accuracy of every
model (CNNs included) from the predicted probabilities saved in `results/probs/`.

### Demo

Hand-drawn digit recognition demo (Gradio) using the final model: an ensemble of 3 improved
CNNs (`v2_full`). It needs the weights `results/cache/mnist/cnn_improved/v2_full_seed{0,1,2}.pt`
(not committed; create them with `python -m experiments.cnn_improved --configs v2_full --seeds 0 1 2`).

```bash
pip install "gradio>=5"
python -m demo.app            # open http://127.0.0.1:7860
python -m demo.app --check    # check the weights: test accuracy should be 99.63%
```

Two tabs: draw a digit (the drawing is converted to the 28×28 MNIST format before prediction),
or sample a random MNIST test image, optionally only from the images the model gets wrong.

## Results

### MNIST: required models and CNN

50,000 train / 10,000 validation / 10,000 test (split fixed with seed 42), using the
configurations chosen on validation in `experiments/run_all.py`. Models with randomness are
run with several seeds and reported as mean ± standard deviation; Naïve Bayes is
deterministic and runs once. Source: `results/tables/mnist_summary_agg.csv`.

| Model | Seeds | Train acc. | Val. acc. | Test acc. | Macro F1 (test) | Training time (s) | Prediction time, 10k images (s) |
|---|---|---|---|---|---|---|---|
| Gaussian Naïve Bayes | 1 | 80.26% | 79.91% | 81.45% | 0.8128 | 0.9 | 0.07 |
| Bernoulli Naïve Bayes | 1 | 83.45% | 83.52% | 84.51% | 0.8432 | 0.4 | 0.06 |
| Logistic Regression | 3 | 92.51% ± 0.05% | 91.74% ± 0.07% | 92.22% ± 0.12% | 0.9211 | 3.8 | 0.01 |
| MLP | 3 | 100.00% ± 0.00% | 98.01% ± 0.13% | 98.25% ± 0.10% | 0.9824 | 23.4 | 0.04 |
| Random Forest | 3 | 99.88% ± 0.01% | 96.65% ± 0.10% | 96.74% ± 0.01% | 0.9672 | 18.2 | 0.61 |
| CNN | 3 | 99.68% ± 0.22% | 98.83% ± 0.14% | 99.02% ± 0.18% | 0.9901 | 141.6 | 1.16 |

Times depend on the machine (CPU, number of cores, load) and are only comparable between
models run on the same machine. Random Forest trains its trees in parallel on all CPU cores.

### MNIST: extensions

| Configuration | Seeds | Val. acc. | Test acc. | Source |
|---|---|---|---|---|
| CNN + data augmentation | 5 | 99.03% | 99.11% ± 0.17% | `seed_ensemble_members.csv` |
| CNN + data augmentation + TTA | 5 | 99.04% | 99.18% ± 0.17% | `seed_ensemble_tta_summary.csv` |
| Seed ensemble of 5 CNNs | 5 | 99.14% | 99.24% | `seed_ensemble.csv` |
| Seed ensemble of 5 CNNs + data augmentation | 5 | 99.39% | 99.38% | `seed_ensemble.csv` |
| Multi-model ensemble, weights chosen on val (MLP, RF, CNN) | 1 | 98.99% | 99.26% | `model_ensemble_summary.csv` |
| Improved CNN (BatchNorm, Dropout, cosine learning rate schedule) | 3 | 99.45% ± 0.03% | 99.55% ± 0.06% | `cnn_improved_agg.csv` |
| **Ensemble of 3 improved CNNs (final model, chosen on val)** | 3 | **99.49%** | **99.63%** | `cnn_improved_agg.csv` |

Robustness to transformed test images (mean of 3 seeds; source `augmentation_robustness.csv`):

| Test images | MLP | MLP + aug | CNN | CNN + aug |
|---|---|---|---|---|
| Original | 98.25% | 98.41% | 99.02% | 99.05% |
| Shifted 2 px | 83.3% | 98.0% | 94.9% | 98.8% |
| Shifted 4 px | 34.2% | 88.4% | 61.2% | 94.1% |
| Rotated 20° | 90.0% | 94.3% | 94.3% | 96.5% |

### Fashion-MNIST

Same split and same configurations as MNIST. Source: `results/tables/fashion_vs_mnist.csv`.

| Model | Fashion-MNIST test acc. | Drop from MNIST |
|---|---|---|
| Gaussian Naïve Bayes | 67.35% | −14.1 points |
| Bernoulli Naïve Bayes | 72.13% | −12.4 points |
| Logistic Regression | 84.08% ± 0.20% | −8.1 points |
| MLP | 87.69% ± 0.50% | −10.6 points |
| Random Forest | 87.20% ± 0.09% | −9.6 points |
| CNN | 90.24% ± 0.20% | −8.8 points |

A seed ensemble of 5 CNNs reaches 91.43%. Augmentation designed for digits (rotation, shift,
noise) lowers the CNN to 86.37%, and TTA on a CNN trained without augmentation lowers it to
80.53% (`fashion_mnist_augmentation_agg.csv`, `fashion_mnist_seed_ensemble_tta_summary.csv`).

### Main findings

- Accuracy follows how much of the image structure each model uses: Naïve Bayes < Logistic
  Regression < Random Forest < MLP < CNN, on both MNIST and Fashion-MNIST.
- The CNN beats the MLP with about the same number of parameters (225k vs 235k). When the
  pixels are shuffled, the MLP barely changes while the CNN drops 2.7 points: the CNN's
  advantage comes from the spatial structure of the image.
- Data augmentation barely changes accuracy on the original test set, but reduces
  overfitting and greatly improves robustness to shifted images. On Fashion-MNIST, the
  digit-oriented transformations hurt.
- Seed ensembles add about 0.3 points for the CNN. TTA only avoids hurting when the model was
  trained with augmentation, and gives no further gain on validation.
- In the improved-CNN ablation, the cosine learning rate schedule is the only change with a
  clear effect: results become more stable (the spread across seeds shrinks about 3 times)
  and a single network beats the ensemble of 5 CNNs.

The full analysis is in `notebooks/main_results.ipynb`.

## Project structure

```
mnist-classifiers/
├── src/
│   ├── data.py                 # loading, normalisation, flattening, splits
│   ├── metrics.py              # accuracy, precision, recall, F1, confusion matrix
│   ├── augmentation.py         # rotation, shift, noise, horizontal flip (NumPy)
│   ├── tta.py                  # test-time augmentation
│   ├── ensemble.py             # soft / hard voting, weight search
│   ├── utils.py                # seeding, timing, saving probabilities
│   ├── plotting.py             # shared figure style
│   └── models/
│       ├── base.py             # shared fit / predict / predict_proba interface
│       ├── naive_bayes.py      # Gaussian and Bernoulli Naïve Bayes
│       ├── logistic_regression.py  # softmax regression with mini-batch GD
│       ├── nn_utils.py         # softmax, cross-entropy shared by LR and MLP
│       ├── mlp.py              # MLP with backpropagation
│       ├── random_forest.py    # decision trees and the forest
│       └── cnn.py              # CNN (from the lectures) and improved CNN
├── experiments/                # experiment scripts
├── demo/                       # Gradio demo (app.py, preprocess.py)
├── notebooks/                  # notebook reproducing the main results
├── tests/                      # tests, including cross-checks against scikit-learn
├── results/
│   ├── tables/                 # result tables (CSV)
│   ├── figures/                # figures
│   └── probs/                  # predicted probabilities on val/test, used by ensembles and the notebook
└── docs/PLAN.md                # work split and schedule
```

## Report and slides

The report and slides are submitted separately and are not stored in this repository.

## References

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. Gradient-based learning applied to document
  recognition. *Proceedings of the IEEE*, 1998. (MNIST)
- H. Xiao, K. Rasul, R. Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking
  machine learning algorithms. arXiv:1708.07747, 2017.
- L. Breiman. Random forests. *Machine Learning*, 2001.
- D. P. Kingma, J. Ba. Adam: a method for stochastic optimization. ICLR, 2015.
- S. Ioffe, C. Szegedy. Batch normalization: accelerating deep network training by reducing
  internal covariate shift. ICML, 2015.
- N. Srivastava et al. Dropout: a simple way to prevent neural networks from overfitting.
  *JMLR*, 2014.
- I. Loshchilov, F. Hutter. SGDR: stochastic gradient descent with warm restarts. ICLR, 2017.
  (cosine learning rate schedule)
- F. Pedregosa et al. Scikit-learn: machine learning in Python. *JMLR*, 2011.
- A. Paszke et al. PyTorch: an imperative style, high-performance deep learning library.
  NeurIPS, 2019.
- Lecture notes for Pattern Recognition (MTH10322), Dr. Truong Phuoc Hung.
