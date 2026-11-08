# Nhận dạng chữ số viết tay bằng các phương pháp phân lớp

[English](README.md) | **Tiếng Việt**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/core-NumPy-013243)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

Đồ án giữa kỳ môn Nhận dạng mẫu. Nhóm tự cài đặt bốn bộ phân lớp (Naïve Bayes, Logistic
Regression, MLP, Random Forest), huấn luyện trên MNIST và so sánh về độ chính xác, thời
gian huấn luyện, thời gian dự đoán và khả năng giải thích. Phần mở rộng: CNN đơn giản,
tăng cường dữ liệu và Fashion-MNIST. scikit-learn chỉ được dùng để đối chiếu kết quả.

![So sánh các mô hình](results/figures/mnist_model_comparison.png)

## Thông tin nhóm

| Họ và tên | MSSV | Phụ trách |
|---|---|---|
| Trần Ngọc Các Uyên | 24280030 | Tiền xử lý dữ liệu, độ đo, Logistic Regression, MLP, CNN |
| Lê Minh Phúc Tiên | 24280107 | Naïve Bayes, Random Forest, tăng cường dữ liệu, Fashion-MNIST |

- **Môn học:** Nhận dạng mẫu (MTH10322)
- **Giảng viên:** TS. Trương Phước Hưng
- **Trường:** Đại học Khoa học tự nhiên, ĐHQG-HCM

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
git clone https://github.com/trnngcccuyn/mnist-classifiers
cd mnist-classifiers
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dữ liệu được tải tự động vào `data/` ở lần chạy đầu tiên.

## Cách chạy

```bash
# Huấn luyện và đánh giá 4 mô hình bắt buộc; bảng lưu vào results/tables/
python -m experiments.run_all --dataset mnist

# Thí nghiệm siêu tham số; biểu đồ lưu vào results/figures/
python -m experiments.hyperparams.logistic_regression
python -m experiments.hyperparams.mlp
python -m experiments.hyperparams.cnn
python -m experiments.hyperparams.naive_bayes
python -m experiments.hyperparams.random_forest

# Phần mở rộng
python -m experiments.cnn_vs_mlp
python -m experiments.augmentation_exp
python -m experiments.seed_ensemble_tta
python -m experiments.cnn_improved
python -m experiments.model_ensemble       # cần chạy run_all trước
python -m experiments.run_all --dataset fashion_mnist
python -m experiments.fashion_vs_mnist
python -m experiments.hyperparams.cnn --dataset fashion_mnist --epochs 30
python -m experiments.augmentation_exp --dataset fashion_mnist
python -m experiments.augmentation_ablation --seeds 0 1 2

# Vẽ lại các hình từ bảng đã lưu, không phải train lại
python -m experiments.make_figures --dataset mnist

# Đối chiếu với scikit-learn
pytest
```

Kết quả chính có thể tái tạo bằng `notebooks/main_results.ipynb`.

## Kết quả

MNIST, 50.000 train / 10.000 validation / 10.000 test (chia cố định với seed 42), dùng cấu
hình đã chọn theo validation trong `experiments/run_all.py`. Mô hình có yếu tố ngẫu nhiên được
chạy với nhiều seed, ghi trung bình ± độ lệch chuẩn; Naïve Bayes tất định nên chạy một lần.
Nguồn: `results/tables/mnist_summary_agg.csv`.

| Mô hình | Số seed | Acc. train | Acc. val. | Acc. test | F1 macro (test) | TG huấn luyện (s) | TG dự đoán 10k ảnh (s) |
|---|---|---|---|---|---|---|---|
| Gaussian Naïve Bayes | 1 | 80.26% | 79.91% | 81.45% | 0.8128 | 0.9 | 0.07 |
| Bernoulli Naïve Bayes | 1 | 83.45% | 83.52% | 84.51% | 0.8432 | 0.4 | 0.06 |
| Logistic Regression | 3 | 92.51% ± 0.05% | 91.74% ± 0.07% | 92.22% ± 0.12% | 0.9211 | 3.8 | 0.01 |
| MLP | 3 | 100.00% ± 0.00% | 98.01% ± 0.13% | 98.25% ± 0.10% | 0.9824 | 23.4 | 0.04 |
| Random Forest | 3 | 99.88% ± 0.01% | 96.65% ± 0.10% | 96.74% ± 0.01% | 0.9672 | 18.2 | 0.61 |
| CNN | 3 | 99.68% ± 0.22% | 98.83% ± 0.14% | 99.02% ± 0.18% | 0.9901 | 141.6 | 1.16 |

Thời gian phụ thuộc máy (CPU, số nhân, máy đang bận) nên chỉ so được giữa các mô hình chạy
trên cùng một máy. Random Forest train các cây song song trên mọi nhân CPU.

## Cấu trúc thư mục

```
mnist-classifiers/
├── src/
│   ├── data.py                 # tải, chuẩn hoá, làm phẳng, chia tập
│   ├── metrics.py              # accuracy, precision, recall, F1, confusion matrix
│   ├── augmentation.py         # xoay, dịch, nhiễu
│   ├── utils.py                # seed, đo thời gian
│   ├── plotting.py             # kiểu vẽ hình dùng chung
│   └── models/
│       ├── base.py             # interface chung fit / predict / predict_proba
│       ├── naive_bayes.py      # Gaussian và Bernoulli Naïve Bayes
│       ├── logistic_regression.py  # softmax regression + mini-batch GD
│       ├── mlp.py              # MLP + backpropagation
│       ├── random_forest.py    # ensemble
│       └── cnn.py              # CNN đơn giản
├── experiments/                # script chạy thí nghiệm
├── notebooks/                  # notebook tái tạo kết quả chính
├── tests/                      # đối chiếu với scikit-learn
├── results/                    # tables/ và figures/
├── report/                     # báo cáo PDF
├── slides/                     # slide trình bày
└── docs/PLAN.md                # phân công và kế hoạch
```

## Báo cáo và slide

- Báo cáo: `report/report.pdf` (TODO)
- Slide: `slides/` (TODO)

## Tài liệu tham khảo

TODO
