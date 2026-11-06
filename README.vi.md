# Nhận dạng chữ số viết tay bằng các phương pháp phân lớp

[English](README.md) | **Tiếng Việt**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/core-NumPy-013243)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

Đồ án giữa kỳ môn Nhận dạng mẫu. Nhóm tự cài đặt bốn bộ phân lớp (Naïve Bayes, Logistic
Regression, MLP, Random Forest), huấn luyện trên MNIST và so sánh về độ chính xác, thời
gian huấn luyện, thời gian dự đoán và khả năng giải thích. Phần mở rộng: CNN đơn giản,
tăng cường dữ liệu và Fashion-MNIST. scikit-learn chỉ được dùng để đối chiếu kết quả.

<!-- TODO: bỏ chú thích khi đã có hình
![So sánh các mô hình](results/figures/model_comparison.png)
-->

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
python -m experiments.run_all --dataset fashion_mnist

# Đối chiếu với scikit-learn
pytest
```

Kết quả chính có thể tái tạo bằng `notebooks/main_results.ipynb`.

## Kết quả

MNIST, 50.000 train / 10.000 validation / 10.000 test, seed 42. TODO: điền sau khi chạy.

| Mô hình | Acc. train | Acc. val. | Acc. test | F1 macro (test) | TG huấn luyện | TG dự đoán |
|---|---|---|---|---|---|---|
| Naïve Bayes | | | | | | |
| Logistic Regression | | | | | | |
| MLP | | | | | | |
| Random Forest | | | | | | |
| CNN | | | | | | |

## Cấu trúc thư mục

```
mnist-classifiers/
├── src/
│   ├── data.py                 # tải, chuẩn hoá, làm phẳng, chia tập
│   ├── metrics.py              # accuracy, precision, recall, F1, confusion matrix
│   ├── augmentation.py         # xoay, dịch, nhiễu
│   ├── utils.py                # seed, đo thời gian
│   └── models/
│       ├── base.py             # interface chung fit / predict / predict_proba
│       ├── naive_bayes.py      # Gaussian Naïve Bayes
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
