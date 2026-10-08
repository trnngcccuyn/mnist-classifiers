# Phân công và kế hoạch 4 tuần

Mốc thời gian giả định bắt đầu từ 09/10/2026. Nếu hạn nộp khác, dời cả bảng cho khớp.

## Phân công

| Hạng mục | Uyên | Tiên |
|---|---|---|
| Hạ tầng chung | `data.py`, `metrics.py`, `run_all.py` | `tests/` đối chiếu scikit-learn |
| Mô hình bắt buộc | Logistic Regression, MLP | Naïve Bayes, Random Forest |
| Thí nghiệm siêu tham số | learning rate, số lớp/nơ-ron (LR, MLP) | số cây, độ sâu (RF), var smoothing (NB) |
| Nâng cao | CNN và so sánh với MLP | Data augmentation, Fashion-MNIST |
| Sản phẩm nộp | Notebook, slide (chủ trì) | Báo cáo PDF (chủ trì ghép + định dạng) |
| Viết báo cáo | Dữ liệu, LR, MLP, CNN, so sánh tổng | Giới thiệu, NB, RF, augmentation, Fashion-MNIST |

Mỗi người review code (pull request) của người kia trước khi merge vào `main`.

## Quy ước chung (chốt trong 2 ngày đầu)

- Mọi mô hình kế thừa `BaseClassifier`: `fit(X_train, y_train, X_val, y_val)`, `predict(X)`, `predict_proba(X)`.
- Chia tập cố định: 50.000 train / 10.000 validation / 10.000 test, `seed = 42`.
- Tập test chỉ dùng để báo cáo số cuối cùng; chọn siêu tham số bằng validation.
- Đầu vào mô hình phẳng: `X` shape `(N, 784)`, float32 trong [0, 1]; CNN reshape lại `(N, 1, 28, 28)`.
- Mỗi người làm trên nhánh riêng (`uyen/...`, `tien/...`), merge qua pull request.

## Tuần 1 (09/10 – 15/10): Hạ tầng + mô hình đơn giản

- [ ] Chung: tạo repo, chốt quy ước ở trên
- [ ] Uyên: `data.py` (tải, chuẩn hoá, làm phẳng, chia tập) — xong trước 12/10 vì Tiên cần dùng
- [ ] Uyên: `metrics.py` (accuracy, precision, recall, F1 macro, confusion matrix)
- [ ] Uyên: bắt đầu Logistic Regression (softmax + mini-batch GD)
- [ ] Tiên: Gaussian Naïve Bayes (log-likelihood, var smoothing)
- [ ] Tiên: cây quyết định (Gini, chọn ngưỡng) làm nền cho Random Forest

**Mốc:** NB chạy được đầu-cuối trên MNIST và ra đủ các độ đo.

## Tuần 2 (16/10 – 22/10): Hoàn tất 4 mô hình bắt buộc

- [ ] Uyên: hoàn tất Logistic Regression, thí nghiệm learning rate
- [ ] Uyên: MLP + backprop, thí nghiệm số lớp / số nơ-ron
- [ ] Tiên: hoàn tất Random Forest (bootstrap, chọn ngẫu nhiên đặc trưng, bỏ phiếu)
- [ ] Tiên: thí nghiệm số cây / độ sâu
- [ ] Tiên: `tests/` đối chiếu 4 mô hình với scikit-learn
- [ ] Uyên: `run_all.py` chạy cả 4 mô hình, đo thời gian huấn luyện + dự đoán

**Mốc:** xong toàn bộ yêu cầu bắt buộc về code. Đây là mốc quan trọng nhất.

## Tuần 3 (23/10 – 29/10): Phần nâng cao + nháp báo cáo

- [ ] Uyên: CNN đơn giản, so sánh với MLP (accuracy, số tham số, thời gian)
- [ ] Tiên: augmentation (xoay, dịch, nhiễu), đánh giá tác động trên MLP/CNN
- [ ] Tiên: chạy lại toàn bộ mô hình trên Fashion-MNIST
- [ ] Chung: mỗi người viết nháp phần phương pháp (công thức + pseudocode) của mình
- [ ] Chung: bảng so sánh tổng + phân tích ưu/nhược điểm

**Mốc:** có đủ mọi con số và biểu đồ cần đưa vào báo cáo.

## Tuần 4 (30/10 – 05/11): Báo cáo, notebook, slide

- [ ] 01/11: đóng băng code, chỉ sửa lỗi
- [ ] Uyên: notebook tái tạo kết quả chính (chạy lại sạch từ đầu một lần)
- [ ] Tiên: ghép báo cáo PDF 10–15 trang, tài liệu tham khảo, bảng phân công
- [ ] Uyên: slide 10–15 phút
- [ ] Chung: hoàn thiện README, clone repo sang máy khác chạy thử theo README
- [ ] Chung: tập trình bày ít nhất 1 lần có bấm giờ
- [ ] Để trống 1–2 ngày cuối làm dự phòng

## Rủi ro cần để ý

- **Random Forest tự cài chạy rất chậm** trên 50.000 × 784. Cách giảm: mỗi nút chỉ xét √784 = 28 đặc trưng, giới hạn số ngưỡng thử (vài mức thay vì mọi giá trị), giới hạn độ sâu, vector hoá bằng NumPy, phát triển trên tập con 5.000 mẫu trước.
- **Augmentation phụ thuộc MLP/CNN của Uyên.** Tiên viết và kiểm tra hàm augmentation trên Logistic Regression/NB trước, tuần 3 chỉ việc cắm vào MLP/CNN.
- **Thời gian chạy thí nghiệm.** Lưu kết quả ra CSV trong `results/tables/`, không chạy lại từ đầu mỗi lần vẽ biểu đồ.
