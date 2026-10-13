# Phân công và kế hoạch

Kế hoạch rút gọn (cập nhật 13/10): hoàn thành trước 29/10, sớm hơn kế hoạch ban đầu 1 tuần.

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
- Commit message viết bằng tiếng Anh; chạy `ruff format .` trước khi commit. Chi tiết: [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- README có hai bản (`README.md` tiếng Anh, `README.vi.md` tiếng Việt): sửa bản nào thì sửa luôn bản kia.

## Giai đoạn 1 (09/10 – 18/10): Hạ tầng + 4 mô hình bắt buộc

- [x] Chung: tạo repo, chốt quy ước ở trên
- [x] Uyên: `data.py` (tải, chuẩn hoá, làm phẳng, chia tập)
- [x] Uyên: `metrics.py` (accuracy, precision, recall, F1 macro, confusion matrix)
- [x] Uyên: tách thí nghiệm siêu tham số thành `experiments/hyperparams/` theo từng mô hình
- [ ] Uyên: Logistic Regression + thí nghiệm learning rate — 14/10
- [ ] Uyên: MLP + backprop, thí nghiệm số lớp / số nơ-ron — 17/10
- [ ] Uyên: `run_all.py` chạy cả 4 mô hình, đo thời gian huấn luyện + dự đoán — 18/10
- [ ] Tiên: Gaussian Naïve Bayes + thí nghiệm var smoothing — 14/10
- [ ] Tiên: cây quyết định — 15/10
- [ ] Tiên: Random Forest + thí nghiệm số cây / độ sâu — 18/10
- [ ] Tiên: `tests/` đối chiếu các mô hình với scikit-learn — 18/10

**Mốc 18/10:** xong toàn bộ code bắt buộc. Đây là mốc quan trọng nhất.

## Giai đoạn 2 (19/10 – 23/10): Phần nâng cao + nháp báo cáo

- [ ] Uyên: CNN đơn giản, so sánh với MLP (accuracy, số tham số, thời gian) — 20/10
- [ ] Tiên: augmentation (xoay, dịch, nhiễu), thử trên Logistic Regression — 20/10
- [ ] Tiên: cắm augmentation vào MLP/CNN, đánh giá tác động — 22/10
- [ ] Tiên: chạy lại toàn bộ mô hình trên Fashion-MNIST — 23/10
- [ ] Chung: mỗi người viết nháp phần phương pháp (công thức + pseudocode) của mình — 22/10
- [ ] Chung: bảng so sánh tổng + phân tích ưu/nhược điểm — 23/10

**Mốc 23/10:** có đủ mọi con số và biểu đồ cần đưa vào báo cáo.

## Giai đoạn 3 (24/10 – 29/10): Báo cáo, notebook, slide, nộp

- [ ] Uyên: notebook tái tạo kết quả chính (chạy lại sạch từ đầu một lần) — 24/10
- [ ] 25/10: đóng băng code, chỉ sửa lỗi
- [ ] Tiên: ghép báo cáo PDF 10–15 trang, tài liệu tham khảo, bảng phân công — 27/10
- [ ] Uyên: slide 10–15 phút — 27/10
- [ ] Chung: hoàn thiện README, clone repo sang máy khác chạy thử theo README — 28/10
- [ ] Chung: tập trình bày ít nhất 1 lần có bấm giờ — 28/10
- [ ] 29/10: dự phòng, nộp, tạo release `v1.0`

## Rủi ro cần để ý

- **Random Forest tự cài chạy rất chậm** trên 50.000 × 784. Cách giảm: mỗi nút chỉ xét √784 = 28 đặc trưng, giới hạn số ngưỡng thử (vài mức thay vì mọi giá trị), giới hạn độ sâu, vector hoá bằng NumPy, phát triển trên tập con 5.000 mẫu trước. Nếu đến 16/10 vẫn quá chậm trên toàn bộ dữ liệu, huấn luyện trên tập con (ví dụ 10.000–20.000 mẫu) và ghi rõ trong báo cáo, để không trễ mốc 18/10.
- **Augmentation phụ thuộc MLP/CNN của Uyên.** Tiên viết và kiểm tra hàm augmentation trên Logistic Regression/NB trước, từ 21/10 chỉ việc cắm vào MLP/CNN.
- **Thời gian chạy thí nghiệm.** Lưu kết quả ra CSV trong `results/tables/`, không chạy lại từ đầu mỗi lần vẽ biểu đồ.
