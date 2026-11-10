"""Demo nhận dạng chữ số viết tay bằng CNN cải tiến (ensemble 3 seed v2_full).  Phụ trách: Uyên

Chạy (từ thư mục gốc repo):
    py -m pip install "gradio>=5"
    py -m demo.app            # mở http://127.0.0.1:7860
    py -m demo.app --check    # kiểm tra trọng số: accuracy trên test phải là 99,63%

Cần trọng số results/cache/mnist/cnn_improved/v2_full_seed{0,1,2}.pt (gitignore). Nếu chưa có:
    py -m experiments.cnn_improved --configs v2_full --seeds 0 1 2

Hai tab:
  1. Vẽ chữ số: ảnh vẽ -> tiền xử lý kiểu MNIST (demo/preprocess.py) -> 3 mạng -> trung bình
     softmax (soft voting, như trong báo cáo).
  2. Ảnh test MNIST: lấy ngẫu nhiên một ảnh test (có thể chỉ lấy ảnh mô hình đoán sai).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.preprocess import to_ink, to_mnist  # noqa: E402
from src.models.cnn import _NetV2  # noqa: E402

WEIGHTS = [ROOT / f"results/cache/mnist/cnn_improved/v2_full_seed{s}.pt" for s in (0, 1, 2)]
TEST_IMAGES = ROOT / "data/MNIST/raw/t10k-images-idx3-ubyte"
TEST_LABELS = ROOT / "data/MNIST/raw/t10k-labels-idx1-ubyte"
SAVED_PROBS = ROOT / "results/probs/mnist/cnn_v2_full_ens3_test.npy"


# ---------------------------------------------------------------- mô hình
def load_nets() -> list[torch.nn.Module]:
    missing = [str(p.relative_to(ROOT)) for p in WEIGHTS if not p.exists()]
    if missing:
        sys.exit(
            "Thiếu trọng số: " + ", ".join(missing) + "\nChạy: py -m experiments.cnn_improved "
            "--configs v2_full --seeds 0 1 2"
        )
    nets = []
    for path in WEIGHTS:
        # Cấu hình v2_full = tham số mặc định của _NetV2 / ImprovedCNN
        net = _NetV2(n_classes=10, channels=(16, 32), hidden=128, batchnorm=True, dropout=0.3)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        nets.append(net.eval())  # eval: BatchNorm dùng thống kê đã học, tắt dropout
    return nets


@torch.no_grad()
def predict(nets, X: np.ndarray) -> np.ndarray:
    """X (N, 28, 28) float32 trong [0, 1] -> xác suất từng mạng, shape (n_nets, N, 10)."""
    x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).reshape(-1, 1, 28, 28)
    return np.stack([F.softmax(net(x), dim=1).numpy() for net in nets])


# ---------------------------------------------------------------- dữ liệu test
def load_test():
    """Đọc thẳng file IDX của MNIST (không cần torchvision tải lại)."""
    if not TEST_IMAGES.exists():
        return None, None
    X = np.fromfile(TEST_IMAGES, np.uint8, offset=16).reshape(-1, 28, 28)
    y = np.fromfile(TEST_LABELS, np.uint8, offset=8).astype(np.int64)
    return X.astype(np.float32) / 255.0, y


def check(nets):
    X, y = load_test()
    if X is None:
        sys.exit(f"Không thấy {TEST_IMAGES.relative_to(ROOT)}")
    P = np.concatenate([predict(nets, X[i : i + 1000]) for i in range(0, len(X), 1000)], axis=1)
    for s, p in enumerate(P):
        print(f"seed {s}: test acc {np.mean(p.argmax(1) == y):.2%}")
    print(f"ensemble 3 seed: test acc {np.mean(P.mean(0).argmax(1) == y):.2%} (báo cáo: 99,63%)")


# ---------------------------------------------------------------- giao diện
def build_ui(nets):
    import gradio as gr

    X_test, y_test = load_test()
    wrong_idx = np.array([], dtype=int)
    if X_test is not None and SAVED_PROBS.exists():
        wrong_idx = np.where(np.load(SAVED_PROBS).argmax(1) != y_test)[0]
    rng = np.random.default_rng()

    def show_28(img28):
        """Phóng ảnh 28x28 lên 224x224 (nearest) để thấy rõ từng pixel mô hình nhìn thấy."""
        return np.kron((img28 * 255).astype(np.uint8), np.ones((8, 8), np.uint8))

    def explain(probs_per_net):
        P = probs_per_net[:, 0]  # (3, 10)
        ens = P.mean(0)
        top = int(ens.argmax())
        lines = ["| Mạng | Dự đoán | Độ tin cậy |", "|---|---|---|"]
        for s, p in enumerate(P):
            lines.append(f"| seed {s} | {p.argmax()} | {p.max():.1%} |")
        lines.append(f"| **Ensemble** | **{top}** | **{ens.max():.1%}** |")
        notes = []
        if len({int(p.argmax()) for p in P}) > 1:
            notes.append("⚠️ Ba mạng không thống nhất: ảnh này khó/mơ hồ.")
        if ens.max() < 0.6:
            notes.append("⚠️ Độ tin cậy thấp: thử vẽ lại rõ hơn, nét liền.")
        labels = {str(k): float(ens[k]) for k in range(10)}
        return labels, "\n".join(lines + ([""] + notes if notes else []))

    def on_draw(value):
        img = value.get("composite") if isinstance(value, dict) else value
        if img is None:
            return None, None, "Hãy vẽ một chữ số (0–9) vào khung bên trái."
        img28 = to_mnist(to_ink(img))
        if img28 is None:
            return None, None, "Hãy vẽ một chữ số (0–9) vào khung bên trái."
        labels, md = explain(predict(nets, img28[None]))
        return show_28(img28), labels, md

    def on_random(only_wrong):
        if X_test is None:
            return None, None, "Không tìm thấy dữ liệu MNIST trong `data/MNIST/raw/`."
        pool = wrong_idx if only_wrong and len(wrong_idx) else np.arange(len(X_test))
        i = int(rng.choice(pool))
        labels, md = explain(predict(nets, X_test[i : i + 1]))
        head = f"**Ảnh test #{i}, nhãn thật: {y_test[i]}**\n\n"
        return show_28(X_test[i]), labels, head + md

    brush = gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed")
    with gr.Blocks(title="Demo MNIST - CNN cải tiến") as demo:
        gr.Markdown(
            "# Nhận dạng chữ số viết tay - CNN cải tiến\n"
            "Ensemble 3 mạng `v2_full` (seed 0, 1, 2), soft voting (test MNIST **99,63%**)\n"
            "Nhóm 6: Trần Ngọc Các Uyên - Lê Minh Phúc Tiên"
        )
        with gr.Tab("✏️ Vẽ chữ số"):
            with gr.Row():
                with gr.Column():
                    pad = gr.Sketchpad(
                        label="Vẽ một chữ số (to, ở giữa khung)",
                        type="numpy",
                        brush=brush,
                        canvas_size=(280, 280),
                    )
                    btn = gr.Button("Dự đoán", variant="primary")
                with gr.Column():
                    seen = gr.Image(label="Mô hình nhìn thấy (28×28 sau tiền xử lý)", height=224)
                    label = gr.Label(num_top_classes=3, label="Xác suất (ensemble)")
                    table = gr.Markdown()
            pad.change(on_draw, pad, [seen, label, table])
            btn.click(on_draw, pad, [seen, label, table])

        with gr.Tab("🎲 Ảnh test MNIST"):
            with gr.Row():
                with gr.Column():
                    only_wrong = gr.Checkbox(
                        label=f"Chỉ lấy ảnh mô hình đoán sai ({len(wrong_idx)}/10.000 ảnh)"
                    )
                    btn2 = gr.Button("Lấy ảnh ngẫu nhiên", variant="primary")
                    seen2 = gr.Image(label="Ảnh test (28×28)", height=224)
                with gr.Column():
                    label2 = gr.Label(num_top_classes=3, label="Xác suất (ensemble)")
                    table2 = gr.Markdown()
            btn2.click(on_random, only_wrong, [seen2, label2, table2])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="tính accuracy trên test rồi thoát")
    parser.add_argument("--share", action="store_true", help="tạo link công khai tạm thời")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    nets = load_nets()
    if args.check:
        check(nets)
    else:
        build_ui(nets).launch(share=args.share)


if __name__ == "__main__":
    main()
