"""CNN đơn giản (phần nâng cao), so sánh với MLP.  Phụ trách: Uyên

Nhận X phẳng (N, 784) như các mô hình khác và tự reshape về (N, 1, 28, 28) bên trong.

Kiến trúc (slide 12 bài B6b), kích thước ảnh ghi bên phải:
    Conv(1 -> 32, 3x3)  - ReLU - MaxPool 2      28 -> 26 -> 13
    Conv(32 -> 64, 3x3) - ReLU - MaxPool 2      13 -> 11 -> 5
    Flatten (64 * 5 * 5 = 1600) - Linear(1600 -> 128) - ReLU - Linear(128 -> 10)
Tổng cộng 225.034 tham số, xấp xỉ MLP 784-256-128-10 (235.146), nên khi so sánh hai mô hình
khác nhau chủ yếu ở cách dùng tham số chứ không phải ở số lượng.

Khác MLP tự viết bằng NumPy: phần feedforward viết tường minh trong _Net.forward, còn đạo hàm
lấy bằng PyTorch autograd và cập nhật bằng torch.optim.Adam (đề cho phép dùng PyTorch làm khung
cho CNN; backprop tự viết đã làm và kiểm tra ở MLP).

Hàm mất mát: cross-entropy trung bình trên batch (F.cross_entropy nhận logits, tự lấy
log-softmax nên ổn định số). predict_proba trả softmax của logits.

Tái tạo được theo seed: torch.manual_seed(seed) trước khi khởi tạo trọng số và một
torch.Generator riêng để xáo dữ liệu mỗi epoch, nên đổi seed là đổi cả khởi tạo lẫn thứ tự
batch (cần cho seed ensemble). Trên GPU kết quả có thể lệch nhẹ giữa các lần chạy do cuDNN.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseClassifier


class _Net(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """x: (B, 1, 28, 28) -> logits (B, n_classes)."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (B, 32, 13, 13)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (B, 64, 5, 5)
        x = torch.flatten(x, 1)  # (B, 1600)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN(BaseClassifier):
    name = "cnn"

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 128,
        n_classes: int = 10,
        augment=None,
        device=None,
        seed: int = 42,
    ):
        """augment: None hoặc hàm (X_img (B, 28, 28) float32, rng) -> mảng cùng shape, áp lên
        từng batch train (không áp lên validation). rng là np.random.Generator theo seed.
        device: None thì dùng GPU nếu có, không thì CPU.
        """
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.n_classes, self.augment, self.seed = n_classes, augment, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _build_net(self) -> nn.Module:
        return _Net(self.n_classes)

    def _make_scheduler(self, opt, total_steps):
        """Lịch learning rate, gọi step() sau mỗi batch. SimpleCNN: lr cố định."""
        return None

    @staticmethod
    def _to_images(X) -> torch.Tensor:
        """(N, 784) -> tensor float32 (N, 1, 28, 28) trên CPU, không chép nếu đã đúng kiểu."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        return torch.from_numpy(X).reshape(-1, 1, 28, 28)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        aug_rng = np.random.default_rng(self.seed)
        self.net_ = self._build_net().to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        X = self._to_images(X_train)
        y = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
        n = len(y)
        steps_per_epoch = (n + self.batch_size - 1) // self.batch_size
        scheduler = self._make_scheduler(opt, steps_per_epoch * self.epochs)
        for _ in range(self.epochs):
            self.net_.train()
            idx = torch.randperm(n, generator=gen)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                xb = X[batch]
                if self.augment is not None:
                    xb_np = self.augment(xb.numpy()[:, 0], aug_rng)
                    xb = self._to_images(xb_np.reshape(len(batch), -1))
                xb, yb = xb.to(self.device), y[batch].to(self.device)

                loss = F.cross_entropy(self.net_(xb), yb)
                opt.zero_grad()
                loss.backward()  # autograd
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item() * len(batch)

            self.history["train_loss"].append(float(total_loss / n))
            if X_val is not None:
                P_val = self.predict_proba(X_val)
                y_val_np = np.asarray(y_val)
                p_true = P_val[np.arange(len(y_val_np)), y_val_np]
                self.history["val_loss"].append(float(-np.mean(np.log(p_true + 1e-12))))
                self.history["val_acc"].append(float(np.mean(P_val.argmax(axis=1) == y_val_np)))
        return self

    @torch.no_grad()
    def predict_proba(self, X, batch_size: int = 1000) -> np.ndarray:
        self.net_.eval()
        X = self._to_images(X)
        out = []
        for start in range(0, len(X), batch_size):
            logits = self.net_(X[start : start + batch_size].to(self.device))
            out.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32, copy=False)

    def n_params(self) -> int:
        """Số tham số, để so với MLP (n_params() của MLP)."""
        net = self.net_ if hasattr(self, "net_") else self._build_net()
        return int(sum(p.numel() for p in net.parameters()))


class _NetV2(nn.Module):
    """Hai khối, mỗi khối 2 lớp conv 3x3 (padding 1, giữ kích thước) rồi max pooling:
    28 -> 14 -> 7. Tuỳ chọn BatchNorm sau mỗi conv và Dropout trước lớp ra."""

    def __init__(self, n_classes=10, channels=(16, 32), hidden=128, batchnorm=True, dropout=0.0):
        super().__init__()
        c1, c2 = channels
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for c_in, c_out in [(1, c1), (c1, c1), (c1, c2), (c2, c2)]:
            # BatchNorm đã có tham số dịch (beta) nên conv không cần bias
            self.convs.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=not batchnorm))
            self.norms.append(nn.BatchNorm2d(c_out) if batchnorm else nn.Identity())
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        """x: (B, 1, 28, 28) -> logits (B, n_classes)."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = F.relu(norm(conv(x)))
            if i % 2 == 1:  # hết một khối
                x = F.max_pool2d(x, 2)  # 28 -> 14 -> 7
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ImprovedCNN(SimpleCNN):
    """CNN cải tiến (phần mở rộng, không thay SimpleCNN trong các thí nghiệm chính).

    So với SimpleCNN: mỗi khối 2 lớp conv thay vì 1 (trường nhìn lớn hơn với cùng số tham số),
    padding giữ kích thước ảnh, BatchNorm (huấn luyện ổn định, cho phép học nhanh hơn),
    Dropout trước lớp ra (giảm overfit), lịch lr cosine giảm dần về 0 (bước cuối nhỏ nên trọng
    số epoch cuối ít dao động). Mặc định 16/32 bộ lọc: tổng 218.586 tham số, xấp xỉ SimpleCNN
    (225.034), nhưng tốn ~1,8 lần phép tính mỗi ảnh. Vòng train, seed, augment: như SimpleCNN.
    """

    name = "cnn_v2"

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 128,
        channels=(16, 32),
        hidden: int = 128,
        batchnorm: bool = True,
        dropout: float = 0.3,
        schedule="cosine",
        n_classes: int = 10,
        augment=None,
        device=None,
        seed: int = 42,
    ):
        super().__init__(lr, epochs, batch_size, n_classes, augment, device, seed)
        self.channels, self.hidden = tuple(channels), hidden
        self.batchnorm, self.dropout, self.schedule = batchnorm, dropout, schedule
        if schedule not in (None, "cosine"):
            raise ValueError("schedule phải là None hoặc 'cosine'")

    def _build_net(self) -> nn.Module:
        return _NetV2(self.n_classes, self.channels, self.hidden, self.batchnorm, self.dropout)

    def _make_scheduler(self, opt, total_steps):
        if self.schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        return None
