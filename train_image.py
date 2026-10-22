"""Train REAL vs AI-GENERATED image classifier - small CNN trained FROM SCRATCH.

Why not EfficientNet-B0 transfer learning (even though that often works better)?
Transfer learning needs to download pretrained ImageNet weights from the internet
at training time. In a locked-down / offline environment that download will fail,
so this script uses no pretrained weights at all - everything is learned from your
own data. This also matches CIFAKE's native resolution well: CIFAKE images are
32x32 natively, so a small CNN at 64x64 avoids the heavy upscale-to-224 blur that
a full ImageNet classifier would otherwise key off instead of real AI artifacts.

Folder structure:
data/image/train/REAL/*.jpg
data/image/train/FAKE/*.jpg
data/image/test/REAL/*.jpg
data/image/test/FAKE/*.jpg

Start with 3000-5000 images/class. CPU-only is fine at this image size.
"""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/image")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--image-size", type=int, default=64)
p.add_argument("--resume", action="store_true", help="continue training from model/image_detector.pt if present")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_tf = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.1, contrast=.1),
    transforms.ToTensor(),
])
test_tf = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

tr = datasets.ImageFolder(Path(args.data) / "train", transform=train_tf)
te = datasets.ImageFolder(Path(args.data) / "test", transform=test_tf)
print("classes:", tr.class_to_idx, "train:", len(tr), "test:", len(te))

train = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


class SmallCNN(nn.Module):
    """4 conv blocks, trained entirely from scratch - no pretrained weights needed."""

    def __init__(self, image_size):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        final = image_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * final * final, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN(args.image_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

best = 0.0
ckpt_path = Path("model/image_detector.pt")
if args.resume and ckpt_path.exists():
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    best = ck.get("best_test_accuracy", 0.0)
    print(f"resumed from checkpoint, previous best_test_accuracy={best:.4f}")
for epoch in range(args.epochs):
    model.train()
    total = correct = 0
    for x, y in train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    scheduler.step()

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test:
            ps.extend(model(x.to(device)).argmax(1).cpu().tolist())
            ys.extend(y.tolist())
    acc = accuracy_score(ys, ps)
    print(f"epoch {epoch + 1}/{args.epochs} train_acc={correct / total:.4f} test_acc={acc:.4f}")
    if acc > best:
        best = acc
        Path("model").mkdir(exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "class_to_idx": tr.class_to_idx,
             "architecture": "small_cnn_scratch", "image_size": args.image_size,
             "best_test_accuracy": float(best)},
            "model/image_detector.pt",
        )

pr, re, f1, _ = precision_recall_fscore_support(ys, ps, average="binary", zero_division=0)
metrics = {
    "architecture": "Small CNN trained from scratch (no pretrained weights)",
    "device": str(device), "image_size": args.image_size,
    "train": len(tr), "test": len(te), "best_test_accuracy": float(best),
    "precision": float(pr), "recall": float(re), "f1": float(f1),
    "confusion_matrix": confusion_matrix(ys, ps).tolist(), "class_to_idx": tr.class_to_idx,
}
Path("model/image_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
