"""Train a two-view (front + back) ResNet18 to regress the PSA grade.

Usage: python image_analysis/modeling/train.py [epochs] [weighted]
"""
from pathlib import Path
import sys
import time

from PIL import Image
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "interim" / "images"
INDEX = ROOT / "data" / "processed" / "index.csv"
MODELS = ROOT / "models"

TF = transforms.Compose(
    [
        transforms.Resize((512, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class CardDataset(Dataset):
    def __init__(self, df):
        self.ids = df["card_id"].tolist()
        self.grades = df["grade"].astype("float32").tolist()

    def __len__(self):
        return len(self.ids)

    def _load(self, card_id, side):
        return TF(Image.open(IMAGES / f"{card_id}_{side}.jpg").convert("RGB"))

    def __getitem__(self, i):
        cid = self.ids[i]
        return self._load(cid, "front"), self._load(cid, "back"), torch.tensor(self.grades[i])


class TwoViewNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )

    def forward(self, front, back):
        feats = torch.cat([self.backbone(front), self.backbone(back)], dim=1)
        return self.head(feats).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    for front, back, y in loader:
        preds.append(model(front.to(device), back.to(device)).cpu())
        ys.append(y)
    p = torch.cat(preds).clamp(1, 10)
    y = torch.cat(ys)
    r = p.round()
    per_grade = [(p[y == g] - g).abs().mean() for g in y.unique()]
    return {
        "mae": (p - y).abs().mean().item(),
        "balanced_mae": torch.stack(per_grade).mean().item(),
        "exact": (r == y).float().mean().item(),
        "within1": ((r - y).abs() <= 1).float().mean().item(),
    }


def grade_weights(train_df):
    """Inverse-sqrt-frequency weights per grade, normalised to average 1."""
    counts = train_df["grade"].value_counts()
    w = counts.pow(-0.5)
    w = w * counts.sum() / (w * counts).sum()
    table = torch.ones(11)
    for grade, value in w.items():
        table[int(grade)] = float(value)
    return table


def baseline_report(train_df, test_df):
    med = train_df["grade"].median()
    for name, c in (("always 10", 10), (f"always {med:g} (median)", med)):
        err = (test_df["grade"] - c).abs()
        balanced = err.groupby(test_df["grade"]).mean().mean()
        exact = (test_df["grade"] == round(c)).mean()
        print(f"baseline {name}: MAE {err.mean():.3f}  balanced MAE {balanced:.3f}  exact {exact:.3f}")


def main(epochs=10, weighted=False, batch_size=16):
    tag = "full_weighted" if weighted else "full"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| run:", tag, flush=True)
    df = pd.read_csv(INDEX)
    loaders = {
        s: DataLoader(
            CardDataset(df[df["split"] == s]),
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=4,
            persistent_workers=True,
        )
        for s in ("train", "val", "test")
    }
    model = TwoViewNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    weights = grade_weights(df[df["split"] == "train"]).to(device)
    MODELS.mkdir(exist_ok=True)
    ckpt = MODELS / f"resnet18_{tag}.pt"
    best = float("inf")
    n_batches = len(loaders["train"])

    for ep in range(1, epochs + 1):
        model.train()
        start = time.time()
        total = 0.0
        for i, (front, back, y) in enumerate(loaders["train"], 1):
            front, back, y = front.to(device), back.to(device), y.to(device)
            opt.zero_grad()
            per_sample = loss_fn(model(front, back), y)
            if weighted:
                per_sample = per_sample * weights[y.long()]
            loss = per_sample.mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            if i % 200 == 0:
                print(f"  epoch {ep}: batch {i}/{n_batches}", flush=True)
        val = evaluate(model, loaders["val"], device)
        print(
            f"epoch {ep} ({(time.time() - start) / 60:.1f} min): "
            f"train loss {total / len(loaders['train'].dataset):.3f} | "
            f"val MAE {val['mae']:.3f} balanced {val['balanced_mae']:.3f} "
            f"exact {val['exact']:.3f} within1 {val['within1']:.3f}",
            flush=True,
        )
        if val["balanced_mae"] < best:
            best = val["balanced_mae"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, loaders["test"], device)
    print(
        f"\nTEST ({tag})  MAE {test['mae']:.3f}  balanced MAE {test['balanced_mae']:.3f}  "
        f"exact {test['exact']:.3f}  within1 {test['within1']:.3f}"
    )
    baseline_report(df[df["split"] == "train"], df[df["split"] == "test"])


if __name__ == "__main__":
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n_epochs, weighted="weighted" in sys.argv[2:])
