"""Train a two-view (front + back) ResNet18 to regress the PSA grade."""
from pathlib import Path
import sys

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
        transforms.Resize((320, 224)),
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
    return {
        "mae": (p - y).abs().mean().item(),
        "exact": (r == y).float().mean().item(),
        "within1": ((r - y).abs() <= 1).float().mean().item(),
    }


def main(epochs=5, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    df = pd.read_csv(INDEX)
    loaders = {
        s: DataLoader(
            CardDataset(df[df["split"] == s]),
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=2,
        )
        for s in ("train", "val", "test")
    }
    model = TwoViewNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    MODELS.mkdir(exist_ok=True)
    best = float("inf")
    n_batches = len(loaders["train"])

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for i, (front, back, y) in enumerate(loaders["train"], 1):
            front, back, y = front.to(device), back.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(front, back), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            if i % 40 == 0:
                print(f"  epoch {ep}: batch {i}/{n_batches}", flush=True)
        val = evaluate(model, loaders["val"], device)
        print(
            f"epoch {ep}: train loss {total / len(loaders['train'].dataset):.3f} | "
            f"val MAE {val['mae']:.3f} exact {val['exact']:.3f} within1 {val['within1']:.3f}",
            flush=True,
        )
        if val["mae"] < best:
            best = val["mae"]
            torch.save(model.state_dict(), MODELS / "resnet18_v1.pt")

    model.load_state_dict(torch.load(MODELS / "resnet18_v1.pt", map_location=device))
    test = evaluate(model, loaders["test"], device)
    print(f"\nTEST  MAE {test['mae']:.3f}  exact {test['exact']:.3f}  within1 {test['within1']:.3f}")
    print("baseline (always 10): MAE 0.716, exact 0.559")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
