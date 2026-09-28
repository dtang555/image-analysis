"""Evaluate the saved model on the test split: per-grade metrics + confusion matrix."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from image_analysis.modeling.train import INDEX, MODELS, ROOT, CardDataset, TwoViewNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(INDEX)
    test = df[df["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(CardDataset(test), batch_size=16, num_workers=2)

    model = TwoViewNet().to(device)
    model.load_state_dict(torch.load(MODELS / "resnet18_v1.pt", map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for front, back, _ in loader:
            preds.append(model(front.to(device), back.to(device)).cpu())

    test["pred"] = torch.cat(preds).clamp(1, 10).numpy()
    test["pred_round"] = test["pred"].round().astype(int)
    test["abs_err"] = (test["pred"] - test["grade"]).abs()
    test["exact"] = (test["pred_round"] == test["grade"]).astype(float)

    table = test.groupby("grade").agg(
        n=("grade", "size"),
        mae=("abs_err", "mean"),
        mean_pred=("pred", "mean"),
        exact=("exact", "mean"),
    )
    print(table.round(3).to_string())

    grades = list(range(1, 11))
    cm = pd.crosstab(test["grade"], test["pred_round"]).reindex(
        index=grades, columns=grades, fill_value=0
    )
    print("\nconfusion matrix (rows = true grade, columns = predicted):")
    print(cm.to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_xticklabels(grades)
    ax.set_yticks(range(10))
    ax.set_yticklabels(grades)
    ax.set_xlabel("predicted grade")
    ax.set_ylabel("true grade")
    for i in range(10):
        for j in range(10):
            if cm.values[i, j]:
                ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=7)
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    print("\nsaved", out / "confusion_matrix.png")


if __name__ == "__main__":
    main()
