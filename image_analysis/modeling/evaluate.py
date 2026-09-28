"""Evaluate a saved model on the test split: per-grade metrics + confusion matrix.

Usage: python -m image_analysis.modeling.evaluate [tag]   (tag: full or full_weighted)
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from image_analysis.modeling.train import INDEX, MODELS, ROOT, CardDataset, TwoViewNet


def main(tag="full"):
    plt.switch_backend("Agg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(INDEX)
    test = df[df["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(CardDataset(test), batch_size=16, num_workers=4)

    model = TwoViewNet().to(device)
    model.load_state_dict(torch.load(MODELS / f"resnet18_{tag}.pt", map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for front, back, _ in loader:
            preds.append(model(front.to(device), back.to(device)).cpu())

    test["pred"] = torch.cat(preds).clamp(1, 10).numpy()
    test["pred_round"] = test["pred"].round().astype(int)
    test["abs_err"] = (test["pred"] - test["grade"]).abs()
    test["exact"] = (test["pred_round"] == test["grade"]).astype(float)
    test["within1"] = ((test["pred_round"] - test["grade"]).abs() <= 1).astype(float)

    table = test.groupby("grade").agg(
        n=("grade", "size"),
        mae=("abs_err", "mean"),
        mean_pred=("pred", "mean"),
        exact=("exact", "mean"),
        within1=("within1", "mean"),
    )
    print(table.round(3).to_string())
    print(
        f"\noverall MAE {test['abs_err'].mean():.3f} | balanced MAE {table['mae'].mean():.3f} | "
        f"exact {test['exact'].mean():.3f} | within1 {test['within1'].mean():.3f}"
    )

    grades = list(range(1, 11))
    cm = pd.crosstab(test["grade"], test["pred_round"]).reindex(
        index=grades, columns=grades, fill_value=0
    )
    print("\nconfusion matrix (rows = true grade, columns = predicted):")
    print(cm.to_string())

    reports = ROOT / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(reports / f"per_grade_{tag}.csv")

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
    fig.tight_layout()
    out = reports / "figures" / f"confusion_matrix_{tag}.png"
    fig.savefig(out, dpi=150)
    print("\nsaved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
