import pandas as pd

df = pd.read_csv("data/processed/index.csv")
print(f"{len(df)} cards with both sides:")
print(df["split"].value_counts().to_string())

print("\ngrade counts by split:")
print(pd.crosstab(df["grade"], df["split"], margins=True).to_string())

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"\nbaseline: always predict 10 -> accuracy {(test['grade'] == 10).mean():.3f}, "
      f"MAE {(test['grade'] - 10).abs().mean():.3f}")
med = train["grade"].median()
print(f"baseline: always predict median ({med:g}) -> "
      f"MAE {(test['grade'] - med).abs().mean():.3f}")
