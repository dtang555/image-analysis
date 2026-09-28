from huggingface_hub import hf_hub_download
import pandas as pd

p = hf_hub_download(
    repo_id="jyesr/pokemon-tcg-grading",
    filename="metadata.csv",
    repo_type="dataset",
    local_dir="data/raw",
)
df = pd.read_csv(p)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(10).to_string())
print("\nmissing values:")
print(df.isna().sum())

for c in df.columns:
    if df[c].nunique() <= 20:
        print(f"\n{c} value counts:")
        print(df[c].value_counts(dropna=False).sort_index())
