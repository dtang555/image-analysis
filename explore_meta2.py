import pandas as pd, tarfile

df = pd.read_csv("data/raw/metadata.csv")
print("rows:", len(df), "| unique cert_id:", df.cert_id.nunique())
print(df.filename.str.extract(r"_(front|back)\.")[0].value_counts(dropna=False))

print("\nTop grade strings:")
print(df.grade.value_counts(dropna=False).head(40))

df["grade_num"] = df.grade.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
print("\nGrade strings with no number:")
print(df[df.grade.notna() & df.grade_num.isna()].grade.value_counts())
print("\nNumeric grade counts:")
print(df.grade_num.value_counts().sort_index())

with tarfile.open("data/raw/data_shard_0000.tar") as t:
    shard = set(t.getnames())
print("\nshard 0 files:", len(shard), "| found in metadata:", len(shard & set(df.filename)))
print("PNG files in shard:", sorted(n for n in shard if n.endswith(".png")))
