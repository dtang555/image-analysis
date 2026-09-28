from huggingface_hub import list_repo_files
import tarfile, collections

files = list_repo_files("jyesr/pokemon-tcg-grading", repo_type="dataset")
print(len(files), "files in repo")
for f in files[:40]:
    print(f)

with tarfile.open("data/raw/data_shard_0000.tar") as t:
    names = t.getnames()
print("entries in shard:", len(names))
print(collections.Counter(n.rsplit(".", 1)[-1] for n in names))
