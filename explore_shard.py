from huggingface_hub import hf_hub_download
import tarfile

path = hf_hub_download(
    repo_id="jyesr/pokemon-tcg-grading",
    filename="data_shard_0000.tar",
    repo_type="dataset",
    local_dir="data/raw",
)
print("Saved to:", path)

with tarfile.open(path) as t:
    printed_meta = False
    for i, m in enumerate(t):
        print(m.name, m.size)
        if not printed_meta and m.name.endswith((".json", ".txt", ".cls", ".csv")):
            content = t.extractfile(m).read().decode("utf-8", errors="replace")
            print("  ->", content[:500])
            printed_meta = True
        if i >= 15:
            break
