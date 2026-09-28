"""Download shards, cache resized images, and build the card index."""
import io
from pathlib import Path
import sys
import tarfile

from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

REPO_ID = "jyesr/pokemon-tcg-grading"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = ROOT / "data" / "interim" / "images"
PROCESSED = ROOT / "data" / "processed"
IMG_W, IMG_H = 352, 512
SIDES = ("front", "back")


def download(n_shards):
    hf_hub_download(REPO_ID, "metadata.csv", repo_type="dataset", local_dir=RAW)
    return [
        Path(
            hf_hub_download(
                REPO_ID, f"data_shard_{i:04d}.tar", repo_type="dataset", local_dir=RAW
            )
        )
        for i in range(n_shards)
    ]


def cache_images(shards):
    IMAGES.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        print("caching", shard.name)
        with tarfile.open(shard) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                out = IMAGES / f"{Path(member.name).stem}.jpg"
                if out.exists():
                    continue
                try:
                    data = tar.extractfile(member).read()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    if img.width > img.height:  # make every image portrait
                        img = img.rotate(90, expand=True)
                    img = img.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
                    img.save(out, quality=92)
                except OSError as err:
                    print("skipping", member.name, err)


def _split(df, test_size, seed=42):
    try:
        return train_test_split(
            df, test_size=test_size, stratify=df["grade"], random_state=seed
        )
    except ValueError:  # a grade has too few cards in this subset
        return train_test_split(df, test_size=test_size, random_state=seed)


def build_index():
    df = pd.read_csv(RAW / "metadata.csv")
    df["card_id"] = df["filename"].str.extract(r"^(\d+)_")[0].astype("int64")
    df = df.dropna(subset=["grade"])
    df = df[~df["grade"].str.startswith("N0")]  # drop "Authentic"
    df["grade_num"] = df["grade"].str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    df = df[df["grade_num"] % 1 == 0]  # drop half grades
    df["qualifier"] = df["grade"].str.extract(r"\((\w+)\)")[0]
    cards = df.drop_duplicates("card_id").copy()
    cards["grade"] = cards["grade_num"].astype(int)
    cards = cards[
        ["card_id", "grade", "qualifier", "card_name", "year", "brand", "subject"]
    ]

    have = {p.stem for p in IMAGES.glob("*.jpg")}
    both = cards["card_id"].map(lambda c: all(f"{c}_{s}" in have for s in SIDES))
    cards = cards[both].reset_index(drop=True)

    train, rest = _split(cards, 0.2)
    val, test = _split(rest, 0.5)
    cards["split"] = "train"
    cards.loc[cards["card_id"].isin(val["card_id"]), "split"] = "val"
    cards.loc[cards["card_id"].isin(test["card_id"]), "split"] = "test"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cards.to_csv(PROCESSED / "index.csv", index=False)
    return cards


def main(n_shards=1):
    shards = download(n_shards)
    cache_images(shards)
    cards = build_index()
    print(f"\n{len(cards)} cards with both sides:")
    print(cards["split"].value_counts().to_string())
    print("\ngrade counts:")
    print(cards["grade"].value_counts().sort_index().to_string())
    train = cards[cards["split"] == "train"]
    test = cards[cards["split"] == "test"]
    print(f"\nbaseline: always predict 10 -> accuracy {(test['grade'] == 10).mean():.3f}, "
          f"MAE {(test['grade'] - 10).abs().mean():.3f}")
    med = train["grade"].median()
    print(f"baseline: always predict median ({med:g}) -> "
          f"MAE {(test['grade'] - med).abs().mean():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
