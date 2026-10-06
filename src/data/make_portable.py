from pathlib import Path

import pandas as pd

from src.config import DATASET_ROOT, TABLES_DIR


def main():
    f = TABLES_DIR / "split_manifest.csv"
    df = pd.read_csv(f)
    df["rel_path"] = [Path(p).relative_to(DATASET_ROOT).as_posix() for p in df["path"]]
    missing = [r for r in df["rel_path"] if not (DATASET_ROOT / r).exists()]
    print("Rows:", len(df), "| missing files:", len(missing))
    print("Example rel_path:", df["rel_path"].iloc[0])
    df.to_csv(f, index=False)
    print("Updated:", f)


if __name__ == "__main__":
    main()