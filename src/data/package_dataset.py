import zipfile

import pandas as pd

from src.config import DATASET_ROOT, PROJECT_ROOT, TABLES_DIR


def main():
    df = pd.read_csv(TABLES_DIR / "split_manifest.csv")
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tomato_clean.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:     # JPEGs are already compressed
        for rel in df["rel_path"]:
            z.write(DATASET_ROOT / rel, arcname=rel)
    with zipfile.ZipFile(out) as z:
        print("Zip test (None = OK):", z.testzip())
        print("Files in zip:", len(z.namelist()), "| expected:", len(df))
    print(f"Size: {out.stat().st_size / 1e6:.0f} MB | Saved: {out}")


if __name__ == "__main__":
    main()