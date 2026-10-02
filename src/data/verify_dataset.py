from collections import Counter
import pandas as pd
from src.config import DATASET_ROOT, INNER_ROOT, TABLES_DIR, CLASS_FOLDERS

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_files(folder):
    return [p for p in folder.rglob("*") if p.is_file()]


def main():
    print("Dataset root :", DATASET_ROOT, "| exists:", DATASET_ROOT.exists())
    print("Nested folder:", INNER_ROOT, "| exists:", INNER_ROOT.exists())
    if INNER_ROOT.exists():
        print("Nested folder contents:", sorted(p.name for p in INNER_ROOT.iterdir()))
    print()

    rows, all_ext = [], Counter()
    for folder, cls in CLASS_FOLDERS.items():
        path = DATASET_ROOT / folder
        if not path.exists():
            print(f"MISSING class folder: {folder}")
            rows.append({"Class": cls, "Folder": folder, "Images": 0, "Non-image files": 0,
                         "Nested copy files": None, "Nested names identical": None})
            continue

        files = list_files(path)
        ext = Counter(p.suffix.lower() for p in files)
        all_ext.update(ext)
        n_img = sum(v for k, v in ext.items() if k in IMG_EXT)

        inner = INNER_ROOT / folder
        inner_n, same = None, None
        if inner.exists():
            inner_files = list_files(inner)
            inner_n = len(inner_files)
            same = {p.name for p in files} == {p.name for p in inner_files}

        rows.append({"Class": cls, "Folder": folder, "Images": n_img,
                     "Non-image files": len(files) - n_img,
                     "Nested copy files": inner_n, "Nested names identical": same})

    df = pd.DataFrame(rows)
    total = df["Images"].sum()
    df["Percent"] = (df["Images"] / total * 100).round(2) if total else 0

    print(df.to_string(index=False))
    print("\nTotal tomato images :", total)
    print("File extensions     :", dict(all_ext))
    nonzero = df[df["Images"] > 0]["Images"]
    if len(nonzero):
        print(f"Largest class: {nonzero.max()} | Smallest class: {nonzero.min()} "
              f"| Imbalance ratio: {nonzero.max() / nonzero.min():.2f}")
    print("Classes found:", (df["Images"] > 0).sum(), "of", len(CLASS_FOLDERS))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "dataset_summary.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()