import pandas as pd
from src.config import TABLES_DIR

inv = pd.read_csv(TABLES_DIR / "image_inventory.csv")
man = pd.read_csv(TABLES_DIR / "clean_manifest.csv")

raw = inv.groupby("class").size().rename("raw_files")
corrupt = inv[inv["corrupted"]].groupby("class").size().rename("corrupted_removed")
valid = inv[~inv["corrupted"]].groupby("class").size().rename("valid_images")
clean = man.groupby("class").size().rename("clean_images")
dark = man[man["brightness"] < 60].groupby("class").size().rename("dark_images_brightness_lt60")

df = pd.concat([raw, corrupt, valid, clean, dark], axis=1).fillna(0).astype(int)
df["exact_duplicates_removed"] = df["valid_images"] - df["clean_images"]
df["percent_of_clean"] = (df["clean_images"] / df["clean_images"].sum() * 100).round(2)
df = df[["raw_files", "corrupted_removed", "exact_duplicates_removed",
         "clean_images", "percent_of_clean", "dark_images_brightness_lt60"]]
df = df.sort_values("clean_images", ascending=False)
df.loc["TOTAL"] = df.sum()

print(df.to_string())
df.to_csv(TABLES_DIR / "cleaning_summary.csv")
print("\nSaved: reports/tables/cleaning_summary.csv")