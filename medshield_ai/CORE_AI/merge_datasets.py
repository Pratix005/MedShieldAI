# CORE_AI/merge_datasets.py
import os
import pandas as pd

base = os.path.join(os.path.dirname(__file__), "data")

# all datasets you uploaded
files = [
    "relationships.tsv",
    "interaction.tsv",
    "interactions.csv",
    "clinicalvariants.tsv",
    "variants.tsv",
    "merged_training_data.csv"
]

dfs = []
for f in files:
    path = os.path.join(base, f)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, sep="\t" if f.endswith(".tsv") else ",", on_bad_lines="skip")
            df["source"] = f
            dfs.append(df)
            print(f"✅ Loaded {f} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")
    else:
        print(f"❌ Missing file: {f}")

if len(dfs) == 0:
    raise RuntimeError("No datasets loaded!")

merged = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Merged {len(merged)} total records from {len(dfs)} datasets")

merged_path = os.path.join(base, "merged_training_data.csv")
merged.to_csv(merged_path, index=False)
print(f"📁 Saved merged file → {merged_path}")
