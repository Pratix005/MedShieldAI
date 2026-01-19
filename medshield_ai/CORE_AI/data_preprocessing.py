import sys, os
import pandas as pd

# ✅ Ensure the parent directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Import the loader properly using package syntax
from CORE_AI.data_loader import load_pharmgkb_data


def preprocess_pharmgkb(df):
    """
    Cleans and prepares the PharmGKB dataset for use.
    - Keeps only relevant columns
    - Drops missing values
    - Removes duplicates
    - Normalizes column names
    """
    # Keep only key columns (if they exist)
    cols_to_keep = ['DrugName', 'GeneSymbol', 'PhenotypeCategory', 'Evidence']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]

    # Drop missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]

    print("✅ Data cleaned successfully!")
    print(f"Remaining records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample rows:\n", df.head(5))
    return df


if __name__ == "__main__":
    print("🔄 Loading PharmGKB data...")
    df = load_pharmgkb_data()

    print("🧹 Preprocessing data...")
    cleaned = preprocess_pharmgkb(df)

    print("\n✅ Preprocessing complete!")

