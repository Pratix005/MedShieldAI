import pandas as pd
import os

def load_pharmgkb_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'relationships.tsv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')
    print(f"✅ Loaded {len(df)} records")
    return df
