from CORE_AI.drug_database import DRUG_DATABASE
import pandas as pd, os

data_path = os.path.join(os.path.dirname(__file__), "CORE_AI", "data", "merged_training_data.csv")
df = pd.read_csv(data_path)

unique_drugs = df["Entity2_name"].dropna().unique()
missing = [d for d in unique_drugs if d not in DRUG_DATABASE]

print(f"🔍 {len(missing)} drugs not found in DRUG_DATABASE:")
print(missing[:30])  # show first 30 missing drugs
