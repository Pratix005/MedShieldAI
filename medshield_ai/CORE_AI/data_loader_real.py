# CORE_AI/data_loader_real.py
import os
import pandas as pd
import numpy as np
from CORE_AI.drug_database import DRUG_DATABASE
from CORE_AI.featurize import build_pair_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_tables():
    drugs = pd.read_csv(os.path.join(DATA_DIR, "drugs.csv"))
    genes = pd.read_csv(os.path.join(DATA_DIR, "genes.csv"))
    inter = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    return drugs, genes, inter

def make_dataset():
    _, _, inter = load_tables()
    X, y = [], []

    for _, row in inter.iterrows():
        drug = row["drug"]
        gene = row["gene"]
        label = int(row["label"])
        smiles = None

        # case-insensitive lookup in DRUG_DATABASE
        for k, v in DRUG_DATABASE.items():
            if k.lower() == drug.lower():
                smiles = v
                break
        if smiles is None:
            # skip unknown drugs
            continue

        feat = build_pair_features(smiles, gene)
        if feat is None:
            continue

        X.append(feat)
        y.append(label)

    if not X:
        raise RuntimeError("No training samples could be built. Check inputs.")

    X = np.stack(X)  # [N, 14]
    y = np.array(y, dtype=np.float32)  # [N]
    return X, y
