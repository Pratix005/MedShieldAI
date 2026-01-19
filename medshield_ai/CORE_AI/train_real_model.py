# CORE_AI/train_real_model.py
# Trains a real MLP on drug–gene pairs using your featurize() pipeline.
# Outputs: CORE_AI/MODELS/trained_mlp.pth and prints Accuracy / ROC-AUC.

import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Import your feature builders
from CORE_AI.featurize import build_pair_features, GENE_LIST
from CORE_AI.chemical_reader import analyze_drug_structure  # indirectly used inside featurize

# --------------------------
# MODEL DEFINITION (always safe to import)
# --------------------------
class DrugGeneMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
# ONLY RUN THIS IF EXECUTED DIRECTLY
# --------------------------
if __name__ == "__main__":
    print("⚙️ Running standalone training script for MedShield AI...")

    # --------------------------
    # Paths
    # --------------------------
    ROOT = os.path.dirname(__file__)
    DATA_DIR = os.path.join(ROOT, "data")
    MODEL_DIR = os.path.join(ROOT, "MODELS")
    os.makedirs(MODEL_DIR, exist_ok=True)

    MERGED_CSV = os.path.join(DATA_DIR, "merged_training_data.csv")
    REL_FALLBACK = os.path.join(DATA_DIR, "relationships.tsv")

    # --------------------------
    # Helper functions
    # --------------------------
    def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        colmap = {
            "drug": ["Entity2_name", "DrugName", "drug", "chemical_name", "ChemicalName"],
            "gene": ["Entity1_name", "GeneSymbol", "gene", "Gene", "GeneName"],
            "relation": ["Association", "relation_type", "PhenotypeCategory", "Evidence"]
        }

        out = {}
        for k, choices in colmap.items():
            for c in choices:
                if c in df.columns:
                    out[k] = c
                    break

        if "drug" not in out or "gene" not in out:
            return pd.DataFrame(columns=["drug", "gene", "relation"])

        sdf = pd.DataFrame({
            "drug": df[out["drug"]].astype(str),
            "gene": df[out["gene"]].astype(str),
            "relation": df[out["relation"]].astype(str) if "relation" in out else ""
        })
        sdf["drug"] = sdf["drug"].str.strip()
        sdf["gene"] = sdf["gene"].str.strip().str.upper()
        sdf = sdf.replace({"": np.nan}).dropna(subset=["drug", "gene"])
        return sdf


    def _make_negatives(positives, all_genes, neg_per_pos=1, seed=42):
        random.seed(seed)
        pos_set = set(positives)
        negs = []
        by_drug = defaultdict(set)
        for d, g in positives:
            by_drug[d].add(g)

        for d, g in positives:
            tries = 0
            while len(negs) < len(positives) * neg_per_pos and tries < 50:
                cand = random.choice(all_genes)
                tries += 1
                if cand == g or (d, cand) in pos_set:
                    continue
                negs.append((d, cand))
                break
        return negs


    class PairDataset(Dataset):
        def __init__(self, feats, labels):
            self.X = torch.tensor(np.array(feats, dtype=np.float32))
            self.y = torch.tensor(np.array(labels, dtype=np.float32)).unsqueeze(1)
        def __len__(self): return len(self.y)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]


    # --------------------------
    # Load Data
    # --------------------------
    if os.path.exists(MERGED_CSV):
        df = pd.read_csv(MERGED_CSV, low_memory=False)
        print("🧾 Columns found in dataset:", df.columns.tolist())
    else:
        if not os.path.exists(REL_FALLBACK):
            raise FileNotFoundError("❌ No merged_training_data.csv or relationships.tsv found in CORE_AI/data/")
        df = pd.read_csv(REL_FALLBACK, sep="\t", on_bad_lines="skip")

    df_std = _standardize_cols(df).dropna(subset=["drug", "gene"]).drop_duplicates()
    supported = set(GENE_LIST)
    df_std = df_std[df_std["gene"].str.upper().isin(supported)].reset_index(drop=True)

    if len(df_std) == 0:
        print("⚠️ Warning: No valid samples found.")
        exit()

    skip_terms = [
        "anxiety", "dementia", "stroke", "coronary", "kidney", "hypercholesterolemia",
        "schizophrenia", "suicide", "anemia", "depressive", "ulcer", "malaria",
        "vasculitis", "toxic", "heart", "cancer", "adenocarcinoma", "neoplasms",
        "transplantation", "psychotic", "metabolic", "gastrointestinal"
    ]
    df_std = df_std[~df_std["drug"].str.lower().apply(lambda d: any(t in d for t in skip_terms))]
    df_std = df_std.reset_index(drop=True)

    pos_pairs = list({(r.drug, r.gene.upper()) for _, r in df_std.iterrows()})
    pos_labels = [1]*len(pos_pairs)
    neg_pairs = _make_negatives(pos_pairs, list(supported), neg_per_pos=1, seed=42)
    neg_labels = [0]*len(neg_pairs)

    all_pairs = pos_pairs + neg_pairs
    all_labels = pos_labels + neg_labels

    feats, kept_labels, kept_pairs = [], [], []
    bad = 0
    for (drug, gene), lab in zip(all_pairs, all_labels):
        vec = build_pair_features(drug, gene)
        if vec is None:
            bad += 1
            continue
        feats.append(vec)
        kept_labels.append(lab)
        kept_pairs.append((drug, gene))

    if len(feats) == 0:
        raise RuntimeError("Featurization produced no samples. Check featurize.py.")

    in_dim = len(feats[0])
    print(f"✅ Featurized {len(feats)} samples (skipped {bad}); feature dim = {in_dim}")

    X_train, X_test, y_train, y_test = train_test_split(
        feats, kept_labels, test_size=0.2, random_state=42, stratify=kept_labels
    )

    train_ds = PairDataset(X_train, y_train)
    test_ds  = PairDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = "cpu"
    model = DrugGeneMLP(in_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    EPOCHS = 20
    for ep in range(1, EPOCHS+1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
        print(f"Epoch {ep}/{EPOCHS} — Loss: {loss_sum/len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        all_p, all_t = [], []
        for xb, yb in test_loader:
            pr = model(xb.to(device)).cpu().numpy().ravel()
            all_p.append(pr)
            all_t.append(yb.numpy().ravel())
        probs = np.concatenate(all_p)
        true  = np.concatenate(all_t)
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(true, preds)
    try:
        auc = roc_auc_score(true, probs)
    except Exception:
        auc = float("nan")

    print("\n✅ Evaluation")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC-AUC : {auc:.3f}")
    print("\nClassification report:")
    print(classification_report(true, preds, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(true, preds))

    model_path = os.path.join(MODEL_DIR, "trained_mlp.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Saved model → {model_path}")
    print("🏁 Training complete. You can now run app.py to start Flask server.")
else:
    print(">>> train_real_model imported (Flask import — no training executed).")
