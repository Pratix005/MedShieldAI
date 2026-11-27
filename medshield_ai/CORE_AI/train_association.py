# CORE_AI/train_association.py
"""Train the drug-gene association model on real labels, and test honestly.

What changed versus train_real_model.py
---------------------------------------
1. REAL NEGATIVES. The old script invented negatives by pairing a drug with a
   random gene. A random pair is overwhelmingly likely to be absent from
   PharmGKB simply because PharmGKB is sparse, so the model could score well by
   learning each gene's base rate. PharmGKB states 'not associated' explicitly;
   those rows are used here instead.

2. SPLIT BY DRUG. The old script used a random split over pairs, so the same
   drug appeared in train and test under different genes. With only six
   descriptors per drug, that leaks. Here all pairs for a given drug go to one
   side (GroupShuffleSplit on drug).

3. A GENE-ONLY BASELINE. The headline question for this model is whether the
   molecule contributes anything beyond "which gene is this". Measured over ten
   diverse drugs, the old checkpoint's output was driven ~24x more by gene
   identity than by the drug. So this script always trains a baseline that sees
   ONLY the gene one-hot, and reports both. If the full model does not clearly
   beat it, the chemistry is not being used and the score should not be
   presented as a per-drug prediction.

Run:  python -m CORE_AI.train_association
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from CORE_AI.featurize import build_pair_features, gene_to_onehot, GENE_LIST
from CORE_AI import drug_database

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
MODEL_DIR = os.path.join(_HERE, "MODELS")
RELATIONSHIPS = os.path.join(DATA_DIR, "relationships.tsv")

SEED = 42
EPOCHS = 400
BATCH = 64
N_SPLITS = 5   # the test set holds ~17 negatives; one split is far too noisy


class DrugGeneMLP(nn.Module):
    """Same topology as train_real_model.DrugGeneMLP so checkpoints interchange."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_pairs() -> pd.DataFrame:
    """Gene-chemical rows with an explicit associated / not-associated verdict."""
    df = pd.read_csv(RELATIONSHIPS, sep="\t", dtype=str).fillna("")
    mask = (((df.Entity1_type == "Gene") & (df.Entity2_type == "Chemical")) |
            ((df.Entity1_type == "Chemical") & (df.Entity2_type == "Gene")))
    gc = df[mask].copy()
    gc["gene"] = np.where(gc.Entity1_type == "Gene", gc.Entity1_name, gc.Entity2_name)
    gc["drug"] = np.where(gc.Entity1_type == "Gene", gc.Entity2_name, gc.Entity1_name)
    gc["gene"] = gc["gene"].str.upper().str.strip()
    gc["drug"] = gc["drug"].str.strip()

    # 'ambiguous' is dropped: it is neither a positive nor a negative, and
    # folding it into either class would reintroduce label noise.
    gc = gc[gc.Association.isin(["associated", "not associated"])]
    gc = gc[gc.gene.isin(GENE_LIST)]
    gc["label"] = (gc.Association == "associated").astype(int)
    return gc.drop_duplicates(["drug", "gene"])[["drug", "gene", "label"]]


def featurise(pairs: pd.DataFrame):
    feats, genes_only, labels, groups = [], [], [], []
    unresolved = 0
    for row in pairs.itertuples(index=False):
        vec = build_pair_features(row.drug, row.gene)
        if vec is None:
            unresolved += 1
            continue
        feats.append(vec)
        genes_only.append(gene_to_onehot(row.gene))
        labels.append(row.label)
        groups.append(row.drug.lower())
    drug_database.flush()
    print(f"[data] featurised {len(feats)} pairs; {unresolved} unresolved names dropped")
    return (np.array(feats, dtype=np.float32),
            np.array(genes_only, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(groups))


def train_once(Xtr, ytr, Xte, yte, seed=SEED):
    """Train to convergence with mini-batches. Returns (model, train_auc, test_auc)."""
    torch.manual_seed(seed)
    model = DrugGeneMLP(Xtr.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # The classes are heavily imbalanced (~484 associated vs ~64 not), so the
    # MINORITY class is up-weighted. Note the direction: weighting negatives by
    # n_neg/n_pos would shrink them further and drive every score to ~1.0.
    n_pos = float(ytr.sum())
    n_neg = float(len(ytr) - n_pos)
    neg_weight = (n_pos / n_neg) if n_neg > 0 else 1.0
    criterion = nn.BCELoss(reduction="none")

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).unsqueeze(1)
    Xte_t = torch.tensor(Xte)
    g = torch.Generator().manual_seed(seed)

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            wb = torch.where(yb > 0.5, torch.tensor(1.0), torch.tensor(neg_weight))
            opt.zero_grad()
            loss = (criterion(model(xb), yb) * wb).mean()
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        p_te = model(Xte_t).numpy().ravel()
        p_tr = model(Xtr_t).numpy().ravel()
    auc_te = roc_auc_score(yte, p_te) if len(set(yte)) > 1 else float("nan")
    auc_tr = roc_auc_score(ytr, p_tr) if len(set(ytr)) > 1 else float("nan")
    return model, auc_tr, auc_te


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    pairs = load_pairs()
    print(f"[data] {len(pairs)} labelled pairs "
          f"({int(pairs.label.sum())} associated, {int((1-pairs.label).sum())} not associated)")
    print(f"[data] {pairs.drug.nunique()} distinct drugs")

    X, Xg, y, groups = featurise(pairs)
    if len(set(y)) < 2:
        raise SystemExit("Only one class survived featurisation; cannot train.")

    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.25, random_state=SEED)
    gene_tr, gene_te, full_tr, full_te, majority = [], [], [], [], []
    model = None

    print(f"Repeated drug-disjoint splits (n={N_SPLITS}). "
          f"'train' AUC shows whether the model can even fit the data.")
    print(f"  {'split':6}{'majority':>10}{'gene train':>12}{'gene test':>11}"
          f"{'full train':>12}{'full test':>11}")
    for i, (tr, te) in enumerate(splitter.split(X, y, groups), 1):
        if len(set(y[te])) < 2:
            print(f"  {i:<6}  test fold has one class only, skipped")
            continue
        _, g_tr, g_te = train_once(Xg[tr], y[tr], Xg[te], y[te], seed=SEED + i)
        model, f_tr, f_te = train_once(X[tr], y[tr], X[te], y[te], seed=SEED + i)
        maj = max(y[te].mean(), 1 - y[te].mean())
        gene_tr.append(g_tr); gene_te.append(g_te)
        full_tr.append(f_tr); full_te.append(f_te); majority.append(maj)
        print(f"  {i:<6}{maj*100:9.1f}%{g_tr:12.3f}{g_te:11.3f}{f_tr:12.3f}{f_te:11.3f}")

    if not full_te:
        raise SystemExit("No usable split; too few negatives.")

    def ms(v):
        return f"{np.mean(v):.3f} +/- {np.std(v):.3f}"

    print()
    print(f"  mean majority-class accuracy : {np.mean(majority)*100:.1f}%")
    print(f"  gene one-hot only  test AUC  : {ms(gene_te)}   (train {ms(gene_tr)})")
    print(f"  gene + chemistry   test AUC  : {ms(full_te)}   (train {ms(full_tr)})")
    print()

    delta = np.mean(full_te) - np.mean(gene_te)
    print(f"[verdict] chemistry changes test ROC-AUC by {delta:+.3f}")
    if np.mean(full_te) <= 0.55:
        print("          Test AUC is at or below chance. This model does not predict")
        print("          drug-gene association on drugs it has not seen. The earlier")
        print("          high accuracy came from randomly generated negatives and from")
        print("          sharing drugs between train and test.")
        print("          Six global descriptors cannot express which enzyme metabolises")
        print("          a molecule; that depends on substructure. Next step is Morgan")
        print("          fingerprints and many more genes, not tuning this model.")
    elif delta < 0.02:
        print("          The molecule contributes essentially nothing beyond gene identity.")
    else:
        print("          The molecular features carry signal beyond the gene prior.")

    out = os.path.join(MODEL_DIR, "association_mlp.pth")
    torch.save(model.state_dict(), out)
    print(f"\n[ok] saved -> {out}")
    print("     Not wired into app.py: app.py still loads trained_mlp.pth.")
