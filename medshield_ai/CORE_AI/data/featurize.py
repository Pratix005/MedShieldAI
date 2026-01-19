# CORE_AI/featurize.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

GENE_LIST = ["CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "VKORC1", "TPMT", "SLCO1B1", "KCNH2"]

def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    # simple scaling
    return np.array([
        mw/800.0, logp/10.0, hbd/10.0, hba/10.0, rot/15.0, rings/10.0
    ], dtype=np.float32)

def gene_to_onehot(gene: str):
    v = np.zeros(len(GENE_LIST), dtype=np.float32)
    if gene in GENE_LIST:
        v[GENE_LIST.index(gene)] = 1.0
    return v

def build_pair_features(smiles: str, gene: str):
    drug_f = smiles_to_features(smiles)
    gene_f = gene_to_onehot(gene)
    if drug_f is None:
        return None
    return np.concatenate([drug_f, gene_f], axis=0)  # length = 6 + 8 = 14
