# CORE_AI/featurize.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from CORE_AI.drug_database import get_smiles

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
    return np.array([mw/800, logp/10, hbd/10, hba/10, rot/15, rings/10], dtype=np.float32)

def gene_to_onehot(gene: str):
    v = np.zeros(len(GENE_LIST), dtype=np.float32)
    if gene in GENE_LIST:
        v[GENE_LIST.index(gene)] = 1.0
    return v

def build_pair_features(drug_name, gene_name):
    """Convert a drug–gene pair into a 14-length feature vector."""
    smiles = get_smiles(drug_name)
    if not smiles:
        print(f"⚠️ No SMILES found for {drug_name}, skipping...")
        return None
    drug_f = smiles_to_features(smiles)
    if drug_f is None:
        return None
    gene_f = gene_to_onehot(gene_name)
    return np.concatenate([drug_f, gene_f], axis=0)
