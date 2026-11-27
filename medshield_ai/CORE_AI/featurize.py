# CORE_AI/featurize.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from CORE_AI.drug_database import get_smiles

GENE_LIST = ["CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "VKORC1", "TPMT", "SLCO1B1", "KCNH2"]

# The descriptor scaling below (mw/800, logp/10, ...) assumes an orally dosed
# small molecule, and the model was trained only on such drugs. Now that names
# resolve live against PubChem, a user can type a peptide or a biologic
# (semaglutide, MW 4114, scales to 5.14 — far outside anything seen in
# training). The model would still emit a confident-looking number, so flag
# those inputs instead of scoring them.
MAX_SMALL_MOLECULE_MW = 1000.0


def in_model_domain(smiles: str):
    """Return (ok, reason). False for molecules outside the training domain."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return False, "structure could not be parsed"
    mw = Descriptors.MolWt(mol)
    if mw > MAX_SMALL_MOLECULE_MW:
        return False, (f"molecular weight {mw:.0f} Da is outside the small-molecule "
                       f"range this model was trained on (<= {MAX_SMALL_MOLECULE_MW:.0f} Da)")
    return True, ""


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
        # ASCII only: this runs inside the Flask request path, and a cp1252
        # Windows console raises UnicodeEncodeError on emoji, taking the app down.
        print(f"[skip] No SMILES found for {drug_name}")
        return None
    drug_f = smiles_to_features(smiles)
    if drug_f is None:
        return None
    gene_f = gene_to_onehot(gene_name)
    return np.concatenate([drug_f, gene_f], axis=0)
