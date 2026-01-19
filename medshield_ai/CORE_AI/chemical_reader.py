from rdkit import Chem
from rdkit.Chem import Descriptors

# ✅ FIXED IMPORT PATH
from CORE_AI.drug_database import DRUG_DATABASE



def create_molecular_graph(mol):
    """Convert molecule into a graph-like structure"""
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return {'atoms': atoms, 'bonds': bonds}


def analyze_drug_structure(drug_name):
    """
    Convert drug name to molecular structure and extract features
    Returns: dict containing SMILES, graph, and molecular descriptors
    """
    # ✅ Case-insensitive lookup
    smiles = None
    for name, smi in DRUG_DATABASE.items():
        if drug_name.lower() == name.lower():
            smiles = smi
            break

    if not smiles:
        raise ValueError(f"Drug '{drug_name}' not found in database")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for drug '{drug_name}'")

    # --- Extract features ---
    molecular_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_bond_donors = Descriptors.NumHDonors(mol)
    h_bond_acceptors = Descriptors.NumHAcceptors(mol)

    graph = create_molecular_graph(mol)

    return {
        'graph': graph,
        'properties': {
            'molecular_weight': molecular_weight,
            'logp': logp,
            'h_bond_donors': h_bond_donors,
            'h_bond_acceptors': h_bond_acceptors
        },
        'smiles': smiles
    }
