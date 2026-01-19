import requests
import time
import json
import os

# Path for local cache
CACHE_FILE = os.path.join(os.path.dirname(__file__), "drug_database.json")

# ✅ Base known SMILES
DEFAULT_DRUG_DATABASE = {
    "Warfarin": "CC(=O)Oc1ccccc1C=O",
    "Clopidogrel": "COC(=O)C1=CC=C(C=C1)C(CSCC2=CC=CC=C2)N3CCCCC3",
    "Codeine": "COC1=C2C3=CC(O)=C4C=CC(O)=CC4(C2=CC=C1)C3O",
    "Azathioprine": "CN1C=NC2=C1N=CN=C2SCC3=CC=CC=C3",
    "Tamoxifen": "CC/C(=C(/CCN(C)C)C1=CC=CC=C1)/C2=CC=CC=C2",
    "Phenytoin": "C1=CC=C2C(=C1)C(=O)NC(=O)N2",
    "Simvastatin": "CCC(C)C(=O)OCC(C)C1CCC(C2C=CCC3=CC=CC=C23)O1",
    "Omeprazole": "COC1=CC=C(C=C1)CNC(=O)C2=CC=C(S2)C(=O)NC3=CC=CC=C3",
    "Metoprolol": "CC(C)NCC(O)COC1=CC=C(C=C1)OC",
    "Carbamazepine": "C1=CC=C2C(=C1)C(=O)NC2=O",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
    "Lisinopril": "CC(C)CC(N)C(=O)N1CCCC1C(=O)O",
    "Amoxicillin": "CC1(C)SCC2NC(=O)C(NC(=O)C(C3=CC=CC=C3)N)C2=O",
    "Fluorouracil": "C1=C(C(=O)NC(=O)N1)F",
    "Oxaliplatin": "C2H6N2O4Pt",
    "Imatinib": "CC1=CC(=C(C=C1)NC(=O)C2=CC=C(C=C2)NC3=NC=CC(=N3)N)C",
    "Doxorubicin": "CC1C2CC3C(C(=O)C4=CC(O)=C(O)C=C4C3=O)C(C(C2(O1)O)O)O",
    "Cisplatin": "Cl[Pt](Cl)(NH3)2",
    "Cyclophosphamide": "C1CN(P(=O)(O1)NCCCl)CCNCCCl",
    "Capecitabine": "CCC(=O)OC1=CN=C(NC2=CC(=O)OC=C2C)N=C1N",
    "Docetaxel": "CC1=C(C(=O)OC2C3C(CC4=C(C3(O1)C(=O)OC(C)C)OC(=O)C(C)(C)C)O4)C5=CC=CC=C5",
    "Rosuvastatin": "CCC(C)C(=O)OCC(C)C1CCC(C2C=CCC3=CC=CC=C23)O1",
    "Paroxetine": "C1CN(CCO1)C2=CC(=C(C=C2)F)C3=CC=CC=C3",
    "Amitriptyline": "CN(C)CCC=C1C2=CC=CC=C2CC3=CC=CC=C13",
    "Fluoxetine": "CC(C)NC1=CC=C(C=C1)OCC2=CC=CC=C2F",
    "Sertraline": "C1CC2=C(C1)C(=CC=C2)C(C3CCNCC3)Cl",
}


def get_smiles(drug_name: str):
    """Return SMILES for a drug. Try local dict first, then query PubChem."""
    if not drug_name:
        return None

    name = drug_name.strip().lower()

    # Local direct or partial match
    for k, v in DRUG_DATABASE.items():
        if k.lower() == name or k.lower() in name:
            return v

    # Query PubChem if not found locally
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT"
        r = requests.get(url, timeout=10)

        if r.status_code == 200:
            smiles = r.text.strip()
            if smiles:
                print(f"✅ Found SMILES for {drug_name}: {smiles[:40]}...")
                DRUG_DATABASE[drug_name] = smiles
                save_cache()
                time.sleep(0.3)
                return smiles
        elif r.status_code == 404:
            # Instead of printing many warnings, log quietly
            with open(os.path.join(os.path.dirname(__file__), "missing_smiles.log"), "a") as log:
                log.write(f"{drug_name}\n")
        else:
            print(f"⚠️ PubChem returned {r.status_code} for {drug_name}")

    except Exception as e:
        with open(os.path.join(os.path.dirname(__file__), "missing_smiles.log"), "a") as log:
            log.write(f"Error for {drug_name}: {e}\n")

    return None


# ✅ Ensure DRUG_DATABASE always exists even before cache load
if "DRUG_DATABASE" not in globals():
    DRUG_DATABASE = DEFAULT_DRUG_DATABASE.copy()
