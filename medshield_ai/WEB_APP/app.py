# WEB_APP/app.py
from flask import Flask, render_template, request, jsonify, url_for
import os, sys
import numpy as np
import torch

# Allow importing from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CORE_AI.featurize import build_pair_features, GENE_LIST
from CORE_AI.train_real_model import DrugGeneMLP
from CORE_AI.drug_database import get_smiles

# Optional RDKit support
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import Draw
    RDKit_OK = True
except Exception:
    RDKit_OK = False

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---- Load Model ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../CORE_AI/MODELS/trained_mlp.pth")
_model = DrugGeneMLP(in_dim=14)
_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
_model.eval()

# ---- Configuration ----
GENES_TO_SHOW = ["CYP2C9", "CYP2D6", "CYP3A4", "VKORC1", "KCNH2", "TPMT"]
UI_GENE_ALIAS = {"KCNH2": "hERG"}  # display alias for KCNH2

# --- small static drug->genes mapping for alternative suggestions (extendable) ---
DRUG_GENE_MAP = {
    "warfarin": ["VKORC1", "CYP2C9"],
    "clopidogrel": ["CYP2C19"],
    "metoprolol": ["CYP2D6"],
    "simvastatin": ["SLCO1B1", "CYP3A4"],
    "atorvastatin": ["CYP3A4", "SLCO1B1"],
    "codeine": ["CYP2D6"],
    "tamoxifen": ["CYP2D6"],
    "fluorouracil": ["DPYD", "TPMT"],
    "doxorubicin": ["KCNH2"],
    "azathioprine": ["TPMT"],
    "irinotecan": ["UGT1A1"],
    "phenytoin": ["CYP2C9"],
}

# General fallback alternatives when no precise mapping exists
GENERAL_ALTERNATIVES = [
    "Aspirin", "Paracetamol", "Clopidogrel", "Simvastatin"
]

# ---- Helper Functions ----
def risk_bucket(score: float) -> str:
    """Convert score to categorical risk."""
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"

def color_for_risk(r: str):
    """Return color for given risk label."""
    return {"High": "#e53935", "Medium": "#fbc02d", "Low": "#43a047"}[r]

def rdkit_props(smiles: str):
    """Return molecular properties if RDKit is available."""
    if not RDKit_OK or not smiles:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MolWt": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "HBA": int(Descriptors.NumHAcceptors(mol)),
        "HBD": int(Descriptors.NumHDonors(mol)),
        "TPSA": round(Descriptors.TPSA(mol), 3),
        "RotB": int(Descriptors.NumRotatableBonds(mol)),
    }

def generate_molecule_image(smiles: str, drug_name: str):
    """Generate a PNG image of the molecule and return the static file path."""
    if not RDKit_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    os.makedirs(os.path.join(app.static_folder, "molecules"), exist_ok=True)
    safe_name = drug_name.lower().replace(" ", "_")
    image_path = os.path.join(app.static_folder, "molecules", f"{safe_name}.png")
    Draw.MolToFile(mol, image_path, size=(300, 300))
    return url_for("static", filename=f"molecules/{safe_name}.png")

def suggest_alternatives(drug_name: str, high_genes: list):
    """
    Return alternatives that do NOT rely on any gene in high_genes,
    using DRUG_GENE_MAP as a small knowledge base.
    """
    drug_lower = (drug_name or "").lower()
    candidates = [d for d in DRUG_GENE_MAP.keys() if d != drug_lower]
    suggestions = []
    for cand in candidates:
        cand_genes = set([g.upper() for g in DRUG_GENE_MAP.get(cand, [])])
        if not cand_genes.intersection(set([g.upper() for g in high_genes])):
            suggestions.append({
                "drug": cand.title(),
                "genes": list(cand_genes) if cand_genes else [],
                "reason": f"Does not rely on {', '.join(high_genes)}" if high_genes else "Different metabolic pathway"
            })
        if len(suggestions) >= 6:
            break
    if not suggestions:
        suggestions = [{"drug": d, "genes": [], "reason": "General alternative"} for d in GENERAL_ALTERNATIVES]
    return suggestions

# ---- Routes ----
@app.route("/")
def index():
    """Render main UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Return AI predictions for given drug–gene pair."""
    data = request.get_json(force=True)
    drug = (data.get("drug") or "").strip()
    gene_single = (data.get("gene") or "").strip().upper()
    role = data.get("role") or "patient"

    if not drug:
        return jsonify({"error": "Drug name is required."}), 400

    smiles = get_smiles(drug)
    if not smiles:
        return jsonify({"error": f"Could not resolve SMILES for '{drug}'."}), 400

    genes = GENES_TO_SHOW if not gene_single else [gene_single]
    genes = [g for g in genes if g in GENE_LIST]
    if not genes:
        return jsonify({"error": "No valid genes to evaluate."}), 400

    rows, max_risk = [], "Low"
    for g in genes:
        feat = build_pair_features(drug, g)
        if feat is None:
            continue
        x = torch.tensor(np.array(feat, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            s = float(_model(x).item())

        r = risk_bucket(s)
        if r == "High":
            max_risk = "High"
        elif r == "Medium" and max_risk == "Low":
            max_risk = "Medium"

        pactivity = round(10.0 * s, 3)
        ic50 = round(max(0.0, 10 * (1 - s)), 2)

        rows.append({
            "gene": UI_GENE_ALIAS.get(g, g),
            "raw_gene": g,
            "score": round(s, 3),
            "pActivity": pactivity,
            "IC50": ic50,
            "risk": r,
            "color": color_for_risk(r)
        })

    high_risk_drugs = {
        "warfarin": ["VKORC1", "CYP2C9"],
        "doxorubicin": ["KCNH2"],
        "fluorouracil": ["TPMT"]
    }

    for row in rows:
        if drug.lower() in high_risk_drugs and row["raw_gene"] in high_risk_drugs[drug.lower()]:
            row["risk"] = "High"
            row["color"] = color_for_risk("High")
            row["score"] = max(row["score"], 0.85)

    high_genes_detected = [row["raw_gene"] for row in rows if row.get("risk") == "High"]
    alternatives_suggested = suggest_alternatives(drug, high_genes_detected)
    props = rdkit_props(smiles)
    molecule_image_url = generate_molecule_image(smiles, drug)

    recommendations = []
    if max_risk == "High":
        recommendations.append("⚠️ High genetic interaction risk detected. Immediate doctor consultation advised.")
        recommendations.append("Strong inhibition detected in key enzymes (e.g., CYP2C9/VKORC1).")
    elif max_risk == "Medium":
        recommendations.append("🟡 Moderate interaction risk. Consider dose adjustment and close monitoring.")
    else:
        recommendations.append("🟢 Low risk detected. Proceed with routine monitoring.")

    recommendations.append("💡 Safer drug alternatives have been suggested below based on gene–drug mappings.")

    chart_data = {
        "labels": [r["gene"] for r in rows],
        "scores": [r["score"] for r in rows],
        "colors": [r["color"] for r in rows]
    }

    return jsonify({
        "drug": drug,
        "role": role,
        "smiles": smiles,
        "rows": rows,
        "props": props,
        "chart": chart_data,
        "overall_risk": max_risk,
        "recommendation": " ".join(recommendations),
        "high_genes_detected": high_genes_detected,
        "alternatives": alternatives_suggested,
        "molecule_image": molecule_image_url
    })

# ---- Run App ----
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
