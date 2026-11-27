# WEB_APP/app.py
from flask import Flask, render_template, request, jsonify, url_for
import json, os, re, sys
import numpy as np
import torch

# Allow importing from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CORE_AI.featurize import build_pair_features, in_model_domain, GENE_LIST
from CORE_AI.train_real_model import DrugGeneMLP
from CORE_AI.drug_database import get_smiles
from CORE_AI import knowledge_base

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
# All eight genes the model supports. CYP2C19 and SLCO1B1 were previously
# omitted, which meant clopidogrel (CYP2C19, level 1A) and the statins
# (SLCO1B1, 1A) could never display their principal association.
GENES_TO_SHOW = list(GENE_LIST)
UI_GENE_ALIAS = {"KCNH2": "hERG"}  # display alias for KCNH2

# The curated knowledge base is loaded by CORE_AI.knowledge_base on import, from
# CORE_AI/data/knowledge_base.json. Rebuild it with:
#     python -m CORE_AI.build_knowledge_base
#
# The hand-written DRUG_GENE_MAP and GENERAL_ALTERNATIVES that used to sit here
# were removed: they drove the "safer alternatives" feature, which recommended
# warfarin in place of a DOAC. DRUG_GENE_MAP was also wrong in places
# (fluorouracil mapped to TPMT; it is a DPYD substrate).
if not knowledge_base.is_available():
    print("[warn] knowledge base is empty. The app will only return model output, "
          "which is exploratory. Run: python -m CORE_AI.build_knowledge_base")

# ---- Helper Functions ----
def risk_bucket(score: float) -> str:
    """Convert a model score to a coarse band (model panel only)."""
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"

def color_for_risk(r: str):
    """Return colour for a risk label. 'Unknown' is grey, never green:
    absence of evidence must not read as reassurance."""
    return {"High": "#e53935", "Medium": "#fbc02d",
            "Low": "#43a047", "Unknown": "#78909c"}.get(r, "#78909c")

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
    # Strip everything that is not a safe filename character. The drug name is
    # user input, and lower()+replace(" ") alone let "../" through into the path.
    safe_name = re.sub(r"[^a-z0-9_.-]", "", drug_name.lower().replace(" ", "_")).strip(".-")
    if not safe_name:
        return None
    image_path = os.path.join(app.static_folder, "molecules", f"{safe_name}.png")
    Draw.MolToFile(mol, image_path, size=(300, 300))
    return url_for("static", filename=f"molecules/{safe_name}.png")

# NOTE: suggest_alternatives() was removed deliberately.
#
# It picked drugs from a 12-entry dict whose gene list did not intersect the
# flagged gene, and called them "safer". That produced actively unsafe output:
# for rivaroxaban flagged on CYP3A4 it recommended warfarin, because warfarin
# was mapped to VKORC1/CYP2C9. Substituting warfarin for a DOAC is a different
# drug class with a different indication, different monitoring and a worse
# bleeding profile for most patients.
#
# Choosing a therapeutic alternative requires indication, comorbidity, renal
# function and interaction data that this project does not have. A drug
# recommendation is also the one output a user is most likely to act on, so it
# is the last thing that should be guessed. If it returns, it must come from
# curated per-indication guidance, not from set arithmetic over a small dict.

# ---- Routes ----
@app.route("/")
def index():
    """Render main UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Look up curated pharmacogenomic evidence for a drug.

    The verdict comes from the curated knowledge base, which traces to PharmGKB
    clinical annotations with stated evidence levels. The ML model is reported
    alongside as an explicitly exploratory signal and never sets the verdict.
    """
    data = request.get_json(force=True)
    drug = (data.get("drug") or "").strip()
    gene_single = (data.get("gene") or "").strip().upper()
    role = data.get("role") or "patient"

    if not drug:
        return jsonify({"error": "Drug name is required."}), 400

    # ---- 1. Curated evidence. This is the part that decides the verdict. ----
    kb_result = knowledge_base.lookup_with_components(drug)
    matches = kb_result["matches"]
    findings = []
    for m in matches:
        for f in m["findings"]:
            findings.append({**f, "drug_component": m["label"]})
    verdict = knowledge_base.assess(findings)

    resolved = None
    if matches:
        resolved = {"label": " + ".join(m["label"] for m in matches),
                    "cid": matches[0]["cid"],
                    "matched_on": matches[0]["matched_on"]}

    # ---- 2. Chemistry. Needed for the structure panel, and for the model. ----
    smiles = get_smiles(matches[0]["label"] if matches else drug) or get_smiles(drug)
    props = rdkit_props(smiles) if smiles else {}
    molecule_image_url = generate_molecule_image(smiles, drug) if smiles else None

    # A drug with curated evidence is still reportable without a structure
    # (several are biologics). Only fail when we have neither.
    if not smiles and not findings:
        return jsonify({"error": f"'{drug}' was not found as a compound and has no "
                                 f"curated pharmacogenomic annotation. Check the "
                                 f"spelling, or try the generic name."}), 404

    genes = GENES_TO_SHOW if not gene_single else [gene_single]
    genes = [g for g in genes if g in GENE_LIST]

    # ---- 3. Exploratory model signal, over the 8 genes it was trained on. ----
    model_ok = bool(smiles) and in_model_domain(smiles)[0]
    rows = []
    for g in (genes if model_ok else []):
        feat = build_pair_features(drug, g)
        if feat is None:
            continue
        x = torch.tensor(np.array(feat, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            s = float(_model(x).item())

        # These are model outputs only. They are NOT affinities, IC50s or
        # probabilities of harm: the model was trained to predict whether a
        # drug-gene pair is recorded in PharmGKB, and on held-out drugs it
        # reaches ROC-AUC ~0.65 against ~0.59 for a gene-identity-only
        # baseline. Reported for transparency, not for decisions.
        rows.append({
            "gene": UI_GENE_ALIAS.get(g, g),
            "raw_gene": g,
            "score": round(s, 3),
            "model_score": round(s, 3),
            "risk": risk_bucket(s),
            "color": color_for_risk(risk_bucket(s)),
            "source": "model (exploratory)"
        })

    # ---- 4. Compose. The verdict is the curated one. ----
    band = verdict["band"]
    curated_genes = sorted({f["gene"] for f in findings})

    recommendations = [verdict["reason"]]
    if band == "High":
        recommendations.append("Discuss with the prescribing clinician before "
                               "starting or continuing this drug. Genotyping may "
                               "be indicated.")
    elif band == "Medium":
        recommendations.append("Clinician review advised; dose adjustment or "
                               "monitoring may apply.")
    elif band == "Unknown":
        recommendations.append("No conclusion can be drawn for this drug from the "
                               "curated evidence available here.")
    if any(f["in_guideline"] for f in findings):
        recommendations.append("At least one finding is backed by a published "
                               "dosing guideline.")
    if kb_result["combination"] and matches:
        recommendations.append("Combination product: findings are listed per component.")

    chart_data = {
        "labels": [r["gene"] for r in rows],
        "scores": [r["score"] for r in rows],
        "colors": [r["color"] for r in rows],
    }

    return jsonify({
        "drug": drug,
        "role": role,
        "resolved": resolved,
        "smiles": smiles,
        "props": props,
        "molecule_image": molecule_image_url,

        # The authoritative part of the response.
        "curated": {
            "available": bool(findings),
            "findings": findings,
            "genes": curated_genes,
            "highest_level": findings[0]["evidence_level"] if findings else "",
            "guideline_backed": any(f["in_guideline"] for f in findings),
            "label_backed": any(f["on_drug_label"] for f in findings),
            "combination": kb_result["combination"],
            "unmatched_components": kb_result["unmatched"],
            "source": "PharmGKB clinical annotations, evidence levels 1A-2B",
        },

        # Exploratory only. Kept separate so it cannot be mistaken for evidence.
        "model": {
            "available": bool(rows),
            "rows": rows,
            "genes_supported": GENES_TO_SHOW,
            "caveat": ("Exploratory. Predicts whether a drug-gene pair is recorded "
                       "in PharmGKB, not severity of harm. Held-out ROC-AUC ~0.65 "
                       "vs ~0.59 for a gene-only baseline. Do not use for decisions."),
        },

        "overall_risk": band,
        "recommendation": " ".join(recommendations),

        # Legacy keys the existing UI reads. rows = model output, which the UI
        # labels as exploratory; alternatives is permanently empty (see above).
        "rows": rows,
        "chart": chart_data,
        "high_genes_detected": curated_genes,
        "alternatives": [],

        "disclaimer": ("Research and educational prototype. Not a medical device "
                       "and not clinical advice. Curated findings are drug-gene "
                       "associations from published literature; they do not account "
                       "for this patient's genotype, dose, comorbidities or other "
                       "medicines. Never change a prescription without a clinician."),
    })


@app.route("/api/genes")
def api_genes():
    """Genes covered by the curated knowledge base, versus by the model."""
    return jsonify({
        "curated_genes": knowledge_base.genes_covered(),
        "model_genes": GENES_TO_SHOW,
        "note": ("The curated knowledge base covers far more genes than the model. "
                 "Genes such as DPYD, HLA-B, UGT1A1 and NUDT15 carry the most "
                 "serious published pharmacogenomic warnings and are not in the "
                 "model's eight."),
    })

# ---- Run App ----
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
