import os
import numpy as np
import torch
import torch.nn as nn
from CORE_AI.chemical_reader import analyze_drug_structure
from CORE_AI.featurize import build_pair_features, GENE_LIST


# ------------------------
# 🧠 Model Definition (MATCHES training)
# ------------------------
class DrugGeneMLP(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ------------------------
# ⚙️ Load trained model
# ------------------------
_model_path = os.path.join(os.path.dirname(__file__), "MODELS", "trained_mlp.pth")
_mlp = DrugGeneMLP(input_dim=14)

if os.path.exists(_model_path):
    state_dict = torch.load(_model_path, map_location="cpu")
    _mlp.load_state_dict(state_dict)
    _mlp.eval()
    print("✅ Loaded trained MLP model.")
else:
    print("⚠️ trained_mlp.pth not found — run train_real_model.py first.")


# ------------------------
# 🩺 Prediction Logic
# ------------------------
def _risk_level(score: float):
    if score >= 0.7: return "High"
    if score >= 0.4: return "Medium"
    return "Low"


def predict_genetic_risks(drug_name):
    analysis = analyze_drug_structure(drug_name)

    scores = []
    for gene in GENE_LIST:
        # build_pair_features resolves the SMILES from the name itself.
        feat = build_pair_features(drug_name, gene)
        if feat is None:
            scores.append(0.0)
            continue
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s = float(_mlp(x).item())
        scores.append(s)

    # gene alias mapping
    gene_alias = {
        "CYP2C9": "CYP2C9",
        "CYP2D6": "CYP2D6",
        "CYP3A4": "CYP3A4",
        "VKORC1": "VKORC1",
        "TPMT": "TPMT",
        "KCNH2": "hERG"
    }

    risks, out_scores = {}, {}
    for g, s in zip(GENE_LIST, scores):
        if g in gene_alias:
            risks[gene_alias[g]] = _risk_level(s)
            out_scores[gene_alias[g]] = s

    return risks, np.array(list(out_scores.values()), dtype=np.float32), analysis


# ------------------------
# 🔬 Optional Test
# ------------------------
if __name__ == "__main__":
    drug = "Warfarin"
    risks, scores, analysis = predict_genetic_risks(drug)
    print("\n🔍 Predictions for", drug)
    for g, r in risks.items():
        print(f"{g}: {r} ({scores[list(risks.keys()).index(g)]:.3f})")

    import matplotlib.pyplot as plt
    plt.bar(risks.keys(), scores)
    plt.ylabel("Affinity Score")
    plt.title(f"Predicted Gene Affinities for {drug}")
    plt.show()
