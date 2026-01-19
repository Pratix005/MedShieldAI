# CORE_AI/train_gnn.py
import os, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from medshield_ai.CORE_AI.data.featurize import build_features
from CORE_AI.chemical_reader import analyze_drug_structure

data_path = os.path.join(os.path.dirname(__file__), "data", "graph_edges.csv")
print(f"🔄 Loading graph data from: {data_path}")
df = pd.read_csv(data_path)

# Build X, y from real features
X, y = build_features(df)
print(f"✅ Features: X={tuple(X.shape)}, y={tuple(y.shape)}")

# Simple split
n = len(X)
split = int(0.8*n)
Xtr, Xval = X[:split], X[split:]
ytr, yval = y[:split], y[split:]

class DrugGeneGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)   # binary for this training example

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.output(x)).squeeze(1)

model = DrugGeneGNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCELoss()

epochs = 20
for ep in range(1, epochs+1):
    model.train()
    opt.zero_grad()
    pred = model(Xtr)
    loss = crit(pred, ytr)
    loss.backward()
    opt.step()

    # quick validation
    model.eval()
    with torch.no_grad():
        pv = model(Xval)
        vloss = crit(pv, yval) if len(Xval)>0 else torch.tensor(0.)
    print(f"Epoch {ep}/{epochs} — loss: {loss.item():.4f} — val: {vloss.item():.4f}")

# Save
model_dir = os.path.join(os.path.dirname(__file__), "MODELS")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "trained_gnn.pth")
torch.save(model.state_dict(), model_path)
print(f"\n✅ Training complete! Saved → {model_path}")
