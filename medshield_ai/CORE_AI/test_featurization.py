from CORE_AI.featurize import build_pair_features

drug = "Warfarin"
gene = "CYP2C9"

# build_pair_features takes a drug NAME and resolves the SMILES itself.
feat = build_pair_features(drug, gene)
if feat is not None:
    print("✅ Featurization successful:", len(feat), "features generated")
else:
    print("❌ Featurization failed")
