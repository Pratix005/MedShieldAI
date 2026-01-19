from CORE_AI.featurize import build_pair_features
from CORE_AI.drug_database import DRUG_DATABASE

drug = "Warfarin"
gene = "CYP2C9"

feat = build_pair_features(DRUG_DATABASE[drug], gene)
if feat is not None:
    print("✅ Featurization successful:", len(feat), "features generated")
else:
    print("❌ Featurization failed")
