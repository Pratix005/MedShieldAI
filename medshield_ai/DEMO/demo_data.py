HIGH_RISK_DEMOS = {
    "Warfarin": {
        "expected": "🚨 HIGH RISK - CYP2C9/VKORC1 dependency",
        "message": "Classic example of genetic sensitivity"
    },
    "Clopidogrel": {
        "expected": "🚨 HIGH RISK - CYP2C19 dependency", 
        "message": "Genetic variants affect antiplatelet efficacy"
    },
    "Codeine": {
        "expected": "🚨 HIGH RISK - CYP2D6 ultra-rapid/poor metabolizers",
        "message": "Fatal respiratory depression risk in ultra-rapid metabolizers"
    }
}
MEDIUM_RISK_DEMOS = {
    "Simvastatin": {
        "expected": "🟡 MEDIUM RISK - SLCO1B1 transport issues",
        "message": "Genetic muscle toxicity risk"
    },
    "Azathioprine": {
        "expected": "🚨 HIGH RISK - TPMT deficiency risk",
        "message": "Life-threatening bone marrow toxicity in poor metabolizers"
    }
}
LOW_RISK_DEMOS = {
    "Aspirin": {
        "expected": "✅ LOW RISK - Minimal genetic dependency",
        "message": "Simple metabolism, wide safety margin"
    },
    "Lisinopril": {
        "expected": "✅ LOW RISK - Standard metabolism",
        "message": "Low genetic variability in response"
    }
}