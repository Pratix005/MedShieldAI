def calculate_overall_risk(genetic_risks, drug_properties, patient_age, ethnicity):
    risk_score = 0
    risk_factors = []

    high_risk_genes = [gene for gene, risk in genetic_risks.items() if risk == 'High']
    if high_risk_genes:
        risk_score += len(high_risk_genes) * 25
        risk_factors.append(f"High genetic dependency: {', '.join(high_risk_genes)}")

    if drug_properties['logp'] > 4:
        risk_score += 15
        risk_factors.append("High lipophilicity → tissue accumulation")

    if drug_properties['molecular_weight'] > 500:
        risk_score += 10
        risk_factors.append("Large molecule → complex metabolism")

    if patient_age > 65:
        risk_score += 10
        risk_factors.append("Elderly → reduced clearance")

    if ethnicity in ['Asian', 'African']:
        risk_score += 5
        risk_factors.append(f"{ethnicity} → variant risk population")

    if risk_score >= 50:
        overall_risk, color = "🚨 HIGH RISK", "red"
    elif risk_score >= 25:
        overall_risk, color = "🟠 MEDIUM RISK", "orange"
    else:
        overall_risk, color = "✅ LOW RISK", "green"

    return {
        'overall_risk': overall_risk,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'color': color
    }

def generate_recommendations(genetic_risks, risk_assessment):
    if risk_assessment['overall_risk'].startswith("🚨"):
        return "Avoid drug or adjust dosage under supervision."
    elif risk_assessment['overall_risk'].startswith("🟠"):
        return "Monitor patient response; genetic testing advised."
    else:
        return "Safe standard use."
