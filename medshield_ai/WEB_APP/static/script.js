async function analyzeDrug() {
    const drug = document.getElementById('drug').value;
    const age = document.getElementById('age').value;
    const ethnicity = document.getElementById('ethnicity').value;

    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = "<p>⏳ Analyzing drug risk... please wait.</p>";

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ drug, age, ethnicity })
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<p style='color:red;'>❌ Error: ${data.error}</p>`;
            return;
        }

        resultDiv.innerHTML = `
            <h2>🔬 ${data.drug}</h2>
            <p><strong>Overall Risk:</strong> <span style="color:${data.risk_assessment.color};">${data.risk_assessment.overall_risk}</span></p>
            <p><strong>Confidence:</strong> ${data.confidence}</p>

            <h3>🧠 Genetic Risk Factors:</h3>
            <ul>
                ${Object.entries(data.genetic_risks).map(([gene, risk]) => `<li>${gene}: ${risk}</li>`).join('')}
            </ul>

            <h3>🧪 Molecular Properties:</h3>
            <ul>
                ${Object.entries(data.molecular_properties).map(([prop, val]) => `<li>${prop}: ${val.toFixed(2)}</li>`).join('')}
            </ul>

            <h3>⚠️ Risk Details:</h3>
            <ul>
                ${data.risk_assessment.risk_factors.map(f => `<li>${f}</li>`).join('')}
            </ul>

            <h3>💡 Recommendation:</h3>
            <p>${data.recommendations}</p>
        `;
    } catch (error) {
        resultDiv.innerHTML = `<p style='color:red;'>Error connecting to backend.</p>`;
        console.error(error);
    }
}
