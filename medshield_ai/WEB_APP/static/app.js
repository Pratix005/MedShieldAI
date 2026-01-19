const predictBtn = document.getElementById("predictBtn");
const output = document.getElementById("output");
const tableBody = document.querySelector("#historyTable tbody");

let historyData = [];

// Chart.js setup
const ctx = document.getElementById("chartCanvas").getContext("2d");
const chart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: [],
    datasets: [
      {
        label: "Interaction Probability (%)",
        data: [],
        backgroundColor: "#64ffda80",
        borderColor: "#64ffda",
        borderWidth: 1,
      },
    ],
  },
  options: {
    scales: {
      y: { beginAtZero: true, max: 100 },
    },
  },
});

predictBtn.addEventListener("click", async () => {
  const drug = document.getElementById("drugInput").value.trim();
  const gene = document.getElementById("geneInput").value.trim();

  if (!drug || !gene) {
    alert("Please enter both Drug and Gene!");
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drug, gene }),
    });

    const result = await response.json();

    if (result.error) {
      output.innerText = `⚠️ ${result.error}`;
      return;
    }

    const pred = parseFloat(result.prediction.replace("%", ""));
    output.innerText = `✅ Prediction for ${result.drug} + ${result.gene}: ${result.prediction}`;

    // Add to history
    const entry = {
      drug: result.drug,
      gene: result.gene,
      prediction: pred,
    };
    historyData.push(entry);
    updateHistoryTable();
    updateChart();

  } catch (err) {
    output.innerText = "❌ Server error: " + err;
  }
});

function updateHistoryTable() {
  tableBody.innerHTML = "";
  historyData.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${row.drug}</td>
      <td>${row.gene}</td>
      <td>${row.prediction.toFixed(2)}%</td>
    `;
    tableBody.appendChild(tr);
  });
}

function updateChart() {
  chart.data.labels = historyData.map((r) => `${r.drug}+${r.gene}`);
  chart.data.datasets[0].data = historyData.map((r) => r.prediction);
  chart.update();
}
