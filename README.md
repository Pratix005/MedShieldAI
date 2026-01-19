🧬 MedShield AI
Personalized Drug–Gene Risk Prediction System

MedShield AI is an AI-driven healthcare prototype designed to predict drug–gene interactions and assess potential risks based on a patient’s genetic profile. The system classifies medications as Safe, Risk, or High-Risk, and intelligently recommends safer alternative drugs when potential adverse interactions are detected.

This project aims to support precision medicine by combining machine learning, bioinformatics, and chemical informatics to improve medication safety and personalization.

🚀 Key Features
🔬 Drug–Gene Risk Prediction

Predicts interaction risk between genes and prescribed drugs

Outputs clear classifications:

✅ Safe

⚠️ Risk

❌ High-Risk

💊 Intelligent Drug Recommendation

Automatically suggests safer alternative medications

Helps reduce adverse drug reactions (ADRs)

Supports clinicians and patients in decision-making

🧪 Molecular Structure Visualization

Uses RDKit to generate and display drug molecule structures

Enables better understanding of chemical properties and interactions


🧠 Machine Learning–Powered

Uses a Multi-Layer Perceptron (MLP) neural network

Trained on curated drug–gene interaction datasets

Model stored as trained_mlp.pth

🧠 Technology Stack
🖥️ Frontend

HTML5

CSS3

JavaScript

Bootstrap (responsive UI)

⚙️ Backend

Python

Flask (REST API)

Pandas & NumPy (data handling)

PyTorch (model inference)

RDKit (chemical informatics)

🤖 Machine Learning Model

Model Type: MLP (Multi-Layer Perceptron)

Input: Drug molecular features + Gene encoding

Output: Risk classification (Safe / Risk / High-Risk)

🏗️ System Architecture

User inputs drug name and gene information

Backend converts drug to molecular descriptors (RDKit)

Gene data is encoded numerically

MLP model predicts interaction risk

System:

Displays risk level

Suggests safer alternatives (if required)

Generates downloadable reports

📈 Use Cases

💉 Personalized medicine decision support

🏥 Clinical research and pharmacogenomics studies

🎓 Educational tool for drug–gene interaction analysis

🧪 Early-stage screening of adverse drug reactions

🔮 Future Enhancements (Planned)

🧬 Blood Report Integration

Use biomarkers (LFT, KFT, CBC, etc.) for enhanced prediction

📁 Patient history & prescription tracking

🌐 Integration with public pharmacogenomics databases (e.g., PharmGKB)

🧠 Advanced models (Graph Neural Networks for molecules)

🔐 Secure authentication & patient data privacy

📱 Mobile-friendly UI / API integration