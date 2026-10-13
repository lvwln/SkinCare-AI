# 🩺 SkinCare-AI: Multimodal Dermatological Classifier & Clinical Guidance System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

A multimodal healthcare AI application designed to assist in dermatological visual classification and automated clinical treatment guidance. Built as part of a portfolio demonstration for research applications in healthcare AI (NUS IRIS Program).

---

## 🌟 Key Features

- **Computer Vision Model:** Fine-tuned **PyTorch ResNet18** transfer learning architecture trained to classify dermatological conditions (Acne, Eczema, Psoriasis).
- **Interactive UI:** Web interface powered by **Streamlit** for instant image upload and real-time visual prediction.
- **Structured Clinical Knowledge Retrieval:** Provides clinical context, recommended primary interventions, and symptom management guidelines alongside visual predictions.
- **Edge Deployment Ready:** Optimized lightweight execution for fast inference on consumer hardware.

---

## 🏗 System Architecture

```text
  [ User Image Upload ]
           │
           ▼
  [ PyTorch Engine (ResNet18) ]  ──►  Visual Classification & Probabilities
           │
           ▼
  [ Clinical Guidance Module ]   ──►  Structured Treatment Protocols & Precautions
           │
           ▼
  [ Streamlit Interface ]        ──►  Real-time Clinical Dashboard
🚀 Getting Started
Prerequisites
Python 3.10+

Git

Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/lvwln/SkinCare-AI.git](https://github.com/lvwln/SkinCare-AI.git)
cd SkinCare-AI
Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash
pip install torch torchvision streamlit pillow torchvision
Run the Streamlit Application:

Bash
streamlit run app.py
Access the dashboard locally at http://localhost:8501.

📁 Repository Structure
Plaintext
SkinCare-AI/
├── app.py             # Main Streamlit web application & inference pipeline
├── train.py           # Model training and fine-tuning script (PyTorch)
├── README.md          # Project documentation
└── .gitignore         # Excludes heavy binaries and datasets from remote tracking
⚠️ Medical Disclaimer
This project is intended strictly for research and portfolio demonstration purposes. It is not designed to replace clinical diagnosis, advice, or treatment from a certified medical professional.

👩‍💻 Author
Samriddhi Sharma