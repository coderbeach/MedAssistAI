# 🩺 MediAssist AI — Advanced Clinical Screening & Report Intelligence Platform

MediAssist AI is a premium, AI-powered healthcare assistant and diagnostic intelligence dashboard designed to streamline patient symptom checking, clinical report summarization, and dermatological/ophthalmological screening. The system features an interactive, animated clinical assistant named **Baymax** to guide users through the process and output a unified health report.

---

## ✨ Features & Architecture

### 1. 🤖 Baymax AI Health Assistant
*   An interactive, animated floating robotic avatar acting as the interface centerpiece.
*   Triggers a collapsable slide-out clinical chat assistant using a custom light-theme UI.
*   Features entrance walk-in and greeting animations.

### 2. 🎛️ Core Diagnostic Modules
*   **Symptom Intelligence:** Multi-select symptom checklists mapped to a PyTorch Multi-Layer Perceptron (MLP) classifier, Random Forest Ensemble, and Decision Tree to evaluate **32 target diseases** (including **Skin Cancer** and **Breast Cancer** detections) with high precision.
*   **Medical Report Summarizer:** Instantly parses uploaded medical PDF reports and generates a clean clinical breakdown.
*   **Skin & Eye Condition Scanner:** Deep Learning screening powered by a fine-tuned ResNet-18 model to analyze dermatological and ocular photographs for quick diagnostic flags.

### 3. 📊 Clinical Insights & Unified Report
*   **Dynamic Results Dashboard:** Live status tracking (Success/Pending/Failed) across all three diagnostic modules.
*   **Unified Health Report:** Combines symptoms, machine learning diagnoses, report summaries, image scan results, and warning alerts into a structured patient profile.
*   **PDF Compiler:** Dynamic print-ready PDF generation with a direct download button.

---

## 📅 Architecture & ML Models
This project showcases comparative machine learning, deep learning, and report compiler technologies:
*   **Decision Tree & Random Forest:** Built using `scikit-learn` to establish baseline diagnostic checks.
*   **PyTorch MLP Network:** Multi-layer neural network with Dropout, ReLU activations, and Adam optimization for specialized disease mapping (including specialized oncology screening rules for breast and skin cancer signs).
*   **Fine-tuned ResNet-18:** PyTorch CNN trained to categorize skin and ocular conditions from clinical image feeds.
*   **PDF Compiler:** Built using `ReportLab` to structure custom clinical reports dynamically.

---

## 📁 Repository Structure
```text
├── app.py                      # Main Streamlit web application (redesigned premium UI/UX)
├── generate_report.py          # PDF compiler script using ReportLab
├── llm_inference.py            # Clinical assistant conversation & inference logic
├── train_image_model.py        # Script to fine-tune ResNet-18 on Flickr medical images
├── train_specialized_model.py  # Script to train Decision Tree, RF, and PyTorch MLP models
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── assets/                     # UI visual assets (including the Baymax avatar)
└── models/                     # Serialized model weights (.pth, .pkl, .json)
```

---

## ⚡ How to Run the App

1.  **Activate Virtual Environment:**
    Ensure you are in the project root directory:
    ```powershell
    .venv\Scripts\activate
    ```

2.  **Launch the Streamlit Server:**
    Run the application using:
    ```powershell
    streamlit run app.py
    ```
    This will spin up the local server and automatically open the application in your default browser at **`http://localhost:8501`**.

---

## 🛠️ Verification Checklist
*   Verify the clean, responsive, professional white/blue healthcare SaaS layout.
*   Click the floating circular **Baymax avatar** on the right side to open the interactive chatbot.
*   Select symptoms under **Symptom Checker** and run the analysis.
*   Upload a sample PDF under the **Report Summarizer** to view the parsed output.
*   Upload an image in the **Skin & Eye Scanner** to trigger the ResNet-18 classifier.
*   View your compiled data in the **Unified Health Report** and click **📥 Download Report PDF** to export.
