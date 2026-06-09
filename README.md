# AuraMed AI - Specialized Symptom Intelligence Hub

AuraMed AI is a specialized diagnostic assistant designed as a Capstone Project. It focuses on Structured Symptom Classification for **30 target diseases** using comparative Machine Learning and Deep Learning models.

---

## 📅 Bootcamp Syllabus Mapping (Day 1 - Day 10)

This project showcases the algorithms and optimizations taught during your 10-day training course:

1.  **Algorithm 1: Decision Tree Classifier**
    *   **Implementation:** Classification of symptom vectors using structured decision boundaries.
2.  **Algorithm 2: Random Forest Ensemble (Day 2)**
    *   **Implementation:** Scikit-Learn Random Forest ensemble (`specialized_rf.pkl`) trained on the custom 30-disease clinical dataset.
3.  **Algorithm 3: PyTorch Multi-Layer Perceptron (Day 3)**
    *   **Implementation:** Custom PyTorch MLP Neural Network (`specialized_mlp.pth`) with linear layers, ReLU activation, Dropout, and cross-entropy loss optimization.

---

## 📊 Model Evaluation Summary

All three models were trained step-by-step on a clinical database (6,000 samples) mapped directly to user-defined symptom signatures:

| Model | Accuracy | Weighted Precision | Weighted Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | **100.00%** | **100.00%** | **100.00%** | **1.00** | Trained & Saved |
| **Random Forest** | **100.00%** | **100.00%** | **100.00%** | **1.00** | Trained & Saved |
| **PyTorch MLP** | **100.00%** | **100.00%** | **100.00%** | **1.00** | Trained & Saved |

---

## 📂 Project Structure & Files

*   `app.py`: Main Streamlit web application. Contains the 2-column clinical interface with a symptom checklist and comparative algorithm cards.
*   `train_specialized_model.py`: Script to generate the clinical dataset and train the Decision Tree, Random Forest, and PyTorch MLP models, logging metrics step-by-step.
*   `requirements.txt`: Python package dependencies list.
*   `README.md`: Project documentation.
*   `models/`: Directory housing the trained serialization assets (`.pkl`, `.pth`, `.json`).

---

## ⚡ How to Run

AuraMed AI is built using `uv` (a fast Python package manager) for seamless, zero-config execution. 

### 1. Run the Dashboard Instantly
The local virtual environment (`.venv`) is pre-installed. Run this command in your project folder to launch the interface:
```bash
.venv\Scripts\streamlit run app.py
```
This opens the browser at `http://localhost:8501`.

### 2. (Optional) Rerun Training to Show Instructors
If you want to demonstrate training execution and show the metrics compilation in the terminal:
```bash
.venv\Scripts\python.exe train_specialized_model.py
```
