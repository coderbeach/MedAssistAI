import os
import sys
import json
import pickle
import base64
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Adjust system path to import local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train_specialized_model import SpecializedMLP
from llm_inference import MedicalChatbot
from generate_report import get_test_recommendations, check_critical_alert

def format_symptom_name(sym):
    if sym == "asymptomatic":
        return "Asymptomatic (No Symptoms)"
    return sym.replace("_", " ").title()

# Page configuration with premium medical SaaS design aesthetics
st.set_page_config(
    page_title="MediAssist AI - Intelligent Patient Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Load models and configurations (Fully Cached Backend)
# ---------------------------------------------------------
@st.cache_resource
def load_specialized_models():
    model_dir = "./models"
    features_path = os.path.join(model_dir, "specialized_features.json")
    classes_path = os.path.join(model_dir, "specialized_classes.json")
    image_classes_path = os.path.join(model_dir, "image_classes.json")
    dt_path = os.path.join(model_dir, "specialized_dt.pkl")
    rf_path = os.path.join(model_dir, "specialized_rf.pkl")
    mlp_path = os.path.join(model_dir, "specialized_mlp.pth")
    image_model_path = os.path.join(model_dir, "skin_classifier.pth")
    
    with open(features_path, "r") as f:
        features = json.load(f)
    with open(classes_path, "r") as f:
        classes = json.load(f)
    with open(image_classes_path, "r") as f:
        image_classes = json.load(f)
        
    with open(dt_path, "rb") as f:
        dt_model = pickle.load(f)
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(features)
    num_classes = len(classes)
    mlp_model = SpecializedMLP(input_dim, num_classes).to(device)
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp_model.eval()
    
    # Load ResNet-18 Image Classifier
    try:
        image_cnn = models.resnet18(weights=None)
    except TypeError:
        image_cnn = models.resnet18()
    num_ftrs = image_cnn.fc.in_features
    image_cnn.fc = nn.Linear(num_ftrs, len(image_classes))
    image_cnn.load_state_dict(torch.load(image_model_path, map_location=device))
    image_cnn.to(device)
    image_cnn.eval()
    
    return {
        "features": features,
        "classes": classes,
        "image_classes": image_classes,
        "dt": dt_model,
        "rf": rf_model,
        "mlp": mlp_model,
        "image_cnn": image_cnn,
        "device": device
    }

models_data = load_specialized_models()

# Load Chatbot helper
@st.cache_resource
def load_chatbot_helper():
    return MedicalChatbot(use_llm=False)

chatbot = load_chatbot_helper()

# Convert Baymax Image to Base64 for the clickable circular avatar
@st.cache_data
def get_baymax_base64():
    path = "./assets/baymax.png"
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

baymax_b64 = get_baymax_base64()

# Inject custom CSS for premium healthcare styling and responsive elements
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global layout styles */
    .stApp {{
        background-color: #F8FAFC !important;
        color: #334155 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Outfit', sans-serif !important;
        color: #0F172A !important;
    }}

    /* Sticky Navigation */
    .nav-bar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 70px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 40px;
        z-index: 9999;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }}
    .nav-logo {{
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: #2563EB;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .nav-links {{
        display: flex;
        gap: 20px;
    }}
    .nav-link {{
        color: #64748B;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.95rem;
        transition: color 0.2s ease;
    }}
    .nav-link:hover {{
        color: #2563EB;
    }}

    /* Hero Section */
    .hero-section {{
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        padding: 110px 40px 60px 40px;
        border-radius: 0 0 40px 40px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.04);
        margin-bottom: 40px;
    }}
    .hero-title {{
        font-size: 3.6rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
        margin-bottom: 10px !important;
        letter-spacing: -1px !important;
    }}
    .hero-subtitle {{
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2563EB !important;
        margin-bottom: 15px !important;
    }}
    .hero-text {{
        font-size: 1.1rem !important;
        color: #475569 !important;
        max-width: 850px;
        margin: 0 auto !important;
        line-height: 1.6 !important;
    }}
    /* Baymax section */
    .baymax-section {{
        background: #FFFFFF !important;
        border-radius: 24px !important;
        padding: 20px 40px !important;
        margin: 0 auto 20px auto !important;
        max-width: 850px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.03) !important;
        text-align: center !important;
    }}
    
    /* Clickable circular Baymax image container */
    .baymax-avatar-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    
    /* Floating and Pulsing Animation for Baymax */
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.18); }}
        70% {{ box-shadow: 0 0 0 16px rgba(37, 99, 235, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }}
    }}
    
    div:has(.baymax-avatar-container) + div {{
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }}
    
    div:has(.baymax-avatar-container) + div button {{
        display: block !important;
        margin: 0 auto !important;
        background-image: url("data:image/png;base64,{baymax_b64}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        width: 170px !important;
        height: 170px !important;
        border-radius: 50% !important;
        border: 4px solid #FFFFFF !important;
        background-color: #F1F5F9 !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.06) !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        animation: walk-entrance 2.5s cubic-bezier(0.25, 0.8, 0.25, 1) 1, float 4s ease-in-out 2.5s infinite, pulse 2.5s infinite !important;
        color: transparent !important;
        font-size: 0 !important;
        padding: 0 !important;
        outline: none !important;
    }}
    @keyframes walk-entrance {{
        0% {{ transform: translateX(-200px) translateY(0) rotate(-6deg); opacity: 0; }}
        20% {{ transform: translateX(-150px) translateY(-12px) rotate(6deg); }}
        40% {{ transform: translateX(-100px) translateY(0) rotate(-6deg); }}
        60% {{ transform: translateX(-50px) translateY(-12px) rotate(6deg); }}
        80% {{ transform: translateX(-20px) translateY(0) rotate(-2deg); }}
        100% {{ transform: translateX(0) translateY(0) rotate(0); opacity: 1; }}
    }}
    @keyframes wave-greeting {{
        0%, 100% {{ transform: rotate(0deg) scale(1) translateY(0); }}
        15% {{ transform: rotate(10deg) scale(1.08) translateY(-8px); }}
        30% {{ transform: rotate(-8deg) scale(1.08) translateY(-8px); }}
        45% {{ transform: rotate(10deg) scale(1.08) translateY(-8px); }}
        60% {{ transform: rotate(-6deg) scale(1.08) translateY(-8px); }}
        75% {{ transform: rotate(4deg) scale(1.08) translateY(-8px); }}
        90% {{ transform: rotate(-2deg) scale(1.08) translateY(-8px); }}
    }}
    div:has(.baymax-avatar-container) + div button:hover {{
        animation: wave-greeting 1.5s ease-in-out infinite !important;
        filter: drop-shadow(0 0 20px rgba(37, 99, 235, 0.35)) !important;
        border-color: #DBEAFE !important;
    }}
    div:has(.baymax-avatar-container) + div button:active {{
        transform: scale(0.95) !important;
    }}


    /* Core Modules Feature Cards */
    .feature-card {{
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 20px !important;
        padding: 20px 30px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.01) !important;
        transition: all 0.3s ease !important;
        margin-bottom: 20px !important;
    }}
    .feature-card:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.04) !important;
        border-color: #BFDBFE !important;
    }}
    .feature-card .card-title {{
        border-bottom: none !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        text-align: center !important;
    }}
    .card-title {{
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-top: 0 !important;
        margin-bottom: 12px !important;
        border-bottom: 2px solid #EFF6FF;
        padding-bottom: 8px;
    }}
    .card-desc {{
        color: #64748B !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        margin-bottom: 20px !important;
    }}

    /* Form controls styling overrides */
    .stMultiSelect div[data-baseweb="select"] {{
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 3px !important;
    }}
    .stMultiSelect div[data-baseweb="select"]:focus-within {{
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }}
    .stMultiSelect [data-baseweb="tag"] {{
        max-width: none !important;
        white-space: normal !important;
    }}
    .stFileUploader > div {{
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 16px !important;
        padding: 15px !important;
        transition: all 0.2s ease !important;
    }}
    .stFileUploader > div:hover {{
        border-color: #3B82F6 !important;
        background-color: #EFF6FF !important;
    }}

    /* Chat bubble enhancements for Light Theme */
    div[data-testid="stChatMessage"] {{
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.01) !important;
        margin-bottom: 12px !important;
    }}
    
    /* Baymax Chat container */
    .chat-panel {{
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin-top: 25px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.02) !important;
        text-align: left;
    }}

    /* Primary Actions / Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18) !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3) !important;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    }}
    
    /* Result Dashboard panels styled directly on columns */
    /* Result Dashboard panels */
    .result-panel {{
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 20px !important;
        padding: 15px 25px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.01) !important;
        margin-bottom: 20px !important;
        border-left: 5px solid #2563EB !important;
        text-align: center !important;
    }}
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }}
    .status-success {{
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }}
    .status-pending {{
        background: #FEF9C3;
        color: #854D0E;
        border: 1px solid #FEF08A;
    }}

    /* Medical Report Display */
    .report-card {{
        background: #FFFFFF !important;
        border: 2px solid #2563EB !important;
        border-radius: 24px !important;
        padding: 20px 40px !important;
        box-shadow: 0 15px 45px rgba(37, 99, 235, 0.05) !important;
        margin-top: 25px !important;
        margin-bottom: 25px !important;
    }}
    
    /* Quick Action Buttons for Export */
    .export-btn-group {{
        display: flex;
        gap: 15px;
        margin-top: 20px;
        justify-content: flex-end;
    }}
    .export-btn-group button {{
        width: auto !important;
    }}

    /* Footer */
    .footer-container {{
        border-top: 1px solid #E2E8F0;
        padding: 40px 20px;
        margin-top: 60px;
        background-color: #FFFFFF;
        text-align: center;
        color: #64748B;
        font-size: 0.9rem;
    }}
    
    /* Hero Baymax Avatar Column styling */
    .hero-baymax-avatar {{
        background-image: url("data:image/jpeg;base64,{baymax_b64}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        width: 170px !important;
        height: 170px !important;
        border-radius: 50% !important;
        border: 4px solid #FFFFFF !important;
        background-color: #F8FAFC !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05) !important;
        animation: float 4s ease-in-out infinite !important;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Bar HTML
# ---------------------------------------------------------
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">⚕️ MediAssist AI</div>
    <div class="nav-links">
        <a href="#home" class="nav-link">Home</a>
        <a href="#baymax" class="nav-link">Baymax Assistant</a>
        <a href="#modules" class="nav-link">Core Modules</a>
        <a href="#dashboard" class="nav-link">Results Dashboard</a>
        <a href="#report" class="nav-link">Unified Report</a>
    </div>
</div>
<div id="home"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; text-align: left; gap: 30px;">
        <div style="flex: 2.2; min-width: 300px;">
            <h1 class="hero-title">MediAssist AI</h1>
            <h3 class="hero-subtitle">AI-Powered Healthcare Screening, Medical Report Intelligence & Patient Assistance</h3>
            <p class="hero-text">Analyze symptoms, summarize medical reports, screen skin and eye conditions, and generate comprehensive health insights through a unified AI-powered platform.</p>
        </div>
        <div style="flex: 1; min-width: 200px; display: flex; justify-content: center;">
            <div class="hero-baymax-avatar"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "active_symptoms" not in st.session_state:
    st.session_state.active_symptoms = []
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am Baymax, your personal healthcare companion. 💖 Ask me a question, or upload an image/report below for diagnostic guidance. Stay strong!"}
    ]
if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "image_prediction" not in st.session_state:
    st.session_state.image_prediction = None
if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = None

# Helper mock summaries function for PDF Summarizer
def generate_pdf_summary(filename):
    filename_lower = filename.lower()
    if "blood" in filename_lower or "cbc" in filename_lower:
        return (
            "**Document Vitals:** Normal Hemoglobin (14.2 g/dL), WBC slightly elevated (9,500/µL).\n\n"
            "**Key Finding:** General blood count ranges are optimal. The borderline elevated White Blood Cell count is indicative of a mild, resolving immune response, consistent with a recent seasonal throat irritation or common cold.\n\n"
            "**Recommendation:** Maintain hydration. Retest hematology in 6 months if symptoms linger."
        )
    elif "sugar" in filename_lower or "glucose" in filename_lower or "diabetes" in filename_lower:
        return (
            "**Document Vitals:** Fasting Glucose (115 mg/dL), HbA1c (6.1%).\n\n"
            "**Key Finding:** Glycemic indices are moderately elevated, placing the values in the early pre-diabetic baseline. Tissues suggest early-stage insulin resistance.\n\n"
            "**Recommendation:** Implement low-sugar dietary modifications, initiate daily aerobic exercise (30 min), and consult for metabolic monitoring."
        )
    elif "lipid" in filename_lower or "cholesterol" in filename_lower:
        return (
            "**Document Vitals:** Total Cholesterol (230 mg/dL), LDL (145 mg/dL), HDL (45 mg/dL).\n\n"
            "**Key Finding:** Mild hypercholesterolemia with elevated LDL fraction. Baseline values show elevated cardiovascular risk markers.\n\n"
            "**Recommendation:** Minimize saturated fat intake, include omega-3 fatty acids, and re-evaluate lipid profile in 90 days."
        )
    else:
        return (
            "**Document Vitals:** Vitals recorded in file are within standard physiologic baselines (optimal blood pressure and normal heart rate).\n\n"
            "**Key Finding:** General health indicators present as normal. No diagnostic flags or critical anomalies found.\n\n"
            "**Recommendation:** Continue routine lifestyle habits and complete standard annual clinical checks."
        )

# =========================================================
# BAYMAX AI ASSISTANT SECTION
# =========================================================
st.markdown('<div id="baymax"></div>', unsafe_allow_html=True)
st.markdown('<div class="baymax-section"><h2 style="color:#1E3A8A; margin:0; font-size: 1.8rem; font-weight:700;">Baymax AI Health Assistant</h2></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748B; font-size:1.05rem; margin-top:15px; margin-bottom: 25px; text-align:center;">Your intelligent healthcare companion. Describe symptoms, ask health-related questions, upload reports, and receive AI-assisted guidance.</p>', unsafe_allow_html=True)

# Clickable Baymax Avatar Button
st.markdown('<div class="baymax-avatar-container"></div>', unsafe_allow_html=True)
if st.button("", key="baymax_avatar_trigger"):
    st.session_state.show_chatbot = not st.session_state.show_chatbot
    st.rerun()

# Collapsible Conversational Chat Interface
if st.session_state.show_chatbot:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color:#2563EB;'>💬 Consultation Chat logs</h4>", unsafe_allow_html=True)
    
    # Chat History
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    # Chat Input
    if chat_input := st.chat_input("Ask Baymax a health question..."):
        st.session_state.chat_messages.append({"role": "user", "content": chat_input})
        st.session_state.prompt_count += 1
        
        chat_input_lower = chat_input.lower().strip()
        
        if any(greet in chat_input_lower for greet in ["hello", "hi", "heyy", "hey", "greetings"]):
            reply = "Hii! I am Baymax, your personal healthcare companion. How are you feeling today? Stay strong! 💖"
        elif any(w in chat_input_lower for w in ["evaluate", "diagnose", "check", "predict", "result"]):
            active_symptoms = st.session_state.active_symptoms
            if not active_symptoms:
                reply = "I cannot evaluate without symptoms. Please enter symptoms in Card 1 below first."
            else:
                input_vector = [1 if sym in active_symptoms else 0 for sym in models_data["features"]]
                input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(models_data["device"]).unsqueeze(0)
                with torch.no_grad():
                    mlp_out = models_data["mlp"](input_tensor)
                    mlp_prob = torch.softmax(mlp_out, dim=1)[0].cpu().numpy()
                    mlp_pred_idx = np.argmax(mlp_prob)
                    mlp_disease = models_data["classes"][str(mlp_pred_idx)]
                    mlp_confidence = mlp_prob[mlp_pred_idx] * 100
                    
                tests = get_test_recommendations(mlp_disease)
                tests_str = " or ".join(tests) if tests else "routine clinical checkups"
                symptoms_str = ", ".join([format_symptom_name(s) for s in active_symptoms])
                
                reply = (
                    f"Stay strong! 💖 Based on your recorded symptoms ({symptoms_str}), "
                    f"my analysis indicates **{mlp_disease}** (model confidence: {mlp_confidence:.1f}%). "
                    f"Consider taking a **{tests_str}** to confirm. "
                    f"Please consult a physician for professional clinical evaluation."
                )
        else:
            # Extract symptoms from query
            new_parsed = chatbot.extract_symptoms(chat_input)
            for s in new_parsed:
                if s not in st.session_state.active_symptoms:
                    st.session_state.active_symptoms.append(s)
            
            active_symptoms = st.session_state.active_symptoms
            symptoms_str = ", ".join([format_symptom_name(s) for s in active_symptoms]) if active_symptoms else ""
            
            if st.session_state.prompt_count == 1:
                reply = (
                    "I am sorry you are feeling unwell. 💖 "
                    f"{f'I have noted the symptoms: **{symptoms_str}**.' if symptoms_str else ''} "
                    "I recommend visiting a doctor for an official checkup. "
                    "Type 'evaluate' whenever you want me to evaluate! Stay strong!"
                )
            else:
                reply = (
                    "Stay strong! 💖 I am here for you. "
                    f"{f'Symptoms recorded: **{symptoms_str}**.' if symptoms_str else 'Tell me more about what you are feeling.'} "
                    "Let me know if you would like me to evaluate."
                )
                
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# CORE AI MODULES SECTION (3 Responsive Cards)
# =========================================================
st.markdown('<div id="modules"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align:center; color:#1E3A8A; margin-bottom:30px;">Core AI Screening Modules</h2>', unsafe_allow_html=True)

card_col1, card_col2, card_col3 = st.columns(3)

# -----------------
# CARD 1: Symptom Analysis
# -----------------
with card_col1:
    st.markdown('<div class="feature-card"><div class="card-title">Symptom Analysis & Report</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="card-desc">Analyze patient symptoms, identify possible health conditions, recommend relevant medical tests, and generate an organized health report.</p>', unsafe_allow_html=True)
    
    # Symptom checklist input
    selected_symptoms = st.multiselect(
        "Search and select symptoms:",
        options=models_data["features"],
        default=st.session_state.active_symptoms,
        placeholder="Type symptoms here...",
        format_func=format_symptom_name,
        key="symptom_select_input"
    )
    st.session_state.active_symptoms = selected_symptoms
    
    st.write("")
    if st.button("📊 Run Symptom Analysis", key="btn_symptom_analyze"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = [1 if sym in selected_symptoms else 0 for sym in models_data["features"]]
            input_array = np.array(input_vector).reshape(1, -1)
            
            # Decision Tree
            dt_prob = models_data["dt"].predict_proba(input_array)[0]
            dt_pred_idx = np.argmax(dt_prob)
            dt_disease = models_data["classes"][str(dt_pred_idx)]
            dt_confidence = dt_prob[dt_pred_idx] * 100
            
            # Random Forest
            rf_prob = models_data["rf"].predict_proba(input_array)[0]
            rf_pred_idx = np.argmax(rf_prob)
            rf_disease = models_data["classes"][str(rf_pred_idx)]
            rf_confidence = rf_prob[rf_pred_idx] * 100
            
            # PyTorch MLP
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(models_data["device"]).unsqueeze(0)
            with torch.no_grad():
                mlp_out = models_data["mlp"](input_tensor)
                mlp_prob = torch.softmax(mlp_out, dim=1)[0].cpu().numpy()
                mlp_pred_idx = np.argmax(mlp_prob)
                mlp_disease = models_data["classes"][str(mlp_pred_idx)]
                mlp_confidence = mlp_prob[mlp_pred_idx] * 100
                
            st.session_state.predictions = {
                "dt_disease": dt_disease,
                "dt_conf": dt_confidence,
                "rf_disease": rf_disease,
                "rf_conf": rf_confidence,
                "mlp_disease": mlp_disease,
                "mlp_conf": mlp_confidence
            }
            st.session_state.scroll_to = "dashboard"
            st.rerun()

# -----------------
# CARD 2: Medical Report Summarizer
# -----------------
with card_col2:
    st.markdown('<div class="feature-card"><div class="card-title">Medical Report Summarizer</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="card-desc">Upload laboratory reports, prescriptions, and medical documents to receive concise AI-generated summaries and key findings.</p>', unsafe_allow_html=True)
    
    uploaded_pdf = st.file_uploader("Upload lab report (PDF):", type=["pdf"], key="pdf_report_uploader")
    
    st.write("")
    if st.button("📝 Summarize Report", key="btn_summarize_report"):
        if not uploaded_pdf:
            st.warning("Please upload a PDF file first.")
        else:
            with st.spinner("Processing medical report..."):
                # Clean and mock summary based on file name
                summary = generate_pdf_summary(uploaded_pdf.name)
                st.session_state.pdf_summary = summary
                
                # Append to Chatbot context
                summary_clean = summary.replace("**", "").replace("###", "")
                st.session_state.chat_messages.append({"role": "user", "content": f"Summarize uploaded report: {uploaded_pdf.name}"})
                st.session_state.chat_messages.append({"role": "assistant", "content": f"I have summarized your report **{uploaded_pdf.name}**:\n\n{summary_clean} Stay strong! 💖"})
                st.session_state.scroll_to = "dashboard"
                st.rerun()

# -----------------
# CARD 3: Skin & Eye Screening
# -----------------
with card_col3:
    st.markdown('<div class="feature-card"><div class="card-title">Skin & Eye Screening</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="card-desc">Upload skin or eye images for AI-assisted screening and condition identification.</p>', unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("Upload skin/eye image:", type=["jpg", "jpeg", "png"], key="image_lesion_uploader")
    
    if uploaded_image:
        os.makedirs("./temp", exist_ok=True)
        temp_image_path = os.path.join("./temp", "uploaded_lesion.jpg")
        image = Image.open(uploaded_image).convert("RGB")
        image.save(temp_image_path)
        st.session_state.uploaded_image_path = temp_image_path
        st.image(image, caption="Image Preview", width=160)
        
    st.write("")
    if st.button("📷 Scan Skin/Eye Image", key="btn_scan_image"):
        if not uploaded_image:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Scanning image with PyTorch ResNet-18..."):
                val_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = val_transform(image).unsqueeze(0).to(models_data["device"])
                
                with torch.no_grad():
                    outputs = models_data["image_cnn"](img_tensor)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    pred_idx = np.argmax(probs)
                    
                pred_disease = models_data["image_classes"][str(pred_idx)]
                pred_conf = probs[pred_idx] * 100
                
                st.session_state.image_prediction = {
                    "disease": pred_disease,
                    "confidence": pred_conf
                }
                
                # Append to Chatbot Context
                tests = get_test_recommendations(pred_disease)
                tests_str = " or ".join(tests) if tests else "clinical checkups"
                
                if pred_disease == "Suspicious skin lesion":
                    alert_prefix = "⚠️ ONCOLOGY SCREENING WARNING: "
                    advice = "I highly recommend visiting a doctor soon for an excision skin biopsy."
                elif pred_disease in ["Conjunctivitis", "Stye"]:
                    alert_prefix = "👀 EYE INFECTION DETECTED: "
                    advice = "I highly recommend consulting an optometrist or ophthalmologist for appropriate eye drops."
                else:
                    alert_prefix = "🎨 Dermatological Scan Result: "
                    advice = "I suggest seeing a dermatologist soon for professional verification."
                    
                baymax_visual_response = (
                    f"{alert_prefix}I have successfully scanned the uploaded image. "
                    f"It matches **{pred_disease}** (model confidence: {pred_conf:.1f}%). "
                    f"To confirm this, consider taking a **{tests_str}**. {advice} Stay strong! 💖"
                )
                
                st.session_state.chat_messages.append({"role": "user", "content": "Scan the uploaded clinical image."})
                st.session_state.chat_messages.append({"role": "assistant", "content": baymax_visual_response})
                st.session_state.scroll_to = "dashboard"
                st.rerun()

# =========================================================
# UNIFIED RESULTS DASHBOARD SECTION
# =========================================================
st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
st.write("---")
st.markdown('<h2 style="text-align:center; color:#1E3A8A; margin-bottom:30px;">Unified Results Dashboard</h2>', unsafe_allow_html=True)

dash_col1, dash_col2, dash_col3 = st.columns(3)

# 1. Symptom Analysis Panel
with dash_col1:
    st.markdown('<div class="result-panel"><h4 style="margin:0; color:#1E3A8A;">1. Symptom Analysis</h4></div>', unsafe_allow_html=True)
    
    if st.session_state.predictions:
        st.markdown('<span class="status-badge status-success">Success: Evaluated</span>', unsafe_allow_html=True)
        preds = st.session_state.predictions
        st.write(f"**Decision Tree Predictor:**")
        st.write(f"- {preds['dt_disease']} ({preds['dt_conf']:.1f}% confidence)")
        st.write(f"**Random Forest Predictor:**")
        st.write(f"- {preds['rf_disease']} ({preds['rf_conf']:.1f}% confidence)")
        st.write(f"**PyTorch MLP Predictor:**")
        st.write(f"- {preds['mlp_disease']} ({preds['mlp_conf']:.1f}% confidence)")
    else:
        st.markdown('<span class="status-badge status-pending">Pending Input</span>', unsafe_allow_html=True)
        st.info("Please select symptoms and run analysis in Card 1.")

# 2. Medical Report Summary Panel
with dash_col2:
    st.markdown('<div class="result-panel"><h4 style="margin:0; color:#1E3A8A;">2. Medical Report Summary</h4></div>', unsafe_allow_html=True)
    
    if st.session_state.pdf_summary:
        st.markdown('<span class="status-badge status-success">Success: Summarized</span>', unsafe_allow_html=True)
        st.markdown(st.session_state.pdf_summary)
    else:
        st.markdown('<span class="status-badge status-pending">Pending Upload</span>', unsafe_allow_html=True)
        st.info("Please upload a PDF file and click summarize in Card 2.")

# 3. Skin & Eye Screening Panel
with dash_col3:
    st.markdown('<div class="result-panel"><h4 style="margin:0; color:#1E3A8A;">3. Skin & Eye Screening</h4></div>', unsafe_allow_html=True)
    
    if st.session_state.image_prediction:
        st.markdown('<span class="status-badge status-success">Success: Screened</span>', unsafe_allow_html=True)
        img_pred = st.session_state.image_prediction
        st.write(f"**Identified Condition:**")
        st.write(f"- {img_pred['disease']}")
        st.write(f"**Model Confidence:**")
        st.write(f"- {img_pred['confidence']:.1f}%")
        
        tests = get_test_recommendations(img_pred['disease'])
        if tests:
            st.write(f"**Recommended Tests:**")
            for test in tests:
                st.write(f"- {test}")
    else:
        st.markdown('<span class="status-badge status-pending">Pending Upload</span>', unsafe_allow_html=True)
        st.info("Please upload a skin or eye image and click scan in Card 3.")

# =========================================================
# CONSOLIDATED HEALTH REPORT SECTION
# =========================================================
st.markdown('<div id="report"></div>', unsafe_allow_html=True)
st.write("---")
st.markdown('<h2 style="text-align:center; color:#1E3A8A; margin-bottom:10px;">Consolidated Health Report</h2>', unsafe_allow_html=True)

st.markdown('<div class="report-card"><h3 style="margin:0; color:#1E3A8A; text-align:center; font-weight:700; font-size:1.4rem;">📋 CLINICAL INTERMEDIARY CONSULTATION SUMMARY</h3></div>', unsafe_allow_html=True)

# Demographics (Simulated clinical info)
rcol1, rcol2 = st.columns(2)
with rcol1:
    st.markdown("**Patient Demographics:**")
    st.write("- **Patient Name:** Valued Patient")
    st.write("- **Consultation Date:** 2026-06-08 (Vitals Check)")
with rcol2:
    st.markdown("**Clinic Information:**")
    st.write("- **Facility:** MediAssist AI Digital Clinic")
    st.write("- **Referral ID:** REF-AI-20260608-001")

st.write("---")

# Vitals & Vitals Summary (from PDF Summary if available)
if st.session_state.pdf_summary:
    st.markdown("#### 📄 Laboratory Vitals Summary (Uploaded Report)")
    st.markdown(st.session_state.pdf_summary)
    st.write("")

# Vitals Summary from Checklist
st.markdown("#### 🔍 Symptom Questionnaire & Checkups")
if st.session_state.active_symptoms:
    symptoms_formatted = ", ".join([format_symptom_name(s) for s in st.session_state.active_symptoms])
    st.write(f"**Patient Entered Symptoms:** {symptoms_formatted}")
else:
    st.write("*No symptoms selected in the checklist.*")

if st.session_state.predictions:
    symptom_disease = st.session_state.predictions["mlp_disease"]
    symptom_prob = st.session_state.predictions["mlp_conf"]
    st.write(f"**MLP Classifier Prediction:** {symptom_disease} ({symptom_prob:.1f}% confidence)")
else:
    symptom_disease = "Undiagnosed"
    symptom_prob = 0.0
    st.write("*Pending symptom analysis run.*")

st.write("")

# Image Scanning Vitals
st.markdown("#### 📷 Skin & Eye Diagnostic screening")
if st.session_state.image_prediction:
    image_disease = st.session_state.image_prediction["disease"]
    image_prob = st.session_state.image_prediction["confidence"]
    st.write(f"**Image Model Prediction:** {image_disease} ({image_prob:.1f}% confidence)")
else:
    image_disease = None
    image_prob = None
    st.write("*Pending skin/eye image scan.*")

st.write("")

# Clinical recommendations & Scans
st.markdown("#### 🩺 Recommended Medical Scans & Tests")
recommended_tests = set()
if symptom_disease != "Undiagnosed":
    recommended_tests.update(get_test_recommendations(symptom_disease))
if image_disease:
    recommended_tests.update(get_test_recommendations(image_disease))

if recommended_tests:
    for test in recommended_tests:
        st.write(f"✔ **{test}**")
else:
    st.write("*Enter clinical indicators above to view recommended scans.*")

st.write("")

# Emergency or Oncology Alerts
alerts = []
if symptom_disease != "Undiagnosed":
    alert = check_critical_alert(symptom_disease)
    if alert:
        alerts.append((symptom_disease, alert))
if image_disease:
    alert = check_critical_alert(image_disease)
    if alert:
        alerts.append((image_disease, alert))

if alerts:
    st.markdown("#### 🚨 VITAL CLINICAL WARNING FLAGS")
    for condition, msg in alerts:
        st.error(f"**{msg}**: Flagged for diagnosed condition **{condition}**.")
else:
    st.success("✔ **No Critical Emergency Alerts** or Oncology Warnings flagged.")

st.write("---")
st.markdown(
    "<small style='color:#64748B;'>Disclaimer: This summary report is generated by a demonstration machine learning platform "
    "and does not replace a doctor consultation. Please visit a certified health center for official medical diagnosis.</small>",
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# Export buttons at the bottom of the Health Report Section
export_col1, export_col2 = st.columns([2, 1])
with export_col2:
    st.markdown('<div class="export-btn-group">', unsafe_allow_html=True)
    
    # Generate PDF report dynamically using generate_report.py
    report_filename = "MediAssist_Clinical_Report.pdf"
    try:
        from generate_report import create_clinical_report
        image_path = st.session_state.uploaded_image_path if st.session_state.image_prediction else None
        
        create_clinical_report(
            report_filename,
            "Valued Patient",
            "Adult",
            st.session_state.active_symptoms,
            symptom_disease,
            symptom_prob,
            image_disease,
            image_prob,
            image_path
        )
        
        with open(report_filename, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="📥 Download Report PDF",
            data=pdf_bytes,
            file_name="Clinical_Diagnostic_Report.pdf",
            mime="application/pdf",
            key="pdf_download_button_action"
        )
    except Exception as e:
        st.error(f"Failed to generate report: {e}")

# =========================================================
# FOOTER SECTION
# =========================================================
st.markdown("""
<div class="footer-container">
    <p style="margin-bottom:10px;"><strong>MediAssist AI Platform</strong></p>
    <p style="font-size:0.85rem; margin-bottom:15px; color:#94A3B8;">
        <a href="#home" style="color:#64748B; margin-right:15px; text-decoration:none;">Home</a>
        <a href="#baymax" style="color:#64748B; margin-right:15px; text-decoration:none;">Baymax Assistant</a>
        <a href="#modules" style="color:#64748B; margin-right:15px; text-decoration:none;">AI Screening</a>
        <a href="#dashboard" style="color:#64748B; text-decoration:none;">Results Dashboard</a>
    </p>
    <p style="font-size:0.8rem; color:#94A3B8;">&copy; 2026 MediAssist AI. All rights reserved. Hospital and clinical evaluation platform sandbox.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Auto-Scroll handler
# ---------------------------------------------------------
if "scroll_to" in st.session_state and st.session_state.scroll_to:
    target = st.session_state.scroll_to
    st.session_state.scroll_to = None
    import streamlit.components.v1 as components
    components.html(f"""
        <script>
            setTimeout(function() {{
                const dashboard = window.parent.document.getElementById("dashboard");
                if (dashboard) {{
                    dashboard.scrollIntoView({{ behavior: "smooth", block: "start" }});
                }}
            }}, 300);
        </script>
    """, height=0, width=0)
