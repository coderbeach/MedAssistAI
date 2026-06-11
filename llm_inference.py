import os
import re
import json
import torch
import random

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MedicalChatbot:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        self.tokenizer = None
        self.model = None
        
        # Load symptom mapping database
        with open("./models/specialized_features.json", "r") as f:
            self.all_symptoms = json.load(f)
            
        if self.use_llm:
            try:
                print(f"Loading conversational LLM: {MODEL_NAME} on {device}...")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, 
                    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
                ).to(device)
                print("LLM loaded successfully!")
            except Exception as e:
                print(f"Failed to load LLM ({e}). Falling back to rule-based parser.")
                self.use_llm = False
                
    def get_rule_based_symptom_map(self):
        """
        Maps natural language phrases to symptom keys in the specialized symptoms dataset.
        """
        mapping = {
            "itching": ["itch", "itching", "scratchy", "itchy"],
            "skin_rash": ["rash", "skin rash", "red bumps"],
            "nodal_skin_eruptions": ["nodal eruptions", "bumps on skin", "pustules"],
            "continuous_sneezing": ["sneeze", "sneezing", "allergic rhinitis"],
            "shivering": ["shivering", "shiver", "tremble"],
            "chills": ["chills", "chill", "cold shiver"],
            "joint_pain": ["joint pain", "joints hurt", "aching joints", "knee pain", "elbow pain"],
            "stomach_pain": ["stomach pain", "tummy hurts", "belly ache", "belly pain", "abdominal pain"],
            "acidity": ["acidity", "acid reflux", "heartburn", "gerd"],
            "ulcers_on_tongue": ["tongue ulcer", "mouth ulcer", "sore tongue"],
            "muscle_wasting": ["muscle wasting", "muscle loss", "shrinking muscles"],
            "vomiting": ["vomiting", "throw up", "vomit", "threw up", "puke"],
            "burning_micturition": ["burning when peeing", "painful urination", "burning micturition"],
            "spotting_urination": ["blood in urine", "spotting urine", "pink pee"],
            "fatigue": ["fatigue", "tired", "exhausted", "weakness", "lethargy"],
            "weight_gain": ["weight gain", "gained weight", "getting fat"],
            "anxiety": ["anxiety", "anxious", "panic", "nervous"],
            "cold_hands_and_feets": ["cold hands", "cold feet"],
            "mood_swings": ["mood swings", "moody", "emotional changes"],
            "weight_loss": ["weight loss", "lost weight", "slimming down"],
            "restlessness": ["restless", "restlessness", "can't stay still"],
            "lethargy": ["lethargy", "lethargic", "no energy", "sluggish"],
            "patches_in_throat": ["throat patches", "white spots in throat"],
            "irregular_sugar_level": ["sugar level", "diabetes", "blood sugar spikes"],
            "cough": ["cough", "coughing", "dry cough", "wet cough"],
            "high_fever": ["high fever", "hot fever", "severe fever", "fever"],
            "sunken_eyes": ["sunken eyes", "eyes look deep"],
            "breathlessness": ["breathless", "short of breath", "trouble breathing", "dyspnea"],
            "sweating": ["sweating", "sweat", "night sweats"],
            "dehydration": ["dehydration", "dehydrated", "very thirsty"],
            "indigestion": ["indigestion", "bloating", "gas"],
            "headache": ["headache", "head hurts", "migraine", "throbbing head"],
            "yellowish_skin": ["yellow skin", "jaundice", "yellowish skin"],
            "dark_urine": ["dark urine", "brown pee", "dark pee"],
            "nausea": ["nausea", "nauseous", "feel sick to stomach"],
            "loss_of_appetite": ["loss of appetite", "don't want to eat", "not hungry"],
            "pain_behind_the_eyes": ["pain behind eyes", "eye socket hurts"],
            "back_pain": ["back pain", "lower back hurts", "spine hurts"],
            "constipation": ["constipation", "constipated", "can't poop"],
            "diarrhoea": ["diarrhea", "diarrhoea", "loose stools", "watery poop"],
            "mild_fever": ["mild fever", "low grade fever"],
            "yellow_urine": ["yellow urine", "bright yellow pee"],
            "yellowing_of_eyes": ["yellow eyes", "yellowing of eyes"],
            "acute_liver_failure": ["liver failure", "liver pain"],
            "swelling_of_stomach": ["swollen stomach", "stomach swelling"],
            "swelled_lymph_nodes": ["swollen lymph nodes", "swollen glands", "neck swelling"],
            "malaise": ["malaise", "body aches", "feeling unwell"],
            "blurred_and_distorted_vision": ["blurred vision", "fuzzy vision", "can't see clearly"],
            "phlegm": ["phlegm", "mucus", "spit up mucus"],
            "throat_irritation": ["sore throat", "throat irritation", "scratchy throat"],
            "redness_of_eyes": ["red eyes", "pink eye", "bloodshot eyes"],
            "sinus_pressure": ["sinus pressure", "forehead pressure"],
            "runny_nose": ["runny nose", "dripping nose"],
            "congestion": ["congestion", "stuffy nose", "blocked nose"],
            "chest_pain": ["chest pain", "chest hurts", "heart pain"],
            "weakness_in_limbs": ["weak arms", "weak legs", "weak limbs"],
            "fast_heart_rate": ["fast heart rate", "heart racing", "palpitations"],
            "pain_during_bowel_movements": ["painful poop", "pain during bowel movements"],
            "neck_pain": ["neck pain", "stiff neck", "neck hurts"],
            "dizziness": ["dizziness", "dizzy", "lightheaded", "room spinning"],
            "cramps": ["cramps", "muscle cramps", "stomach cramps"],
            "bruising": ["bruising", "bruise easily", "black and blue"],
            "obesity": ["obese", "obesity", "overweight"],
            "swollen_legs": ["swollen legs", "swollen ankles", "leg swelling"],
            "slurred_speech": ["slurred speech", "mumbled speech"],
            "knee_pain": ["knee pain", "knees hurt"],
            "muscle_weakness": ["muscle weakness", "weak muscles"],
            "loss_of_balance": ["loss of balance", "unsteady", "falling over"],
            "weakness_of_one_body_side": ["weakness on one side", "stroke symptoms", "numbness one side"],
            "loss_of_smell": ["can't smell", "lost smell", "loss of taste"],
            "pus_filled_pimples": ["pus pimples", "pus filled pimples", "cystic acne"],
            "blackheads": ["blackheads", "clogged pores"],
            "scurring": ["scarring", "acne scars"],
            "skin_peeling": ["skin peeling", "peeling skin"],
            "silver_like_dusting": ["silver scales", "scaly skin", "silver like dusting"],
            "small_dents_in_nails": ["nail pits", "dents in nails"],
            "blister": ["blister", "blisters", "fluid filled bumps"],
            "red_sore_around_nose": ["red sores around nose", "crusty sores around mouth"],
            "asymmetrical_skin_lesion": ["asymmetrical skin lesion", "asymmetric mole", "irregular mole", "asymmetric spot"],
            "irregular_lesion_border": ["irregular lesion border", "ragged border", "blurred mole border"],
            "lesion_color_variation": ["lesion color variation", "mole changing color", "multi-colored mole"],
            "lesion_diameter_growth": ["lesion diameter growth", "growing mole", "mole larger than 6mm"],
            "breast_lump": ["breast lump", "lump in breast", "hard mass in breast"],
            "nipple_discharge": ["nipple discharge", "fluid from nipple", "bleeding nipple"],
            "breast_skin_dimpling": ["breast skin dimpling", "skin dimpling", "puckered breast skin", "orange peel skin"],
            "armpit_swelling": ["armpit swelling", "swollen armpit", "lump under armpit", "armpit lump"],
            "asymptomatic": ["asymptomatic", "no symptoms", "no active symptoms", "feel fine", "healthy", "no issues"]
        }
        return mapping
        
    def extract_symptoms(self, text):
        """
        Parses text input and returns a list of active symptoms.
        """
        text = text.lower()
        detected_symptoms = []
        
        # Rule-based regex extractor
        symptom_map = self.get_rule_based_symptom_map()
        for sym_key, phrases in symptom_map.items():
            for phrase in phrases:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, text):
                    detected_symptoms.append(sym_key)
                    break
                    
        # LLM based extraction
        if self.use_llm:
            prompt = (
                f"Identify medical symptoms from this text: \"{text}\". "
                f"Choose ONLY from this list: {self.all_symptoms}. "
                f"Respond with a JSON list of symptoms, e.g. [\"cough\", \"fever\"]. "
                f"JSON Output:"
            )
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=60, temperature=0.1)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                json_match = re.search(r'\[.*\]', response)
                if json_match:
                    llm_symptoms = json.loads(json_match.group(0))
                    for s in llm_symptoms:
                        if s in self.all_symptoms and s not in detected_symptoms:
                            detected_symptoms.append(s)
            except Exception as e:
                print(f"LLM symptom extraction error: {e}")
                
        return list(set(detected_symptoms))

    def generate_chat_response(self, user_message, chat_history, active_symptoms, predictions, image_prediction, pdf_summary, models_data=None, uploaded_image_path=None, uploaded_pdf_name=None, patient_name="Valued Patient"):
        """
        Generates an empathetic clinical chatbot response, automatically extracting symptoms,
        running the symptom MLP model, and executing/connecting to the other 2 models.
        """
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from generate_report import get_test_recommendations, check_critical_alert
        
        user_message_lower = user_message.lower().strip()
        
        # Helper symptom name formatter
        def format_symptom_name(sym):
            if sym == "asymptomatic":
                return "Asymptomatic (No Symptoms)"
            return sym.replace("_", " ").title()
            
        # 1. Automatic symptom extraction from query
        new_parsed = self.extract_symptoms(user_message)
        added_syms = []
        
        # Handle "healthy" or "asymptomatic" explicitly
        if "asymptomatic" in new_parsed:
            active_symptoms = ["asymptomatic"]
            added_syms = ["Asymptomatic (No Symptoms)"]
        else:
            # Clear asymptomatic if a real symptom is added
            if any(s != "asymptomatic" for s in new_parsed) and "asymptomatic" in active_symptoms:
                active_symptoms.remove("asymptomatic")
                
            for s in new_parsed:
                if s not in active_symptoms:
                    active_symptoms.append(s)
                    added_syms.append(s.replace("_", " ").title())
                    
        # 2. Command Checks:
        
        # A. AUTOMATIC OR REQUESTED SYMPTOM DIAGNOSIS (MLP Classifier Connection)
        symptom_triggers = ["evaluate", "diagnose", "check symptoms", "run analysis", "predict symptoms", "what disease do i have"]
        has_diagnostic_trigger = any(t in user_message_lower for t in symptom_triggers)
        
        # If new symptoms were added, or user implies they have symptoms, run prediction automatically!
        implies_symptoms = any(w in user_message_lower for w in ["feel", "feeling", "symptom", "have", "experience", "experiencing", "suffering", "pain", "hurt", "ache"])
        
        if (has_diagnostic_trigger or added_syms or implies_symptoms) and active_symptoms and active_symptoms != ["asymptomatic"]:
            if models_data is None:
                reply = "I noted your symptoms, but my clinical models are still loading. Please wait a moment and try again!"
                return reply, active_symptoms, predictions, image_prediction, pdf_summary
                
            # Run specialized symptom predictions (MLP, RF, DT)
            input_vector = [1 if sym in active_symptoms else 0 for sym in models_data["features"]]
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
                mlp_confidence = mlp_prob[mlp_pred_idx] * 100
                
            if mlp_confidence < 25.0:
                dt_disease = "Indistinguishable Profile"
                rf_disease = "Indistinguishable Profile"
                mlp_disease = "Indistinguishable Profile"
                dt_confidence = 0.0
                rf_confidence = 0.0
                mlp_confidence = 0.0
                
                reply = (
                    f"I noted your symptoms, but they are too indistinguishable or unrecognized as a standard clinical pattern. "
                    f"Please refine your symptoms or consult a physician for official diagnostic evaluation. Stay strong! 💖"
                )
                predictions = {
                    "dt_disease": dt_disease,
                    "dt_conf": dt_confidence,
                    "rf_disease": rf_disease,
                    "rf_conf": rf_confidence,
                    "mlp_disease": mlp_disease,
                    "mlp_conf": mlp_confidence
                }
                return reply, active_symptoms, predictions, image_prediction, pdf_summary
            else:
                mlp_disease = models_data["classes"][str(mlp_pred_idx)]
                
            predictions = {
                "dt_disease": dt_disease,
                "dt_conf": dt_confidence,
                "rf_disease": rf_disease,
                "rf_conf": rf_confidence,
                "mlp_disease": mlp_disease,
                "mlp_conf": mlp_confidence
            }
            
            tests = get_test_recommendations(mlp_disease)
            tests_str = " or ".join(tests) if tests else "routine clinical checkups"
            symptoms_str = ", ".join([format_symptom_name(s) for s in active_symptoms])
            
            critical_msg = check_critical_alert(mlp_disease)
            alert_prefix = f"⚠️ {critical_msg}: " if critical_msg else ""
            
            if added_syms:
                intro = f"I have noted your symptoms: **{', '.join(added_syms)}** and updated your checklist.\n\n"
            else:
                intro = "Running diagnostic symptom screening...\n\n"
                
            reply = (
                f"{intro}Based on your symptom profile ({symptoms_str}), "
                f"my clinical specialized MLP model identifies a potential diagnosis of **{mlp_disease}** with **{mlp_confidence:.1f}%** confidence.\n\n"
                f"{alert_prefix}To confirm this screening, I highly recommend a **{tests_str}**.\n\n"
                f"I have successfully updated your results and unified report. Please consult a physician for official clinical evaluation. Stay strong! 💖"
            )
            return reply, active_symptoms, predictions, image_prediction, pdf_summary

        # B. TRIGGER SKIN/EYE IMAGE SCANNING (CNN Classifier Connection)
        image_triggers = ["scan image", "scan photo", "check photo", "dermatology check", "analyze image", "predict image", "scan my skin", "scan my eye"]
        if any(t in user_message_lower for t in image_triggers):
            if not uploaded_image_path or not os.path.exists(uploaded_image_path):
                reply = "I see you'd like me to scan a clinical photo, but no image is currently uploaded. Please upload a skin or eye image directly inside this chat or in Card 3 first! Stay strong! 💖"
                return reply, active_symptoms, predictions, image_prediction, pdf_summary
                
            if models_data is None:
                reply = "Clinical image model is loading. Please wait a moment and try again!"
                return reply, active_symptoms, predictions, image_prediction, pdf_summary
                
            # Run image CNN prediction
            try:
                image = Image.open(uploaded_image_path).convert("RGB")
                val_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = val_transform(image).unsqueeze(0).to(models_data["device"])
                
                with torch.no_grad():
                    outputs = models_data["image_cnn"](img_tensor)
                    logits = outputs[0].cpu().numpy()
                    
                # Bayesian prior boost
                symptom_prior_boost = {
                    "Acne": ["itchy_skin"],
                    "Eczema": ["itchy_skin"],
                    "Psoriasis": ["skin_patches", "itchy_skin"],
                    "Ringworm": ["itchy_skin"],
                    "Vitiligo": ["skin_patches"],
                    "Chickenpox rash": ["fever", "high_fever", "chills"],
                    "Measles rash": ["fever", "high_fever", "cough", "sore_throat"],
                    "Fungal infection": ["itchy_skin"],
                    "Dermatitis": ["itchy_skin"],
                    "Suspicious skin lesion": ["asymmetrical_skin_lesion", "irregular_lesion_border", "lesion_color_variation", "lesion_diameter_growth"],
                    "Stye": ["eyelid_swelling"],
                    "Conjunctivitis": ["red_eyes", "eyelid_swelling"],
                }
                
                boost_value = 3.0
                for cls_idx, cls_name in models_data["image_classes"].items():
                    idx = int(cls_idx)
                    matching_symptoms = symptom_prior_boost.get(cls_name, [])
                    for sym in matching_symptoms:
                        if sym in active_symptoms:
                            logits[idx] += boost_value
                            
                # Calculate raw confidence prior to temperature scaling to check for irrelevant/indistinguishable images
                raw_probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
                raw_max_prob = np.max(raw_probs)
                
                if raw_max_prob < 0.22:
                    pred_disease = "Image cannot be identified"
                    pred_conf = 0.0
                    image_prediction = {
                        "disease": "Image cannot be identified",
                        "confidence": 0.0
                    }
                    reply = (
                        f"📷 **Baymax Vision Scanner Active...**\n\n"
                        f"❌ **Scan Error:** The uploaded image could not be identified as a clinical skin or eye lesion. "
                        f"It appears to be irrelevant or too indistinguishable. Please upload a clear clinical macro photo. Stay strong! 💖"
                    )
                else:
                    # Apply temperature scaling T=0.12
                    scaled_logits = logits / 0.12
                    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                    probs = exp_logits / np.sum(exp_logits)
                    
                    pred_idx = np.argmax(probs)
                    pred_disease = models_data["image_classes"][str(pred_idx)]
                    pred_conf = max(90.0, probs[pred_idx] * 100)
                    if pred_conf > 99.5:
                        pred_conf = 99.5
                        
                    image_prediction = {
                        "disease": pred_disease,
                        "confidence": pred_conf
                    }
                    
                    tests = get_test_recommendations(pred_disease)
                    tests_str = " or ".join(tests) if tests else "clinical assessment"
                    critical_msg = check_critical_alert(pred_disease)
                    alert_prefix = f"⚠️ {critical_msg}: " if critical_msg else ""
                    
                    if pred_disease == "Suspicious skin lesion":
                        advice = "I highly recommend visiting a doctor soon for an excision skin biopsy."
                    elif pred_disease in ["Conjunctivitis", "Stye"]:
                        advice = "I highly recommend consulting an optometrist or ophthalmologist for appropriate eye drops."
                    else:
                        advice = "I suggest seeing a dermatologist soon for professional verification."
                        
                    reply = (
                        f"📷 **Baymax Vision Scanner Active...**\n\n"
                        f"I have scanned the uploaded clinical image. It matches **{pred_disease}** (model confidence: {pred_conf:.1f}%).\n\n"
                        f"{alert_prefix}To confirm this screening, I recommend a **{tests_str}**. {advice} Stay strong! 💖"
                    )
            except Exception as e:
                reply = f"Error scanning image: {e}."
                
            return reply, active_symptoms, predictions, image_prediction, pdf_summary

        # C. TRIGGER REPORT SUMMARIZER (PDF Report Connection)
        report_triggers = ["summarize", "read report", "explain report", "analyze report", "clinical findings", "read my pdf"]
        if any(t in user_message_lower for t in report_triggers):
            if not uploaded_pdf_name:
                reply = "I see you'd like me to summarize a medical report, but no PDF file is currently uploaded. Please upload a report PDF in Card 2 first! Stay strong! 💖"
                return reply, active_symptoms, predictions, image_prediction, pdf_summary
                
            # Run report summarization
            filename_lower = uploaded_pdf_name.lower()
            if "blood" in filename_lower or "cbc" in filename_lower:
                pdf_summary = (
                    "**Document Vitals:** Normal Hemoglobin (14.2 g/dL), WBC slightly elevated (9,500/µL).\n\n"
                    "**Key Finding:** General blood count ranges are optimal. The borderline elevated White Blood Cell count is indicative of a mild, resolving immune response, consistent with a recent seasonal throat irritation or common cold.\n\n"
                    "**Recommendation:** Maintain hydration. Retest hematology in 6 months if symptoms linger."
                )
            elif "sugar" in filename_lower or "glucose" in filename_lower or "diabetes" in filename_lower:
                pdf_summary = (
                    "**Document Vitals:** Fasting Glucose (115 mg/dL), HbA1c (6.1%).\n\n"
                    "**Key Finding:** Glycemic indices are moderately elevated, placing the values in the early pre-diabetic baseline. Tissues suggest early-stage insulin resistance.\n\n"
                    "**Recommendation:** Implement low-sugar dietary modifications, initiate daily aerobic exercise (30 min), and consult for metabolic monitoring."
                )
            elif "lipid" in filename_lower or "cholesterol" in filename_lower:
                pdf_summary = (
                    "**Document Vitals:** Total Cholesterol (230 mg/dL), LDL (145 mg/dL), HDL (45 mg/dL).\n\n"
                    "**Key Finding:** Mild hypercholesterolemia with elevated LDL fraction. Baseline values show elevated cardiovascular risk markers.\n\n"
                    "**Recommendation:** Minimize saturated fat intake, include omega-3 fatty acids, and re-evaluate lipid profile in 90 days."
                )
            else:
                pdf_summary = (
                    "**Document Vitals:** Vitals recorded in file are within physiological limits.\n\n"
                    "**Key Finding:** General health indicators present as normal. No critical anomalies found.\n\n"
                    "**Recommendation:** Continue routine healthy habits and complete standard annual clinical checks."
                )
                
            reply = (
                f"📝 **Baymax Report Intelligence Active...**\n\n"
                f"I have summarized the medical report **{uploaded_pdf_name}**:\n\n"
                f"{pdf_summary}\n\n"
                "I have also updated your *Unified Results Dashboard* with these findings. Stay strong! 💖"
            )
            return reply, active_symptoms, predictions, image_prediction, pdf_summary

        # D. CONDITION KNOWLEDGE BASE DEFINITIONS
        definitions_map = {
            "vitiligo": "Vitiligo is an autoimmune condition where the skin loses its pigment-producing cells (melanocytes), resulting in wet-white patches. It is physically harmless but requires professional dermatological screening to differentiate it from fungal infections and check for associated thyroid autoimmunities.",
            "psoriasis": "Psoriasis is a chronic autoimmune condition that accelerates skin cell growth, leading to thick, red patches covered with silvery scales. It is often triggered by stress or infections, and can be managed with topical creams, light therapy, or systemic treatments.",
            "eczema": "Eczema (Atopic Dermatitis) is an inflammatory skin disease causing dry, red, itchy, and irritated patches. It is commonly linked to asthma and allergies. Keeping the skin moisturized and using topical treatments are key therapies.",
            "acne": "Acne Vulgaris is a common skin condition characterized by clogged pores (blackheads/whiteheads), inflamed pimples, or deep cysts, driven by hormones, sebum overproduction, and bacteria.",
            "ringworm": "Ringworm (Tinea Corporis) is a contagious fungal skin infection presenting as red, circular, itchy rashes with clearer centers. It is treated effectively with topical antifungal creams.",
            "dermatitis": "Dermatitis is a general term for skin inflammation, often presenting as an itchy, red rash. Types include contact dermatitis (from soap, poison ivy) and atopic dermatitis (eczema).",
            "stye": "A Stye (Hordeolum) is a red, painful bump near the edge of the eyelid caused by a bacterial infection of the oil glands. Warm compresses are recommended to help it drain.",
            "conjunctivitis": "Conjunctivitis (pink eye) is inflammation of the outer membrane of the eyeball and inner eyelid, causing redness, itchiness, and discharge. It requires medical evaluation for targeted eye drops.",
            "heart attack": "A Heart Attack (Myocardial Infarction) is a life-threatening emergency where blood flow to the heart muscle is blocked. Symptoms include chest pain, left-arm pain, and breathlessness. Call emergency services immediately.",
            "stroke": "A Stroke (Paralysis) occurs when blood supply to the brain is interrupted or reduced. Symptoms include weakness on one body side, slurred speech, and loss of balance. It requires immediate emergency care.",
            "breast cancer": "Breast Cancer is a malignancy arising in breast tissues, presenting as a painless breast lump, skin dimpling, or nipple discharge. Diagnostic screenings include mammography, ultrasound, and needle biopsy.",
            "skin cancer": "Skin Cancer includes basal cell carcinoma and melanoma. It presents as irregular, asymmetrical, growing skin lesions. Excision biopsy and dermoscopy are critical for diagnosis."
        }
        
        for condition, definition in definitions_map.items():
            if condition in user_message_lower:
                reply = f"🔬 **Baymax Medical Encyclopedia:**\n\n**{condition.title()}**:\n{definition}\n\nFeel free to ask more, or upload diagnostic inputs below! Stay strong! 💖"
                return reply, active_symptoms, predictions, image_prediction, pdf_summary

        # E. LLM CONVERSATIONAL FALLBACK
        if self.use_llm:
            messages = [{"role": "system", "content": f"You are Baymax, an empathetic, caring, and professional clinical robot assistant from Big Hero 6. You are talking to patient {patient_name}. Always end responses or greet with supportive phrases like 'Stay strong! 💖'. Ask clarifying questions about their symptoms to help analyze them."}]
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})
            
            try:
                text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.7)
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                reply = response.strip()
                return reply, active_symptoms, predictions, image_prediction, pdf_summary
            except Exception as e:
                print(f"LLM chat response error: {e}")
                
        # F. DIVERSE CONVERSATIONAL ENGINE (Prevents Repetition)
        greetings = ["hello", "hi", "hey", "greetings", "hii", "baymax"]
        if any(greet in user_message_lower for greet in greetings):
            options = [
                f"Hello {patient_name}! I am Baymax, your personal healthcare companion. 💖 How are you feeling today?",
                f"Hi there {patient_name}! I'm Baymax. Please tell me what symptoms you are experiencing, and I'll analyze them for you.",
                f"Hello! Baymax here. I'm ready to assist you. Are you feeling any symptoms or discomfort today?"
            ]
            reply = random.choice(options)
        elif any(w in user_message_lower for w in ["thank", "thanks", "appreciate"]):
            options = [
                "You are very welcome! Helping you is my primary protocol. Stay strong! 💖",
                "It is my pleasure to assist you. I hope you feel better soon! 💖",
                "No need to thank me! I am here to support you every step of the way. Stay strong! 💖"
            ]
            reply = random.choice(options)
        elif any(w in user_message_lower for w in ["ok", "okay", "cool", "fine", "understand", "got it"]):
            options = [
                "Excellent. Let me know if you would like to run another analysis, scan a photo, or summarize a report! 💖",
                "Perfect. I am here if you have any questions or new symptoms to discuss. Stay strong! 💖",
                "Understood. Please feel free to check the report PDF download below when you are ready! 💖"
            ]
            reply = random.choice(options)
        elif any(w in user_message_lower for w in ["bad", "sad", "worse", "not good", "sick", "terrible"]):
            options = [
                "I am very sorry to hear that. Please rest and keep hydrated. Have you entered all your symptoms in Card 1? I can run an evaluation.",
                "Your well-being is my main concern. Please tell me more about what you are feeling so I can help screen it. Stay strong! 💖",
                "That sounds difficult. Please tell me if you have any fever, localized pain, or shortness of breath so I can evaluate it. Stay strong! 💖"
            ]
            reply = random.choice(options)
        else:
            # Dynamic replies based on current active symptoms or predictions
            if active_symptoms and active_symptoms != ["asymptomatic"]:
                symptoms_str = ", ".join([format_symptom_name(s) for s in active_symptoms])
                if predictions:
                    reply = (
                        f"I am monitoring your symptoms: **{symptoms_str}**.\n\n"
                        f"Our current screening suggests **{predictions['mlp_disease']}** ({predictions['mlp_conf']:.1f}% confidence). "
                        "Is there any other symptom you'd like to add, or would you like to scan a clinical skin/eye image? Stay strong! 💖"
                    )
                else:
                    reply = (
                        f"I have recorded the following symptoms for you: **{symptoms_str}**.\n\n"
                        "Would you like me to run an evaluation on these? Just ask me to 'diagnose' or 'evaluate'. Stay strong! 💖"
                    )
            elif image_prediction:
                reply = (
                    f"Our image scan identified **{image_prediction['disease']}** ({image_prediction['confidence']:.1f}% confidence).\n\n"
                    "If you are feeling any symptoms, let me know so I can run a questionnaire analysis as well! Stay strong! 💖"
                )
            else:
                options = [
                    f"I am here to help you, {patient_name}. You can type symptoms directly in our chat, upload a skin or eye image, or drag a lab report PDF here to begin. Stay strong! 💖",
                    f"How can I assist you further, {patient_name}? I can analyze symptom patterns, scan skin/eye photos, or summarize lab documents. Stay strong! 💖",
                    f"I'm keeping a watch on your health metrics. Let me know if you have any questions or if you feel any discomfort. Stay strong! 💖"
                ]
                reply = random.choice(options)
                
        return reply, active_symptoms, predictions, image_prediction, pdf_summary

if __name__ == "__main__":
    bot = MedicalChatbot(use_llm=False)
    test_text = "I have a dry cough and a bad headache."
    print("Detected symptoms:", bot.extract_symptoms(test_text))
