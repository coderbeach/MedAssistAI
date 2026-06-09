import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MedicalChatbot:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        self.tokenizer = None
        self.model = None
        
        # Load symptom mapping database
        with open("./models/symptom_features.json", "r") as f:
            self.all_symptoms = json.load(f)
            
        if self.use_llm:
            try:
                print(f"Loading conversational LLM: {MODEL_NAME} on {device}...")
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
        Maps natural language phrases to symptom keys in the 132 symptoms dataset.
        """
        mapping = {
            "itching": ["itch", "itching", "scratchy"],
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
            "red_sore_around_nose": ["red sores around nose", "crusty sores around mouth"]
        }
        return mapping
        
    def extract_symptoms(self, text):
        """
        Parses text input (user query or chatbot log) and returns a list of active symptoms.
        Uses LLM parsing if active, otherwise falls back to a regex keyword matcher.
        """
        text = text.lower()
        detected_symptoms = []
        
        # Rule-based regex extractor (highly robust)
        symptom_map = self.get_rule_based_symptom_map()
        for sym_key, phrases in symptom_map.items():
            for phrase in phrases:
                pattern = r'\b' + re.escape(phrase) + r'\b'
                if re.search(pattern, text):
                    detected_symptoms.append(sym_key)
                    break # Match found for this key, move to next symptom
                    
        # LLM based extraction (to demonstrate Day 7 capabilities)
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
                    # Validate against known keys
                    for s in llm_symptoms:
                        if s in self.all_symptoms and s not in detected_symptoms:
                            detected_symptoms.append(s)
            except Exception as e:
                print(f"LLM symptom extraction error: {e}")
                
        return list(set(detected_symptoms))

    def generate_chat_response(self, chat_history, user_message):
        """
        Generates a chatbot reply to hold the medical intake conversation.
        """
        if self.use_llm:
            # Format conversational prompt
            messages = [{"role": "system", "content": "You are a professional and empathetic healthcare assistant. Ask clarifying questions about their symptoms to help analyze them."}]
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})
            
            try:
                text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.7)
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return response.strip()
            except Exception as e:
                print(f"LLM chat response error: {e}")
                
        # Empathetic conversational script fallback (if LLM is disabled or fails)
        user_message_lower = user_message.lower()
        if any(w in user_message_lower for w in ["hello", "hi", "hey"]):
            return "Hello! I am your AI Clinical Assistant. What symptoms are you experiencing today?"
        elif any(w in user_message_lower for w in ["pain", "hurt", "ache"]):
            return "I am sorry to hear that you are in pain. Can you describe where the pain is located and if you have other symptoms like fever, nausea, or sweating?"
        elif any(w in user_message_lower for w in ["chest pain", "heart hurts", "stroke"]):
            return "⚠️ Chest pain or sudden numbness is a serious symptom. Please tell me if you have shortness of breath, left-arm pain, or dizziness, and consult emergency services immediately if severe."
        else:
            return "Thank you for sharing. I have noted these details. Feel free to list any other symptoms you have, or select them from the checklist, and we can generate your diagnosis and clinical report!"

if __name__ == "__main__":
    # Test symptom mapping
    # Create mock symptom files for stand-alone testing
    os.makedirs("./models", exist_ok=True)
    with open("./models/symptom_features.json", "w") as f:
        json.dump(["cough", "high_fever", "headache", "itching", "chest_pain"], f)
        
    bot = MedicalChatbot(use_llm=False)
    test_text = "I have a dry cough, bad headache and feel dizzy since yesterday."
    print("Detected symptoms:", bot.extract_symptoms(test_text))
