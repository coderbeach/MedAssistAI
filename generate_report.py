import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def get_test_recommendations(disease):
    """
    Returns specific clinical scans/tests for a diagnosed disease.
    """
    scans_map = {
        "Fungal Infection": ["Skin Scraping (KOH Prep)", "Fungal Culture"],
        "Allergy": ["Allergy Skin Prick Test", "Serum IgE Blood Test"],
        "Gastroesophageal Reflux Disease (GERD / Acid Reflux)": ["Upper Endoscopy (EGD)", "Esophageal pH Monitoring"],
        "Chronic Cholestasis": ["Liver Function Test (LFT)", "Abdominal Ultrasound", "MRCP Scan"],
        "Drug Reaction": ["Complete Blood Count (CBC)", "Skin Biopsy (if severe)"],
        "Peptic Ulcer Disease": ["H. pylori Urea Breath Test", "Upper Endoscopy (EGD)"],
        "AIDS / HIV": ["HIV ELISA & Western Blot Test", "CD4 Cell Count", "Viral Load Test"],
        "Diabetes Mellitus (Type 1 & Type 2)": ["HbA1c Blood Test", "Fasting Blood Sugar (FBS)", "Oral Glucose Tolerance Test"],
        "Gastroenteritis": ["Stool Culture", "Electrolyte Panel Test"],
        "Bronchial Asthma": ["Spirometry Lung Function Test", "Chest X-ray"],
        "Hypertension (High blood pressure)": ["Electrocardiogram (ECG)", "Basic Metabolic Panel"],
        "Migraine": ["Brain MRI (to rule out other causes)", "Neurological Exam"],
        "Cervical Spondylosis": ["Cervical Spine X-ray", "Cervical Spine MRI"],
        "Paralysis (Brain Hemorrhage / Stroke)": ["Brain CT Scan (Emergency)", "Brain MRI", "Carotid Ultrasound"],
        "Jaundice": ["Bilirubin Levels (Total/Direct)", "Liver Ultrasound", "Hepatitis Panel"],
        "Malaria": ["Thick and Thin Blood Smears", "Rapid Diagnostic Test (RDT)"],
        "Chickenpox": ["Tzanck Smear", "Varicella Zoster PCR"],
        "Dengue Fever": ["Dengue NS1 Antigen Test", "CBC (Platelet count monitoring)"],
        "Typhoid Fever": ["Blood Culture Test", "Widal Agglutination Test"],
        "Hepatitis A": ["Anti-HAV IgM Blood Test", "LFT"],
        "Hepatitis B": ["HBsAg Blood Test", "Anti-HBc IgM Test"],
        "Hepatitis C": ["Anti-HCV Antibody Test", "HCV RNA PCR Test"],
        "Hepatitis D": ["Anti-HDV Blood Test", "HDV RNA PCR"],
        "Hepatitis E": ["Anti-HEV IgM Blood Test", "LFT"],
        "Alcoholic Hepatitis": ["Liver Panel", "Prothrombin Time (PT/INR)", "Abdominal Ultrasound"],
        "Tuberculosis (TB)": ["Sputum Acid-Fast Bacilli (AFB) Smear", "Chest X-ray", "Mantoux Skin Test / IGRA"],
        "Common Cold": ["Clinical Assessment (No scans usually needed)"],
        "Pneumonia": ["Chest X-ray", "Sputum Culture", "Pulse Oximetry"],
        "Dimorphic Hemorrhoids (Piles)": ["Anoscopy", "Sigmoidoscopy"],
        "Heart Attack (Myocardial Infarction)": ["12-Lead ECG (Emergency)", "Serum Troponin Level Test (Serial)", "Coronary Angiography"],
        "Varicose Veins": ["Venous Duplex Ultrasound"],
        "Hypothyroidism": ["TSH (Thyroid Stimulating Hormone) Test", "Free T4 Test"],
        "Hyperthyroidism": ["TSH Test", "Free T3/T4 Test", "Radioactive Iodine Uptake Scan"],
        "Hypoglycemia": ["Fasting Blood Glucose Test", "Insulin Level Test"],
        "Osteoarthritis": ["Joint X-rays", "Joint Fluid Analysis"],
        "Rheumatoid Arthritis": ["Rheumatoid Factor (RF) Test", "Anti-CCP Antibody", "ESR and CRP Inflammation Markers"],
        "Paroxysmal Positional Vertigo (PPV)": ["Dix-Hallpike Maneuver Assessment", "Electronystagmography (ENG)"],
        "Acne Vulgaris": ["Clinical Evaluation"],
        "Urinary Tract Infection (UTI)": ["Urinalysis", "Urine Culture & Sensitivity"],
        "Psoriasis": ["Clinical Evaluation", "Skin Biopsy (rarely)"],
        "Impetigo": ["Wound Culture", "Clinical Assessment"],
        "Appendicitis": ["Abdominal Ultrasound (Emergency)", "Contrast-Enhanced Abdominal CT Scan", "CBC (checking white blood cells)"],
        "Chronic Obstructive Pulmonary Disease (COPD)": ["Spirometry", "Chest CT Scan", "Arterial Blood Gas (ABG)"],
        "COVID-19": ["SARS-CoV-2 RT-PCR Test", "Rapid Antigen Test", "Chest CT (if severe lung involvement)"],
        "Influenza": ["Rapid Influenza Diagnostic Test (RIDT)", "Flu PCR Test"],
        "Kidney Stones": ["Non-contrast CT Abdomen and Pelvis (Gold Standard)", "Renal Ultrasound", "Urinalysis"],
        "Gastrointestinal Bleeding": ["Upper Endoscopy", "Colonoscopy", "CBC (checking Hemoglobin/Hematocrit)"],
        "Iron Deficiency Anemia": ["Complete Blood Count (CBC)", "Serum Ferritin Test", "Iron Panel (TIBC, Serum Iron)"],
        "Transient Ischemic Attack (TIA)": ["Brain MRI/CT Scan", "Carotid Duplex Ultrasound", "ECG"],
        "Acute Gastritis": ["H. pylori Stool Antigen Test", "Upper Endoscopy"],
        "Acute Sinusitis": ["Clinical Evaluation", "Sinus CT Scan (only if chronic or complicated)"],
        "Malignant Melanoma": ["Dermoscopy Examination", "Excision Skin Biopsy (Critical)", "Sentinel Lymph Node Biopsy"],
        "Basal Cell Carcinoma": ["Skin Biopsy (Diagnostic)", "Shave or Punch Biopsy"],
        "Herpes Simplex Labialis": ["Viral Culture", "HSV-1 PCR Test"],
        "Alopecia Areata": ["Scalp Biopsy", "Thyroid & Autoimmune blood screens"],
        "Skin Cancer": ["Excision Skin Biopsy", "Dermoscopy", "Sentinel Lymph Node Biopsy"],
        "Breast Cancer": ["Diagnostic Mammogram", "Breast Ultrasound", "Core Needle Biopsy", "Breast MRI"]
    }
    
    return scans_map.get(disease, ["General Routine Blood Work", "Clinical Evaluation"])

def check_critical_alert(disease):
    """
    Checks if a diagnosed condition is a life-threatening critical alert or cancer.
    """
    critical_diseases = [
        "Heart Attack (Myocardial Infarction)", 
        "Paralysis (Brain Hemorrhage / Stroke)", 
        "Appendicitis", 
        "Transient Ischemic Attack (TIA)", 
        "Gastrointestinal Bleeding"
    ]
    cancer_diseases_skin = [
        "Malignant Melanoma", 
        "Basal Cell Carcinoma", 
        "Actinic Keratosis",
        "Skin Cancer"
    ]
    cancer_diseases_breast = [
        "Breast Cancer"
    ]
    
    if disease in critical_diseases:
        return "CRITICAL EMERGENCY ALERT"
    elif disease in cancer_diseases_skin:
        return "ONCOLOGY SCREENING WARNING (POSSIBLE SKIN CANCER)"
    elif disease in cancer_diseases_breast:
        return "ONCOLOGY SCREENING WARNING (POSSIBLE BREAST CANCER)"
    return None

def create_clinical_report(output_filename, patient_name, patient_age, symptoms_list, symptom_disease, symptom_prob, image_disease=None, image_prob=None, image_path=None):
    """
    Generates a beautifully structured PDF clinical consultation report.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    # Setup styles
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate blue
    warning_color = colors.HexColor("#C53030")   # Red
    dark_text = colors.HexColor("#2D3748")       # Dark charcoal
    light_bg = colors.HexColor("#EDF2F7")        # Light gray
    
    # Modify default styles in-place
    styles['Normal'].textColor = dark_text
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 14
    
    # Create unique ParagraphStyles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        alignment=0, # Left-aligned
        spaceAfter=15
    )
    
    header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    label_style = ParagraphStyle(
        'LabelText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=primary_color
    )
    
    alert_style = ParagraphStyle(
        'AlertText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=warning_color
    )
    
    # Title
    story.append(Paragraph("CLINICAL CONSULTATION & DIAGNOSTIC REPORT", title_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # Patient Demographics Table
    demo_data = [
        [Paragraph("Patient Name:", label_style), Paragraph(patient_name, styles['Normal']), 
         Paragraph("Age / Gender:", label_style), Paragraph(f"{patient_age} / Undefined", styles['Normal'])],
        [Paragraph("Report Date:", label_style), Paragraph("2026-06-08 (Consultation Date)", styles['Normal']),
         Paragraph("Report Reference:", label_style), Paragraph("REF-AI-20260608-001", styles['Normal'])]
    ]
    demo_table = Table(demo_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, primary_color),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 0.2 * inch))
    
    # ---------------------------------------------------------
    # Background Diagnostic Alerts
    # ---------------------------------------------------------
    alerts = []
    symptom_alert = check_critical_alert(symptom_disease)
    if symptom_alert:
        alerts.append((symptom_disease, symptom_alert))
    
    if image_disease:
        image_alert = check_critical_alert(image_disease)
        if image_alert:
            alerts.append((image_disease, image_alert))
            
    if alerts:
        story.append(Paragraph("BACKGROUND SCREENING & CLINICAL ALERTS", header_style))
        for condition, alert_msg in alerts:
            alert_box_data = [[
                Paragraph(f"⚠️ {alert_msg}: Flagged based on findings for {condition}.", alert_style)
            ]]
            alert_table = Table(alert_box_data, colWidths=[7.0*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFF5F5")),
                ('BOX', (0,0), (-1,-1), 1.5, warning_color),
                ('PADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(alert_table)
            story.append(Spacer(1, 0.1 * inch))
        story.append(Spacer(1, 0.1 * inch))
        
    # ---------------------------------------------------------
    # Symptom Analysis Section
    # ---------------------------------------------------------
    story.append(Paragraph("SYMPTOM ANALYSIS & TABULAR CLASSIFIER INFERENCE", header_style))
    
    # Format symptom list
    symptoms_text = ", ".join([s.replace("_", " ").title() for s in symptoms_list]) if symptoms_list else "No structured symptoms entered."
    
    symptom_table_data = [
        [Paragraph("Reported Symptoms:", label_style), Paragraph(symptoms_text, styles['Normal'])],
        [Paragraph("Primary Diagnosis prediction:", label_style), Paragraph(f"{symptom_disease} (Confidence: {symptom_prob:.2f}%)", styles['Normal'])],
        [Paragraph("Algorithm Model Used:", label_style), Paragraph("PyTorch Multi-Layer Perceptron (MLP) / Scikit-Learn Random Forest", styles['Normal'])]
    ]
    symptom_table = Table(symptom_table_data, colWidths=[2.2*inch, 4.8*inch])
    symptom_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(symptom_table)
    story.append(Spacer(1, 0.25 * inch))
    
    # ---------------------------------------------------------
    # Image Analysis Section (if applicable)
    # ---------------------------------------------------------
    if image_disease:
        story.append(Paragraph("EXTERNAL IMAGE CLASSIFICATION & DERMATOLOGY INFERENCE", header_style))
        
        image_table_data = [
            [Paragraph("Dermatological Finding:", label_style), Paragraph(f"{image_disease} (Confidence: {image_prob:.2f}%)", styles['Normal'])],
            [Paragraph("Scope Constraints:", label_style), Paragraph("External Surface Anatomical Analysis Only (Rashes, Acne, Lesions)", styles['Normal'])],
            [Paragraph("Algorithm Model Used:", label_style), Paragraph("PyTorch Convolutional Neural Network (Transfer Learning via ResNet-18)", styles['Normal'])]
        ]
        
        # If image path is provided and exists, we can embed it
        if image_path and os.path.exists(image_path):
            try:
                # Resize image for display in report
                img_widget = Image(image_path, width=1.8*inch, height=1.8*inch)
                image_row_data = [
                    [Paragraph("Uploaded Lesion Visual Scan:", label_style), img_widget],
                ]
                img_desc_table = Table(image_row_data, colWidths=[2.2*inch, 4.8*inch])
                img_desc_table.setStyle(TableStyle([
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('PADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(img_desc_table)
                story.append(Spacer(1, 0.05 * inch))
            except Exception as e:
                print(f"Error embedding image in PDF: {e}")
                
        image_table = Table(image_table_data, colWidths=[2.2*inch, 4.8*inch])
        image_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 0.25 * inch))
        
    # ---------------------------------------------------------
    # Scans and Diagnostic Scans Recommended Section
    # ---------------------------------------------------------
    story.append(Paragraph("RECOMMENDED CLINICAL SCANS & PROCEDURAL TESTS", header_style))
    story.append(Paragraph("Based on the AI model inference and clinical alert criteria, the following diagnostic scans and medical tests are recommended for absolute clinical verification:", styles['Normal']))
    story.append(Spacer(1, 0.05 * inch))
    
    recommended_scans = set()
    recommended_scans.update(get_test_recommendations(symptom_disease))
    if image_disease:
        recommended_scans.update(get_test_recommendations(image_disease))
        
    scan_bullets = []
    for scan in recommended_scans:
        scan_bullets.append([Paragraph("•", label_style), Paragraph(f"<b>{scan}</b>", styles['Normal'])])
        
    scan_table = Table(scan_bullets, colWidths=[0.3*inch, 6.7*inch])
    scan_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(scan_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # ---------------------------------------------------------
    # Clinical Disclaimer
    # ---------------------------------------------------------
    story.append(Spacer(1, 0.1 * inch))
    disclaimer_header = Paragraph("⚠️ IMPORTANT CLINICAL NOTICE & DISCLAIMER", ParagraphStyle('DiscHead', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor("#718096")))
    disclaimer_text = Paragraph(
        "This report is generated by a clinical demonstration machine learning system trained for academic and Capstone evaluation. "
        "It does NOT substitute for professional medical consultation, lab diagnostic testing, or clinical evaluations. "
        "If you are experiencing severe symptoms, immediate emergency room consultation or a visit to a registered medical practitioner is required. "
        "The diagnostic suggestions represent probabilistic inferences based on training datasets (DermNet, Kaushil268) and should not be used as clinical diagnosis.", 
        ParagraphStyle('DiscBody', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor("#718096"))
    )
    
    disc_table = Table([[disclaimer_header], [disclaimer_text]], colWidths=[7.0*inch])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(disc_table)
    
    # Build Document
    doc.build(story)
    print(f"Generated Clinical PDF Report at: {output_filename}")

if __name__ == "__main__":
    # Test generation
    create_clinical_report(
        "test_report.pdf", 
        "John Doe", 
        "45", 
        ["cough", "breathlessness", "fatigue"], 
        "Chronic Obstructive Pulmonary Disease (COPD)", 
        98.5, 
        "Atopic Dermatitis", 
        92.1
    )
