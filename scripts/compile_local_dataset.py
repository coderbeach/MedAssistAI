import os
import shutil
import hashlib

PROJECT_DIR = r"c:\Users\Nisarga N\OneDrive\Documents\My Projects\Healthcare-AI-Capstone"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(PROJECT_DIR, "dataset", "raw")
METADATA_PATH = os.path.join(PROJECT_DIR, "dataset", "metadata.json")

# Class mapping from local directories to our 13 target classes
SKIN_IMAGES_MAP = {
    "Acne_Vulgaris": "acne",
    "Atopic_Dermatitis": "eczema",
    "Psoriasis": "psoriasis",
    "Tinea_Corporis": "ringworm",
    "Vitiligo": "vitiligo",
    "Hordeolum": "stye",
    "Malignant_Melanoma": "suspicious_lesion",
    "Basal_Cell_Carcinoma": "suspicious_lesion",
    "Contact_Dermatitis": "dermatitis",
    "Seborrheic_Dermatitis": "dermatitis"
}

EXTERNAL_IMAGES_MAP = {
    "Acne": "acne",
    "Eczema": "eczema",
    "Psoriasis": "psoriasis",
    "Ringworm": "ringworm",
    "Vitiligo": "vitiligo",
    "Chickenpox_rash": "chickenpox",
    "Measles_rash": "measles",
    "Fungal_infection": "fungal_infection",
    "Dermatitis": "dermatitis",
    "Suspicious_skin_lesion": "suspicious_lesion",
    "Stye": "stye",
    "Conjunctivitis": "conjunctivitis",
    "Normal_Eye": "normal_eye"
}

def copy_files(src_dir, mapping, source_name):
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return
        
    copied_count = 0
    for folder_name, target_class in mapping.items():
        sub_folder = os.path.join(src_dir, folder_name)
        if not os.path.exists(sub_folder):
            continue
            
        target_dir = os.path.join(RAW_DIR, target_class)
        os.makedirs(target_dir, exist_ok=True)
        
        for file in os.listdir(sub_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                # Avoid duplicating files that have already been copied
                src_file_path = os.path.join(sub_folder, file)
                
                # Check file hash or rename to prevent namespace collision
                file_hash = hashlib.md5(f"{source_name}_{folder_name}_{file}".encode()).hexdigest()[:12]
                ext = os.path.splitext(file)[1]
                new_filename = f"{target_class}_{source_name}_{file_hash}{ext}"
                dest_file_path = os.path.join(target_dir, new_filename)
                
                shutil.copy2(src_file_path, dest_file_path)
                copied_count += 1
                
    print(f"Copied {copied_count} files from '{source_name}' ({os.path.basename(src_dir)}) to raw dataset.")

def main():
    print("=========================================================")
    print(" Aggregating Local & Pre-Downloaded Clinical Images...")
    print("=========================================================")
    
    # 1. Copy from skin_images splits
    copy_files(os.path.join(DATA_DIR, "skin_images", "train"), SKIN_IMAGES_MAP, "skin_images")
    copy_files(os.path.join(DATA_DIR, "skin_images", "val"), SKIN_IMAGES_MAP, "skin_images")
    
    # 2. Copy from external_images splits
    copy_files(os.path.join(DATA_DIR, "external_images", "train"), EXTERNAL_IMAGES_MAP, "external_images")
    copy_files(os.path.join(DATA_DIR, "external_images", "val"), EXTERNAL_IMAGES_MAP, "external_images")
    
    print("\nDataset aggregation completed successfully!")

if __name__ == "__main__":
    main()
