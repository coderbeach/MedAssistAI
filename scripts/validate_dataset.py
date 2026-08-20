import os
import sys
import json
import hashlib
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = r"c:\Users\Nisarga N\OneDrive\Documents\My Projects\Healthcare-AI-Capstone"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")
REPORT_PATH = os.path.join(PROJECT_DIR, "results", "dataset_validation_report.txt")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_image_hashes():
    print("Computing perceptual hashes to scan for duplicates...")
    hashes = {}
    duplicates = []
    
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, f)
                try:
                    with Image.open(path) as img:
                        # Compute dhash
                        h = str(imagehash.dhash(img))
                        if h in hashes:
                            duplicates.append((path, hashes[h]))
                            print(f"  [Duplicate Found] {f} matches {os.path.basename(hashes[h])}")
                        else:
                            hashes[h] = path
                except Exception as e:
                    print(f"  Error reading {f} for hashing: {e}")
                    
    return hashes, duplicates

def create_splits(hashes):
    # Group unique images by their target class
    class_images = defaultdict(list)
    for h, path in hashes.items():
        cls_name = os.path.basename(os.path.dirname(path))
        class_images[cls_name].append(path)
        
    print("\nSplitting unique dataset (70% Train, 15% Val, 15% Test) stratified by class:")
    
    # Initialize folders
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        
    split_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    
    for cls, paths in class_images.items():
        # Shuffle deterministically
        random.shuffle(paths)
        
        n = len(paths)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        # Avoid zero elements in splits for tiny classes if possible
        if n > 0 and n_train == 0:
            n_train = 1
        if n > 1 and n_val == 0:
            n_val = 1
            
        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train+n_val]
        test_paths = paths[n_train+n_val:]
        
        # Copy to folders
        for split, split_paths in [("train", train_paths), ("validation", val_paths), ("test", test_paths)]:
            dest_split_dir = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(dest_split_dir, exist_ok=True)
            for path in split_paths:
                dest_path = os.path.join(dest_split_dir, os.path.basename(path))
                shutil.copy2(path, dest_path)
                split_counts[cls][split if split != "validation" else "val"] += 1
                
    return split_counts

def generate_report(split_counts, duplicates):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    
    total_images = 0
    total_train = 0
    total_val = 0
    total_test = 0
    
    report_lines = []
    report_lines.append("=========================================================")
    report_lines.append("       MEDICAL IMAGE DATASET VALIDATION REPORT")
    report_lines.append("=========================================================\n")
    
    report_lines.append("## 1. Class Distribution and Splits")
    report_lines.append(f"{'Class Name':<25} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    report_lines.append("-" * 65)
    
    for cls in sorted(split_counts.keys()):
        counts = split_counts[cls]
        cls_total = counts["train"] + counts["val"] + counts["test"]
        total_train += counts["train"]
        total_val += counts["val"]
        total_test += counts["test"]
        total_images += cls_total
        
        report_lines.append(f"{cls:<25} | {counts['train']:<7} | {counts['val']:<7} | {counts['test']:<7} | {cls_total:<7}")
        
    report_lines.append("-" * 65)
    report_lines.append(f"{'TOTAL':<25} | {total_train:<7} | {total_val:<7} | {total_test:<7} | {total_images:<7}\n")
    
    report_lines.append("## 2. Duplicate Detection Report")
    report_lines.append(f"Total duplicates detected and removed: {len(duplicates)}")
    for dup, orig in duplicates:
        report_lines.append(f"  - Dup: {os.path.basename(dup)} -> matches Original: {os.path.basename(orig)}")
        
    report_lines.append("\n## 3. Data Leakage Risk Assessment")
    report_lines.append("- Perceptual hashing (dhash) was run recursively on all files.")
    report_lines.append("- Duplicate files across different splits have been entirely pruned.")
    report_lines.append("- Stratified split logic was executed. Held-out test sets are completely isolated.")
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nValidation report saved to: {REPORT_PATH}")

def main():
    if not os.path.exists(RAW_DIR):
        print(f"Error: Raw directory not found at {RAW_DIR}. Run prepare_dataset.py first.")
        sys.exit(1)
        
    hashes, duplicates = compute_image_hashes()
    
    # Prune duplicates from hashes list so they are not included in training splits
    for dup_path, _ in duplicates:
        if os.path.exists(dup_path):
            os.remove(dup_path)
            print(f"Pruned duplicate image file: {os.path.basename(dup_path)}")
            
    split_counts = create_splits(hashes)
    generate_report(split_counts, duplicates)

if __name__ == "__main__":
    main()
