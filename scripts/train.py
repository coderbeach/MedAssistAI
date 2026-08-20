import os
import sys
import argparse
PROJECT_DIR = r"c:\Users\Nisarga N\OneDrive\Documents\My Projects\Healthcare-AI-Capstone"
sys.path.append(PROJECT_DIR)

from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Medical Image Classifier Training Script")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(PROJECT_DIR, "configs", "baseline_resnet.yaml"),
        help="Path to YAML training configuration"
    )
    args = parser.parse_args()
    
    # Define directories
    dataset_dir = os.path.join(PROJECT_DIR, "dataset")
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")
    output_dir = os.path.join(PROJECT_DIR, "models", "resnet50_run")
    
    # Run training (force CPU or auto-detect CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_model(
        config_path=args.config,
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=output_dir,
        device=device
    )

if __name__ == "__main__":
    import torch
    main()
